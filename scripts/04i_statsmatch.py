"""04i -- closed-form statistics matching as a TTA arm (DEV only).

Why this exists
---------------
`04f_grad_align.py` showed the gradient step has a mean cosine of about -0.04
with the supervised direction, while the per-image best step is worth +0.27 dB
on average.  A gradient step that has to be tuned per image to a factor of 100
in step length is a bad estimator: its variance, not its direction, is what
kills it.

A global colour/luminance cast is the dominant underwater failure, and it is a
LOW-DIMENSIONAL correction that can be solved in closed form instead of
descended into.  This script measures that arm honestly on DEV:

  identity      frozen model output, no correction              (reference)
  lab_match     per-image Lab mean/std matching to source clean  (closed form)
  lab_mean      per-image Lab mean matching only                 (closed form)
  rgb_match     per-image RGB mean/std matching                  (closed form)
  gamma         per-image grey-world gain                        (closed form)
  supervised    per-image least-squares affine to GT             (ceiling only)

The supervised arm is the label-using ceiling for the SAME parameterisation:
if `lab_match` is close to it, the closed-form estimate is doing the work; if
`supervised` is far above all of them, a global correction is the wrong family
regardless of how it is estimated.

Every arm is applied to the frozen source model's output, so nothing here
adapts network weights.  Results are DEV-only (spec §5.2/§11: no test set is
touched while choosing anything).

Usage:
  bash scripts/gpu_run.sh 04i_statsmatch.py --stream LSUI2UIEB --dev-domain target --n 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    adapter_ckpt_path, backbone_suffix, load_config, open_stream, resolve_ckpt, run_dir, save_json,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


# --------------------------------------------------------------------------- #
# colour space helpers (numpy, [0,1], HWC)
# --------------------------------------------------------------------------- #
def rgb_to_lab(im: np.ndarray) -> np.ndarray:
    """sRGB [0,1] HWC -> CIE Lab (D65).  Same maths as the differentiable version."""
    c = im.reshape(-1, 3)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
    xyz = c @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    t = xyz / white
    t = np.where(t > 0.008856, np.cbrt(t), (903.3 * t + 16.0) / 116.0)
    lab = np.stack([116.0 * t[:, 1] - 16.0,
                    500.0 * (t[:, 0] - t[:, 1]),
                    200.0 * (t[:, 1] - t[:, 2])], axis=1)
    return lab.reshape(im.shape)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    flat = lab.reshape(-1, 3)
    L, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], axis=1)
    t = np.where(f ** 3 > 0.008856, f ** 3, (116.0 * f - 16.0) / 903.3)
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = t * white
    minv = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]], dtype=np.float64)
    lin = xyz @ minv.T
    lin = np.clip(lin, 0.0, None)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).reshape(lab.shape)


def source_stats(split_dir: str, n: int = 200) -> dict:
    """Clean-source statistics that every arm is allowed to use (source data only)."""
    from brctta.data import list_stems, load_rgb, resize_eval

    stems = list_stems(split_dir)[:n]
    labs, rgbs = [], []
    for s in stems:
        g = os.path.join(split_dir, "GT", s)
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            if os.path.exists(g + ext):
                g = g + ext
                break
        im = np.asarray(resize_eval(load_rgb(g)), dtype=np.float64) / 255.0
        labs.append(rgb_to_lab(im).reshape(-1, 3))
        rgbs.append(im.reshape(-1, 3))
    lab = np.concatenate(labs, 0)
    rgb = np.concatenate(rgbs, 0)
    return {
        "lab_mean": lab.mean(0).tolist(), "lab_std": (lab.std(0) + 1e-6).tolist(),
        "rgb_mean": rgb.mean(0).tolist(), "rgb_std": (rgb.std(0) + 1e-6).tolist(),
        "n": len(stems),
    }


# --------------------------------------------------------------------------- #
# closed-form arms: each takes y and returns a corrected y
# --------------------------------------------------------------------------- #
def arm_identity(y, st, x=None):
    return y


def arm_lab_match(y, st, x=None):
    lab = rgb_to_lab(y)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    tgt_mu = np.asarray(st["lab_mean"])
    tgt_sd = np.asarray(st["lab_std"])
    # keep L free to move but cap the gain so a flat/degenerate image cannot be
    # stretched into noise; a and b get the full match (that is the colour cast)
    gain = np.clip(tgt_sd / sd, 0.5, 2.0)
    out = (lab.reshape(-1, 3) - mu) * gain + tgt_mu
    out[:, 0] = np.clip(out[:, 0], 0.0, 100.0)
    return lab_to_rgb(out.reshape(lab.shape))


def arm_lab_mean(y, st, x=None):
    lab = rgb_to_lab(y)
    mu = lab.reshape(-1, 3).mean(0)
    out = lab.reshape(-1, 3) - mu + np.asarray(st["lab_mean"])
    out[:, 0] = np.clip(out[:, 0], 0.0, 100.0)
    return lab_to_rgb(out.reshape(lab.shape))


def arm_rgb_match(y, st, x=None):
    mu, sd = y.reshape(-1, 3).mean(0), y.reshape(-1, 3).std(0) + 1e-6
    gain = np.clip(np.asarray(st["rgb_std"]) / sd, 0.5, 2.0)
    out = (y.reshape(-1, 3) - mu) * gain + np.asarray(st["rgb_mean"])
    return np.clip(out, 0.0, 1.0).reshape(y.shape)


def arm_gamma(y, st, x=None):
    mu = y.reshape(-1, 3).mean(0) + 1e-6
    gain = np.clip(np.asarray(st["rgb_mean"]) / mu, 0.6, 1.7)
    return np.clip(y * gain.reshape(1, 1, 3), 0.0, 1.0)


def arm_supervised(y, st, gt):
    """Least-squares per-channel affine to GT -- ceiling for this family only."""
    out = np.empty_like(y)
    for c in range(3):
        a = np.stack([y[..., c].ravel(), np.ones(y[..., c].size)], 1)
        coef, *_ = np.linalg.lstsq(a, gt[..., c].ravel(), rcond=None)
        out[..., c] = (a @ coef).reshape(y.shape[:2])
    return np.clip(out, 0.0, 1.0)


def _match_to(y, tgt_mean, tgt_std, gain_clip=(0.5, 2.0), blend=1.0):
    mu, sd = y.reshape(-1, 3).mean(0), y.reshape(-1, 3).std(0) + 1e-6
    gain = np.clip(np.asarray(tgt_std) / sd, *gain_clip)
    out = (y.reshape(-1, 3) - mu) * gain + np.asarray(tgt_mean)
    out = blend * out + (1.0 - blend) * y.reshape(-1, 3)
    return np.clip(out, 0.0, 1.0).reshape(y.shape)


def arm_input_rgb(y, st, x=None):
    """Match the output's RGB mean/std to the INPUT's -- same scene, no labels."""
    return _match_to(y, x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0))


def arm_input_rgb_half(y, st, x=None):
    return _match_to(y, x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0), blend=0.5)


def arm_input_mean(y, st, x=None):
    """Mean-only correction: removes the cast, leaves contrast to the model."""
    mu = y.reshape(-1, 3).mean(0)
    tgt = x.reshape(-1, 3).mean(0)
    return np.clip((y.reshape(-1, 3) - mu + tgt), 0.0, 1.0).reshape(y.shape)


def arm_input_lab(y, st, x=None):
    lab_y = rgb_to_lab(y)
    lab_x = rgb_to_lab(x)
    out = _match_to(lab_y, lab_x.reshape(-1, 3).mean(0), lab_x.reshape(-1, 3).std(0),
                    gain_clip=(0.5, 2.0))
    lab = np.empty_like(lab_y)
    lab[...] = out
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    return lab_to_rgb(lab)

ARMS = {
    "identity": arm_identity,
    "input_rgb": arm_input_rgb,
    "input_rgb_half": arm_input_rgb_half,
    "input_mean": arm_input_mean,
    "input_lab": arm_input_lab,
    "lab_match": arm_lab_match,
    "lab_mean": arm_lab_mean,
    "rgb_match": arm_rgb_match,
    "gamma": arm_gamma,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--dev-domain", default="target")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--stats-n", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dev_ds = sc["source"] if args.dev_domain == "source" else sc["target"]
    dev_rel = cfg["splits"][dev_ds][args.split]
    dev_dir = os.path.join(cfg["paths"]["data_root"], dev_rel)
    src_dir = os.path.join(cfg["paths"]["data_root"], cfg["splits"][sc["source"]]["train"])

    st = source_stats(src_dir, n=args.stats_n)
    spec = StreamSpec(name=f"{args.stream}_statsmatch_{dev_ds}", source_dataset=dev_ds,
                      target_split=args.split, seed=0, order="shuffle",
                      eval_size=cfg["eval"]["size"])
    stream = open_stream(cfg, dev_ds, args.split, spec)

    adapter = adapter_ckpt_path(cfg, sc["source"], args.backbone)
    model = build_adapted(resolve_ckpt(cfg, sc, args.backbone), device=device,
                          hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                          adapter_ckpt=adapter)
    model.eval()

    names = list(ARMS) + ["supervised", "input_vs_gt_psnr"]
    acc = {k: [] for k in names}
    n = min(args.n, len(stream)) if args.n else len(stream)
    with torch.no_grad():
        for i in range(n):
            x, gt, stem = stream.take(i)
            y = model.predict(x.unsqueeze(0).to(device))[0].detach().cpu().float()
            y = y.permute(1, 2, 0).numpy().astype(np.float64)
            g = gt.permute(1, 2, 0).numpy().astype(np.float64)
            xn = x.permute(1, 2, 0).numpy().astype(np.float64)
            for k, fn in ARMS.items():
                out = np.ascontiguousarray(fn(y, st, xn)).transpose(2, 0, 1)
                acc[k].append(M.per_image(torch.from_numpy(out.astype(np.float32)), gt))
            sup = np.ascontiguousarray(arm_supervised(y, st, g)).transpose(2, 0, 1)
            acc.setdefault("input_vs_gt_psnr", []).append(
                M.per_image(x, gt))
            acc["supervised"].append(M.per_image(torch.from_numpy(sup.astype(np.float32)), gt))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}]", flush=True)

    base = np.array([r["psnr"] for r in acc["identity"]])
    rows = {}
    for k, v in acc.items():
        p = np.array([r["psnr"] for r in v])
        d = p - base
        rows[k] = {
            "psnr": float(p.mean()), "ssim": float(np.mean([r["ssim"] for r in v])),
            "dPSNR_mean": float(d.mean()), "dPSNR_median": float(np.median(d)),
            "NAR_0": float((d < 0).mean()), "worst10_dPSNR": float(np.sort(d)[:max(1, len(d) // 10)].mean()),
            "p90_dPSNR": float(np.percentile(d, 90)),
        }
    res = {
        "stream": args.stream, "dev": f"{dev_ds}/{args.split}", "n": n,
        "backbone": args.backbone, "theta0": "source_adapter" if adapter else "frozen_backbone",
        "source_stats_from": src_dir, "stats": st, "arms": rows,
        "note": ("Closed-form, per-image, zero-parameter-learning correction applied to the frozen "
                 "model output. `supervised` is the label-using ceiling for the same affine family and "
                 "is never a method."),
    }
    out = os.path.join(run_dir(cfg, "statsmatch", f"{args.stream}{backbone_suffix(args.backbone)}"),
                       "statsmatch.json")
    save_json(res, out)
    print(json.dumps(rows, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can a source-fitted degradation model recover the PER-IMAGE colour correction?

Why this experiment
-------------------
`scripts/_diag_affine.py` measured the gap precisely: on LSUI->UIEB, a single
dataset-level affine buys +0.17 dB, but a **per-image** affine buys +6.865 dB.
Meanwhile every statistics prior (`lab_match`, `rgb_match`, `gamma`, source
prior matching) LOSES PSNR, and the residual adapter's gateable headroom is
~+0.05 dB.  So the label-free signal that is actually missing is *which
per-image correction this particular scene needs* -- not a global colour cast.

The degradation model `D: clean -> degraded` (fitted on SOURCE pairs only) is a
candidate source of exactly that signal.  If `y` is the correct clean image of
the scene, then re-degrading it should reproduce the camera's observation `x`:

    a*, b* = argmin_{a,b}  || D(a*y + b) - x ||_1  +  lam * ||(a,b)-(1,0)||^2

This is entirely label-free (`x` is the input, `D` is source-trained) and it is
per-image, so it has the right degree of freedom.  The question this script
answers on DEV only:

  1. Is the residual actually lower at the TRUE clean image than at `y`?
     (`resid(gt) < resid(y)`)  -- if not, `D` is wrong and the idea is dead.
  2. Does minimising the residual select an affine that raises PSNR?
  3. How much of the +6.9 dB oracle affine ceiling does it recover?

Arms
----
  identity      : y as-is
  oracle        : per-image affine fitted to GT (ceiling, label-using, never a method)
  redegrade     : per-image affine fitted by minimising the re-degradation residual
  redegrade_reg : same, with a strong prior pulling (a,b) to (1,0)
  stats         : affine matching source clean-image statistics (the failed baseline)

Everything is dev-split only.  The locked test split is untouched.

Usage
-----
    bash scripts/gpu_run.sh 04j_affine_redegrade.py --stream LSUI2UIEB --n 40
    bash scripts/gpu_run.sh 04j_affine_redegrade.py --stream UIEB2LSUI --n 40
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from brctta.data import StreamSpec  # noqa: E402
from brctta.degrade import Degrader  # noqa: E402
from brctta.metrics import psnr, ssim  # noqa: E402
from brctta.model import build_adapted  # noqa: E402

from common import (adapter_ckpt_path, load_config, open_stream, resolve_ckpt,  # noqa: E402
                    run_dir, save_json, split_dir, backbone_suffix)


def load_degrader(cfg, source, device, tag=""):
    """Load the source-fitted degradation model, or None if it was never trained."""
    path = os.path.join(cfg["paths"]["out_root"], f"degrader_{source}{tag}", "degrader.pth")
    if not os.path.exists(path):
        return None, path
    payload = torch.load(path, map_location="cpu")
    net = Degrader(width=int(payload.get("width", 32)))
    net.load_state_dict(payload["degrader"])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, path


def fit_affine_redegrade(degrader, y, x, lam=0.0, steps=40, lr=0.05):
    """Per-image (scale, offset) minimising ||D(a*y+b) - x||_1 (+ lam prior to identity).

    Parameterised per channel (3 scales + 3 offsets = 6 numbers), initialised at
    the identity so the arm degrades gracefully to `identity` if the gradient is
    uninformative.  Returns (a, b) as tensors of shape (1,3,1,1).
    """
    a = torch.ones(1, 3, 1, 1, device=y.device, requires_grad=True)
    b = torch.zeros(1, 3, 1, 1, device=y.device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        z = torch.clamp(y * a + b, 0.0, 1.0)
        loss = (degrader(z) - x).abs().mean()
        if lam > 0:
            loss = loss + lam * ((a - 1.0) ** 2).mean() + lam * (b ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return a.detach(), b.detach()


def fit_affine_oracle(y, gt):
    """Least-squares per-channel affine y -> gt (label-using ceiling)."""
    a = torch.ones(1, 3, 1, 1, device=y.device)
    b = torch.zeros(1, 3, 1, 1, device=y.device)
    ys = y[0].reshape(3, -1)
    gs = gt[0].reshape(3, -1)
    for c in range(3):
        yc = ys[c] - ys[c].mean()
        gc = gs[c] - gs[c].mean()
        denom = (yc * yc).sum().clamp_min(1e-8)
        a[0, c, 0, 0] = (yc * gc).sum() / denom
        b[0, c, 0, 0] = gs[c].mean() - a[0, c, 0, 0] * ys[c].mean()
    return a, b


def apply_ab(y, a, b):
    return torch.clamp(y * a + b, 0.0, 1.0)


def resid(degrader, z, x):
    with torch.no_grad():
        return float((degrader(z) - x).abs().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-adapter", action="store_true",
                    help="apply the source-pretrained adapter before the affine (theta_0 = source_adapter)")
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    source = stream_cfg["source"]
    target_ds = stream_cfg["target"]

    degrader, dpath = load_degrader(cfg, source, device)
    if degrader is None:
        print(f"[fatal] no degradation model at {dpath}")
        print("        run: bash scripts/gpu_run.sh 00e_degrade_train.py --source " + source)
        return 2
    print(f"[affine] degrader={dpath} source={source} target={target_ds} device={device}")

    ack = adapter_ckpt_path(cfg, source, args.backbone) if args.use_adapter else None
    model = build_adapted(resolve_ckpt(cfg, stream_cfg, args.backbone), device=device,
                          backbone=args.backbone, adapter_ckpt=ack)
    print(f"[affine] theta_0 = {'source_adapter' if ack else 'frozen_backbone'}")

    spec = StreamSpec(name=f"{args.stream}_affine", source_dataset=source,
                      target_split=f"{target_ds}/dev", seed=args.seed, order="shuffle")
    stream = open_stream(cfg, target_ds, "dev", spec)
    n = min(args.n, len(stream))

    rows = []
    for i in range(n):
        x_t, gt_t, stem = stream.take(i)
        x = x_t.unsqueeze(0).to(device)
        gt = gt_t.unsqueeze(0).to(device)
        with torch.no_grad():
            y = model.predict(x)

        row = {"i": i, "stem": stem,
               "psnr_identity": psnr(y, gt), "ssim_identity": ssim(y, gt),
               "resid_y": resid(degrader, y, x), "resid_gt": resid(degrader, gt, x)}

        a_o, b_o = fit_affine_oracle(y, gt)
        y_o = apply_ab(y, a_o, b_o)
        row["psnr_oracle"] = psnr(y_o, gt)
        row["a_oracle"] = a_o.flatten().tolist()
        row["b_oracle"] = b_o.flatten().tolist()

        for tag, lam in (("redegrade", 0.0), ("redegrade_reg", 0.30)):
            a_f, b_f = fit_affine_redegrade(degrader, y, x, lam=lam, steps=args.steps, lr=args.lr)
            y_f = apply_ab(y, a_f, b_f)
            row[f"psnr_{tag}"] = psnr(y_f, gt)
            row[f"ssim_{tag}"] = ssim(y_f, gt)
            row[f"resid_out_{tag}"] = resid(degrader, y_f, x)
            row[f"a_{tag}"] = a_f.flatten().tolist()
            row[f"b_{tag}"] = b_f.flatten().tolist()

        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}  id {row['psnr_identity']:.3f}  "
                  f"oracle {row['psnr_oracle']:.3f}  redegrade {row['psnr_redegrade']:.3f}")

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        d = v - np.array([r["psnr_identity"] for r in rows], dtype=np.float64)
        return {"key": key, "psnr": float(v.mean()), "dPSNR": float(d.mean()),
                "median_dPSNR": float(np.median(d)),
                "nar0": float((d < 0).mean()),
                "worst10": float(np.mean(np.sort(d)[:max(1, len(d) // 10)])),
                "frac_pos": float((d > 0).mean())}

    summary = {"stream": args.stream, "backbone": args.backbone, "n": n,
               "theta0": "source_adapter" if ack else "frozen_backbone",
               "degrader": dpath, "steps": args.steps, "lr": args.lr,
               "arms": {k: agg(k) for k in ("psnr_identity", "psnr_oracle",
                                            "psnr_redegrade", "psnr_redegrade_reg")},
               "resid": {"y_mean": float(np.mean([r["resid_y"] for r in rows])),
                         "gt_mean": float(np.mean([r["resid_gt"] for r in rows])),
                         "gt_lower_frac": float(np.mean([r["resid_gt"] < r["resid_y"] for r in rows]))}}

    y_res = np.array([r["resid_y"] for r in rows])
    y_out = np.array([r["resid_out_redegrade"] for r in rows])
    d_psnr = np.array([r["psnr_redegrade"] - r["psnr_identity"] for r in rows])
    if len(rows) > 3 and y_res.std() > 1e-9 and d_psnr.std() > 1e-9:
        summary["corr_resid_reduction_vs_dPSNR"] = float(np.corrcoef(y_res - y_out, d_psnr)[0, 1])

    out = run_dir(cfg, "affine_redegrade", f"{args.stream}{backbone_suffix(args.backbone)}")
    save_json(summary, os.path.join(out, "affine_redegrade.json"))
    save_json(rows, os.path.join(out, "rows.json"))

    print("\n=== per-image affine on DEV ===")
    for k in ("psnr_identity", "psnr_oracle", "psnr_redegrade", "psnr_redegrade_reg"):
        s = summary["arms"][k]
        print(f"  {k:16s} psnr {s['psnr']:.4f}  dPSNR {s['dPSNR']:+.4f}  "
              f"med {s['median_dPSNR']:+.4f}  NAR0 {s['nar0']:.3f}  worst10 {s['worst10']:+.4f}")
    print(f"  residual at y   {summary['resid']['y_mean']:.6f}")
    print(f"  residual at gt  {summary['resid']['gt_mean']:.6f}")
    print(f"  gt residual lower than y in {summary['resid']['gt_lower_frac'] * 100:.1f}% of images")
    if "corr_resid_reduction_vs_dPSNR" in summary:
        print(f"  corr(residual reduction, dPSNR) = {summary['corr_resid_reduction_vs_dPSNR']:+.4f}")
    print(f"[affine] wrote {out}/affine_redegrade.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

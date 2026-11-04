"""_diag_affine.py -- verify the supervised-affine ceiling and characterise the
target-domain tone gap.  DEV ONLY, diagnostic, not part of the pipeline.

Questions this answers, in order:
 1. Is `arm_supervised` (+7.3 dB on LSUI2UIEB/UIEB-dev) real, or a numpy/
    channel-order artefact?  Recompute it two independent ways.
 2. How much of it is a *global* (dataset-level) affine rather than per-image?
    A per-image affine is a 6-parameter oracle; a dataset-level affine is what a
    deployable calibrator could hope to reach.
 3. Do target GT statistics actually differ from source GT statistics?  If they
    do, "match to source prior" is not merely hard, it is wrong -- which would
    explain why every prior-matching arm lost PSNR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import adapter_ckpt_path, load_config, open_stream, resolve_ckpt, save_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def fit_affine(y: np.ndarray, ref: np.ndarray):
    """Per-channel least squares y -> ref.  Returns (matrix 3x3, offset 3)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in range(3):
        X = np.stack([y[..., c].ravel(), np.ones(y[..., c].size)], 1)
        coef, *_ = np.linalg.lstsq(X, ref[..., c].ravel(), rcond=None)
        A[c, c] = coef[0]
        b[c] = coef[1]
    return A, b


def apply_affine(y: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    flat = y.reshape(-1, 3) @ A.T + b
    return np.clip(flat, 0.0, 1.0).reshape(y.shape)


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    m = float(np.mean((a - b) ** 2))
    return 99.0 if m <= 1e-12 else float(10 * np.log10(1.0 / m))


def stats_of(imgs):
    st = []
    for im in imgs:
        flat = im.reshape(-1, 3)
        st.append(np.concatenate([flat.mean(0), flat.std(0)]))
    st = np.stack(st)
    return {"rgb_mean": st[:, :3].mean(0).tolist(), "rgb_std": st[:, 3:].mean(0).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    ap.add_argument("--splits", default="dev,train")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = adapter_ckpt_path(cfg, sc["source"], args.backbone)
    model = build_adapted(resolve_ckpt(cfg, sc, args.backbone), device=device,
                          hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                          adapter_ckpt=adapter)
    model.eval()

    out = {}
    for split in [s for s in args.splits.split(",") if s]:
        tgt_ds, out_split = (sc["target"], split) if split != "train" else (sc["target"], "train")
        spec = StreamSpec(name=f"diag_{args.stream}_{tgt_ds}_{out_split}",
                          source_dataset=tgt_ds, target_split=out_split, seed=0,
                          order="shuffle", eval_size=cfg["eval"]["size"])
        stream = open_stream(cfg, tgt_ds, out_split, spec)
        n = min(args.n, len(stream))
        ys, gs, per_psnr = [], [], []
        with torch.no_grad():
            for i in range(n):
                x, gt, stem = stream.take(i)
                y = model.predict(x.unsqueeze(0).to(device))[0].detach().cpu().float()
                yn = y.permute(1, 2, 0).numpy().astype(np.float64)
                gn = gt.permute(1, 2, 0).numpy().astype(np.float64)
                ys.append(yn)
                gs.append(gn)
                per_psnr.append(psnr_np(yn, gn))

        # (1) identity cross-check: numpy vs the package metric
        cross = max(abs(psnr_np(ys[i], gs[i]) - M.psnr(
            torch.from_numpy(np.ascontiguousarray(ys[i]).transpose(2, 0, 1).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(gs[i]).transpose(2, 0, 1).astype(np.float32))))
            for i in range(min(4, n)))

        # (2) per-image affine (oracle) vs dataset-level affine (one map for all)
        per_img = [psnr_np(apply_affine(*[ys[i], *fit_affine(ys[i], gs[i])]), gs[i]) for i in range(n)]
        gains = np.array([fit_affine(ys[i], gs[i])[0][c, c] for i in range(n) for c in range(3)])
        offs = np.array([fit_affine(ys[i], gs[i])[1][c] for i in range(n) for c in range(3)])
        # dataset-level map = mean of the per-image diagonal fits (one map for all)
        A = np.diag(np.array([np.mean([fit_affine(ys[i], gs[i])[0][c, c] for i in range(n)])
                              for c in range(3)]))
        b = np.array([np.mean([fit_affine(ys[i], gs[i])[1][c] for i in range(n)]) for c in range(3)])
        dataset_level = [psnr_np(apply_affine(ys[i], A, b), gs[i]) for i in range(n)]

        out[out_split] = {
            "target_dataset": tgt_ds, "n": n,
            "identity_psnr_mean": float(np.mean(per_psnr)),
            "identity_psnr_p10_p90": [float(np.percentile(per_psnr, 10)), float(np.percentile(per_psnr, 90))],
            "numpy_vs_package_max_abs_diff": float(cross),
            "per_image_affine_psnr_mean": float(np.mean(per_img)),
            "per_image_affine_gain_dB": float(np.mean(per_img) - np.mean(per_psnr)),
            "dataset_affine_psnr_mean": float(np.mean(dataset_level)),
            "dataset_affine_gain_dB": float(np.mean(dataset_level) - np.mean(per_psnr)),
            "dataset_affine_A": A.diagonal().tolist(), "dataset_affine_b": b.tolist(),
            "per_image_scale_mean": float(gains.mean()), "per_image_scale_sd": float(gains.std()),
            "per_image_offset_mean": float(offs.mean()), "per_image_offset_sd": float(offs.std()),
            "y_stats": stats_of(ys), "gt_stats": stats_of(gs),
        }
        print(f"[{out_split}] identity {out[out_split]['identity_psnr_mean']:.3f} | "
              f"per-image affine {out[out_split]['per_image_affine_psnr_mean']:.3f} "
              f"({out[out_split]['per_image_affine_gain_dB']:+.3f}) | "
              f"dataset affine {out[out_split]['dataset_affine_psnr_mean']:.3f} "
              f"({out[out_split]['dataset_affine_gain_dB']:+.3f}) | "
              f"numpy-vs-package {cross:.2e}", flush=True)

    # source clean statistics for the same comparison
    src_dir = os.path.join(cfg["paths"]["data_root"], cfg["splits"][sc["source"]]["train"])
    from brctta.data import list_stems, load_rgb, resize_eval
    imgs = []
    for s in list_stems(src_dir)[:args.n]:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = os.path.join(src_dir, "GT", s + ext)
            if os.path.exists(p):
                imgs.append(np.asarray(resize_eval(load_rgb(p)), dtype=np.float64) / 255.0)
                break
    out["source_gt_stats"] = stats_of(imgs)
    out["source_gt_stats"]["n"] = len(imgs)
    print("source GT stats:", json.dumps(out["source_gt_stats"]))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "manifests",
                        f"diag_affine_{args.stream}.json")
    save_json(out, os.path.abspath(path))
    print("wrote", os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

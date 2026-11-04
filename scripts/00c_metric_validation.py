"""00c -- metric + protocol validation against an independent reference.

Why: the whole §11 gate compares our absolute numbers with published ones.  If
our PSNR/SSIM differ from the reference implementation, or if our resampling
differs from the one the source model was trained/tested with, then every
absolute comparison is meaningless -- and the failure would be silent, because
the numbers would still look plausible.

Three checks, in increasing order of strength:

  A. metric identity   our psnr/ssim vs an independent implementation
                       (skimage.metrics) on identical arrays.  Must agree to ~1e-9
                       for PSNR and ~1e-6 for SSIM on images without flat regions.

  B. range/shape       psnr() on x==y must be inf; psnr() must be scale-consistent
                       with MSE, i.e. 10*log10(1/mse).

  C. protocol replay   the frozen source model on a target test split, through our
                       stream+metric code, compared with the number the upstream
                       project recorded for the same weights.  A large gap here
                       means the protocol (resize, range, channel order) differs,
                       not the metric.

Usage: python scripts/00c_metric_validation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import PairedStream, StreamSpec, load_rgb, resize_eval, resolve_path, to_tensor  # noqa: E402
from brctta.model import build_adapted  # noqa: E402

# Upstream-recorded numbers for the checkpoints in weights_root (README/ledger of
# the MambaCue project).  Used only as a protocol sanity check, never as a result.
UPSTREAM = {
    "LSUI": {"psnr": 31.2323, "ssim": 0.9357, "mse": 104.35},
    "UIEB": {"psnr": 24.4629, "ssim": 0.9224, "mse": 392.46},
    "UFO": {"psnr": 28.6112, "ssim": 0.8789, "mse": 113.96},
}


def check_a(stream) -> dict:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim

    rows = []
    for i in range(4):
        x, gt, stem = stream.take(i)
        with torch.no_grad():
            pred = (x * 0.72 + 0.11).clamp(0, 1)  # non-trivial perturbation
        a = pred.permute(1, 2, 0).numpy().astype(np.float64)
        b = gt.permute(1, 2, 0).numpy().astype(np.float64)
        rows.append({
            "stem": stem,
            "psnr_ours": M.psnr(pred, gt),
            "psnr_skimage": float(sk_psnr(b, a, data_range=1.0)),
            "ssim_ours": M.ssim(pred, gt),
            # The comparison that can actually be exact: skimage's Wang-variant
            # SSIM (Gaussian 11x11 sigma 1.5, no sample covariance, K1/K2
            # stabilisation) is the same estimator ours implements.  Matching
            # against skimage's *default* (uniform 7x7) would only measure how
            # different two SSIM definitions are, which is not a validation.
            "ssim_skimage_wang": float(sk_ssim(b, a, data_range=1.0, channel_axis=2,
                                               gaussian_weights=True, sigma=1.5,
                                               use_sample_covariance=False, win_size=11)),
            "ssim_skimage_default": float(sk_ssim(b, a, data_range=1.0, channel_axis=2)),
        })
    d_p = max(abs(r["psnr_ours"] - r["psnr_skimage"]) for r in rows)
    d_s = max(abs(r["ssim_ours"] - r["ssim_skimage_wang"]) for r in rows)
    d_s_def = max(abs(r["ssim_ours"] - r["ssim_skimage_default"]) for r in rows)
    return {"rows": rows, "max_abs_dPSNR": d_p, "max_abs_dSSIM": d_s,
            "max_abs_dSSIM_vs_skimage_default": d_s_def,
            "ok": bool(d_p < 1e-6 and d_s < 1e-6),
            "note": ("PSNR must agree exactly (same formula). SSIM is validated against "
                     "skimage's Wang variant (gaussian_weights=True, sigma=1.5, "
                     "use_sample_covariance=False, win 11), which is the same estimator; "
                     "agreement to 1e-6 means the implementation is right. The default "
                     "uniform-7x7 SSIM differs by design and is reported only to show that "
                     "difference is a definition gap, not a bug.")}


def check_b() -> dict:
    rng = np.random.RandomState(0)
    a = torch.from_numpy(rng.rand(3, 64, 64).astype(np.float32))
    # Documented convention: identical images return the cap 99.0 dB rather than
    # +inf, because +inf poisons every downstream mean/sum.  The check asserts
    # the cap, not infinity.
    p_ident = float(M.psnr(a, a))
    ident = {"psnr_identical": p_ident, "psnr_on_identical_capped": bool(p_ident >= 99.0)}
    b = (a + 0.1 * torch.from_numpy(rng.randn(3, 64, 64).astype(np.float32))).clamp(0, 1)
    mse = M.mse(a, b)
    implied = 10.0 * np.log10(1.0 / max(mse, 1e-20))
    return {"mse": mse, "psnr": M.psnr(a, b), "psnr_from_mse": implied,
            "abs_diff": abs(M.psnr(a, b) - implied),
            "ok": bool(ident["psnr_on_identical_capped"] and abs(M.psnr(a, b) - implied) < 1e-9),
            **ident}


def check_c(cfg, ds: str, split: str, limit: int, device: str) -> dict:
    ckpt = resolve_ckpt(cfg, {"source": ds, "source_ckpt": f"{ds}/best_model.pth"}, "mambacue")
    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"])
    spec = StreamSpec(name=f"validate_{ds}", source_dataset=ds,
                      target_split=f"{ds}/{split}", order="natural", seed=0)
    stream = PairedStream(split_dir(cfg, ds, split), spec)
    n = len(stream) if limit <= 0 else min(limit, len(stream))
    ps, ss, ms = [], [], []
    with torch.no_grad():
        for i in range(n):
            x, gt, stem = stream.take(i)
            y = model(x.unsqueeze(0).to(device))[0].cpu()
            m = M.per_image(y, gt)
            ps.append(m["psnr"]); ss.append(m["ssim"]); ms.append(m["mse"])
    got = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "mse": float(np.mean(ms)), "n": n}
    ref = UPSTREAM.get(ds)
    out = {"split": split, "our_protocol": got, "upstream_recorded": ref}
    if ref:
        out["dPSNR"] = got["psnr"] - ref["psnr"]
        out["dSSIM"] = got["ssim"] - ref["ssim"]
        # 0.5 dB / 0.01 SSIM is the tolerance for "same protocol, different split/eval detail"
        out["same_protocol"] = bool(abs(out["dPSNR"]) < 0.5 and abs(out["dSSIM"]) < 0.01)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="UIEB")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = StreamSpec(name="metric_val", source_dataset="UIEB",
                      target_split="UIEB/test", order="natural", seed=0)
    stream = PairedStream(split_dir(cfg, "UIEB", "test"), spec)

    report = {}
    report["A_metric_identity"] = check_a(stream)
    report["B_range_consistency"] = check_b()
    report["C_protocol_replay"] = check_c(cfg, args.dataset, args.split, args.limit, device)

    print("== A. metric identity vs skimage")
    for r in report["A_metric_identity"]["rows"]:
        print(f"   {r['stem'][:40]:<40} PSNR ours={r['psnr_ours']:.6f} "
              f"sk={r['psnr_skimage']:.6f} | SSIM ours={r['ssim_ours']:.6f} "
              f"sk_wang={r['ssim_skimage_wang']:.6f} sk_def={r['ssim_skimage_default']:.6f}")
    print(f"   max|dPSNR|={report['A_metric_identity']['max_abs_dPSNR']:.2e}  "
          f"max|dSSIM vs skimage-Wang|={report['A_metric_identity']['max_abs_dSSIM']:.2e}  "
          f"ok={report['A_metric_identity']['ok']}")
    print(f"   (skimage default uniform-7x7 SSIM differs by "
          f"{report['A_metric_identity']['max_abs_dSSIM_vs_skimage_default']:.2e} -- definition gap)")

    print("\n== B. PSNR/MSE consistency")
    b = report["B_range_consistency"]
    print(f"   psnr={b['psnr']:.6f} from_mse={b['psnr_from_mse']:.6f} diff={b['abs_diff']:.2e} "
          f"ok={b['ok']}")

    print("\n== C. protocol replay (frozen source on its own test split)")
    c = report["C_protocol_replay"]
    print(f"   ours     {c['our_protocol']}")
    print(f"   upstream {c['upstream_recorded']}")
    if "dPSNR" in c:
        print(f"   dPSNR={c['dPSNR']:+.4f} dB  dSSIM={c['dSSIM']:+.4f}  "
              f"same_protocol={c['same_protocol']}")

    out = os.path.join(run_dir(cfg, "manifests"), "metric_validation.json")
    save_json(report, out)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Can the per-image affine be PREDICTED instead of fitted to the label?

WHY THIS EXISTS
---------------
`_diag_affine.py` measured that a per-image per-channel affine applied to the
model output is worth **+6.9 dB** on UIEB/dev, while a single dataset-level
affine is worth only +0.17 dB.  That is the largest tractable headroom found
anywhere in this project -- an order of magnitude more than the residual
adapter's gateable +0.006..0.049 dB.

The +6.9 dB number is, however, label-fitted, so it is a ceiling, not a method.
The question this script answers is whether a *label-free at deployment* version
exists: predict the 6 numbers from the input alone, having learned the mapping on
SOURCE pairs where ground truth is available (spec §5.1 explicitly allows source
supervision offline).

Two predictors, both deliberately tiny, because a big network would just be a
second enhancer and the point is to isolate the calibration component:

  ``stats``  an MLP on global colour statistics of x and of y_source
             (~2k params).  Domain shift enters only through those statistics,
             so this is the most likely to *fail visibly* if the mapping does
             not transfer -- which is what we want to know.
  ``cnn``    a 3-layer fully-convolutional head producing low-resolution
             per-channel gain/bias maps, upsampled and applied to y
             (~30k params).  Slightly less global, still not an enhancer.

Both are trained with L1 on the *reconstruction*, not on the affine parameters
directly: the target parameters are only identified up to what the loss cares
about, and regressing the oracle affine directly would waste capacity on
parameters that do not affect the image.

The evaluation is on the TARGET dev split.  Source splits are reported as a
sanity check (a predictor that cannot even help in-domain cannot transfer).

Usage
    bash scripts/gpu_run.sh 04k_affine_predict.py --source LSUI --target UIEB
    bash scripts/gpu_run.sh 04k_affine_predict.py --source UIEB --target LSUI
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from brctta.data import StreamSpec  # noqa: E402
from brctta.metrics import per_image  # noqa: E402
from brctta.model import build_adapted  # noqa: E402
from common import (adapter_ckpt_path, load_config, open_stream, resolve_ckpt,  # noqa: E402
                    run_dir, save_json)


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #
def global_stats(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,12]: per-channel mean/std of x, plus of its log-response.

    Log-response is included because underwater attenuation is multiplicative in
    the linear domain; a gain error shows up additively in log space, which is
    where the affine lives.
    """
    m = x.mean(dim=(2, 3))
    s = x.std(dim=(2, 3))
    lm = torch.log1p(9.0 * x.clamp_min(0)).mean(dim=(2, 3))
    return torch.cat([m, s, lm, (m - lm)], dim=1)


class StatsMLP(nn.Module):
    """Global-statistics -> 6 affine parameters (3 gains, 3 offsets)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, x):
        p = self.net(global_stats(x))
        gain = 1.0 + 0.5 * torch.tanh(p[:, :3])     # in [0.5, 1.5]
        bias = 0.25 * torch.tanh(p[:, 3:])          # in [-0.25, 0.25]
        return gain.view(-1, 3, 1, 1), bias.view(-1, 3, 1, 1)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
    )


class AffineCNN(nn.Module):
    """x (+y) -> low-res per-channel gain/bias maps, upsampled to full size."""

    def __init__(self, width: int = 24, out_res: int = 8):
        super().__init__()
        self.out_res = out_res
        self.body = nn.Sequential(_block(6, width), nn.AvgPool2d(2),
                                  _block(width, 2 * width), nn.AvgPool2d(2),
                                  _block(2 * width, 2 * width))
        self.head = nn.Conv2d(2 * width, 6, 1)
        # start at the identity so an untrained head is a no-op, not a wrecking ball
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, y):
        h = self.body(torch.cat([x, y], dim=1))
        h = F.interpolate(h, size=(self.out_res, self.out_res), mode="bilinear", align_corners=False)
        p = self.head(h)
        p = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False)
        gain = 1.0 + 0.5 * torch.tanh(p[:, :3])
        bias = 0.25 * torch.tanh(p[:, 3:])
        return gain, bias


def apply_affine(y, gain, bias):
    return torch.clamp(y * gain + bias, 0.0, 1.0)


def oracle_affine(y, gt):
    """Per-channel least squares y -> gt, as a [B,3,1,1]-shaped pair."""
    ys = y.reshape(y.shape[0], 3, -1)
    gs = gt.reshape(gt.shape[0], 3, -1)
    ym, gm = ys.mean(-1, keepdim=True), gs.mean(-1, keepdim=True)
    yc, gc = ys - ym, gs - gm
    denom = (yc * yc).sum(-1, keepdim=True).clamp_min(1e-8)
    a = (yc * gc).sum(-1, keepdim=True) / denom
    b = gm - a * ym
    return a.view(-1, 3, 1, 1), b.view(-1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def gather(cfg, ds, split, source, seed, limit, device, ckpt, adapter, max_n=600):
    """Frozen-model outputs for a split, cached in memory (it is small)."""
    spec = StreamSpec(name=f"affpred_{ds}_{split}", source_dataset=source,
                      target_split=f"{ds}/{split}", seed=seed)
    stream = open_stream(cfg, ds, split, spec)
    n = min(limit or len(stream), len(stream), max_n)
    model = build_adapted(ckpt, device=device, backbone="mambacue", adapter_ckpt=adapter)
    xs, ys, gs = [], [], []
    with torch.no_grad():
        for i in range(n):
            x_t, gt_t, _ = stream.take(i)
            x = x_t.unsqueeze(0).to(device)
            xs.append(x)
            ys.append(model.predict(x))
            gs.append(gt_t.unsqueeze(0).to(device))
    return torch.cat(xs), torch.cat(ys), torch.cat(gs)


def psnr_t(y, gt):
    mse = ((y - gt) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="LSUI")
    ap.add_argument("--target", default="UIEB")
    ap.add_argument("--backbone", default="mambacue")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stream_cfg = next(s for s in cfg["streams"] if s["source"] == args.source)
    ckpt = resolve_ckpt(cfg, stream_cfg, args.backbone)
    adapter = adapter_ckpt_path(cfg, args.source, args.backbone)
    print(f"[affpred] source={args.source} target={args.target} "
          f"theta_0={'source_adapter' if adapter else 'frozen_backbone'}", flush=True)

    print("[affpred] gathering frozen outputs ...", flush=True)
    xs, ys, gs = gather(cfg, args.source, "train", args.source, args.seed,
                        args.limit, device, ckpt, adapter)
    xtr_s, ytr_s, gtr_s = xs, ys, gs
    xd, yd, gd = gather(cfg, args.target, "dev", args.source, args.seed,
                        args.limit, device, ckpt, adapter)
    print(f"  source/train n={len(xs)}  target/dev n={len(xd)}", flush=True)

    # reference points, all on target/dev
    ref = {"identity": psnr_t(yd, gd)}
    a_o, b_o = oracle_affine(yd, gd)
    ref["oracle_per_image"] = psnr_t(apply_affine(yd, a_o, b_o), gd)
    # dataset-level affine: one pair of (a,b) for the whole split, fitted to target
    # dev labels -- that is the ceiling of "calibrate once", shown for scale only
    a_d = a_o.mean(0, keepdim=True).expand_as(a_o)
    b_d = b_o.mean(0, keepdim=True).expand_as(b_o)
    ref["oracle_dataset_level"] = psnr_t(apply_affine(yd, a_d, b_d), gd)
    print(f"  identity {ref['identity']:.4f} | per-image oracle {ref['oracle_per_image']:.4f} "
          f"| dataset-level oracle {ref['oracle_dataset_level']:.4f}", flush=True)

    results = {}
    for name in ("stats", "cnn"):
        model = StatsMLP().to(device) if name == "stats" else AffineCNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(1, args.epochs * max(1, len(xs) // args.batch)),
            pct_start=0.2)
        n = len(xs)
        hist = []
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=device)
            tot, nb = 0.0, 0
            for i in range(0, n - args.batch + 1, args.batch):
                idx = perm[i:i + args.batch]
                xb, yb, gb = xtr_s[idx], ytr_s[idx], gtr_s[idx]
                if name == "stats":
                    gain, bias = model(xb)
                else:
                    gain, bias = model(xb, yb)
                loss = (apply_affine(yb, gain, bias) - gb).abs().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss)
                nb += 1
            hist.append(tot / max(nb, 1))
            if ep % max(1, args.epochs // 8) == 0 or ep == args.epochs - 1:
                print(f"  [{name}] epoch {ep:3d} train L1 {hist[-1]:.5f}", flush=True)

        model.eval()
        with torch.no_grad():
            if name == "stats":
                gain_t, bias_t = model(xd)
                gain_s, bias_s = model(xtr_s[:min(200, len(xtr_s))])
            else:
                gain_t, bias_t = model(xd, yd)
                gain_s, bias_s = model(xtr_s[:min(200, len(xtr_s))],
                                       ytr_s[:min(200, len(ytr_s))])
            y_pred_t = apply_affine(yd, gain_t, bias_t)
            y_pred_s = apply_affine(ytr_s[:min(200, len(ytr_s))], gain_s, bias_s)
            gt_s = gtr_s[:min(200, len(gtr_s))]

        psnr_tgt = psnr_t(y_pred_t, gd)
        psnr_src = psnr_t(y_pred_s, gt_s)
        psnr_src0 = psnr_t(ytr_s[:min(200, len(ytr_s))], gt_s)
        m = per_image(y_pred_t[0], gd[0])
        # per-image delta distribution on the target split
        p_before = (10.0 * torch.log10(1.0 / ((yd - gd) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)))
        p_after = (10.0 * torch.log10(1.0 / ((y_pred_t - gd) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)))
        delta = (p_after - p_before).cpu().numpy()
        k = max(1, int(round(len(delta) * 0.10)))
        worst = np.sort(delta)[:k]
        results[name] = {
            "psnr_target": psnr_tgt,
            "dPSNR_target": psnr_tgt - ref["identity"],
            "frac_improved": float((delta > 0).mean()),
            "median_delta": float(np.median(delta)),
            "worst10_delta": float(worst.mean()),
            "psnr_source_before": psnr_src0,
            "psnr_source_after": psnr_src,
            "dPSNR_source": psnr_src - psnr_src0,
            "train_l1_history": hist,
            "params": sum(p.numel() for p in model.parameters()),
            "sample_per_image": m,
        }
        print(f"  [{name}] TARGET dev PSNR {psnr_tgt:.4f} (d {psnr_tgt - ref['identity']:+.4f}) "
              f"improved {results[name]['frac_improved']:.2f} worst10 {results[name]['worst10_delta']:+.3f} "
              f"| SOURCE d {results[name]['dPSNR_source']:+.4f}", flush=True)

    best = max(("stats", "cnn"), key=lambda k: results[k]["psnr_target"])
    oracle_gap = ref["oracle_per_image"] - ref["identity"]
    verdict = {
        "predictable_frac_of_oracle": (results[best]["dPSNR_target"] / oracle_gap) if oracle_gap > 1e-6 else None,
        "beats_dataset_level": results[best]["psnr_target"] > ref["oracle_dataset_level"],
        "best_predictor": best,
        "summary": (
            f"{best} recovers {results[best]['dPSNR_target']:+.3f} dB of the "
            f"{oracle_gap:+.3f} dB per-image affine headroom on {args.target}/dev"
        ),
    }
    print(f"\n[affpred] {verdict['summary']}")
    print(f"[affpred] vs dataset-level oracle ({ref['oracle_dataset_level']:.4f}): "
          f"{'BEATS' if verdict['beats_dataset_level'] else 'does NOT beat'} it")

    out = run_dir(cfg, "affine_predict", f"{args.source}2{args.target}")
    payload = {"source": args.source, "target": args.target, "backbone": args.backbone,
               "theta_0": "source_adapter" if adapter else "frozen_backbone",
               "n_source": len(xs), "n_target": len(xd),
               "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
               "reference": ref, "predictors": results, "verdict": verdict,
               "note": ("Predictors are trained on SOURCE pairs and evaluated on TARGET "
                        "dev. The oracle rows are label-fitted and are ceilings, never "
                        "methods. dataset_level_oracle uses target-dev labels for one "
                        "shared affine, i.e. it is still label-using; it is shown to "
                        "separate 'per-image' from 'one global correction'.")}
    save_json(payload, os.path.join(out, "affine_predict.json"))
    print(f"[affpred] wrote {os.path.join(out, 'affine_predict.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

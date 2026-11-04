#!/usr/bin/env python3
"""Is the degradation model D actually able to reproduce a degradation?

WHY: `04j_affine_redegrade.py` showed the re-degradation residual does NOT
prefer the ground truth over the model output (GT residual lower in only
42-50% of images, corr with dPSNR = -0.36..-0.43).  Two hypotheses:

  H1 D is under-trained / too weak      -> overfitting 32 pairs drives L1 to ~0
  H2 the formulation is unidentifiable  -> even overfitting cannot fit

This distinguishes them with a single cheap run.  If H1 holds, D is not useless
as a *simulator* (turning clean source images into plausible degraded inputs),
which is a legitimate training-time use that needs no target data -- unlike the
verification use that failed.

Usage: bash scripts/gpu_run.sh _probe_degrader.py --source LSUI --steps 400
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from brctta.data import StreamSpec  # noqa: E402
from brctta.degrade import Degrader, redegradation_loss  # noqa: E402
from common import load_config, open_stream, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="LSUI")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = StreamSpec(name=f"probe_degrade_{args.source}", source_dataset=args.source,
                      target_split=f"{args.source}/train", seed=0)
    stream = open_stream(cfg, args.source, "train", spec)
    n = min(args.n, len(stream))
    xs, gs = stream.take_batch(list(range(n)))[:2]
    x = xs.to(device)
    g = gs.to(device)

    net = Degrader(width=args.width).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)
    hist = []
    for step in range(args.steps):
        # fresh random crop of rows so the net cannot memorise one tile
        idx = torch.randint(0, n, (min(args.rows, n),))
        loss = redegradation_loss(net, g[idx], x[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            hist.append({"step": step, "loss": float(loss)})
            print(f"  step {step:4d}  L1 {float(loss):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        full = float(redegradation_loss(net, g, x))
        # can it tell clean from degraded?  residual on GT vs on a blurred version
        blur = torch.nn.functional.avg_pool2d(g, 5, 1, 2)
        r_gt = float((net(g) - x).abs().mean())
        r_blur = float((net(blur) - x).abs().mean())
    verdict = ("capacity_ok" if full < 0.02 else
               ("partially_fits" if full < 0.05 else "cannot_fit_formulation_or_capacity"))
    print(f"\n[probe] overfit L1 on {n} pairs = {full:.5f} (start {hist[0]['loss']:.5f})")
    print(f"[probe] residual  GT {r_gt:.5f}  |  blurred-GT {r_blur:.5f}   "
          f"({'discriminates' if r_gt < r_blur else 'does NOT discriminate'})")
    print(f"[probe] verdict: {verdict}")

    out = {"source": args.source, "n": n, "steps": args.steps, "lr": args.lr,
           "width": args.width, "history": hist, "overfit_l1": full,
           "resid_gt": r_gt, "resid_blur": r_blur,
           "discriminates": bool(r_gt < r_blur), "verdict": verdict,
           "interpretation": (
               "D fits the source degradation well enough to be used as a SHIFT "
               "SIMULATOR (clean source -> plausible degraded input) for training "
               "the risk controller, even though its residual failed as a "
               "test-time verification signal." if verdict == "capacity_ok" else
               "D is not usable; the re-degradation idea needs reformulating.")}
    path = os.path.join(cfg["paths"]["out_root"], "manifests", f"degrader_probe_{args.source}.json")
    save_json(out, path)
    print(f"[probe] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

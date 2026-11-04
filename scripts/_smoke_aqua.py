"""Smoke test: can the risk-controlled adapter sit on the AquaClarity backbone?

Checks, on a handful of real images:
  * the frozen AquaClarity checkpoint loads exactly (no missing/unexpected keys)
  * theta_0 == frozen model bit-for-bit (the zero-init adapter property)
  * a candidate update produces a finite gradient and finite loss
  * wall-clock per image for forward, and for a k-step candidate

Usage: python scripts/_smoke_aqua.py --dataset UIEB --n 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, resolve_ckpt, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior, self_supervised_loss  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UIEB")
    ap.add_argument("--source", default="LSUI")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = resolve_ckpt(cfg, {"source": args.source, "source_ckpt": None}, "aqua")
    print("ckpt", ckpt)

    t0 = time.time()
    model = build_adapted(ckpt, device=device, hidden=32, backbone="aqua")
    print(f"built in {time.time()-t0:.1f}s  "
          f"backbone params={sum(p.numel() for p in model.backbone.parameters()):,}  "
          f"adapter params={sum(p.numel() for p in model.adapter.parameters()):,}")

    spec = StreamSpec(name="smoke", source_dataset=args.dataset,
                      target_split=f"{args.dataset}/dev", seed=0)
    stream = PairedStream(split_dir(cfg, args.dataset, "dev"), spec)
    prior = load_prior(os.path.join(cfg["paths"]["out_root"], "manifests",
                                    f"prior_{args.source}.json"))

    theta0 = model.snapshot()
    for i in range(args.n):
        x, gt, stem = stream.take(i)
        xb = x.unsqueeze(0).to(device)
        t = time.time()
        with torch.no_grad():
            y = model(xb)
        dt = time.time() - t
        # theta_0 must equal the frozen backbone exactly
        y0 = model.backbone(xb)
        diff = float((y - y0.clamp(0, 1)).abs().max())
        print(f"  {stem}: forward {dt*1000:.0f} ms  range=[{y.min():.3f},{y.max():.3f}]  "
              f"|y - frozen|_max={diff:.2e}")

    x, gt, stem = stream.take(0)
    xb = x.unsqueeze(0).to(device)
    opt = torch.optim.SGD(model.adapter_params(), lr=1e-3)
    t = time.time()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = self_supervised_loss(model, xb, prior=prior, with_consistency=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0))
        opt.step()
    dt = (time.time() - t) / 3
    print(f"  3-step candidate: loss={float(loss):.5f} gnorm={gn:.4f} "
          f"{dt*1000:.0f} ms/step  finite={torch.isfinite(loss).item()}")
    print(f"  adapter moved by {model.update_magnitude(theta0):.4f} (L2)")
    model.restore(theta0)
    print(f"  after restore, |theta-theta0|={model.update_magnitude(theta0):.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""01b -- pre-train the residual adapter on SOURCE pairs (supervised, source-only).

Why this step exists.

The diagnosis (scripts/04e_ceiling.py, scripts/04g_objective.py) is unambiguous:

  * the adapter has plenty of plasticity -- with a label loss on the target it
    gains +1.45 dB on UIEB dev;
  * every hand-designed no-reference objective we tried either loses PSNR
    (best: rgb, -0.03 dB mean, -1.8 dB worst-10%) or is neutral within noise.

So the bottleneck is not the gate and not the adapter capacity: it is that a
hand-written image prior is a weak stand-in for "what a good restoration looks
like".  The honest fix is not another prior -- it is to *learn* what a good
residual looks like from the source domain, which the protocol explicitly allows
(spec §5.1: the source model was trained on source pairs; target labels are what
is forbidden).

How it stays legal.

  theta_0 = backbone (frozen, source-trained) + adapter (trained here on SOURCE
  pairs only).  No target image, label or statistic enters this training.  The
  target is only used later, unlabeled, by the online step, and only for the
  accept/keep/reset decision.  So "theta_0 is a source-only model" still holds;
  it is simply a better source-only model than the bare backbone.

Consequence for the baselines: because theta_0 now already contains a learned
residual, the honest no-adaptation reference is "theta_0 with no online update",
reported as `source_adapter`.  `source_frozen` (bare backbone) stays in the table
as the weaker reference, and the *TTA gain* is measured against
`source_adapter`, never against the bare backbone.

Usage:
    python scripts/01b_adapter_pretrain.py --source LSUI --epochs 6
    python scripts/01b_adapter_pretrain.py --source LSUI --epochs 1 --limit 200   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def ssim_loss(y: torch.Tensor, gt: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable SSIM (the same formulation the metric uses), as a loss term."""
    c = y.shape[1]
    coords = torch.arange(win, dtype=y.dtype, device=y.device) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, win, 1) * (g / g.sum()).view(1, 1, 1, win)
    g = g.expand(c, 1, win, win).contiguous()
    pad = win // 2
    mu_y = F.conv2d(y, g, padding=pad, groups=c)
    mu_g = F.conv2d(gt, g, padding=pad, groups=c)
    s_y = F.conv2d(y * y, g, padding=pad, groups=c) - mu_y ** 2
    s_g = F.conv2d(gt * gt, g, padding=pad, groups=c) - mu_g ** 2
    s_yg = F.conv2d(y * gt, g, padding=pad, groups=c) - mu_y * mu_g
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_y * mu_g + c1) * (2 * s_yg + c2)) / \
               ((mu_y ** 2 + mu_g ** 2 + c1) * (s_y + s_g + c2))
    return 1.0 - ssim_map.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default="LSUI", choices=["LSUI", "UIEB", "UFO", "EUVPd"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--w-ssim", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    spec = StreamSpec(name=f"pretrain_{args.source}", source_dataset=args.source,
                      target_split=f"{args.source}/train", seed=0)
    train = open_stream(cfg, args.source, "train", spec)
    n = len(train) if args.limit <= 0 else min(args.limit, len(train))
    print(f"source={args.source} train n={n} epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} backbone={args.backbone}", flush=True)

    base_ckpt = resolve_ckpt(cfg, {"source": args.source,
                                   "source_ckpt": f"{args.source}/best_model.pth"},
                             args.backbone)
    model = build_adapted(base_ckpt, device=device, hidden=args.hidden, backbone=args.backbone)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_ad = sum(p.numel() for p in model.adapter.parameters())
    print(f"backbone {n_bb:,} frozen | adapter {n_ad:,} trainable", flush=True)

    opt = torch.optim.AdamW(model.adapter_params(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.2)
    order = np.arange(n)
    best = {"epoch": -1, "loss": float("inf")}
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        np.random.RandomState(ep).shuffle(order)
        run, cnt = 0.0, 0
        for b in range(steps_per_epoch):
            idxs = order[b * args.batch:(b + 1) * args.batch]
            if len(idxs) < 2:
                continue
            xs, gts = [], []
            for i in idxs:
                x, gt, _ = train.take(int(i))
                xs.append(x); gts.append(gt)
            xb = torch.stack(xs).to(device)
            gtb = torch.stack(gts).to(device)
            y = model(xb)
            l1 = (y - gtb).abs().mean()
            loss = l1 + args.w_ssim * ssim_loss(y, gtb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(idxs); cnt += len(idxs)
        mean_loss = run / max(cnt, 1)
        hist.append({"epoch": ep, "loss": mean_loss, "seconds": time.time() - t0})
        print(f"  epoch {ep}: loss={mean_loss:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if mean_loss < best["loss"]:
            best = {"epoch": ep, "loss": mean_loss}

    d = run_dir(cfg, f"adapter_pretrain_{args.source}" + ("" if args.backbone == "mambacue" else f"_{args.backbone}"))
    out_ckpt = os.path.join(d, "adapter.pth")
    torch.save({"adapter": model.adapter.state_dict(),
                "source": args.source, "epochs": args.epochs,
                "best_epoch": best["epoch"], "best_loss": best["loss"],
                "hidden": args.hidden, "backbone": args.backbone,
                "base_ckpt": base_ckpt}, out_ckpt)
    sha = hashlib.sha256(open(out_ckpt, "rb").read()).hexdigest()
    save_json({"checkpoint": out_ckpt, "sha256": sha, "history": hist,
               "best": best, "n_train": n, "params_adapter": n_ad,
               "protocol": ("adapter trained on SOURCE train pairs only; no target data of "
                            "any kind; backbone frozen")},
              os.path.join(d, "adapter_pretrain.json"))
    print(f"\nwrote {out_ckpt}\nsha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

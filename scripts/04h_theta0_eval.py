"""04h -- is theta_0 actually better?  (dev only, no test data)

`scripts/01b_adapter_pretrain.py` fits the residual adapter on SOURCE pairs, but
the training loss barely moves (0.05030 -> 0.05015 over three epochs on LSUI).
Two very different explanations produce that curve:

  (a) the adapter is learning, but the backbone is already close on its own
      domain, so there is little left to gain on SOURCE data -- while the
      cross-domain gain (the thing we care about) could still be large;
  (b) gradients are effectively dead and the adapter is not learning at all.

This script separates them, and answers the question that actually matters:
does theta_0 (= backbone + source-trained residual) beat the bare backbone on
the TARGET dev split?

  --overfit N   train on N images for --steps steps and print L1 as it goes.
                A working adapter drives this to near zero; a broken one flatlines.
  --splits ...  measure PSNR/SSIM of frozen vs adapter on each requested split.

Everything here runs on dev/train splits -- never the locked test sets.

Usage:
    python scripts/04h_theta0_eval.py --overfit 8 --steps 240
    python scripts/04h_theta0_eval.py --splits LSUI/train,UIEB/dev,LSUI/dev,UFO/dev
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (adapter_ckpt_path, load_config, open_stream, resolve_ckpt,  # noqa: E402
                    run_dir, save_json, split_dir)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def dev_spec(ds: str, split: str, tag: str) -> StreamSpec:
    return StreamSpec(name=tag, source_dataset=ds, target_split=f"{ds}/{split}", seed=0)


def run_overfit(cfg, args, device: str) -> dict:
    """Fit N images hard.  A live gradient turns L1 into noise; a dead one flatlines."""
    ds = args.source
    spec = dev_spec(ds, "train", f"overfit_{ds}")
    stream = open_stream(cfg, ds, "train", spec)
    n = min(args.overfit, len(stream))
    ckpt = resolve_ckpt(cfg, {"source": ds, "source_ckpt": f"{ds}/best_model.pth"}, args.backbone)
    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"], backbone=args.backbone)
    opt = torch.optim.AdamW(model.adapter_params(), lr=args.lr, weight_decay=0.0)

    xs, gts = [], []
    for i in range(n):
        x, gt, _ = stream.take(i)
        xs.append(x)
        gts.append(gt)
    xb = torch.stack(xs).to(device)
    gtb = torch.stack(gts).to(device)

    with torch.no_grad():
        base = float((model(xb) - gtb).abs().mean())
    print(f"[overfit] n={n} backbone-only L1 = {base:.5f}", flush=True)

    hist = []
    for step in range(args.steps):
        y = model(xb)
        loss = (y - gtb).abs().mean() + 0.5 * _ssim_loss(y, gtb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0))
        opt.step()
        if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
            with torch.no_grad():
                l1 = float((model(xb) - gtb).abs().mean())
            hist.append({"step": step, "l1": l1, "loss": float(loss.detach()), "grad_norm": gn})
            print(f"  step {step:4d}  l1={l1:.5f}  loss={float(loss.detach()):.5f}  |g|={gn:.4f}",
                  flush=True)
    final = hist[-1]["l1"]
    return {"n": n, "backbone_l1": base, "final_l1": final,
            "reduction": (base - final) / max(base, 1e-9), "history": hist,
            "works": bool(final < 0.8 * base)}


def _ssim_loss(y, gt, win=11, sigma=1.5):
    import torch.nn.functional as F

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
    smap = ((2 * mu_y * mu_g + c1) * (2 * s_yg + c2)) / \
           ((mu_y ** 2 + mu_g ** 2 + c1) * (s_y + s_g + c2))
    return 1.0 - smap.mean()


def measure(cfg, args, device: str, ds: str, split: str, adapter_ckpt: str | None) -> dict:
    spec = dev_spec(ds, split, f"theta0_{ds}_{split}")
    stream = open_stream(cfg, ds, split, spec)
    n = min(args.n, len(stream))
    src = args.source or ds
    ckpt = resolve_ckpt(cfg, {"source": src, "source_ckpt": f"{src}/best_model.pth"}, args.backbone)
    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"],
                          backbone=args.backbone, adapter_ckpt=adapter_ckpt)
    ps, ss, ms = [], [], []
    with torch.no_grad():
        for i in range(n):
            x, gt, _ = stream.take(i)
            y = model(x.unsqueeze(0).to(device))[0]
            m = M.per_image(y, gt)
            ps.append(m["psnr"]); ss.append(m["ssim"]); ms.append(m["mse"])
    return {"split": f"{ds}/{split}", "n": n,
            "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "mse": float(np.mean(ms)),
            "arm": "source_adapter" if adapter_ckpt else "frozen_backbone",
            "adapt_delta_psnr": float(np.mean(ps)) - 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default="LSUI", choices=["LSUI", "UIEB", "UFO", "EUVPd"])
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--splits", default="")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    ck = adapter_ckpt_path(cfg, args.source, args.backbone)
    print(f"adapter checkpoint for source={args.source}: {ck}", flush=True)

    if args.overfit:
        out["overfit"] = run_overfit(cfg, args, device)
        print(f"[overfit] reduction {out['overfit']['reduction']*100:.1f}% "
              f"-> adapter {'LEARNS' if out['overfit']['works'] else 'IS NOT LEARNING'}",
              flush=True)

    if args.splits:
        rows = []
        for spec in args.splits.split(","):
            spec = spec.strip()
            if not spec:
                continue
            ds, split = spec.split("/")
            a = measure(cfg, args, device, ds, split, None)
            b = measure(cfg, args, device, ds, split, ck) if ck else None
            row = {"split": spec, "frozen": a, "adapter": b}
            if b:
                row["delta_psnr"] = b["psnr"] - a["psnr"]
                row["delta_ssim"] = b["ssim"] - a["ssim"]
            rows.append(row)
            if b:
                print(f"  {spec:<14} frozen {a['psnr']:7.4f}/{a['ssim']:.4f}  "
                      f"adapter {b['psnr']:7.4f}/{b['ssim']:.4f}  "
                      f"dPSNR {row['delta_psnr']:+.4f}  dSSIM {row['delta_ssim']:+.4f}",
                      flush=True)
            else:
                print(f"  {spec:<14} frozen {a['psnr']:7.4f}/{a['ssim']:.4f}  (no adapter)",
                      flush=True)
        out["splits"] = rows

    d = run_dir(cfg, f"theta0_eval_{args.source}" + ("" if args.backbone == "mambacue" else f"_{args.backbone}"))
    save_json(out, os.path.join(d, "theta0_eval.json"))
    print("wrote", os.path.join(d, "theta0_eval.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

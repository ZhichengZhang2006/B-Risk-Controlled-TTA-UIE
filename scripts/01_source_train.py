"""01 -- source-domain training (spec §13 "source train").

Trains the MambaCue backbone on one source domain with paired supervision, then
stores the checkpoint plus a hash and the exact command line.  This exists so
the pipeline is reproducible from scratch; the shipped runs may also start from
external MambaCue checkpoints, in which case the provenance is recorded in
``source_provenance.json`` instead.

Usage:
    python scripts/01_source_train.py --dataset LSUI --epochs 40 --out runs/src_LSUI.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.data import PairedStream, StreamSpec, collate  # noqa: E402
from brctta.mambacue import myModel  # noqa: E402


def ssim_loss(pred, gt, win: int = 11):
    from brctta.metrics import ssim as ssim_fn
    # differentiable 1 - SSIM on a 1-channel luma proxy, cheap and stable
    p = (0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]).unsqueeze(1)
    g = (0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]).unsqueeze(1)
    mu_p = F.avg_pool2d(p, win, 1, win // 2)
    mu_g = F.avg_pool2d(g, win, 1, win // 2)
    sp = F.avg_pool2d(p * p, win, 1, win // 2) - mu_p ** 2
    sg = F.avg_pool2d(g * g, win, 1, win // 2) - mu_g ** 2
    spg = F.avg_pool2d(p * g, win, 1, win // 2) - mu_p * mu_g
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_p * mu_g + c1) * (2 * spg + c2)) / ((mu_p ** 2 + mu_g ** 2 + c1) * (sp + sg + c2))
    return 1.0 - s.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True, choices=["LSUI", "UIEB", "UFO"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-train", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_spec = StreamSpec(name=f"src_{args.dataset}", source_dataset=args.dataset,
                         target_split=f"{args.dataset}/train", seed=0)
    tr = PairedStream(split_dir(cfg, args.dataset, "train"), tr_spec, cache=False)
    if args.limit_train:
        tr.stems = tr.stems[: args.limit_train]
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)

    model = myModel(hidden_channels=32, scan_mode="four", output_heads="both").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    out = args.out or os.path.join(cfg["paths"]["out_root"], "source", f"{args.dataset}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        acc = []
        for b in dl:
            x, y = b["input"].to(device), b["gt"].to(device)
            pred = model(x)
            loss = F.l1_loss(pred, y) + 0.5 * ssim_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            acc.append(float(loss.detach()))
        ep_loss = sum(acc) / max(len(acc), 1)
        print(f"[{args.dataset}] epoch {ep + 1}/{args.epochs} loss={ep_loss:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time() - t0:.0f}s", flush=True)
        if ep_loss < best:
            best = ep_loss
            torch.save(model.state_dict(), out)

    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    meta = {
        "dataset": args.dataset, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "train_pairs": len(tr), "best_train_loss": best,
        "checkpoint": out, "sha256": sha, "wall_seconds": time.time() - t0,
        "argv": sys.argv,
    }
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("saved", out, "sha256", sha[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

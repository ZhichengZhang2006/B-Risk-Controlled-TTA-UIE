"""09 -- strong FROZEN comparators under our protocol (spec §11).

§11 forbids judging the risk controller only against the same weak backbone it
wraps: the comparison set must contain a strong frozen model *and* a strong TTA.
This script produces the frozen arms.

Two distinct reference roles, deliberately kept apart because conflating them is
the most common way this field's tables become unfair:

  cross_domain  ckpt trained on the SOURCE dataset, evaluated on the TARGET test
                split.  Same permission as ours (no target labels, no target
                training).  THIS is the fair "strong frozen" comparator.

  in_domain     ckpt trained on the TARGET dataset's own training split.  It has
                seen target labels, so it is a "target supervision reference"
                (spec §11), an absolute coordinate only -- never a gate.

Nothing here adapts: theta is frozen.

Usage:
    python scripts/09_strong_frozen.py --eval-dataset UIEB --ckpt-dataset LSUI --role cross_domain
    python scripts/09_strong_frozen.py --eval-dataset UIEB --ckpt-dataset UIEB --role in_domain
    python scripts/09_strong_frozen.py --eval-dataset UIEB --ckpt-dataset LSUI --limit 3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta import metrics as M  # noqa: E402
from brctta.backbones import build_aqua, parameter_counts  # noqa: E402
from brctta.data import EVAL_SIZE, StreamSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-dataset", required=True, choices=["UIEB", "LSUI", "UFO", "EUVPd"])
    ap.add_argument("--ckpt-dataset", required=True, choices=["UIEB", "LSUI", "UFO", "EUVPd"])
    ap.add_argument("--role", default="cross_domain", choices=["cross_domain", "in_domain"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backbone", default="aqua", choices=["aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = StreamSpec(
        name=f"{args.backbone}_{args.ckpt_dataset}2{args.eval_dataset}_{args.split}",
        source_dataset=args.ckpt_dataset,
        target_split=f"{args.eval_dataset}/{args.split}",
        seed=args.seed, eval_size=EVAL_SIZE,
    )
    stream = open_stream(cfg, args.eval_dataset, args.split, spec)
    n = len(stream) if args.limit <= 0 else min(args.limit, len(stream))

    ckpt = resolve_ckpt(cfg, {"source": args.ckpt_dataset, "source_ckpt": None}, args.backbone)
    model = build_aqua(ckpt, device)

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            x, gt, stem = stream.take(i)
            y = model(x.unsqueeze(0).to(device))[0].cpu()
            m = M.per_image(y, gt)
            m["stem"] = stem
            rows.append(m)
            if (i + 1) % 25 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] {stem} psnr={m['psnr']:.4f}", flush=True)

    secs = time.time() - t0
    out = {
        "method": "strong_frozen", "role": args.role,
        "backbone": args.backbone,
        "ckpt_dataset": args.ckpt_dataset, "eval_dataset": args.eval_dataset,
        "split": args.split, "n": n, "eval_size": EVAL_SIZE,
        "protocol": "whole-image BICUBIC resize 256x256, no crop/pad, frozen weights, no adaptation",
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "mse": float(np.mean([r["mse"] for r in rows])),
        "wall_seconds": secs, "per_image_seconds": secs / max(n, 1),
        "order_digest": spec.digest(),
        "checkpoint": ckpt,
        "ckpt_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
        "params": parameter_counts(model),
        "per_image": rows if n <= 40 else rows[:5],
    }
    d = run_dir(cfg, f"strong_frozen_{args.role}_{args.eval_dataset}")
    save_json(out, os.path.join(d, f"{args.backbone}_{args.ckpt_dataset}2{args.eval_dataset}.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

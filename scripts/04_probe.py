"""04 -- hyper-parameter probe for the adapter (DEV ONLY, cheap).

The risk controller can only be meaningful if the candidate update actually
moves the output: too small and every candidate is a no-op (all labels 0), too
large and every candidate destroys the image (all labels 1).  This script
measures, on a handful of dev images, the *magnitude* of the candidate effect so
the step size can be fixed before any test split is opened.

Usage:
    python scripts/04_probe.py --stream LSUI2UIEB --n 48
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ckpt_path, load_config, run_dir, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta import metrics as M  # noqa: E402
from brctta.adapt import AdaptConfig  # noqa: E402
from brctta.baselines import GatedAdaptation  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--grid", default="1x0.001,3x0.003,5x0.01,8x0.02")
    ap.add_argument("--gate", default="always", choices=["always", "oracle", "random"])
    ap.add_argument("--dev-domain", default="source", help="source | <dataset> for an auxiliary dev domain")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0),
              "free/total GB:", [round(x / 1e9, 2) for x in torch.cuda.mem_get_info()])

    ds = stream_cfg["source"]
    prior_ds = ds
    probe_ds = ds if args.dev_domain == "source" else args.dev_domain
    spec = StreamSpec(name=f"probe_{args.stream}_{probe_ds}", source_dataset=probe_ds,
                      target_split=f"{probe_ds}/{args.split}", seed=stream_cfg["seed"])
    stream = PairedStream(split_dir(cfg, probe_ds, args.split), spec)
    print(f"probe domain: {probe_ds}/{args.split}  prior: {prior_ds}")

    ckpt = ckpt_path(cfg, stream_cfg["source_ckpt"])
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{prior_ds}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    print("prior:", prior_path if prior else "NONE")
    rows = []
    for cell in args.grid.split(","):
        steps, lr = cell.split("x")
        acfg = AdaptConfig(steps=int(steps), lr=float(lr), consistency=True, memory=0)
        model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"])
        eng = GatedAdaptation(model, kind=args.gate, cfg=acfg, method=f"probe_{args.gate}_{cell}", prior=prior)
        t0 = time.time()
        s = eng.run(stream, limit=args.n)
        s["wall_seconds"] = time.time() - t0
        s["per_image_seconds"] = s["wall_seconds"] / args.n
        rows.append({"cell": cell, **{k: s[k] for k in (
            "psnr_prequential", "ssim_prequential", "dPSNR_mean", "dPSNR_median",
            "NAR_0", "worst10_dPSNR", "accept_rate", "benefit_retention", "wall_seconds", "per_image_seconds")}})
        print(json.dumps(rows[-1]), flush=True)

    out = run_dir(cfg, "probe", f"{args.stream}_{probe_ds}")
    save_json({"stream": args.stream, "n": args.n, "gate": args.gate, "rows": rows},
              os.path.join(out, "probe.json"))
    print("wrote", os.path.join(out, "probe.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

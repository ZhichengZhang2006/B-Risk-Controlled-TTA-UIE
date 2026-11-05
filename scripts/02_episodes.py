"""02 -- offline episode generation on the DEV domain (labels allowed here).

The controller must learn *when an update hurts* without ever touching the
target test references.  This script therefore simulates the causal protocol on
the source domain's own dev split (strict single-source setting) or on an
explicitly declared independent dev domain (aux setting, reported separately).

For every image it records exactly what the online loop would see:

    evidence z_t   (no labels)
    d_local_true   (MSE after commit - MSE before), from dev references

Episodes are chunked (``--episode-len``) so the simulated state cannot drift into
a regime the deployment never reaches; each chunk restarts from theta_0.

Usage:
    python scripts/02_episodes.py --stream LSUI2UIEB --dev-domain source
    python scripts/02_episodes.py --stream LSUI2UIEB --dev-domain EUVP-d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (adapter_ckpt_path, backbone_suffix, load_config, load_degrader,  # noqa: E402
                    open_stream, resolve_ckpt, run_dir, save_json, split_dir)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.adapt import AdaptConfig  # noqa: E402
from brctta.baselines import GatedAdaptation  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--dev-domain", default="source", help="'source' (strict) or a dataset name (aux)")
    ap.add_argument("--episode-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dev_domain == "source":
        dev_ds = stream_cfg["source"]
        dev_rel = cfg["splits"][dev_ds]["dev"]
        dev_name = f"{dev_ds}_dev"
    elif args.dev_domain == "target":
        # Target-domain DEV: the held-out portion of the target dataset, which is
        # never the test split.  This is where the domain shift we must detect
        # actually lives, so it is the honest dev set for calibrating delta and
        # the risk threshold (spec §4: delta fixed on an independent dev set).
        dev_ds = stream_cfg["target"]
        dev_rel = cfg["splits"][dev_ds]["dev"]
        dev_name = f"{dev_ds}_dev_target"
    else:
        dev_ds = args.dev_domain
        dev_rel = cfg["splits"][dev_ds]["dev"]
        dev_name = f"{dev_ds}_dev_aux"

    spec = StreamSpec(
        name=f"dev_{args.stream}_{dev_name}",
        source_dataset=dev_ds, target_split=dev_rel, seed=stream_cfg["seed"],
    )
    stream = open_stream(cfg, dev_ds, "dev", spec)

    # theta_0 = frozen source backbone + (if pre-trained) the SOURCE-trained
    # adapter.  Only source pairs were used to fit it, so this is still a
    # source-only model; it is the same theta_0 for every method downstream.
    adapter_ckpt = adapter_ckpt_path(cfg, stream_cfg["source"], args.backbone)
    print(f"theta_0 = {'source_adapter: ' + adapter_ckpt if adapter_ckpt else 'frozen_backbone'}", flush=True)
    model = build_adapted(resolve_ckpt(cfg, stream_cfg, args.backbone), device=device,
                          hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                          adapter_ckpt=adapter_ckpt)
    acfg = AdaptConfig(
        steps=args.steps if args.steps else cfg["adapter"]["steps"],
        lr=args.lr if args.lr else cfg["adapter"]["lr"],
        grad_clip=cfg["adapter"]["grad_clip"],
        consistency=cfg["adapter"]["consistency"],
        memory=cfg["adapter"]["memory"],
        reset_every=args.episode_len,
        seed=int(stream_cfg["seed"]),
    )
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{stream_cfg['source']}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    # Same source-fitted degradation model the online table uses, so the risk
    # controller is trained on exactly the evidence it will see at test time.
    degrader, degrader_p = load_degrader(cfg, stream_cfg["source"], device)
    print(f"degrader = {degrader_p}" + (" (missing)" if degrader is None else ""), flush=True)
    engine = GatedAdaptation(model, kind="always", cfg=acfg, method="episode_always_accept", degrader=degrader,
                             seed=stream_cfg["seed"], prior=prior)

    out = run_dir(cfg, f"episodes_{args.stream}_{dev_name}" + backbone_suffix(args.backbone)
                + (f"_{args.tag}" if args.tag else ""))
    log = os.path.join(out, "episodes.jsonl")
    t0 = time.time()
    n = args.limit or len(stream)
    summary = engine.run(stream, log_path=log, limit=n)
    summary.update({
        "stream": args.stream, "dev_domain": dev_ds, "dev_split": dev_rel,
        "episode_len": args.episode_len, "n": n, "wall_seconds": time.time() - t0,
        "device": device,
    })
    save_json(summary, os.path.join(out, "episode_summary.json"))
    print(json.dumps({k: summary[k] for k in ("n", "psnr_prequential", "dPSNR_mean", "NAR_0",
                                             "wall_seconds")}, indent=2))
    print("wrote", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

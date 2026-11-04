"""06 -- episodic mode (§5.2 auxiliary protocol).

Episodic TTA may use the *current* image and resets between episodes, so it is a
different protocol from the causal online table and is reported separately.  We
provide it because the spec asks for both, and because the gap between the two
tables is itself evidence about drift.

Usage:
    python scripts/06_episodic.py --stream LSUI2UIEB --episode-len 1 --limit 200
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
from brctta.evaluate import compare, format_table  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402

EPISODIC_NOTE = (
    "Episodic protocol: theta resets to theta_0 at each episode boundary and the "
    "update for an episode may use that episode's own image(s). Do NOT compare "
    "these numbers to the causal online table (spec §5.2)."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--episode-len", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = stream_cfg["source"]

    spec = StreamSpec(name=f"{args.stream}_episodic", source_dataset=ds,
                      target_split=f"{stream_cfg['target']}/test", seed=stream_cfg["seed"])
    stream = open_stream(cfg, stream_cfg["target"], "test", spec)
    n = args.limit or len(stream)

    ckpt = resolve_ckpt(cfg, stream_cfg, args.backbone)
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{ds}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    degrader, _ = load_degrader(cfg, stream_cfg["source"], device)
    acfg = AdaptConfig(
        steps=args.steps or cfg["adapter"]["steps"],
        lr=args.lr or cfg["adapter"]["lr"],
        consistency=cfg["adapter"]["consistency"], memory=0,
        reset_every=args.episode_len,
    )

    out = run_dir(cfg, f"{args.stream}_episodic" + (f"_{args.tag}" if args.tag else ""))
    rows = []
    for method, kind in (("ep_always_accept", "always"), ("ep_oracle", "oracle")):
        model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"],
                              backbone=args.backbone,
                              adapter_ckpt=adapter_ckpt_path(cfg, stream_cfg["source"], args.backbone))
        eng = GatedAdaptation(model, kind=kind, cfg=acfg, method=method, prior=prior, degrader=degrader)
        t0 = time.time()
        print(f"[{args.stream}] episodic({args.episode_len}) {method} n={n}", flush=True)
        s = eng.run(stream, log_path=os.path.join(out, f"action_log_{method}.jsonl"), limit=n)
        s.update({"wall_seconds": time.time() - t0, "protocol": "episodic",
                  "episode_len": args.episode_len, "n": n})
        rows.append(s)

    payload = {"stream": args.stream, "protocol": "episodic", "episode_len": args.episode_len,
               "note": EPISODIC_NOTE, "methods": compare(rows, reference="ep_always_accept")}
    save_json(payload, os.path.join(out, "episodic_results.json"))
    print(format_table(rows))
    print("wrote", os.path.join(out, "episodic_results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

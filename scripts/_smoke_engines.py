#!/usr/bin/env python3
"""Construct every engine arm on 2 real images before the long run.

WHY this exists: three consecutive `run_main3.sh` launches died inside
`02_episodes.py` with `TypeError: __init__() got an unexpected keyword argument
'degrader'` because `OnlineAdaptation` grew a new parameter and the subclasses in
`baselines.py` did not forward it.  Each failure cost a launch-and-inspect cycle,
and would have cost far more if it had happened after an hour of evaluation.

A 30-second construction smoke test converts that class of error (signature
drift between the engine and its six arms) into a fast local failure.

Usage: bash scripts/gpu_run.sh _smoke_engines.py --stream LSUI2UIEB --n 2
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from brctta.adapt import AdaptConfig, OnlineAdaptation  # noqa: E402
from brctta.baselines import (FrozenBaseline, GatedAdaptation,  # noqa: E402
                             SimpleThresholdGate, StochasticRestoration)
from brctta.data import StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402
from common import (adapter_ckpt_path, load_config, load_degrader, open_stream,  # noqa: E402
                   resolve_ckpt, split_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="LSUI2UIEB")
    ap.add_argument("--backbone", default="mambacue")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    ckpt = resolve_ckpt(cfg, sc, args.backbone)
    ack = adapter_ckpt_path(cfg, sc["source"], args.backbone)
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{sc['source']}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    degrader, dpath = load_degrader(cfg, sc["source"], device)

    spec = StreamSpec(name=f"smoke_{args.stream}", source_dataset=sc["source"],
                      target_split=f"{sc['target']}/dev", seed=sc["seed"])
    stream = open_stream(cfg, sc["target"], "dev", spec)
    print(f"[smoke] stream={args.stream} n={len(stream)} ckpt={os.path.basename(ckpt)} "
          f"theta0={'source_adapter' if ack else 'frozen'} degrader={'yes' if degrader else 'MISSING'}")

    acfg = AdaptConfig(steps=cfg["adapter"]["steps"], lr=cfg["adapter"]["lr"])
    arms = {
        "source_frozen": lambda m: FrozenBaseline(m, device=device, method="source_frozen"),
        "source_adapter": lambda m: FrozenBaseline(m, device=device, method="source_adapter"),
        "always_accept": lambda m: GatedAdaptation(m, kind="always", cfg=acfg, method="always_accept",
                                                   prior=prior, degrader=degrader),
        "random_gate": lambda m: GatedAdaptation(m, kind="random", p=0.5, cfg=acfg, method="random_gate",
                                                 prior=prior, degrader=degrader),
        "oracle": lambda m: GatedAdaptation(m, kind="oracle", cfg=acfg, method="oracle",
                                            prior=prior, degrader=degrader),
        "simple_gate": lambda m: SimpleThresholdGate(m, tau=0.0, cfg=acfg, method="simple_gate",
                                                     prior=prior, degrader=degrader),
        "stochastic": lambda m: StochasticRestoration(m, p=0.05, cfg=acfg, method="stochastic",
                                                      prior=prior, degrader=degrader),
        "ours": lambda m: OnlineAdaptation(m, controller=None, cfg=acfg, method="ours",
                                           device=device, prior=prior, degrader=degrader),
    }

    failures = []
    for name, make in arms.items():
        try:
            model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"],
                                  backbone=args.backbone,
                                  adapter_ckpt=None if name == "source_frozen" else ack)
            eng = make(model)
            log = os.path.join(cfg["paths"]["out_root"], "_smoke", f"{args.stream}_{name}.jsonl")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            # a null controller means "ours" behaves as keep-everything here; the
            # point is construction + one forward + evidence vector, not quality
            summary = eng.run(stream, log_path=log, limit=args.n)
            keys = ("psnr_prequential", "ssim_prequential", "accept_rate")
            print(f"  {name:16s} ok  " + "  ".join(f"{k}={summary.get(k)}" for k in keys))
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)))
            print(f"  {name:16s} FAIL {exc!r}")
            traceback.print_exc(limit=3)

    if failures:
        print(f"\n[smoke] {len(failures)} arm(s) FAILED: {[f[0] for f in failures]}")
        return 1
    print(f"\n[smoke] all {len(arms)} arms construct and run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

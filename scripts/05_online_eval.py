"""05 -- causal online evaluation of every method on one locked transfer.

Runs, in the SAME stream order and with the SAME budget:
    source_frozen, always_accept, random_gate, oracle, ours
and writes one row per method into runs/<stream>/online_results.json, plus the
per-image action log for ``ours`` (action_log.jsonl) that the replay check uses.

The target test references are read only here, only for scoring, and never feed
back into any state (the engine has no code path that reads ``gt`` except the
metric calls in this script).

Usage:
    python scripts/05_online_eval.py --stream LSUI2UIEB [--limit 200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (adapter_ckpt_path, backbone_suffix, load_config, load_degrader, open_stream,  # noqa: E402
                    resolve_ckpt, run_dir, save_json, split_dir, theta0_label)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.adapt import AdaptConfig, OnlineAdaptation  # noqa: E402
from brctta.baselines import (  # noqa: E402
    FrozenBaseline, GatedAdaptation, SimpleThresholdGate, StochasticRestoration,
)
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.evaluate import compare, format_table  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402
from brctta.risk import RiskController  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--methods", default="source_frozen,source_adapter,always_accept,random_gate,"
                                            "oracle,simple_gate,stochastic,ours")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--episode-len", type=int, default=0, help=">0 switches to episodic chunking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic-p", type=float, default=-1.0,
                    help="per-element restore probability for the CoTTA-style arm; "
                         "set equal to the gate's measured reset rate after the first pass")
    ap.add_argument("--tag", default="")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_ds, target_split = stream_cfg["target"], "test"

    spec = StreamSpec(
        name=args.stream, source_dataset=stream_cfg["source"],
        target_split=f"{target_ds}/test", seed=stream_cfg["seed"] + args.seed,
    )
    stream = open_stream(cfg, target_ds, target_split, spec)
    n = args.limit or len(stream)

    ckpt = resolve_ckpt(cfg, stream_cfg, args.backbone)
    # theta_0 carries the SOURCE-pretrained residual when it exists.  Every arm
    # below starts from this same theta_0, so the comparison is unaffected --
    # what changes is that "do nothing" is a strong reference, not a weak one.
    adapter_ckpt = adapter_ckpt_path(cfg, stream_cfg["source"], args.backbone)
    print(f"theta_0 = {theta0_label(cfg, stream_cfg['source'], args.backbone)}", flush=True)
    # The self-supervised objective is anchored on the SOURCE domain's colour
    # prior.  It is loaded from the source dataset (never the target) and is part
    # of theta_0's definition, so every method gets the same one.
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{stream_cfg['source']}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    # Re-degradation model D: clean -> degraded, fitted on SOURCE pairs only.
    # Supplies the per-image ||D(y) - x|| evidence.  Missing -> the keys stay
    # absent and the controller sees NaN -> 0.0, so a run is never blocked, but
    # online_results.json records which case it was.
    degrader, degrader_p = load_degrader(cfg, stream_cfg["source"], device)
    print(f"[05] degrader={degrader_p} " + ("(missing)" if degrader is None else "ok"))
    if prior is None:
        print("[warn] no source prior; self-supervised objective will run unanchored")
    ctrl_path = os.path.join(run_dir(cfg, f"risk_{args.stream}" + backbone_suffix(args.backbone)),
                             "risk_controller.json")
    controller = RiskController.load(ctrl_path) if os.path.exists(ctrl_path) else None
    if controller is None:
        print("[warn] no risk controller found; 'ours' will behave like always_accept")

    acfg = AdaptConfig(
        steps=args.steps or cfg["adapter"]["steps"],
        lr=args.lr or cfg["adapter"]["lr"],
        grad_clip=cfg["adapter"]["grad_clip"],
        consistency=cfg["adapter"]["consistency"],
        memory=cfg["adapter"]["memory"],
        reset_every=args.episode_len,
    )

    out = run_dir(cfg, args.stream + backbone_suffix(args.backbone)
                  + (f"_{args.tag}" if args.tag else ""))
    rows = []
    for method in args.methods.split(","):
        method = method.strip()
        if not method:
            continue
        # ``source_frozen`` must stay the BARE backbone, otherwise it silently
        # becomes a duplicate of ``source_adapter`` and the table loses the
        # reference that shows what the source-pretrained residual is worth.
        use_adapter = None if method == "source_frozen" else adapter_ckpt
        model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"],
                              backbone=args.backbone, adapter_ckpt=use_adapter)
        t0 = time.time()
        print(f"[{args.stream}] running {method} (n={n}) ...", flush=True)

        if method == "source_frozen":
            eng = FrozenBaseline(model, device=device)
            s = eng.run(stream, log_path=os.path.join(out, "action_log_source.jsonl"), limit=n)
        elif method == "source_adapter":
            eng = FrozenBaseline(model, device=device, method="source_adapter")
            s = eng.run(stream, log_path=os.path.join(out, "action_log_source_adapter.jsonl"), limit=n)
        elif method == "always_accept":
            eng = GatedAdaptation(model, kind="always", cfg=acfg, method=method, seed=stream_cfg["seed"], prior=prior, degrader=degrader)
            s = eng.run(stream, log_path=os.path.join(out, "action_log_always.jsonl"), limit=n)
        elif method == "random_gate":
            eng = GatedAdaptation(model, kind="random", p=0.5, cfg=acfg, method=method, seed=stream_cfg["seed"], prior=prior, degrader=degrader)
            s = eng.run(stream, log_path=os.path.join(out, "action_log_random.jsonl"), limit=n)
        elif method == "oracle":
            eng = GatedAdaptation(model, kind="oracle", cfg=acfg, method=method, seed=stream_cfg["seed"], prior=prior, degrader=degrader)
            s = eng.run(stream, log_path=os.path.join(out, "action_log_oracle.jsonl"), limit=n)
        elif method == "simple_gate":
            eng = SimpleThresholdGate(model, tau=0.0, cfg=acfg, method=method, prior=prior, degrader=degrader)
            s = eng.run(stream, log_path=os.path.join(out, "action_log_simple_gate.jsonl"), limit=n)
        elif method == "stochastic":
            # Rate-matched to the gate: P3 compares the two policies at the same
            # expected amount of state destruction, so the arm must use the reset
            # rate the gate actually produced on this very stream.  A negative
            # --stochastic-p means "auto"; the gate row must already be present.
            p_use = args.stochastic_p
            if p_use < 0:
                gate = next((r for r in rows if r["method"] == "ours"), None)
                if gate is None:
                    raise SystemExit("stochastic with auto p must run after 'ours'")
                p_use = float(gate["reset_rate"])
                p_use = min(max(p_use, 1e-4), 0.5)
            print(f"[{args.stream}] stochastic p={p_use:.4f} (matched reset rate)", flush=True)
            eng = StochasticRestoration(model, p=p_use, cfg=acfg, method=method,
                                        prior=prior, seed=stream_cfg["seed"])
            s = eng.run(stream, log_path=os.path.join(out, "action_log_stochastic.jsonl"), limit=n)
            s["matched_p"] = p_use
        elif method == "ours":
            eng = OnlineAdaptation(model, controller=controller, cfg=acfg, method=method, device=device, prior=prior, degrader=degrader)
            s = eng.run(stream, log_path=os.path.join(out, "action_log.jsonl"), limit=n)
        else:
            raise ValueError(method)

        s["wall_seconds"] = time.time() - t0
        s["stream"] = args.stream
        s["limit"] = n
        s["episode_len"] = args.episode_len
        rows.append(s)
        print(format_table([s]), flush=True)

    cmp = compare(rows, reference="source_frozen")
    payload = {
        "stream": args.stream,
        "target_split": f"{target_ds}/{target_split}",
        "n": n,
        "order_digest": spec.digest(),
        "adapter_cfg": acfg.__dict__,
        "controller": ctrl_path if os.path.exists(ctrl_path) else None,
        "backbone": args.backbone,
        "theta0": theta0_label(cfg, stream_cfg["source"], args.backbone),
        "theta0_ckpt": adapter_ckpt,
        "degrader": degrader_p if degrader is not None else None,
        "device": device,
        "methods": cmp,
    }
    save_json(payload, os.path.join(out, "online_results.json"))
    print()
    print(format_table(cmp))
    print("wrote", os.path.join(out, "online_results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

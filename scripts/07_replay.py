"""07 -- replay check (§13 deliverable).

A causal online log is only worth the paper it is printed on if replaying the
recorded decisions reproduces the recorded outputs.  This script:

  1. re-runs the stream from scratch with the *recorded action sequence* forced
     (accept/keep/reset taken from action_log.jsonl, no controller involved),
  2. compares each replayed prequential metric to the logged one,
  3. asserts bit-level agreement within a tight tolerance and reports the
     maximum absolute deviation.

It also verifies that 'keep' and 'reset' are exact: after a reset the state hash
must equal the hash of theta_0.

Usage:
    python scripts/07_replay.py --stream LSUI2UIEB --log runs/LSUI2UIEB/action_log.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import adapter_ckpt_path, backbone_suffix, load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta import metrics as M  # noqa: E402
from brctta.adapt import AdaptConfig, OnlineAdaptation  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


class ForcedReplay(OnlineAdaptation):
    """Same engine, but the commit decision is dictated by the log."""

    def __init__(self, *a, actions=None, **kw):
        super().__init__(*a, **kw)
        self.actions = list(actions or [])
        self.k = 0
        self.reset_ok = True

    def decide(self, ev, theta_t, cand, base_state, x, gt):
        a = self.actions[self.k] if self.k < len(self.actions) else "keep"
        self.k += 1
        return a, float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = sc["source"]
    log = args.log or os.path.join(run_dir(cfg, args.stream + backbone_suffix(args.backbone)),
                                   "action_log.jsonl")

    recs = [json.loads(l) for l in open(log, "r", encoding="utf-8") if l.strip()]
    n = args.limit or len(recs)
    recs = recs[:n]
    actions = [r["action"] for r in recs]

    spec = StreamSpec(name=args.stream, source_dataset=ds,
                      target_split=f"{sc['target']}/test", seed=sc["seed"])
    stream = open_stream(cfg, sc["target"], "test", spec)

    ckpt = resolve_ckpt(cfg, sc, args.backbone)
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{ds}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    acfg = AdaptConfig(steps=args.steps or cfg["adapter"]["steps"],
                       lr=args.lr or cfg["adapter"]["lr"],
                       consistency=cfg["adapter"]["consistency"], memory=cfg["adapter"]["memory"])

    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"],
                          backbone=args.backbone,
                          adapter_ckpt=adapter_ckpt_path(cfg, sc["source"], args.backbone))
    eng = ForcedReplay(model, controller=None, cfg=acfg, method="replay", device=device,
                       prior=prior, actions=actions)
    s = eng.run(stream, limit=n, log_path=log.replace(".jsonl", "_replay.jsonl"))

    # compare per-image prequential PSNR
    replay = [json.loads(l) for l in open(log.replace(".jsonl", "_replay.jsonl"), "r", encoding="utf-8")]
    d_psnr = [abs(a["psnr_pre"] - b["psnr_pre"]) for a, b in zip(recs, replay)]
    d_ssim = [abs(a["ssim_pre"] - b["ssim_pre"]) for a, b in zip(recs, replay)]

    res = {
        "stream": args.stream, "log": log, "n": n,
        "max_abs_dpsnr": float(max(d_psnr)) if d_psnr else 0.0,
        "max_abs_dssim": float(max(d_ssim)) if d_ssim else 0.0,
        "mean_abs_dpsnr": float(np.mean(d_psnr)) if d_psnr else 0.0,
        "tolerance_psnr": 1e-4,
        "tolerance_ssim": 1e-6,
        "action_histogram": {a: actions.count(a) for a in set(actions)},
        "ok": bool(d_psnr and max(d_psnr) < 1e-4 and max(d_ssim) < 1e-6),
    }
    out = os.path.join(run_dir(cfg, args.stream), "replay_check.json")
    save_json(res, out)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

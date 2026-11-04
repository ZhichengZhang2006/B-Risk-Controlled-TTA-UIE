"""03 -- train + calibrate the risk controller on DEV episodes (never on test).

Inputs : episodes.jsonl produced by 02_episodes.py (evidence + d_local_true).
Outputs: risk_controller.json with mean/std/coef, threshold, temperature, delta.

Two-stage split of the DEV episodes:
    fit split  (first 70%)  -> logistic regression weights
    cal split  (last 30%)   -> temperature scaling + threshold selection
The threshold targets a false-accept-rate budget (--target-far) so that the
decision rule is fixed *before* the target test split is ever opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, run_dir, save_json  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.risk import RiskController, choose_delta, risk_labels  # noqa: E402
from brctta.signals import EVIDENCE_KEYS, evidence_array  # noqa: E402


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--episodes", default=None, help="path to episodes.jsonl")
    ap.add_argument("--delta-quantile", type=float, default=None)
    ap.add_argument("--target-far", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.episodes:
        ep_path = args.episodes
    else:
        base = run_dir(cfg, f"episodes_{args.stream}_source_dev")
        ep_path = os.path.join(base, "episodes.jsonl")
    rows = read_jsonl(ep_path)

    X = evidence_array([r["evidence"] for r in rows])
    d_local = np.asarray([r["d_local_true"] for r in rows], dtype=np.float64)

    q = args.delta_quantile if args.delta_quantile is not None else cfg["risk"]["delta_quantile"]
    delta = choose_delta(d_local, q)
    y = risk_labels(d_local, delta)

    # chronological split: the last 30% of the episode stream is the calibration
    # block, which mimics "later data used only for calibration".
    n = len(X)
    cut = max(10, int(0.7 * n))
    Xf, yf = X[:cut], y[:cut]
    Xc, yc = X[cut:], y[cut:]
    dc = d_local[cut:]

    ctrl = RiskController(keys=list(EVIDENCE_KEYS), delta=delta)
    ctrl.fit(Xf, yf, l2=cfg["risk"]["l2"], iters=cfg["risk"]["iters"])
    cal = ctrl.calibrate(Xc, yc)
    far = args.target_far if args.target_far is not None else cfg["risk"]["target_far"]
    thr = ctrl.select_threshold(Xc, yc, dc, target_far=far)
    ctrl.reset_threshold = min(0.95, ctrl.threshold + 0.15)
    ctrl.meta = {
        "stream": args.stream, "episodes": ep_path, "n_total": n, "n_fit": len(Xf), "n_cal": len(Xc),
        "delta_quantile": q, "delta": delta, "target_far": far,
        "pos_rate": float(y.mean()), "cal": cal, "threshold": thr,
        "episode_order": "chronological",
    }
    out = args.out or os.path.join(run_dir(cfg, f"risk_{args.stream}"), "risk_controller.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ctrl.save(out)

    print(json.dumps({
        "stream": args.stream, "n": n, "delta": round(delta, 6),
        "pos_rate": round(float(y.mean()), 4),
        "threshold": round(ctrl.threshold, 4), "reset_threshold": round(ctrl.reset_threshold, 4),
        "temperature": round(ctrl.temperature, 3),
        "cal_far": round(thr["far"], 4), "cal_benefit_kept": round(thr["benefit_kept"], 4),
    }, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

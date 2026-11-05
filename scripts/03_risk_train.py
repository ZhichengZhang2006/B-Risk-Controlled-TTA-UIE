"""03 -- train + calibrate the risk controller on DEV episodes (never on test).

Inputs : episodes.jsonl from 02_episodes.py (evidence + d_local_true).
Outputs: risk_controller.json (standardisation, coefficients, threshold, delta, meta).

What changed from the first version, and why
--------------------------------------------
The first version fitted 19 features on ~62 rows and then selected the accept
threshold on the next ~27 rows.  Two consequences, both visible in the first full
run: (a) a 19-parameter model on 62 rows is mostly noise, (b) with 27 calibration
rows and a 10% false-accept budget the threshold was driven below almost every
predicted probability, so the gate accepted 1.1% of updates -- indistinguishable
from "never update".

This version keeps the same *decision rule* (one linear score, one threshold) but
makes the two estimates honest:

  1. **Feature selection inside the fold.**  Candidate feature counts k in {3, 5, 19}
     are compared by 5-fold out-of-fold AUC on the dev episodes, where the top-k
     features themselves are re-selected on each training fold.  Selecting the best
     of 19 signals on the same rows that score it is optimistic; this is not.
  2. **Threshold from cross-fitted probabilities.**  The accept threshold is chosen
     on the concatenated out-of-fold predictions, so the false-accept rate is
     estimated from all dev rows rather than a single 27-row block.
  3. **Abstain when the evidence is uninformative.**  If the best out-of-fold AUC is
     below 0.5 the controller cannot certify any update on this stream; it then
     sets the threshold to -inf (accept nothing) and records
     mode="abstain_no_signal".  That is a declared rule, fixed before the test split
     is opened, and it is the honest behaviour of a risk controller: refusing to act
     when it cannot estimate risk.  Refusing is not the same as controlling risk,
     and the report says so.

Class weights balance the two labels inside every fit.

Usage:
    python scripts/03_risk_train.py --stream LSUI2UIEB
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

K_GRID = (3, 5, 19)
N_FOLDS = 5


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc(y: np.ndarray, s: np.ndarray) -> float:
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    npos, nneg = int((y > 0).sum()), int((y <= 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[y > 0].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def class_weights(y: np.ndarray) -> np.ndarray:
    npos = max(int((y > 0).sum()), 1)
    nneg = max(int((y <= 0).sum()), 1)
    w = np.where(y > 0, len(y) / (2.0 * npos), len(y) / (2.0 * nneg))
    return w


def select_topk(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    uni = [abs(auc(y, X[:, j]) - 0.5) if np.isfinite(X[:, j]).any() else 0.0 for j in range(X.shape[1])]
    return list(np.argsort(uni)[::-1][:k])


def cv_probs(X: np.ndarray, y: np.ndarray, k: int, folds: int, seed: int):
    """Out-of-fold probabilities, with feature selection redone on each train fold."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    probs = np.full(len(y), np.nan)
    for i, te in enumerate(parts):
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        cols = select_topk(X[tr], y[tr], k)
        c = RiskController(keys=[EVIDENCE_KEYS[j] for j in cols])
        c.fit(X[tr][:, cols], y[tr], l2=1e-2, sample_weight=class_weights(y[tr]))
        probs[te] = [c.proba(X[r, cols]) for r in te]
    return probs


def pick_threshold(probs: np.ndarray, y: np.ndarray, target_far: float) -> dict:
    harmful = y.astype(bool)
    best = None
    grid = np.unique(np.round(probs, 4))
    curve = []
    for thr in grid:
        acc = probs <= thr
        if harmful.sum() == 0:
            continue
        far = float((acc & harmful).sum() / harmful.sum())
        kept = float((acc & ~harmful).sum() / max((~harmful).sum(), 1))
        curve.append({"threshold": float(thr), "far": far, "benefit_kept": kept,
                      "accept_rate": float(acc.mean())})
        if far <= target_far and (best is None or kept > best["benefit_kept"]):
            best = {"threshold": float(thr), "far": far, "benefit_kept": kept,
                    "accept_rate": float(acc.mean())}
    if best is None:
        best = {"threshold": float(np.min(probs)) if len(probs) else 0.0,
                "far": float("nan"), "benefit_kept": float("nan"),
                "accept_rate": float((probs <= np.min(probs)).mean()) if len(probs) else 0.0,
                "note": "no threshold met the FAR budget; fell back to the most conservative cut"}
    return {"best": best, "curve": curve}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--delta-quantile", type=float, default=None)
    ap.add_argument("--target-far", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    target_far = args.target_far if args.target_far is not None else cfg["risk"]["target_far"]

    # ---- model selection: how many signals does the dev pool support? ----
    oof = {}
    for k in K_GRID:
        if k > X.shape[1]:
            continue
        p = cv_probs(X, y, k, N_FOLDS, args.seed)
        oof[k] = {"auc": auc(y, p), "probs": p}
    best_k = max(oof, key=lambda k: (oof[k]["auc"], -k)) if oof else 3
    best_auc = oof[best_k]["auc"] if oof else float("nan")

    thr_info = pick_threshold(oof[best_k]["probs"], y, target_far)
    mode = "risk_controlled"
    if not np.isfinite(best_auc) or best_auc < 0.5:
        # Declared rule: no certifiable signal on dev -> do not accept anything.
        mode = "abstain_no_signal"
        thr_info["best"] = {"threshold": float("-inf"), "far": float("nan"),
                            "benefit_kept": 0.0, "accept_rate": 0.0,
                            "note": "out-of-fold AUC <= 0.5 on dev; controller abstains by design"}

    # ---- final model on all dev rows, using the selected feature count ----
    cols = select_topk(X, y, best_k)
    ctrl = RiskController(keys=[EVIDENCE_KEYS[j] for j in cols], delta=delta)
    ctrl.fit(X[:, cols], y, l2=1e-2, iters=cfg["risk"]["iters"], sample_weight=class_weights(y))
    ctrl.temperature = 1.0            # threshold comes from cross-fitted probabilities
    ctrl.threshold = float(thr_info["best"]["threshold"])
    ctrl.reset_threshold = float(min(0.95, ctrl.threshold + 0.15)) if mode == "risk_controlled" else 1.01

    uni = {k: auc(y, X[:, j]) for j, k in enumerate(EVIDENCE_KEYS)}
    ctrl.meta = {
        "stream": args.stream, "episodes": ep_path, "n": len(rows),
        "delta_quantile": q, "delta": delta, "pos_rate": float(y.mean()),
        "mode": mode, "k_features": best_k, "selected_keys": ctrl.keys,
        "oof_auc_by_k": {str(k): float(v["auc"]) for k, v in oof.items()},
        "oof_auc": float(best_auc),
        "univariate_auc": {k: (None if not np.isfinite(v) else float(v)) for k, v in uni.items()},
        "threshold": ctrl.threshold, "reset_threshold": ctrl.reset_threshold,
        "target_far": target_far, "threshold_info": thr_info["best"],
        "risk_coverage": thr_info["curve"],
        "n_folds": N_FOLDS, "cv_seed": args.seed,
        "calibration": "none: threshold selected on cross-fitted dev probabilities",
        "note": ("Dev-only fitting. Feature count and threshold were chosen before the "
                 "test split was read; nothing here is tuned on test."),
    }
    out = args.out or os.path.join(run_dir(cfg, f"risk_{args.stream}"), "risk_controller.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ctrl.save(out)

    print(json.dumps({
        "stream": args.stream, "n": len(rows), "mode": mode,
        "delta": round(delta, 8), "pos_rate": round(float(y.mean()), 4),
        "k_features": best_k, "selected_keys": ctrl.keys,
        "oof_auc_by_k": ctrl.meta["oof_auc_by_k"], "oof_auc": round(float(best_auc), 4),
        "threshold": (ctrl.threshold if np.isfinite(ctrl.threshold) else "-inf"),
        "target_far": target_far,
        "dev_accept_rate": round(float(thr_info["best"].get("accept_rate") or 0.0), 4),
        "dev_far": thr_info["best"].get("far"),
        "dev_benefit_kept": thr_info["best"].get("benefit_kept"),
    }, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

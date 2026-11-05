#!/usr/bin/env python3
"""Which evidence keys actually separate a helpful update from a harmful one?

WHY THIS IS THE CENTRAL DIAGNOSTIC
----------------------------------
The first full causal run produced a gate that accepts 1.1% of updates on
LSUI2UIEB, i.e. it collapses to "never update".  That is *safe*, but it means
the evidence vector carries too little information for the controller to
separate good updates from bad ones -- or that the controller is misspecified.

This script answers the question directly and cheaply, from an action log that
already exists: for every evidence key, compute the AUC for predicting
``beneficial`` (d_local_true > 0).  AUC = 0.5 means the signal is noise; the
gate cannot work no matter how it is thresholded or calibrated.  AUC far from
0.5 in the *wrong* direction (e.g. 0.3) is more interesting still: the signal is
informative but with the opposite sign to the assumption, which is exactly the
"loss went down but the image got worse" failure the spec warns about.

Also reported:
  * a logistic model on ALL keys with 5-fold cross-validated AUC, so the
    multi-key combination is judged out-of-sample (a 19-key linear model on 89
    samples overfits happily in-sample);
  * the best single key and its AUC, which is the honest baseline a learned
    controller must beat;
  * the oracle accept-rate and the gain available at that rate.

Usage:
    bash scripts/gpu_run.sh 04l_signal_auc.py --log runs/online_LSUI2UIEB/action_log.jsonl
    bash scripts/gpu_run.sh 04l_signal_auc.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from brctta.signals import EVIDENCE_KEYS  # noqa: E402


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), NaN-safe.

    Implemented by hand rather than importing sklearn so the number does not
    depend on a library version -- same reason the metrics module avoids
    skimage at eval time.
    """
    mask = np.isfinite(score)
    y, s = y_true[mask], score[mask]
    n_pos = int((y > 0).sum())
    n_neg = int((y <= 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logistic(X, y, l2=1e-2, iters=400, lr=0.1):
    """Plain gradient-descent logistic regression, mean-imputed features."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.where(np.isfinite(X), X, mu)
    Z = (Z - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w, mu, sd


def _predict(w, mu, sd, X):
    Z = np.where(np.isfinite(X), X, mu)
    Z = (Z - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))


def cv_auc(X, y, folds=5, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    scores = np.zeros(len(y))
    for i, te in enumerate(parts):
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        w, mu, sd = _fit_logistic(X[tr], y[tr])
        scores[te] = _predict(w, mu, sd, X[te])
    return auc(y, scores), scores


def analyse(path: str) -> dict:
    rows = read_jsonl(path)
    if not rows:
        return {"path": path, "error": "empty log"}
    # SIGN CONVENTION (this was wrong in the first version of this script and
    # made every row look harmful): `ActionRecord.d_local_true` is
    # ``mse_post - mse_pre``, i.e. a LOSS delta where LOWER is better.  A
    # beneficial update therefore has d_local_true < 0.  The PSNR columns are
    # the human-readable view and have the opposite sign.
    d_local = np.array([r["d_local_true"] for r in rows], dtype=np.float64)
    d_psnr = np.array([r["psnr_post"] - r["psnr_pre"] for r in rows], dtype=np.float64)
    y = (d_local < 0).astype(np.float64)

    # The evidence vector is NESTED under "evidence" in the jsonl record.  An
    # earlier version of this script read r.get(k) at the top level, got NaN for
    # all 19 keys, used no keys at all, and then confidently printed
    # "no_signal" -- a silent false negative that would have killed the whole
    # mechanism for the wrong reason.
    def ev(r, k):
        e = r.get("evidence")
        return e.get(k, np.nan) if isinstance(e, dict) else np.nan

    cols, keys_used = [], []
    for k in EVIDENCE_KEYS:
        v = np.array([ev(r, k) for r in rows], dtype=np.float64)
        finite_frac = np.isfinite(v).mean()
        if np.isfinite(v).sum() >= max(8, 0.5 * len(v)) and np.nanstd(v) > 1e-12:
            cols.append(v)
            keys_used.append(k)
    X = np.stack(cols, axis=1) if cols else np.zeros((len(rows), 0))

    per_key = {}
    for j, k in enumerate(keys_used):
        a = auc(y, X[:, j])
        per_key[k] = {"auc": a, "abs_dev": abs(a - 0.5) if np.isfinite(a) else 0.0,
                      "mean_pos": float(np.nanmean(X[y > 0, j])) if (y > 0).any() else None,
                      "mean_neg": float(np.nanmean(X[y <= 0, j])) if (y <= 0).any() else None}
    ranked = sorted(per_key.items(), key=lambda kv: -kv[1]["abs_dev"])

    multi_auc = None
    if len(keys_used) >= 2 and len(rows) >= 20:
        multi_auc, _ = cv_auc(X, y)
    # A single key's AUC is invariant to monotone transforms, so the best single
    # key IS the best single-threshold rule -- no fitting needed, which is why
    # "best_single_auc" below is computed directly from the per-key table.
    pos_rate = float(y.mean())
    out = {
        "log": path, "n": len(rows), "pos_rate_beneficial": pos_rate,
        "d_local_mean_mse": float(d_local.mean()),
        "d_local_median_mse": float(np.median(d_local)),
        "dPSNR_mean": float(d_psnr.mean()), "dPSNR_median": float(np.median(d_psnr)),
        "keys_used": keys_used, "per_key": per_key,
        "ranked": [{"key": k, "auc": v["auc"], "abs_dev": v["abs_dev"]} for k, v in ranked],
        "best_single_key": ranked[0][0] if ranked else None,
        "best_single_auc": ranked[0][1]["auc"] if ranked else None,
        "multi_key_cv_auc": multi_auc,
        "verdict": None,
    }
    best = ranked[0][1]["auc"] if ranked else float("nan")
    if not keys_used:
        out["verdict"] = "no_keys: evidence vector was empty (bug, not a finding)"
    elif not np.isfinite(best) or abs(best - 0.5) < 0.05:
        out["verdict"] = "no_signal: no evidence key separates helpful from harmful updates"
    elif abs(best - 0.5) >= 0.05 and best < 0.5:
        out["verdict"] = f"informative_but_inverted: best key {ranked[0][0]} AUC {best:.3f} < 0.5"
    else:
        out["verdict"] = f"usable_signal: best key {ranked[0][0]} AUC {best:.3f}"
    if multi_auc is not None:
        out["verdict"] += f"; all-keys CV AUC {multi_auc:.3f}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.all:
        # action logs land in runs/<STREAM>/action_log.jsonl (the arm name is in
        # the filename), not in a runs/online_* subdirectory -- an earlier glob
        # for runs/online_*/ found nothing and the diagnostic silently reported
        # "no action logs found".
        primary = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "action_log.jsonl")))
        primary = [p for p in primary if "_smoke" not in p and "_dbg" not in p]
        # Also analyse the always-accept log: it contains the full mix of good and
        # bad updates, so it is the right sample for asking "is there a signal?".
        # The gated arm's own log is dominated by 'keep' rows and under-reports
        # signal quality.
        always = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "action_log_always.jsonl")))
        logs = primary + [p for p in always if "_smoke" not in p]
    elif args.log:
        logs = [args.log]
    else:
        print("[auc] pass --log PATH or --all")
        return 2

    if not logs:
        print("[auc] no action logs found")
        return 2

    all_out = {}
    for path in logs:
        parent = os.path.basename(os.path.dirname(path))
        name = parent if os.path.basename(path) == "action_log.jsonl" else             f"{parent}/{os.path.basename(path)[:-6]}"
        res = analyse(path)
        all_out[name] = res
        if "error" in res:
            print(f"[auc] {name}: {res['error']}")
            continue
        print(f"\n=== {name}  n={res['n']}  beneficial_rate={res['pos_rate_beneficial']:.3f}")
        print(f"    d_local (loss, lower=better): mean {res['d_local_mean_mse']:+.3e} "
              f"median {res['d_local_median_mse']:+.3e}")
        print(f"    dPSNR: mean {res['dPSNR_mean']:+.4f} dB  median {res['dPSNR_median']:+.4f} dB")
        print(f"    {'key':22s} {'AUC':>7s}  {'|AUC-0.5|':>9s}   mean(beneficial) mean(harmful)")
        for item in res["ranked"][:10]:
            k = item["key"]
            v = res["per_key"][k]
            mp = f"{v['mean_pos']:+.4e}" if v["mean_pos"] is not None else "     n/a"
            mn = f"{v['mean_neg']:+.4e}" if v["mean_neg"] is not None else "     n/a"
            print(f"    {k:22s} {v['auc']:7.3f}  {v['abs_dev']:9.3f}   {mp} {mn}")
        print(f"    VERDICT: {res['verdict']}")

    out_path = args.out or os.path.join(ROOT, "runs", "manifests", "signal_auc.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_out, fh, indent=2)
    print(f"\n[auc] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""04m -- does the evidence vector separate helpful from harmful updates on DEV?

This is the design-time question the whole method depends on, and it is asked
only on dev (the always-accept proposal pool from 02_episodes), never on test.
Every number here is therefore a legitimate input to designing the controller;
nothing is selected after looking at a test metric.

Why a dedicated script: ``04l_signal_auc`` reads *test* action logs and is a
descriptive post-mortem.  This one reads ``runs/episodes_*/episodes.jsonl``,
which is the dev pool the controller is actually fitted on, and it evaluates the
multi-key model with fold-wise feature selection so the reported AUC is not
inflated by selecting the best of 19 keys on the same rows.

Reported per pool and pooled
  * univariate AUC of every evidence key, with the sign of the effect;
  * the label rate for two definitions of "harmful":
      mse_delta  : d_local_true > delta (delta = dev median, spec §4)
      psnr_drop  : dPSNR_candidate < 0
  * cross-validated AUC of a logistic model on all keys (out-of-fold);
  * cross-validated AUC of a logistic model whose features are chosen *inside*
    each fold (top-k by univariate AUC), which is what an honest "pick the useful
    signals" procedure would achieve.

Reading the result
  CV AUC ~ 0.5          -> no usable signal; a gate can only learn "do nothing"
  0.5 < CV AUC < 0.65   -> weak signal; a gate can shave the left tail but will
                           also reject benefit (this is what the first run showed)
  CV AUC >= 0.7         -> the evidence supports a real accept/reject decision

Usage:
    bash scripts/gpu_run.sh 04m_pool_auc.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, save_json  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.signals import EVIDENCE_KEYS  # noqa: E402


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc(y, s):
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


def fit_logistic(Z, y, l2=1e-2, iters=500, lr=0.1):
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    return w


def design(X):
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (np.where(np.isfinite(X), X, mu) - mu) / sd
    return Z, mu, sd


def cv_auc(X, y, folds=5, seed=0, topk=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    scores = np.zeros(len(y))
    for i, te in enumerate(parts):
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        cols = np.arange(X.shape[1])
        if topk and topk < X.shape[1]:
            # select features inside the fold, never on the held-out rows
            uni = [abs(auc((y[tr] > 0).astype(float), X[tr, j]) - 0.5) for j in cols]
            cols = np.argsort(uni)[::-1][:topk]
        Ztr, mu, sd = design(X[tr][:, cols])
        w = fit_logistic(Ztr, y[tr])
        Zte = (np.where(np.isfinite(X[te][:, cols]), X[te][:, cols], mu) - mu) / sd
        scores[te] = 1.0 / (1.0 + np.exp(-np.clip(np.hstack([Zte, np.ones((len(te), 1))]) @ w, -30, 30)))
    return auc(y, scores)


def analyse_pool(path, delta_quantile=0.5):
    rows = read_jsonl(path)
    if len(rows) < 10:
        return None
    d_local = np.array([r["d_local_true"] for r in rows], dtype=np.float64)
    dc_psnr = np.array([r.get("psnr_cand", np.nan) - r["psnr_pre"] for r in rows], dtype=np.float64)
    if not np.isfinite(dc_psnr).all():  # legacy log: post is the candidate under always-accept
        dc_psnr = np.array([r["psnr_post"] - r["psnr_pre"] for r in rows], dtype=np.float64)

    cols, keys = [], []
    for k in EVIDENCE_KEYS:
        v = np.array([(r.get("evidence") or {}).get(k, np.nan) for r in rows], dtype=np.float64)
        if np.isfinite(v).sum() >= max(10, 0.8 * len(v)) and np.nanstd(v) > 1e-12:
            cols.append(v)
            keys.append(k)
    X = np.stack(cols, axis=1) if cols else np.zeros((len(rows), 0))

    out = {"path": path, "n": len(rows), "keys": keys,
           "d_local_mean": float(d_local.mean()), "d_local_median": float(np.median(d_local)),
           "psnr_cand_mean": float(dc_psnr.mean()),
           "psnr_cand_neg_frac": float((dc_psnr < 0).mean())}
    delta = float(np.quantile(d_local, delta_quantile))
    out["delta_median_mse"] = delta

    for name, y in (("mse_delta", (d_local > delta).astype(float)),
                    ("psnr_drop", (dc_psnr < 0).astype(float))):
        entry = {"pos_rate": float(y.mean()), "per_key": {}}
        for j, k in enumerate(keys):
            a = auc(y, X[:, j])
            entry["per_key"][k] = {"auc": a, "abs_dev": abs(a - 0.5) if np.isfinite(a) else 0.0}
        ranked = sorted(entry["per_key"].items(), key=lambda kv: -kv[1]["abs_dev"])
        entry["ranked"] = [{"key": k, "auc": v["auc"]} for k, v in ranked]
        if len(keys) >= 2 and len(rows) >= 25:
            entry["cv_auc_all_keys"] = cv_auc(X, y)
            entry["cv_auc_top5_inside_fold"] = cv_auc(X, y, topk=5)
            entry["cv_auc_top3_inside_fold"] = cv_auc(X, y, topk=3)
        out[name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    pools = sorted(glob.glob(os.path.join(cfg["paths"]["out_root"], "episodes_*", "episodes.jsonl")))
    if not pools:
        print("[pool] no dev episode pools found")
        return 2

    all_out = {}
    for path in pools:
        name = os.path.basename(os.path.dirname(path))
        res = analyse_pool(path)
        if not res:
            continue
        all_out[name] = res
        print(f"\n=== {name}  n={res['n']}  keys={len(res['keys'])}")
        print(f"    d_local (MSE) mean {res['d_local_mean']:+.2e} median {res['d_local_median']:+.2e}"
              f"   dPSNR_cand mean {res['psnr_cand_mean']:+.3f} dB, harm frac {res['psnr_cand_neg_frac']:.3f}")
        for lab in ("mse_delta", "psnr_drop"):
            e = res[lab]
            top = ", ".join(f"{r['key']} {r['auc']:.2f}" for r in e["ranked"][:5])
            cv = (f"  CV all={e.get('cv_auc_all_keys', float('nan')):.3f}"
                  f" top5={e.get('cv_auc_top5_inside_fold', float('nan')):.3f}"
                  f" top3={e.get('cv_auc_top3_inside_fold', float('nan')):.3f}")
            print(f"    [{lab}] pos_rate={e['pos_rate']:.3f}  top: {top}{cv}")

    out = args.out or os.path.join(cfg["paths"]["out_root"], "manifests", "pool_auc_dev.json")
    save_json(all_out, out)
    print(f"\n[pool] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

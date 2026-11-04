"""Result summarisation + the negative-transfer metrics of spec §9.2.

Every method is scored the same way from the same per-image records, so the
comparison table cannot drift between methods.

Terminology used below
----------------------
prequential : the metric of the output produced *before* the update (the number
              that matters under the causal protocol).
final       : the metric of the output produced after committing/keeping -- used
              only for negative-transfer accounting.
dPSNR       : psnr_post - psnr_pre for the same image.  Negative = the update
              hurt this image.
NAR_0       : fraction of images whose update reduced PSNR (negative-transfer
              ratio).  NAR_0.1dB / NAR_0.5dB are the margin-qualified variants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

MAIN_KEYS = ("psnr_prequential", "ssim_prequential", "mse_prequential")


def summarize(records: Sequence, method: str = "ours") -> Dict:
    if not records:
        return {"method": method, "n": 0}

    psnr_pre = np.array([r.psnr_pre for r in records], dtype=np.float64)
    psnr_post = np.array([r.psnr_post for r in records], dtype=np.float64)
    ssim_pre = np.array([r.ssim_pre for r in records], dtype=np.float64)
    ssim_post = np.array([r.ssim_post for r in records], dtype=np.float64)
    mse_pre = np.array([r.mse_pre for r in records], dtype=np.float64)
    mse_post = np.array([r.mse_post for r in records], dtype=np.float64)

    d_psnr = psnr_post - psnr_pre
    d_ssim = ssim_post - ssim_pre

    actions = np.array([r.action for r in records])
    n = len(records)
    n_accept = int((actions == "accept").sum())
    n_reset = int((actions == "reset").sum())
    n_keep = n - n_accept - n_reset

    accepted = actions == "accept"
    hurt = d_psnr < 0
    would_hurt = np.array([bool(r.harmful) for r in records])

    false_accept = float((accepted & hurt).sum() / max(hurt.sum(), 1))
    harmful_accept_ratio = float((accepted & would_hurt).sum() / max(accepted.sum(), 1))
    beneficial = ~hurt
    frr = float(((~accepted) & beneficial).sum() / max(beneficial.sum(), 1))

    oracle_gain = float(np.clip(d_psnr, 0, None).sum())
    realised_gain = float(np.where(accepted, d_psnr, 0.0).sum())
    regret = oracle_gain - realised_gain

    k = max(1, int(round(n * 0.10)))
    worst = np.argsort(d_psnr)[:k]

    return {
        "method": method,
        "n": n,
        # --- main table (prequential) ---
        "psnr_prequential": float(psnr_pre.mean()),
        "ssim_prequential": float(ssim_pre.mean()),
        "mse_prequential": float(mse_pre.mean()),
        "psnr_median": float(np.median(psnr_pre)),
        "ssim_median": float(np.median(ssim_pre)),
        # --- post-commit view ---
        "psnr_final": float(psnr_post.mean()),
        "ssim_final": float(ssim_post.mean()),
        "mse_final": float(mse_post.mean()),
        # --- negative transfer (§9.2) ---
        "dPSNR_mean": float(d_psnr.mean()),
        "dPSNR_median": float(np.median(d_psnr)),
        "dSSIM_mean": float(d_ssim.mean()),
        "dSSIM_median": float(np.median(d_ssim)),
        "NAR_0": float((d_psnr < 0).mean()),
        "NAR_0.1dB": float((d_psnr < -0.1).mean()),
        "NAR_0.5dB": float((d_psnr < -0.5).mean()),
        "worst10_dPSNR": float(d_psnr[worst].mean()),
        "accept_rate": n_accept / n,
        "keep_rate": n_keep / n,
        "reset_rate": n_reset / n,
        "false_accept_rate": false_accept,
        "harmful_update_accept_ratio": harmful_accept_ratio,
        "beneficial_false_reject_rate": frr,
        "benefit_retention": (realised_gain / oracle_gain) if oracle_gain > 1e-9 else float("nan"),
        "regret_dB_per_image": regret / n,
        "update_seconds_mean": float(np.mean([r.update_seconds for r in records])),
    }


METRIC_KEYS_FOR_TABLE: List[str] = [
    "psnr_prequential", "ssim_prequential", "psnr_final", "ssim_final",
    "dPSNR_mean", "NAR_0", "worst10_dPSNR", "accept_rate", "false_accept_rate",
    "benefit_retention",
]


def compare(summaries: Sequence[Dict], reference: str = "source_frozen") -> List[Dict]:
    """Attach deltas against a reference row (usually the frozen source model)."""
    ref = next((s for s in summaries if s["method"] == reference), None)
    out = []
    for s in summaries:
        row = dict(s)
        if ref is not None and s is not ref:
            for k in MAIN_KEYS:
                row[f"d_{k}_vs_ref"] = s[k] - ref[k]
        out.append(row)
    return out


def format_table(summaries: Sequence[Dict], keys: Sequence[str] = METRIC_KEYS_FOR_TABLE) -> str:
    """Fixed-width table for the terminal / report."""
    head = f"{'method':<18}" + "".join(f"{k[:14]:>15}" for k in keys)
    lines = [head, "-" * len(head)]
    for s in summaries:
        row = f"{s.get('method', '?'):<18}"
        for k in keys:
            v = s.get(k, float("nan"))
            row += f"{v:>15.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'nan':>15}"
        lines.append(row)
    return "\n".join(lines)

"""Result summarisation + the negative-transfer metrics of spec §9.2.

Every method is scored the same way from the same per-image records, so the
comparison table cannot drift between methods.

Terminology used below
----------------------
prequential : the metric of the output produced *before* the update (the number
              that matters under the causal protocol).  This is the main table.
realised    : ``psnr_post - psnr_pre``.  The output produced after the decision
              minus the one produced before it, i.e. what this step actually did
              to the stream.  A keep/reset leaves the state alone, so this is 0.
candidate   : ``psnr_cand - psnr_pre``.  The quality of the *proposed* state
              theta_+, whether or not it was committed.  This is the utility the
              gate is supposed to predict; measuring it after the decision (the
              pre-fix behaviour) made every rejected update look harmless because
              the rejected state was never evaluated.

Why both views are reported
---------------------------
A gate can look excellent on the realised view by refusing everything: every
step then has ``dPSNR == 0``, so NAR_0, worst-10 % and regret all read as zero.
That is not risk control, it is inaction.  The candidate view exposes the pool
the gate faced (its true harm rate) and the accept/reject rates expose how much
of the available benefit it actually took.  §9.2's negative-transfer numbers are
therefore reported on the realised view (what the stream experienced) *and* on
the candidate view (what the decision was worth), never mixed.

Legacy records (written before the candidate metrics existed) lack
``psnr_cand``; they are scored with the post values and the summary carries
``candidate_labels_legacy: true`` so a downstream table cannot silently treat
them as candidate measurements.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

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

    cand_psnr = np.array([getattr(r, "psnr_cand", np.nan) for r in records], dtype=np.float64)
    cand_ssim = np.array([getattr(r, "ssim_cand", np.nan) for r in records], dtype=np.float64)
    cand_mse = np.array([getattr(r, "mse_cand", np.nan) for r in records], dtype=np.float64)
    legacy = bool(np.isfinite(cand_psnr).sum() < len(records))
    if legacy:  # score old logs on the realised view, flagged as such
        cand_psnr = np.where(np.isfinite(cand_psnr), cand_psnr, psnr_post)
        cand_ssim = np.where(np.isfinite(cand_ssim), cand_ssim, ssim_post)
        cand_mse = np.where(np.isfinite(cand_mse), cand_mse, mse_post)

    # realised view: what the stream actually experienced
    d_psnr = psnr_post - psnr_pre
    d_ssim = ssim_post - ssim_pre
    # candidate view: what the proposed update was worth
    dc_psnr = cand_psnr - psnr_pre
    dc_ssim = cand_ssim - ssim_pre

    actions = np.array([r.action for r in records])
    n = len(records)
    n_accept = int((actions == "accept").sum())
    n_reset = int((actions == "reset").sum())
    n_keep = n - n_accept - n_reset

    accepted = actions == "accept"
    cand_hurt = dc_psnr < 0
    cand_beneficial = ~cand_hurt

    false_accept = float((accepted & cand_hurt).sum() / max(int(cand_hurt.sum()), 1))
    harmful_accept_ratio = float((accepted & cand_hurt).sum() / max(n_accept, 1))
    frr = float(((~accepted) & cand_beneficial).sum() / max(int(cand_beneficial.sum()), 1))

    oracle_gain = float(np.clip(dc_psnr, 0, None).sum())
    realised_gain = float(np.where(accepted, dc_psnr, 0.0).sum())
    regret = oracle_gain - realised_gain

    k = max(1, int(round(n * 0.10)))
    worst_real = np.argsort(d_psnr)[:k]
    worst_cand = np.argsort(dc_psnr)[:k]

    return {
        "method": method,
        "n": n,
        "candidate_labels_legacy": legacy,
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
        # --- negative transfer, realised (§9.2) ---
        "dPSNR_mean": float(d_psnr.mean()),
        "dPSNR_median": float(np.median(d_psnr)),
        "dSSIM_mean": float(d_ssim.mean()),
        "dSSIM_median": float(np.median(d_ssim)),
        "NAR_0": float((d_psnr < 0).mean()),
        "NAR_0.1dB": float((d_psnr < -0.1).mean()),
        "NAR_0.5dB": float((d_psnr < -0.5).mean()),
        "worst10_dPSNR": float(d_psnr[worst_real].mean()),
        # --- candidate pool (what the decision was worth) ---
        "dPSNR_cand_mean": float(dc_psnr.mean()),
        "dPSNR_cand_median": float(np.median(dc_psnr)),
        "NAR_0_cand": float((dc_psnr < 0).mean()),
        "worst10_dPSNR_cand": float(dc_psnr[worst_cand].mean()),
        # --- action statistics ---
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
    "psnr_prequential", "ssim_prequential", "dPSNR_mean", "dPSNR_cand_mean",
    "NAR_0", "NAR_0_cand", "worst10_dPSNR", "accept_rate", "false_accept_rate",
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

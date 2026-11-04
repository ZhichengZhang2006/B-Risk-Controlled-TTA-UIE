"""Candidate evidence for risk estimation (spec §7.2).

Each function below maps ``(theta_t, theta_+, x)`` to scalar evidence.  The
vector is intentionally *diverse*: parameter-space magnitude, output-space
displacement, proxy-objective movement, physical plausibility, and gradient
conflict.  No reference image participates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .losses import _highpass, rgb_prior_loss

EVIDENCE_KEYS: List[str] = [
    "upd_l2_cur",      # ||theta_+ - theta_t||
    "upd_l2_anchor",   # ||theta_+ - theta_0||
    "upd_rel_anchor",  # ||theta_+ - theta_0|| / ||theta_0||
    "dout_cur",        # mean |y_+(x) - y_t(x)|
    "dout_src",        # mean |y_+(x) - y_0(x)|
    "loss_ss",         # proxy objective of theta_+ on x
    "dloss_ss",        # proxy objective change vs theta_t
    "clip_ratio",      # fraction of saturated pixels in y_+
    "color_balance",   # std of channel means of y_+
    "sat_mean",        # mean chroma of y_+
    "contrast_ratio",  # std(y_+) / std(y_t)
    "hf_resid",        # high-frequency share of (x - y_+)
    "grad_cosine",     # cos(g_t, EMA(g))
    "upd_step_norm",   # ||g_t|| (size of the candidate step)
    # --- re-degradation evidence (spec §7.2), filled by OnlineAdaptation when a
    # source-fitted degradation model is supplied.  These are the only keys that
    # constrain the output's RELATION to its own input rather than its statistics.
    "redegrade_cur",   # ||D(y_+) - x||
    "redegrade_t",     # ||D(y_t) - x||
    "redegrade_src",   # ||D(y_0) - x||
    "dloss_redeg",     # redegrade_cur - redegrade_t  (<0 = candidate explains input better)
    "dredeg_src",      # redegrade_cur - redegrade_src (<0 = better than frozen model)
]


def _l2(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for k, v in a.items():
        total += float(((v.detach() - b[k].detach()) ** 2).sum())
    return total ** 0.5


def _chroma(y: torch.Tensor) -> torch.Tensor:
    mx = y.max(dim=1, keepdim=True).values
    mn = y.min(dim=1, keepdim=True).values
    return (mx - mn) / (mx + 1e-6)


def compute_evidence(
    model,
    x: torch.Tensor,
    state_t: Dict[str, torch.Tensor],
    state_plus: Dict[str, torch.Tensor],
    state_0: Dict[str, torch.Tensor],
    y_t: torch.Tensor,
    y_plus: torch.Tensor,
    loss_ss_t: float,
    loss_ss_plus: float,
    grad_cosine: float = 0.0,
    grad_norm: float = 0.0,
    anchor_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Assemble one evidence row.  All tensors are batched [B,3,H,W]."""
    if anchor_norm is None:
        anchor_norm = _l2(state_0, {k: torch.zeros_like(v) for k, v in state_0.items()}) + 1e-8

    clip = ((y_plus <= 1e-4) | (y_plus >= 1 - 1e-4)).float().mean()
    # colour balance = spread of the per-channel means (0 = perfectly grey)
    cb = y_plus.mean(dim=(2, 3)).std(dim=1).mean()
    sat = _chroma(y_plus).mean()
    eps = 1e-6
    contrast = y_plus.std(dim=(1, 2, 3)).mean() / (y_t.std(dim=(1, 2, 3)).mean() + eps)
    hf = _highpass(x - y_plus).abs().mean() / (_highpass(x).abs().mean() + eps)

    return {
        "upd_l2_cur": _l2(state_plus, state_t),
        "upd_l2_anchor": _l2(state_plus, state_0),
        "upd_rel_anchor": _l2(state_plus, state_0) / anchor_norm,
        "dout_cur": float((y_plus - y_t).abs().mean()),
        "dout_src": float("nan"),  # filled by fill_dout_src (needs y_0)
        "loss_ss": float(loss_ss_plus),
        "dloss_ss": float(loss_ss_plus - loss_ss_t),
        "clip_ratio": float(clip),
        "color_balance": float(cb),
        "sat_mean": float(sat),
        "contrast_ratio": float(contrast),
        "hf_resid": float(hf),
        "grad_cosine": float(grad_cosine),
        "upd_step_norm": float(grad_norm),
    }


@torch.no_grad()
def fill_dout_src(ev: Dict[str, float], y_plus: torch.Tensor, y0: torch.Tensor) -> Dict[str, float]:
    ev["dout_src"] = float((y_plus - y0).abs().mean())
    return ev


def evidence_array(rows: List[Dict[str, float]]) -> np.ndarray:
    """Stack evidence rows into a matrix in EVIDENCE_KEYS order.

    Uses ``.get(k, nan)`` rather than ``r[k]``: logs written before a new
    evidence key existed must stay loadable, and ``RiskController`` already maps
    NaN to 0.0 so an absent signal degrades to "no information" instead of
    crashing a replay of an old run.
    """
    return np.asarray([[float(r.get(k, float("nan"))) for k in EVIDENCE_KEYS] for r in rows],
                      dtype=np.float64)

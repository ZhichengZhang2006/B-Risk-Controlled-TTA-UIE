"""Risk controller: maps candidate evidence -> P(update is harmful), then acts.

Why a learned controller and not a hand-tuned threshold?  §7.2 requires the
decision to use *multiple* heterogeneous signals whose scale differs by orders
of magnitude (weight L2 vs pixel differences vs ratios).  A calibrated
probabilistic map over standardised evidence is the smallest model that can
combine them without per-dataset hand tuning -- and it keeps the decision rule
auditable: one linear score, one threshold, one probability.

Training labels come from a labelled *dev* split (never the test split): for an
episode we know d_local = L(theta_+) - L(theta_t), and an update is "harmful"
when d_local > delta.  delta itself is fixed on dev (§4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import EVIDENCE_KEYS


# --------------------------------------------------------------------------- #
# standardisation + logistic model (numpy only: no sklearn dependency at run time)
# --------------------------------------------------------------------------- #
@dataclass
class RiskController:
    keys: List[str] = field(default_factory=lambda: list(EVIDENCE_KEYS))
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    threshold: float = 0.5
    reset_threshold: float = 0.9
    temperature: float = 1.0
    delta: float = 0.0
    meta: Dict[str, float] = field(default_factory=dict)

    # -- fit ---------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l2: float = 1e-3,
        iters: int = 500,
        lr: float = 0.1,
    ) -> "RiskController":
        """L2-regularised logistic regression by full-batch gradient descent."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        Z = (X - self.mean) / self.std

        w = np.zeros(Z.shape[1], dtype=np.float64)
        b = 0.0
        n = len(Z)
        for _ in range(iters):
            logits = Z @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            grad_w = Z.T @ (p - y) / n + l2 * w
            grad_b = float((p - y).mean())
            w -= lr * grad_w
            b -= lr * grad_b
        self.coef, self.intercept = w, b
        return self

    # -- inference ---------------------------------------------------------
    def proba(self, x: Dict[str, float] | np.ndarray) -> float:
        if isinstance(x, dict):
            x = np.asarray([x.get(k, 0.0) for k in self.keys], dtype=np.float64)
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        z = (x - self.mean) / self.std
        logit = float(z @ self.coef + self.intercept)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit / max(self.temperature, 1e-6), -30, 30)))
        return float(p)

    def decide(self, x) -> Tuple[str, float]:
        """Return (action, p_harm) with action in {'accept','keep','reset'}.

        Three-way rule from §4: accept when the candidate looks safe; otherwise
        *keep* the committed state, unless the candidate looks so bad that the
        accumulated drift is the likely culprit -- then *reset* to theta_0.
        """
        p = self.proba(x)
        if p <= self.threshold:
            return "accept", p
        if p > self.reset_threshold:
            return "reset", p
        return "keep", p

    # -- calibration -------------------------------------------------------
    def calibrate(self, X: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> dict:
        """Platt-style temperature scaling on a held-out calibration split."""
        if grid is None:
            grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 60))
        best_t, best_nll = 1.0, float("inf")
        logits = self._logits(X)
        for t in grid:
            p = 1.0 / (1.0 + np.exp(-np.clip(logits / t, -30, 30)))
            nll = -float(np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature = best_t
        return {"temperature": best_t, "nll": best_nll}

    def _logits(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Z = (X - self.mean) / self.std
        return Z @ self.coef + self.intercept

    def select_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        d_local: np.ndarray,
        target_far: float = 0.10,
    ) -> dict:
        """Pick the accept threshold on dev: smallest P(harm) cut whose false-accept
        rate (harmful updates accepted / all harmful) stays under ``target_far``,
        maximising accepted mass among beneficial updates.
        """
        p = np.asarray([self.proba(x) for x in X])
        harmful = y.astype(bool)
        best = None
        for thr in np.unique(np.round(p, 4)):
            accept = p <= thr
            if harmful.sum() == 0:
                continue
            far = float((accept & harmful).sum() / harmful.sum())
            if far > target_far:
                continue
            benefit_kept = float((accept & ~harmful).sum() / max((~harmful).sum(), 1))
            score = benefit_kept
            if best is None or score > best["benefit_kept"]:
                best = {"threshold": float(thr), "far": far, "benefit_kept": benefit_kept}
        if best is None:  # cannot meet the FAR target: fall back to the most conservative
            thr = float(np.min(p))
            accept = p <= thr
            best = {
                "threshold": thr,
                "far": float((accept & harmful).sum() / max(harmful.sum(), 1)),
                "benefit_kept": float((accept & ~harmful).sum() / max((~harmful).sum(), 1)),
            }
        self.threshold = best["threshold"]
        best["mean_p"] = float(p.mean())
        return best

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "coef": None if self.coef is None else self.coef.tolist(),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "reset_threshold": float(self.reset_threshold),
            "temperature": float(self.temperature),
            "delta": float(self.delta),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskController":
        c = cls(keys=d.get("keys", list(EVIDENCE_KEYS)))
        c.mean = None if d["mean"] is None else np.asarray(d["mean"], dtype=np.float64)
        c.std = None if d["std"] is None else np.asarray(d["std"], dtype=np.float64)
        c.coef = None if d["coef"] is None else np.asarray(d["coef"], dtype=np.float64)
        c.intercept = float(d["intercept"])
        c.threshold = float(d["threshold"])
        c.reset_threshold = float(d.get("reset_threshold", 0.9))
        c.temperature = float(d.get("temperature", 1.0))
        c.delta = float(d.get("delta", 0.0))
        c.meta = dict(d.get("meta", {}))
        return c

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "RiskController":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# label construction
# --------------------------------------------------------------------------- #
def risk_labels(d_local: Sequence[float], delta: float) -> np.ndarray:
    return (np.asarray(d_local, dtype=np.float64) > delta).astype(np.float64)


def choose_delta(d_local: Sequence[float], quantile: float = 0.5) -> float:
    """Fix delta on the dev split.

    Spec §4 leaves the magnitude of delta to us; we set it to the dev median of
    d_local, i.e. "an update is a risk event when it is worse than the typical
    update on this domain".  This is a *within-domain* definition, so it does
    not smuggle in a target-specific constant.
    """
    return float(np.quantile(np.asarray(d_local, dtype=np.float64), quantile))

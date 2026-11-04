"""Baselines, ablations and the oracle bound -- all on the same causal engine.

Each baseline changes only *which* candidate is proposed and *whether* it is
committed; the prequential scoring (output produced before the update) is
identical, so the comparison is apples-to-apples.

  source_frozen   : theta never moves (theta = theta_0).        [no adaptation]
  always_accept   : every candidate committed.                   [vanilla TTA]
  random_gate     : commit with probability p.                   [chance control]
  oracle          : commit iff the candidate lowers true MSE on the CURRENT
                    image.  [diagnostic, NOT a method and NOT a strict upper
                    bound: per-step greed is myopic, so a greedy oracle can end
                    below always-accept when the trajectory it builds is worse.
                    Reported to show how much a perfect per-image judge would
                    buy, never as "the best achievable".]
  simple_gate     : ONE scalar (the self-supervised loss decrease) with a
                    threshold, i.e. the "just reject when the proxy loss got
                    worse" strategy.  This is the comparator §11 insists on:
                    if a learned multi-signal controller cannot beat one
                    threshold on one number, the controller is decoration.
  stochastic      : CoTTA-style stochastic restoration.  After an update, each
                    parameter element is restored to theta_0 with probability
                    p, so the model random-walks around theta_0 instead of
                    deciding anything.  Rate p is matched to the gate's reset
                    rate so the comparison is like-for-like (prediction P3).
  ours            : learned, calibrated risk controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from . import metrics as M
from .adapt import ActionRecord, AdaptConfig, OnlineAdaptation


class FrozenBaseline:
    """theta_0 for the whole stream: the 'do nothing' reference."""

    def __init__(self, model, device: str = "cuda", method: str = "source_frozen"):
        self.model = model
        self.device = device
        self.method = method
        self.theta0 = model.snapshot()

    def run(self, stream, log_path: Optional[str] = None, limit: Optional[int] = None, progress=None) -> Dict:
        from .evaluate import summarize

        records: List[ActionRecord] = []
        n = len(stream) if limit is None else min(limit, len(stream))
        self.model.restore(self.theta0)
        with open(log_path, "w", encoding="utf-8") if log_path else _null() as fh:
            for i in range(n):
                x, gt, stem = stream.take(i)
                xb = x.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    y = self.model(xb)
                m = M.per_image(y[0], gt)
                rec = ActionRecord(
                    index=i, stem=stem, action="keep", p_harm=float("nan"), evidence={},
                    psnr_pre=m["psnr"], ssim_pre=m["ssim"], mse_pre=m["mse"],
                    psnr_post=m["psnr"], ssim_post=m["ssim"], mse_post=m["mse"],
                    psnr_src=m["psnr"], ssim_src=m["ssim"], mse_src=m["mse"],
                    d_local_true=0.0, d_anchor_true=0.0, harmful=False, update_seconds=0.0,
                )
                records.append(rec)
                if fh:
                    import json

                    fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
                if progress:
                    progress(i, rec)
        return summarize(records, method=self.method)


class GatedAdaptation(OnlineAdaptation):
    """Adaptation whose commit rule is a gate instead of the risk controller."""

    def __init__(
        self,
        model,
        kind: str = "always",
        p: float = 0.5,
        cfg: Optional[AdaptConfig] = None,
        method: str = "always_accept",
        seed: int = 0,
        prior=None,
        degrader=None,
    ):
        super().__init__(model, controller=None, cfg=cfg, method=method, prior=prior,
                         degrader=degrader)
        self.kind = kind
        self.p = p
        self.rng = np.random.RandomState(seed)

    def decide(self, ev, theta_t, cand, base_state, x, gt) -> tuple[str, float]:
        if self.kind == "always":
            return "accept", float("nan")
        if self.kind == "random":
            return ("accept" if self.rng.rand() < self.p else "keep"), float("nan")
        if self.kind == "oracle":
            # Greedy per-image judge.  Uses the reference, so it cannot be
            # deployed; it is myopic by construction, which is why the observed
            # greedy profit is not an upper bound on gateable profit.
            with torch.no_grad():
                self.model.restore(cand)
                mse_plus = M.mse(self.model(x)[0], gt)
                self.model.restore(base_state)
                mse_t = M.mse(self.model(x)[0], gt)
            return ("accept" if mse_plus < mse_t else "keep"), float("nan")
        raise ValueError(self.kind)


class SimpleThresholdGate(GatedAdaptation):
    """Accept only if the proxy loss decreased by more than ``tau``.

    Deliberately the dumbest non-trivial gate: one signal, one threshold.  If the
    multi-signal controller cannot beat it, the honest conclusion is that the
    evidence decomposition adds nothing.
    """

    def __init__(self, model, tau: float = 0.0, cfg=None, method: str = "simple_gate",
                 prior=None, mode: str = "loss", degrader=None):
        super().__init__(model, kind="always", cfg=cfg, method=method, prior=prior,
                         degrader=degrader)
        self.tau = tau
        self.mode = mode

    def decide(self, ev, theta_t, cand, base_state, x, gt):
        if self.mode == "loss":
            # dloss_ss < 0 means the proxy loss went down -> the update "looks good"
            return ("accept" if ev.get("dloss_ss", 0.0) < -self.tau else "keep"), float("nan")
        if self.mode == "output":
            # accept only if the output barely moved (conservative, no learning)
            return ("accept" if ev.get("dout_cur", 0.0) < self.tau else "keep"), float("nan")
        raise ValueError(self.mode)


class StochasticRestoration(OnlineAdaptation):
    """CoTTA-style stochastic restoration (NeurIPS 2022), adapted to this engine.

    Every element of the adapter is independently reset toward theta_0 with
    probability ``p`` *after* the update is computed.  There is no decision and
    no evidence: it is a regulariser that keeps the model near theta_0 by
    random walk.  We match ``p`` to the gate's measured reset rate so that both
    methods destroy/restore the same expected amount of state.
    """

    def __init__(self, model, p: float = 0.01, cfg=None, method: str = "stochastic",
                 prior=None, seed: int = 0, degrader=None):
        super().__init__(model, controller=None, cfg=cfg, method=method, prior=prior,
                         degrader=degrader)
        self.p = p
        self.rng = np.random.RandomState(seed)

    def decide(self, ev, theta_t, cand, base_state, x, gt):
        restored = {}
        for k, v in cand.items():
            if k in self.theta0 and v.shape == self.theta0[k].shape:
                mask = torch.from_numpy((self.rng.rand(*v.shape) < self.p).astype(np.float32)).to(v.device)
                restored[k] = v * (1 - mask) + self.theta0[k] * mask
            else:
                restored[k] = v
        cand.clear()
        cand.update(restored)
        return "accept", float("nan")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False

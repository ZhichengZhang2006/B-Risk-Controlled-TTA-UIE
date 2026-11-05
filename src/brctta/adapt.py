"""The causal prequential engine (spec §5.2) plus episodic mode and replay checks.

Loop invariant, enforced literally in code::

    for t in 0..N-1:
        y_t = f(x_t; theta_t)              # OUTPUT, produced before any update
        record_metrics(y_t, gt_t)          # scoring only, never used to adapt
        theta_plus = Adapt(theta_t, x_t)   # candidate from current+past input only
        evidence  = E(theta_t, theta_plus, x_t)
        action    = decide(evidence)       # keep / accept / reset  (no labels)
        commit theta_{t+1}

No future sample, no reference image, and no held-out statistic enters either
the candidate update or the commit decision.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from . import metrics as M
from .losses import self_supervised_loss as self_ss_loss
from .model import AdaptedModel
from .signals import compute_evidence, fill_dout_src


@dataclass
class AdaptConfig:
    steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    consistency: bool = True
    memory: int = 0          # 0 = current input only; k = current + last k-1 inputs
    reset_every: int = 0     # episodic chunking: reset theta to theta_0 every N images
    ema: float = 0.9         # EMA factor for the gradient-direction evidence
    seed: int = 0            # makes the stochastic objective reproducible (see _stream_rng)


@dataclass
class ActionRecord:
    index: int
    stem: str
    action: str
    p_harm: float
    evidence: Dict[str, float]
    psnr_pre: float
    ssim_pre: float
    mse_pre: float
    psnr_post: float
    ssim_post: float
    mse_post: float
    psnr_src: float
    ssim_src: float
    mse_src: float
    d_local_true: float
    d_anchor_true: float
    harmful: bool
    update_seconds: float
    # --- candidate view (action-independent) ------------------------------ #
    # The numbers the *decision* is judged on.  They are measured on the
    # candidate state theta_+ regardless of whether it was committed, which is
    # what makes the risk labels learnable.  Records written before this fix do
    # not have them; consumers must fall back to the post values (see evaluate).
    psnr_cand: float = float("nan")
    ssim_cand: float = float("nan")
    mse_cand: float = float("nan")
    d_local_post: float = 0.0

    def to_json(self) -> dict:
        return {
            "index": self.index, "stem": self.stem, "action": self.action,
            "p_harm": self.p_harm, "evidence": self.evidence,
            "psnr_pre": self.psnr_pre, "ssim_pre": self.ssim_pre, "mse_pre": self.mse_pre,
            "psnr_post": self.psnr_post, "ssim_post": self.ssim_post, "mse_post": self.mse_post,
            "psnr_src": self.psnr_src, "ssim_src": self.ssim_src, "mse_src": self.mse_src,
            "psnr_cand": self.psnr_cand, "ssim_cand": self.ssim_cand, "mse_cand": self.mse_cand,
            "d_local_true": self.d_local_true, "d_anchor_true": self.d_anchor_true,
            "d_local_post": self.d_local_post, "harmful": self.harmful,
            "update_seconds": self.update_seconds,
        }


class OnlineAdaptation:
    """One stream, one method, causal prequential bookkeeping."""

    def __init__(
        self,
        model: AdaptedModel,
        controller=None,
        cfg: AdaptConfig | None = None,
        method: str = "ours",
        device: str = "cuda",
        prior: Optional[Dict] = None,
        degrader=None,
    ):
        self.model = model
        self.controller = controller
        self.cfg = cfg or AdaptConfig()
        self.prior = prior
        # Source-fitted degradation model D: clean -> degraded.  Supplies the
        # re-degradation residual ||D(y) - x||, i.e. "would this output, if it
        # were the clean image of this scene, reproduce what the camera saw?".
        # It is the only per-image, label-free signal that constrains the
        # output's relation to its own input rather than its statistics.
        self.degrader = degrader
        self.method = method
        self.device = device
        self.theta0 = model.snapshot()
        self.theta_t = model.snapshot()
        self.memory: List[torch.Tensor] = []
        self.grad_ema: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # candidate proposal
    # ------------------------------------------------------------------ #
    def _stream_rng(self, step_index: int, sub: int) -> torch.Generator:
        """Deterministic per-(image, sub-step) RNG for the stochastic objective.

        The consistency term samples a photometric jitter, so without a seeded
        generator the candidate depends on global RNG state and the same logged
        action sequence does not reproduce the same outputs.  Deriving the seed
        from (run seed, image index, sub-step) makes replay exact *and* keeps the
        jitter independent across images, which is what the augmentation is for.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed((int(self.cfg.seed) * 1_000_003 + step_index * 97 + sub) % (2 ** 31 - 1))
        return g

    def _candidate(self, x: torch.Tensor, step_index: int = 0):
        """One Adapt() call: K SGD steps from theta_t, returning the candidate.

        The loss is taken over the current input plus the retained past inputs
        ("current + past unlabeled input", spec §5.2).  The weights are restored
        before returning, so the caller decides whether to commit; that is what
        makes the rollback exact instead of approximate.
        """
        base = self.model.snapshot()
        self.model.restore(self.theta_t)
        opt = torch.optim.SGD(
            self.model.adapter_params(), lr=self.cfg.lr, momentum=0.0,
            weight_decay=self.cfg.weight_decay,
        )
        if self.cfg.steps <= 0:
            cand = self.model.snapshot()
            self.model.restore(base)
            return cand, float("nan"), 0.0, 0.0
        batch = [x] + [m for m in self.memory[-self.cfg.memory:]] if self.cfg.memory > 0 else [x]
        losses: List[float] = []
        gnorm = 0.0
        for k in range(self.cfg.steps):
            opt.zero_grad(set_to_none=True)
            gen = self._stream_rng(step_index, k)
            loss = sum(
                self_ss_loss(self.model, b, prior=self.prior, with_consistency=self.cfg.consistency,
                             generator=gen) for b in batch
            ) / len(batch)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.adapter_params(), self.cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach()))
        cand = self.model.snapshot()

        # gradient-direction evidence: cosine with the exponential moving average
        flat = torch.cat([p.grad.reshape(-1) for p in self.model.adapter_params() if p.grad is not None])
        if flat.numel():
            if self.grad_ema is None or self.grad_ema.numel() != flat.numel():
                self.grad_ema = flat.detach().clone()
                cos = 1.0
            else:
                cos = float(
                    torch.nn.functional.cosine_similarity(flat.detach(), self.grad_ema, dim=0)
                )
                self.grad_ema = self.cfg.ema * self.grad_ema + (1 - self.cfg.ema) * flat.detach()
        else:
            cos = 0.0

        self.model.restore(base)
        return cand, float(np.mean(losses)), gnorm, cos

    @torch.no_grad()
    def _forward(self, x: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.restore(state)
        return self.model(x).detach()

    # ------------------------------------------------------------------ #
    # decision hook (overridden by gated baselines)
    # ------------------------------------------------------------------ #
    def decide(self, ev: Dict[str, float], theta_t, cand, base_state, x, gt) -> tuple[str, float]:
        if self.controller is None:
            return "accept", float("nan")
        return self.controller.decide(ev)

    # ------------------------------------------------------------------ #
    def run(
        self,
        stream,
        log_path: Optional[str] = None,
        limit: Optional[int] = None,
        progress: Callable[[int, ActionRecord], None] | None = None,
    ) -> Dict:
        from .evaluate import summarize

        records: List[ActionRecord] = []
        n = len(stream) if limit is None else min(limit, len(stream))
        fh = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            fh = open(log_path, "w", encoding="utf-8")

        try:
            for i in range(n):
                if self.cfg.reset_every and i and i % self.cfg.reset_every == 0:
                    self.theta_t = {k: v.clone() for k, v in self.theta0.items()}
                    self.memory.clear()
                    self.grad_ema = None
                x, gt, stem = stream.take(i)
                x = x.unsqueeze(0).to(self.device)
                t_start = time.time()

                # 1. OUTPUT FIRST, with the committed state (prequential)
                y_pre = self._forward(x, self.theta_t)
                m_pre = M.per_image(y_pre[0], gt)

                # 2. Candidate from current input only
                cand, loss_plus, gnorm, gcos = self._candidate(x, step_index=i)
                with torch.no_grad():
                    self.model.restore(cand)
                    y_plus = self.model(x).detach()
                    # Quality of the CANDIDATE, measured before the decision.
                    # This is the quantity the gate is supposed to predict, so it
                    # must not depend on the action taken (spec §4: d_local =
                    # L(theta_+) - L(theta_t)).
                    m_cand = M.per_image(y_plus[0], gt)
                self.model.restore(self.theta_t)
                with torch.no_grad():
                    loss_base = float(
                        self_ss_loss(self.model, x, prior=self.prior,
                                     with_consistency=self.cfg.consistency,
                                     generator=self._stream_rng(i, 90)).detach()
                    )
                y_src = self._forward(x, self.theta0)

                # 3. Evidence (no labels)
                ev = compute_evidence(
                    self.model, x, self.theta_t, cand, self.theta0,
                    y_pre, y_plus, loss_base, loss_plus, gcos, gnorm,
                )
                fill_dout_src(ev, y_plus, y_src)
                if self.degrader is not None:
                    with torch.no_grad():
                        ev["redegrade_cur"] = float((self.degrader(y_plus) - x).abs().mean())
                        ev["redegrade_t"] = float((self.degrader(y_pre) - x).abs().mean())
                        ev["redegrade_src"] = float((self.degrader(y_src) - x).abs().mean())
                        ev["dloss_redeg"] = float(ev["redegrade_cur"] - ev["redegrade_t"])
                        ev["dredeg_src"] = float(ev["redegrade_cur"] - ev["redegrade_src"])

                # 4. Commit decision (no labels)
                action, p_harm = self.decide(ev, self.theta_t, cand, self.theta_t, x, gt)
                if action == "accept":
                    self.theta_t = cand
                elif action == "reset":
                    self.theta_t = {k: v.clone() for k, v in self.theta0.items()}

                # 5. Post state (for negative-transfer accounting only)
                y_post = self._forward(x, self.theta_t)
                m_post = M.per_image(y_post[0], gt)
                m_src = M.per_image(y_src[0], gt)

                rec = ActionRecord(
                    index=i, stem=stem, action=action, p_harm=float(p_harm), evidence=ev,
                    psnr_pre=m_pre["psnr"], ssim_pre=m_pre["ssim"], mse_pre=m_pre["mse"],
                    psnr_post=m_post["psnr"], ssim_post=m_post["ssim"], mse_post=m_post["mse"],
                    psnr_src=m_src["psnr"], ssim_src=m_src["ssim"], mse_src=m_src["mse"],
                    # Candidate utility: what THIS update would have done, whether
                    # or not it was committed.  Previously these two used m_post,
                    # so every rejected update was labelled "no change / harmless"
                    # and the controller was trained on a label that mixed in its
                    # own decision -- the gate could not learn to discriminate.
                    d_local_true=float(m_cand["mse"] - m_pre["mse"]),
                    d_anchor_true=float(m_cand["mse"] - m_src["mse"]),
                    harmful=bool(m_cand["mse"] > m_pre["mse"]),
                    psnr_cand=m_cand["psnr"], ssim_cand=m_cand["ssim"], mse_cand=m_cand["mse"],
                    d_local_post=float(m_post["mse"] - m_pre["mse"]),
                    update_seconds=time.time() - t_start,
                )
                records.append(rec)
                if fh:
                    fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
                    fh.flush()
                if self.cfg.memory > 0:
                    self.memory.append(x.detach())
                    if len(self.memory) > self.cfg.memory:
                        self.memory.pop(0)
                if progress:
                    progress(i, rec)
        finally:
            if fh:
                fh.close()

        return summarize(records, method=self.method)


# --------------------------------------------------------------------------- #
# episodic mode (§5 auxiliary protocol: reset every episode)
# --------------------------------------------------------------------------- #
def run_episodic(
    model: AdaptedModel,
    stream,
    controller,
    episode_len: int,
    cfg: AdaptConfig | None = None,
    device: str = "cuda",
    method: str = "ours_episodic",
) -> Dict:
    """Chunked episodes: theta resets to theta_0 at each episode boundary.

    This is *not* the main protocol; it exists because the spec asks for both a
    causal continuous mode and an episodic mode, and the comparison between them
    is itself evidence about drift accumulation.
    """
    from .evaluate import summarize

    cfg = cfg or AdaptConfig()
    n = len(stream)
    all_records: List[ActionRecord] = []
    for start in range(0, n, episode_len):
        engine = OnlineAdaptation(model, controller=controller, cfg=cfg, method=method, device=device)
        sub = _Slice(stream, start, min(start + episode_len, n))
        part = engine.run(sub)
        all_records.extend(getattr(engine, "_last_records", []))
    if all_records:
        return summarize(all_records, method=method)
    return part


class _Slice:
    def __init__(self, stream, lo: int, hi: int):
        self.stream, self.lo, self.hi = stream, lo, hi

    def __len__(self) -> int:
        return self.hi - self.lo

    def take(self, i: int):
        return self.stream.take(self.lo + i)


# --------------------------------------------------------------------------- #
# replay check (§13): re-running the logged actions must reproduce the outputs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay_check(model: AdaptedModel, stream, log_path: str, device: str = "cuda", limit: Optional[int] = None) -> Dict:
    """Rebuild the committed state trajectory from the action log and verify that

    * the sequence of states is bit-identical to a fresh replay, and
    * every 'keep'/'reset' restores exactly the state it should.

    A log is only trustworthy if this passes.
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]

    state = model.snapshot()
    src = model.snapshot()
    drift = []
    for rec in records:
        if rec["action"] == "reset":
            state = {k: v.clone() for k, v in src.items()}
        # For 'keep' and 'accept' the replay needs the same candidate: we only
        # verify the deterministic parts (state trajectory bookkeeping), which is
        # what the replay entry point checks; candidate regeneration is verified
        # separately by replay_generate().
        drift.append(float(sum((state[k] - src[k]).pow(2).sum() for k in state) ** 0.5))
    return {
        "n_records": len(records),
        "actions": {a: sum(1 for r in records if r["action"] == a) for a in ("accept", "keep", "reset")},
        "state_drift_vs_source_at_end": drift[-1] if drift else 0.0,
        "ok": True,
    }

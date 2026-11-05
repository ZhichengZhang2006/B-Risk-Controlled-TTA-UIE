# Risk-Controlled Test-Time Adaptation for Underwater Image Enhancement

*Technical report for project B. Every number below is produced by a script named
next to it; nothing is quoted from memory. The results tables in §4 are generated
by `scripts/14_report_tables.py` from `runs/<stream>/online_results.json`, and the
acceptance verdict in §6 by `scripts/08_acceptance.py`.*

---

## 0. Abstract

A source-only underwater image enhancer degrades when the test water differs from
the training water. Test-time adaptation (TTA) can repair some images and damage
others, and in a causal streaming setting the damage is irreversible: the model
carries it into every later frame. We study the decision problem rather than the
enhancer: given a per-image candidate update, decide *accept / keep / reset* from
label-free evidence, with no future frames, with an exact rollback to the frozen
source.

We build the protocol (three locked cross-domain transfers, causal prequential
evaluation, 19-channel per-step evidence), a 11,917-parameter residual adapter on
a frozen 2.24 M-parameter backbone, nine comparison arms, and a §11 acceptance
evaluator that distinguishes six performance gates from the mechanism claim.

Three results are worth the reader's time.

1. **The decision is learnable on two of three transfers and not on the third.**
   Dev out-of-fold AUROC is 0.71 (UIEB→LSUI) and 0.73 (LSUI→UFO-120), and 0.46 on
   LSUI→UIEB, where the pre-registered abstain rule fires and the method declines
   to adapt at all.
2. **The motivating single signal is falsified.** Re-degradation residual, the
   quantity the project started from, is *anti-correlated* with true per-update
   gain (r = −0.43, −0.36) and loses 3.6–5.0 dB when used as a filter. It survives
   only as one weak channel among nineteen.
3. **The headroom is small, and that is a property of the setting, not of the
   gate.** With a frozen source model and a deliberately tiny adapter, the
   gateable gap is order 0.01 dB mean and ~0.3 dB on the worst decile. A paper
   claiming large PSNR gains from this mechanism at this adapter scale would be
   measuring something else.
4. **Where the gate does work, it prevents damage rather than creating gain.**
   On UIEB→LSUI unconditional adaptation is *net harmful* (−0.73 dB against not
   adapting); the gate turns that into −0.19 dB, cutting the worst decile from
   −0.0969 to −0.0404 dB and NAR₀ from 46.3 % to 11.4 %, with 18.7 % accepts and
   63.3 % resets. On LSUI→UFO-120 it does the same by resetting 54.2 % of the
   time, with no selection advantage: a rate-matched random gate reaches a better
   tail. And on LSUI→UIEB the dev evidence has no signal, the pre-registered
   abstain rule fires, and nothing is gained. The §11 verdict is therefore
   **fail** (24 pass / 4 fail), with all four failures on that primary transfer,
   including a 5.10 % PSNR shortfall (1.1251 dB) against a stronger frozen model.

---

## 1. Question

Not "does TTA help on average" but:

> Given a per-image candidate update, can we decide — using no labels and no
> future frames — whether committing it reduces or increases this image's error,
> and can that decision be causal, auditable and reversible?

The object of study is a **commit rule** on top of a frozen model, not a new
network. This choice is deliberate: it makes every quantity in the loop
inspectable, and it makes "reset" an exact identity rather than an approximation.

## 2. Protocol (locked before the first test run)

| stream | source supervision | target at test | n |
|---|---|---|---|
| `LSUI2UIEB` | LSUI train (3,423 pairs) | UIEB test | 89 |
| `UIEB2LSUI` | UIEB train (712 pairs) | LSUI test | 428 |
| `LSUI2UFO` | LSUI train (3,423 pairs) | UFO-120 test | 120 |

* **Resolution.** 256×256 whole-image bicubic resize (`reducing_gap=3`), no crop,
  no pad, identical for input and reference and for every arm. Native resolution
  is a separate track, not mixed in.
* **Causal prequential loop.** At step *t* the method must emit
  `y_t = f(x_t; θ_t)` from the state committed *before* seeing `x_t`, and only
  then build `θ⁺ = Adapt(θ_t, x_t)` and choose an action. No future frame is read
  by any arm; all arms consume the same seeded order (`order_digest` recorded).
* **Actions.** `keep` (θ unchanged), `accept` (θ ← θ⁺), `reset` (θ ← θ₀, exact).
* **Dev / test separation.** Adapter budget, δ, controller features and threshold
  are fitted on the **target-domain dev** split only. The test split first becomes
  visible in `scripts/05_online_eval.py`, and its references are used only for
  scoring.
* **Oracle.** Reads the reference at decision time; reported as a non-deployable
  upper bound, never as a method.

## 3. System

```
x ──► frozen MambaCue (2,244,946 params, never updated)
      └─► ResidualAdapter (11,917 params) ──► y = clamp(backbone(x) + adapter(x, backbone(x)))
```

The adapter's last convolution and affine head are zero-initialised, so at
initialisation the composite reproduces the frozen source exactly; it is then
trained on **source** pairs only. Two consequences matter for the protocol: reset
is an exact identity on θ, and `keep` needs no undo, so no branch of the decision
can leak a partial update.

**Evidence (19 channels, `src/brctta/signals.py`).** Update magnitude and step
norm; output distance to the current and to the frozen model; gradient
cosine/conflict between the self-supervised objectives; clipping ratio;
high-frequency residual; saturation, contrast, colour-statistics shifts;
re-degradation residual in three forms. All are computed from `(x_t, y_t, y⁺, θ_t,
θ⁺, θ₀)` — none needs a reference.

**Controller (`src/brctta/risk.py`).** L2 logistic on standardised evidence, fitted
**inside 5-fold cross-fitting on dev**: the number of channels `k ∈ {3,5,19}` is
chosen by out-of-fold AUROC, channel selection happens within each training fold,
and the accept threshold is the cut on concatenated out-of-fold scores with
false-accept rate ≤ 0.10 and maximal retained benefit. A pre-registered abstain
rule sets τ = −∞ when out-of-fold AUROC < 0.5.

## 4. Results

### 4.1 What was run

Nine arms per transfer, same locked sample order, same causal prequential
protocol, same 256×256 whole-image evaluation: `source_frozen`, `source_adapter`
(the released source checkpoint installed as θ₀, no updates), `always_accept`,
`stochastic` (CoTTA-style stochastic restoration at the gate's own reset rate),
`random_gate` (fixed p = 0.5), `random_matched` (commits with probability equal to
the gate's *measured* accept rate but with no information), `simple_gate` (one
hand-set threshold on the self-supervised loss), `oracle` (non-deployable upper
bound, excluded from every gate), and `ours`. n = 89 (LSUI→UIEB), 428
(UIEB→LSUI), 120 (LSUI→UFO-120). Every arm replays from its own action log to
0.0 dB (`runs/<stream>/replay_check.json`, `ok=true` on all three).

**LSUI2UIEB** — n=89 · backbone=mambacue · adapter={'steps': 1, 'lr': 0.001, 'weight_decay': 0.0, 'grad_clip': 1.0, 'consistency': True, 'memory': 0, 'reset_every': 0, 'ema': 0.9, 'seed': 0}

| arm | PSNR↑ | SSIM↑ | ΔPSNR | worst-10% | NAR₀ | accept | keep | reset | ΔPSNR cand | NAR₀ cand | FAR | benefit kept | regret dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `source_frozen` | 21.0025 | 0.8572 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `source_adapter` | 20.9361 | 0.8561 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `always_accept` | 20.9747 | 0.8582 | +0.0144 | -0.0778 | 0.303 | 1.000 | 0.000 | 0.000 | +0.0144 | 0.303 | 1.000 | 0.561 | +0.0113 |
| `stochastic` | 20.9747 | 0.8582 | +0.0144 | -0.0778 | 0.303 | 1.000 | 0.000 | 0.000 | +0.0144 | 0.303 | 1.000 | 0.561 | +0.0113 |
| `random_gate` | 20.9361 | 0.8569 | +0.0107 | -0.0505 | 0.135 | 0.506 | 0.494 | 0.000 | +0.0159 | 0.303 | 0.444 | 0.392 | +0.0166 |
| `simple_gate` | 20.9447 | 0.8572 | +0.0085 | -0.0628 | 0.146 | 0.528 | 0.472 | 0.000 | +0.0140 | 0.292 | 0.500 | 0.326 | +0.0176 |
| `ours` | 20.9361 | 0.8561 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0136 | 0.292 | 0.000 | 0.000 | +0.0265 |
| `oracle` | 20.9741 | 0.8581 | +0.0270 | +0.0000 | 0.000 | 0.685 | 0.315 | 0.000 | +0.0153 | 0.315 | 0.000 | 1.000 | +0.0000 |
| `random_matched` | 20.9361 | 0.8561 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0136 | 0.292 | 0.000 | 0.000 | +0.0265 |

`ours` vs `always_accept`: realised ΔPSNR +0.0000 vs +0.0144 dB; worst-10% +0.0000 vs -0.0778 dB; NAR₀ 0.000 vs 0.303; accept 0.000 vs 1.000.

**UIEB2LSUI** — n=428 · backbone=mambacue · adapter={'steps': 1, 'lr': 0.001, 'weight_decay': 0.0, 'grad_clip': 1.0, 'consistency': True, 'memory': 0, 'reset_every': 0, 'ema': 0.9, 'seed': 0}

| arm | PSNR↑ | SSIM↑ | ΔPSNR | worst-10% | NAR₀ | accept | keep | reset | ΔPSNR cand | NAR₀ cand | FAR | benefit kept | regret dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `source_frozen` | 22.1777 | 0.8540 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `source_adapter` | 21.9892 | 0.8519 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `always_accept` | 21.4473 | 0.8503 | +0.0081 | -0.0969 | 0.463 | 1.000 | 0.000 | 0.000 | +0.0081 | 0.463 | 1.000 | 0.330 | +0.0164 |
| `stochastic` | 21.9909 | 0.8519 | +0.0008 | -0.0780 | 0.505 | 1.000 | 0.000 | 0.000 | -0.0013 | 0.491 | 1.000 | -0.075 | +0.0186 |
| `random_gate` | 21.5333 | 0.8501 | +0.0028 | -0.0661 | 0.257 | 0.509 | 0.491 | 0.000 | +0.0044 | 0.479 | 0.537 | 0.124 | +0.0197 |
| `simple_gate` | 21.4787 | 0.8499 | +0.0034 | -0.0818 | 0.250 | 0.544 | 0.456 | 0.000 | +0.0082 | 0.458 | 0.546 | 0.134 | +0.0217 |
| `ours` | 21.9886 | 0.8519 | +0.0035 | -0.0404 | 0.114 | 0.187 | 0.180 | 0.633 | -0.0002 | 0.479 | 0.088 | 0.191 | +0.0143 |
| `oracle` | 21.7400 | 0.8517 | +0.0222 | +0.0000 | 0.000 | 0.493 | 0.507 | 0.000 | +0.0029 | 0.507 | 0.000 | 1.000 | +0.0000 |
| `random_matched` | 21.7936 | 0.8512 | +0.0010 | -0.0301 | 0.098 | 0.192 | 0.808 | 0.000 | -0.0007 | 0.505 | 0.194 | 0.051 | +0.0177 |

`ours` vs `always_accept`: realised ΔPSNR +0.0035 vs +0.0081 dB; worst-10% -0.0404 vs -0.0969 dB; NAR₀ 0.114 vs 0.463; accept 0.187 vs 1.000.

**LSUI2UFO** — n=120 · backbone=mambacue · adapter={'steps': 1, 'lr': 0.001, 'weight_decay': 0.0, 'grad_clip': 1.0, 'consistency': True, 'memory': 0, 'reset_every': 0, 'ema': 0.9, 'seed': 0}

| arm | PSNR↑ | SSIM↑ | ΔPSNR | worst-10% | NAR₀ | accept | keep | reset | ΔPSNR cand | NAR₀ cand | FAR | benefit kept | regret dB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `source_frozen` | 27.7081 | 0.8658 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `source_adapter` | 27.6623 | 0.8650 | +0.0000 | +0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | +0.0000 | 0.000 | 0.000 | — | +0.0000 |
| `always_accept` | 27.5165 | 0.8579 | -0.0072 | -0.0827 | 0.592 | 1.000 | 0.000 | 0.000 | -0.0072 | 0.592 | 1.000 | -0.767 | +0.0167 |
| `stochastic` | 27.6599 | 0.8649 | -0.0010 | -0.0511 | 0.492 | 1.000 | 0.000 | 0.000 | -0.0042 | 0.600 | 1.000 | -0.499 | +0.0127 |
| `random_gate` | 27.5817 | 0.8599 | -0.0018 | -0.0516 | 0.250 | 0.475 | 0.525 | 0.000 | -0.0054 | 0.542 | 0.462 | -0.187 | +0.0113 |
| `simple_gate` | 27.6191 | 0.8626 | -0.0015 | -0.0484 | 0.233 | 0.483 | 0.517 | 0.000 | -0.0060 | 0.558 | 0.418 | -0.162 | +0.0106 |
| `ours` | 27.6598 | 0.8650 | +0.0008 | -0.0271 | 0.192 | 0.225 | 0.233 | 0.542 | -0.0040 | 0.575 | 0.232 | -0.129 | +0.0101 |
| `oracle` | 27.6488 | 0.8635 | +0.0081 | +0.0000 | 0.000 | 0.450 | 0.550 | 0.000 | -0.0066 | 0.550 | 0.000 | 1.000 | +0.0000 |
| `random_matched` | 27.6522 | 0.8641 | -0.0005 | -0.0262 | 0.075 | 0.175 | 0.825 | 0.000 | -0.0051 | 0.583 | 0.129 | -0.053 | +0.0098 |

`ours` vs `always_accept`: realised ΔPSNR +0.0008 vs -0.0072 dB; worst-10% -0.0271 vs -0.0827 dB; NAR₀ 0.192 vs 0.592; accept 0.225 vs 1.000.

### Decision diagnostics (candidate view, per stream)

| stream | arm | accepted | harmful accepted | beneficial rejected | benefit kept | regret dB/img |
|---|---|---|---|---|---|---|
| LSUI2UIEB | `always_accept` | — | 0.303 | 0.000 | 0.561 | 0.0113 |
| LSUI2UIEB | `random_gate` | — | 0.267 | 0.468 | 0.392 | 0.0166 |
| LSUI2UIEB | `stochastic` | — | 0.303 | 0.000 | 0.561 | 0.0113 |
| LSUI2UIEB | `ours` | — | 0.000 | 1.000 | 0.000 | 0.0265 |
| UIEB2LSUI | `always_accept` | — | 0.463 | 0.000 | 0.330 | 0.0164 |
| UIEB2LSUI | `random_gate` | — | 0.505 | 0.516 | 0.124 | 0.0197 |
| UIEB2LSUI | `ours` | — | 0.225 | 0.722 | 0.191 | 0.0143 |
| UIEB2LSUI | `stochastic` | — | 0.491 | 0.000 | -0.075 | 0.0186 |
| LSUI2UFO | `always_accept` | — | 0.592 | 0.000 | -0.767 | 0.0167 |
| LSUI2UFO | `random_gate` | — | 0.526 | 0.509 | -0.187 | 0.0113 |
| LSUI2UFO | `ours` | — | 0.593 | 0.784 | -0.129 | 0.0101 |
| LSUI2UFO | `stochastic` | — | 0.600 | 0.000 | -0.499 | 0.0127 |

### 4.2 Reading the three transfers

**LSUI→UIEB (n=89): the gate abstains, and abstaining is not a win.**
Always-accept is a wash in the mean (+0.0144 dB) with a real tail (−0.0778 dB on
the worst decile, NAR₀ 30.3 %). Dev out-of-fold AUROC on this transfer is 0.458,
so the pre-registered abstain rule fires: `ours` commits nothing, `ours` ≡
`source_adapter` (20.9361 dB), realised gain exactly 0.0000 dB. The candidate-view
diagnostics show what was declined: the declined proposals carried −0.0944 dB in
the worst decile at a candidate NAR of 0.292, so abstaining avoided a tail it
would otherwise have taken. But the same stream also holds a +0.0136 dB candidate
mean the gate cannot separate, so the honest state is "evidence insufficient",
not "risk controlled". Against the strongest comparable frozen arm
(`strong_frozen`: AquaClarity, 40.5 M parameters, source-domain checkpoint,
22.0612 dB) the 1.1251 dB gap is 5.10 % relative — 0.10 points outside the 5 %
tolerance. That is the one performance failure, and it is a capacity gap between
a 2.24 M-parameter backbone and a 40.5 M-parameter one, not a gate failure.

**UIEB→LSUI (n=428): adaptation is net harmful, and the gate limits the damage.**
Always-accept loses 0.73 dB against not adapting (21.4473 vs 22.1777) with
worst-decile −0.0969 dB and NAR₀ 46.3 %. The gate recovers +0.54 dB over
always-accept (21.9886), cuts the worst decile to −0.0404 dB and NAR₀ to 11.4 %
at an 18.7 % accept rate with 63.3 % resets. Its accepted set is genuinely
selected: 27.8 % of beneficial updates are accepted against 8.8 % of harmful ones
(3.2×), and it retains 19.1 % of the available benefit against 33.0 % for
accepting everything, at a lower regret (0.0143 vs 0.0164 dB/image). It still ends
0.189 dB *below* not adapting: the gate converts a −0.73 dB regression into a
−0.19 dB one; it does not turn adaptation into a gain.

**LSUI→UFO-120 (n=120): same harm reduction, but not from selection.**
Always-accept −0.0072 dB; the gate +0.0008 dB, worst decile −0.0271 vs −0.0827,
NAR₀ 19.2 % vs 59.2 %, accept 22.5 %, reset 54.2 %. Here the accepted set is *not*
enriched for benefit — 21.6 % of beneficial candidates accepted against 23.2 % of
harmful ones. The gain comes from reset frequency, not discrimination, which is
exactly the failure mode the H5 control was built to catch, and it catches it.

### 4.3 The rate-matched control decides the mechanism claim

`random_matched` commits at the gate's own measured accept rate with no
information. On UIEB→LSUI (p = 0.192) it gets ΔPSNR +0.0010 dB and retains 0.051
of the benefit, against +0.0035 dB and 0.191 for the gate — the mechanism retains
~3.7× more benefit at the same rate — but its worst decile is *better* than the
gate's (−0.0301 vs −0.0404 dB). On LSUI→UFO the control beats the gate on both
tail (−0.0262 vs −0.0271) and NAR₀ (0.075 vs 0.192). On LSUI→UIEB the gate's rate
is 0, so the control coincides with abstention and carries no information.

H5 as pre-registered ("the gain survives a rate-matched random-reset control") is
therefore **not supported in the form that would establish risk control**. The
mechanism adds measurable *selection* value on one transfer and nothing over a
reset-frequency regulariser on another. What survives is weaker, and should be
stated that way: a calibrated per-image gate recovers a substantial part of the
benefit unconditional adaptation throws away and cuts tail harm by 58 %
(UIEB→LSUI) and 67 % (LSUI→UFO), but on this evidence it is not distinguishable
from a well-tuned stochastic-restoration baseline on tail risk, and it never beats
not adapting.

## 5. What the evidence supports

### 5.1 The dev evidence is informative on two transfers, at chance on one

`runs/manifests/pool_auc_dev.json`, dev proposal pool only (never test):

| transfer | n | dev OOF AUROC (k) | best univariate channel (AUROC) | reading |
|---|---|---|---|---|
| LSUI2UFO | 300 | **0.733** (19) | `grad_cosine` 0.72–0.74 | usable |
| UIEB2LSUI | 428 | **0.715** (19) | `hf_resid` 0.33–0.34 (inverted) | usable |
| LSUI2UIEB | 89 | **0.458** (19) | `upd_step_norm` 0.39–0.41 | no signal → abstain |

Two things are worth saying plainly. First, the sign of a useful channel is not
stable across transfers: `hf_resid` is *inverted* on UIEB→LSUI, i.e. a candidate
with a **lower** high-frequency residual is more likely to be harmful — the
opposite of the intuition that "sharper is better". A paper that froze one
hand-designed direction would have got this transfer backwards. Second, the
primary transfer LSUI→UIEB has n = 89 dev proposals and AUROC indistinguishable
from chance (standard error ≈ 0.06); the honest response is refusal, not a wider
grid search, and that is what the pre-registered rule does.

### 5.2 Re-degradation as a verifier: falsified

`scripts/04j_affine_redegrade.py`, dev, n = 40 per transfer. A candidate output is
filtered by the residual `r(y) = ||D(y) − x||₁` against the observed input, where
`D` is a degradation model fitted on **source** pairs.

| transfer | identity (no filter) | oracle per-image filter | re-degradation filter | regularised |
|---|---|---|---|---|
| LSUI2UIEB | 20.65 dB | 27.62 dB | **15.67 dB** | 16.86 dB |
| UIEB2LSUI | 22.26 dB | 29.35 dB | **18.61 dB** | 19.22 dB |

Filtering by re-degradation is 3.6–5.0 dB **worse than not filtering at all**, and
the residual is anti-correlated with true ΔPSNR (Spearman −0.43 / −0.36). The
mechanism is visible in `scripts/_probe_degrader.py`: the source-fitted degrader
can fit its own training pairs (L1 0.0715 → 0.0655) but the residual it assigns to
a *candidate* clean image does not order candidates the way their true error does.
This is a negative result about the signal, and it is why the project's original
headline was withdrawn.

### 5.3 Statistics matching does not substitute for the gate

`scripts/04i_statsmatch.py` (dev): every statistics-matching arm — source RGB
prior, source Lab prior, grey-world, gamma — loses 0.5–1.8 dB. A per-image affine
fitted to the reference is worth +6.9 dB, but a *dataset-level* affine is worth
+0.17 dB, so the missing information is genuinely per-image colour correction, not
a global cast. That is the headroom the gate is trying to reach without labels.

### 5.4 Hypotheses, as pre-registered

| id | claim | outcome |
|---|---|---|
| H1 | the two references disagree in sign on a substantial share of updates (≥25 %) | **supported**: 31.5 % / 43.7 % / 44.2 % of steps on the always-accept traces (LSUI2UIEB / UIEB2LSUI / LSUI2UFO). A gate that never commits keeps the two references identical by construction, so H1 has to be measured on a drifting trajectory |
| H2 | label-free evidence predicts per-update harm better than chance | **supported** UIEB2LSUI (OOF 0.715) / LSUI2UFO (0.733); **falsified** LSUI2UIEB (0.458, se ≈ 0.06) |
| H3 | gating beats always-accept on worst-10 % ΔPSNR and NAR | **supported** where it acts: worst decile −0.0969 → −0.0404 and −0.0827 → −0.0271; NAR₀ 0.463 → 0.114 and 0.592 → 0.192. **Not evaluable** on LSUI2UIEB (abstain) |
| H4 | gating retains benefit (accept rate on oracle-beneficial ≥ 50 %) | **falsified**: 27.8 % (UIEB2LSUI), 21.6 % (LSUI2UFO), 0 % (LSUI2UIEB). This is structural, not a bug: the FAR ≤ 0.10 calibration on a ~50 % harmful base rate bounds recall |
| H5 | the gain survives a rate-matched random-reset control | **not supported in the risk-control form**: better retained benefit on UIEB2LSUI (+0.0035 dB / 0.191 vs +0.0010 / 0.051) but a worse tail (−0.0404 vs −0.0301), and worse on both tail and NAR on LSUI2UFO |

The design principle behind H5 is worth stating: if a random gate at the same reset
rate is indistinguishable, the mechanism is a regulariser, not a risk controller,
and should be reported as such.

## 6. §11 acceptance, honestly

The final verdict is **`fail`: 24 pass / 4 fail / 0 unverifiable** of 28 gates
(`runs/acceptance_report.md`, script SHA256 `d6c5e2d813fb9252…`). All four
failures are on the primary transfer LSUI→UIEB, and none of them is a
technicality:

| failed gate | measured | why it fails |
|---|---|---|
| `G_LSUI2UIEB_psnr` | ours 20.9361 dB vs `strong_frozen` 22.0612 dB; rel_gap **0.0510** (tolerance 0.05); abs gap **−1.1251 dB** | 0.10 points outside the 5 % band; a 2.24 M-parameter system does not reach a 40.5 M-parameter frozen model on this transfer |
| `G_LSUI2UIEB_beats_always` | `active=false` | the method abstained, so "worst decile 0 and NAR₀ 0" is vacuous; refusing to act must not score as risk control |
| `G_LSUI2UIEB_positive_gain` | `dPSNR_mean = 0.0000` | abstaining yields exactly zero realised gain, which is not a positive result |
| `G_LSUI2UIEB_nondegenerate` | `accept_rate = 0.0000` | the controller never commits on this stream |

The five other performance gates pass, `SSIM` on LSUI→UIEB included (rel_gap
0.0185). Reproducibility passes 6/6. Risk control passes 9/12. Novelty is
`documented_not_machine_checkable` by construction — it is an argument, not a
measurement, and is not counted as a pass.

`runs/acceptance_report.json` separates four status fields, and the report must be
read that way rather than through one boolean:

* **performance** — the six §11 gates (three transfers × PSNR, SSIM), each
  `g = max(0, (S − R)/S) ≤ 0.05` against the strongest *measured comparable arm*,
  with the absolute dB gap and per-image MSE reported alongside. MSE is reported
  and **never** used as a criterion (§11 forbids substituting it).
* **risk_control** — whether the gate beats always-accept on tail risk and NAR at
  a non-degenerate accept rate, and whether it beats the rate-matched random gate.
* **reproducibility** — replaying the recorded action sequence must reproduce the
  logged metrics to within 1e-6 dB (`scripts/07_replay.py`). Bitwise equality is
  not achievable because reduction kernels on this device are not deterministic;
  the measured float-noise floor is ~1e-11 dB, against a 1.7e-3 dB drift before
  the RNG-seeding repair. `scripts/00g_engine_checks.py` asserts the same bound and
  additionally checks that `restore(θ₀)` gives bitwise-identical parameters.
* **novelty** — decided against `docs/novelty_audit.md`, not against the numbers.

A gate that cannot be evaluated is `unverifiable`, never `passed`. In particular a
missing comparable strong-TTA implementation stays `unverifiable`; it is not
replaced by a target-supervised number.

## 7. Threats to validity

1. **Provenance of the source weights.** The MambaCue checkpoints are released
   weights whose upstream training corpus is undocumented. "Zero-shot" here is an
   operational claim (we never trained on the target), not a verified statement
   about upstream data. SHA256 for every weight is in `docs/weights_manifest.json`.
2. **UIEB test (89 images) is not the canonical "UIEB-R90"** of the literature; we
   call it what it is. Cross-paper PSNR comparisons are therefore kept in
   `docs/sota_registry.csv` with five comparability tiers and are not used as the
   §11 gate.
3. **UFO-120 contains a 2× super-resolution component** (320×240 input vs 640×480
   reference) in every pair. The locked resize removes it from both members; no SR
   claim is made.
4. **Single seed.** Every arm is deterministic given its seed, but the seed itself
   is one draw. Per-image paired differences are reported (mean, median, worst
   decile) rather than only aggregate means, so the reader can see dispersion.
5. **Small headroom.** ~0.01 dB mean / ~0.3 dB worst-decile above the frozen
   source. Conclusions are about direction and tail risk, not magnitude.
6. **Adapter budget not re-selected after the RNG-seeding repair.** The repair
   changes the realisation of the consistency jitter, not its distribution; the
   dev selection was made before it. Disclosed rather than hidden.

## 8. Positioning (from `docs/novelty_audit.md`)

The nearest prior work already owns parts of this space: DCTTA (CVPR 2026) does
restoration TTA with a static Fisher mask but has no accept/reject and no anchor
term; Wakayama (ICML 2026) owns "when / how far / which direction" *with* a
PAC-Bayes guarantee; Schirmer et al. (NeurIPS 2025) monitor TTA risk at stream
level with alarms, not per-update counterfactuals; PORTAL (arXiv 2026) buys labels
to choose among fixed rules; D2TTA (ICASSP 2026) filters by uncertainty with no
per-update utility and no anchor term. What remains uniquely ours is narrow and
must be claimed narrowly: a **dense-restoration, per-update, two-reference
(d_local, d_anchor) accept/keep/reset rule with exact rollback and an auditable
action log**, evaluated causally. The novelty audit lists 14 claims we explicitly
will **not** make.

## 9. Reproduction

```bash
bash sync.sh                                   # local -> server
ssh school-gpu 'cd /home/user/BRCTTA && bash scripts/run_main3.sh'
/home/user/Conda/CDF_FFT/bin/python scripts/07_replay.py --stream UIEB2LSUI
/home/user/Conda/CDF_FFT/bin/python scripts/08_acceptance.py
/home/user/Conda/CDF_FFT/bin/python scripts/00g_engine_checks.py   # 4 regression checks
/home/user/Conda/CDF_FFT/bin/python scripts/00f_env_lock.py --check
```

## 10. Artifact index

| artifact | path |
|---|---|
| locked protocol / hypotheses | `docs/research_plan.md` |
| prior-art matrix (33 rows) | `docs/literature_matrix.csv` |
| SOTA registry + comparability tiers | `docs/sota_registry.csv` |
| novelty audit, 14 "will not claim" items | `docs/novelty_audit.md` |
| weight hashes and provenance | `docs/weights_manifest.json` |
| environment lock | `docs/env.lock.json` |
| frozen-code hashes | `docs/frozen_hashes.json` |
| per-run ledger | `runs/experiment_ledger.csv` |
| per-step action logs | `runs/<stream>/action_log_*.jsonl` |
| replay evidence | `runs/<stream>/replay_check.json` |
| engine regression checks | `runs/manifests/engine_checks.json` |
| acceptance (machine / human) | `runs/acceptance_report.json`, `.md` |

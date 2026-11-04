# Research plan — Risk-Controlled Test-Time Adaptation for Underwater Image Enhancement

> Status: living document. Sections marked **[locked]** may not change after the
> first test-set evaluation; changing them requires a dated entry in
> `docs/experiment_ledger.csv` and an explicit statement in the final report.

---

## 1. Question

Source-only underwater image enhancement (UIE) models fail on a *different*
underwater domain. Test-time adaptation (TTA) can fix some images and damage
others. The question is not "does TTA help on average" but:

> **Given a per-image candidate update, can we decide — using no labels and no
> future frames — whether committing it reduces or increases this image's error,
> and can that decision be made causal, auditable and reversible?**

The object of study is therefore a *risk-controlled commit rule* sitting on top of
a frozen source model, not a new enhancer architecture.

## 2. Hypothesis decomposition

| id | claim | falsified if |
|----|-------|--------------|
| H1 | The two references (θ_t and θ_0) disagree in sign on a substantial share of updates | sign disagreement < 10% of updates |
| H2 | Labelled-free evidence predicts per-update harm better than chance | AUROC of the learned controller ≈ 0.5 on held-out dev |
| H3 | Gating beats always-accept on tail risk (worst-10% ΔPSNR) and NAR | no improvement at matched accept rate |
| H4 | Gating retains benefit (does not just refuse everything) | accept rate on oracle-beneficial updates < 50% |
| H5 | The gain survives a rate-matched control (random reset at the same rate) | random gate at matched rate is statistically indistinguishable |

H1–H5 are all measurable within the delivered code. H5 is the one that decides
whether the *decision mechanism* (as opposed to the regularising effect of
occasionally resetting) is doing the work.

## 3. Protocol **[locked]**

### 3.1 Transfers (§5.1)

| stream | source (train) | target (test) | images |
|--------|----------------|---------------|--------|
| `LSUI2UIEB` | LSUI train | UIEB test | 89 |
| `UIEB2LSUI` | UIEB train | LSUI test | 428 |
| `LSUI2UFO`  | LSUI train | UFO-120 test | 120 |

Resolution: **256×256, whole image resized (bicubic, `reducing_gap=3`), no crop,
no pad, identical for input and reference, identical for every method.** Native
resolution is a separate, later track and is not mixed into these tables.

### 3.2 Causal prequential loop (§5.2)

For t = 1..T, in one fixed order shared by every method:

1. `y_t = f(x_t; θ_t)` — the output is produced and scored **before** any update
   from `x_t` is applied.
2. `θ⁺ = Adapt(θ_t, x_t [, past unlabeled inputs])` — candidate.
3. **Evidence** `e_t` is computed from `(x_t, y_t, y⁺, θ_t, θ⁺, θ_0)` only.
4. **Action** `a_t ∈ {accept, keep, reset}` from `e_t`.
5. `θ_{t+1} = θ⁺` if accept, `θ_t` if keep, `θ_0` if reset.

No future image is ever read. No reference is read at decision time. The oracle
gate does read the reference and is therefore reported as a **non-deployable
upper bound**, never as a method.

`episodic` mode (reset every episode, update may use the episode's own image) is a
**separate table** and is never compared against the online table.

### 3.3 Where δ and the threshold come from

* `δ` (risk event threshold) = median of `d_local` on the **dev** split.
* Controller threshold = smallest cut whose false-accept rate ≤ 0.10 on the **dev**
  calibration block (chronological 70/30 split of the dev episodes).
* The adapter budget (steps × lr) is selected on **dev only**, by maximising the
  oracle ceiling among cells with at least 0.05 dB of gateable headroom.

Dev sets used, and why:

| use | split | justification |
|-----|-------|---------------|
| adapter budget, δ, threshold | target-domain **dev** split (UIEB/val, LSUI/val, UFO/val) | the shift being calibrated for exists only against the target domain; target *test* is never touched |
| auxiliary robustness check | EUVP-d val (independent third domain) | disclosed separately; never a locked target |

### 3.4 What is forbidden

* Using target test references for anything except scoring.
* Selecting the threshold, δ, adapter budget, seeds or stream order after seeing
  test results; keeping only favourable orders.
* Reporting MSE-based acceptance criteria (§11 forbids substituting them for the
  5% PSNR/SSIM rule).
* Mixing paper numbers computed at other resolutions or splits into one ranking
  (see `docs/sota_registry.csv`, which separates five comparability tiers).
* Claiming the episodic table and the online table measure the same thing.

## 4. Implementation

```
src/brctta/
  mambacue/       frozen backbone (vendored from the user's own MambaCue)
  model.py        ResidualAdapter + AdaptedModel (state snapshot/restore)
  losses.py       self-supervised objective (hf prior, RGB/Lab prior, TV, consistency)
  signals.py      the 14 evidence channels of §7.2
  risk.py         L2 logistic risk model, calibration, threshold selection
  adapt.py        the causal engine, action log, replay
  baselines.py    frozen / always / random / oracle
  evaluate.py     the negative-transfer metric suite of §9.2
  data.py         locked streams, resizing, manifests
```

**Plasticity is confined to an 11,917-parameter residual adapter** on top of a
2,244,946-parameter frozen backbone. The adapter's last convolution and its affine
head are zero-initialised, so `θ_0` reproduces the frozen source **exactly**
(verified: `max|y − y_0| = 0.0`). Two consequences that make the protocol honest:

* `reset` is an exact identity on θ, so "full state rollback" is exact by
  construction rather than by approximation.
* `keep` needs no undo, so no branch of the decision can leak a partial update.

## 5. Metrics (§9.2)

Primary: prequential PSNR / SSIM (the score of `y_t`, produced before the update).
Negative-transfer suite: mean and median ΔPSNR / ΔSSIM, `NAR_0`, `NAR_δ`,
worst-10% mean ΔPSNR, accept / keep / reset rates, false-accept rate,
harmful-update acceptance ratio, beneficial-update false-reject rate, benefit
retention, regret versus the per-image local oracle.

Δ is always measured against **the state the method actually had** (prequential
baseline), never against a state reconstructed after the fact.

## 6. Known ambiguities, resolved explicitly

1. **"UIEB-R90"** appears once in §5.1 with no definition. Resolved as: the UIEB
   paired test protocol of 90 held-out images (the T90 protocol family in the
   literature). Recorded here so the choice is auditable rather than implicit.
2. **UFO-120 contains a 2× super-resolution component** (320×240 input vs
   640×480 reference) in every one of its 1,200/300/120 pairs. The locked 256×256
   whole-image resize removes the SR component from both members. No SR claim is
   made anywhere.
3. **The dev set must come from the target domain.** A source-domain dev split
   cannot calibrate a detector of target-domain drift. Target *dev* is used;
   target *test* remains untouched until scoring.
4. **The adapter is tiny on purpose.** A larger adapter would give a bigger
   headroom and a more impressive number, but it would also change the object of
   study from "is this update safe to commit" to "can a big adapter be trained
   unsupervised". Headroom is measured, not manufactured (see §7).

## 7. Measured headroom (dev only, before any test run)

`headroom = PSNR(oracle gate) − PSNR(always accept)`, on dev, n≈60.

| stream / dev | cell | always ΔPSNR | oracle ΔPSNR | headroom (dB) | worst-10 improvement |
|---|---|---|---|---|---|
| LSUI2UIEB / UIEB-val | 1×0.001 | −0.0138 | +0.0181 | **0.088** | 0.309 |
| LSUI2UIEB / UIEB-val | 3×0.003 | −0.1583 | +0.124 | 1.564 | 1.904 |
| LSUI2UIEB / EUVP-d | 1×0.001 | +0.0141 | — | — | 0.057 |

The raw-headroom maximum (3×0.003) is **not** selected: its *ceiling* is 0.6 dB
below the 1×0.001 ceiling, i.e. it wins by destroying the image and letting the
oracle rescue it. Selection therefore maximises the oracle ceiling among cells
with headroom ≥ 0.05 dB. This rule was written down before the test run and is
applied identically to all three streams.

The honest reading of these numbers: with a frozen source model and an
11.9k-parameter adapter, the *achievable* gain is small (≈0.09 dB mean, ≈0.31 dB on
the worst decile). That is the regime the paper must argue in — the contribution
is tail-risk control under a small but non-zero headroom, not a large PSNR jump.

## 8. Execution order

1. `scripts/00_audit.py`, `00b_source_stats.py` — manifests and source priors.
2. `scripts/04b_select_adapt.py <stream> --dev-domain target` — adapter budget.
3. `scripts/02_episodes.py <stream> --dev-domain target` — dev episodes.
4. `scripts/03_risk_train.py <stream>` — δ, controller fit, calibration.
5. `scripts/05_online_eval.py <stream>` — the five methods on the locked test split.
6. `scripts/07_replay.py <stream>` — replay must reproduce the log.
7. `scripts/06_episodic.py <stream>` — separate episodic table.
8. `scripts/08_acceptance.py` — §11 gates, applied mechanically.

## 9. Risks to the result, stated in advance

* **Headroom may be too small for a stable gate.** If the oracle ceiling is within
  noise of always-accept, H3 cannot be established and the honest conclusion is
  "at this adapter scale, gating is not measurable on this transfer".
* **The controller may learn "never accept".** Guarded by reporting accept rate
  and the beneficial-update false-reject rate next to every headline number.
* **Sign disagreement between the two references may be rare.** If H1 fails the
  two-reference decomposition collapses to a single reference; that would be
  reported as a negative result about the mechanism, not hidden.

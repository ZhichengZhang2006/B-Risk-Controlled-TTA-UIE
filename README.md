# B — Risk-Controlled Test-Time Adaptation for Underwater Image Enhancement

Can a test-time adaptation (TTA) method be made **safe** on a source→target
underwater transfer, i.e. can it accept the updates that help and reject the ones
that hurt, using **no target labels**?

This repository is the full experimental record: the frozen evaluation protocol,
every diagnostic that led to the design, the negative results, and the acceptance
report generated against the spec's §11 gate.

---

## Headline result (read this first)

The honest headline is **not** "our method beats SOTA". It is:

> A calibrated gate over cheap per-step evidence separates helpful from harmful
> test-time updates on **two of the three locked transfers** (dev out-of-fold
> AUROC 0.71 / 0.73) and buys tail-risk reduction over accepting every update at a
> fraction of the accept rate. On the **third** transfer the same evidence has no
> dev-predictive power (AUROC 0.44–0.46) and the pre-registered abstain rule
> refuses to act; we report that as a negative result instead of tuning until it
> passes. The gateable headroom above a frozen source model is small (order
> 0.01 dB mean, ~0.3 dB on the worst decile), and the single-channel story that
> motivated the work — re-degradation residual as a verifier — is **falsified**:
> it loses 3–5 dB when used as a filter (see §"What failed").
>
> The §11 acceptance verdict is **fail (24 pass / 4 fail / 0 unverifiable)**, with
> all four failures on the primary LSUI→UIEB transfer — including a 5.10 % PSNR
> shortfall against a stronger frozen model. Where the gate does help, it
> *prevents damage*: it turns a −0.73 dB regression into −0.19 dB on UIEB→LSUI and
> cuts the worst decile by 58 %. It never turns adaptation into a gain over not
> adapting, and a rate-matched random gate matches or beats it on tail risk on two
> of three streams. That comparison is the reason the mechanism claim is stated
> narrowly.

The rule is reported in two coordinate systems that must never be mixed: the
**§11 comparable-method gate** (this repository's own measured arms) and
**published numbers** (other papers at their own resolutions and splits, kept in
`docs/sota_registry.csv` with five comparability tiers).

Everything below documents how that conclusion was reached and how it is
measured. Numbers live in `runs/acceptance_report.md` and
`runs/experiment_ledger.csv`; they are never quoted from memory.

---

## What is being tested

**Transfer (spec §5.1), locked before any test run:**

| stream | source supervision | target at test time | target split |
|---|---|---|---|
| `LSUI2UIEB` | LSUI train (3423 pairs) | UIEB test (89) | 256×256 whole-image resize |
| `UIEB2LSUI` | UIEB train (712 pairs) | LSUI test (428) | 256×256 whole-image resize |
| `LSUI2UFO`  | LSUI train (3423 pairs) | UFO-120 test (120) | 256×256 whole-image resize |

**Protocol (spec §5.2): causal prequential.**
At step `t` the method must first emit `y_t = f(x_t; θ_t)` using only the state
committed *before* seeing `x_t`. Only then may it look at `x_t` (and past inputs)
to build a candidate `θ_t⁺`, and decide `keep / accept / reset`. No future frame is
ever visible; every arm consumes the identical stream order (seeded permutation
recorded as `order_digest`).

**Actions (spec §4):**

- `θ_0` — frozen source model (backbone + source-trained residual adapter)
- `θ_t⁺ = Adapt(θ_t, x_t)` — one candidate update
- `accept` / `keep` / `reset` (reset returns to `θ_0`, exactly)
- `d_local = L(θ_t⁺) − L(θ_t)`,  `d_anchor = L(θ_t⁺) − L(θ_0)` — the two references
- risk event: `d_local > δ`, with `δ` fixed on the **dev** split

The two references exist because they answer different questions: `d_local` says
"is this step better than where I am" (it can drift), `d_anchor` says "is this
better than doing nothing at all" (spec §4's warning). Both are computed
label-free, from the self-supervised objective, and both are input to the gate.

---

## Architecture

```
x ──► frozen backbone (MambaCue or AquaClarity) ──► y_source ──┐
                                                              ├─► y = clamp(y_source + r, 0, 1)
x, y_source ──► ResidualAdapter (11,917 params) ──► r ────────┘
```

The adapter's last conv and affine are **zero-initialised**, so `θ_0` is
*bitwise identical* to the frozen source model (verified: `max|y − y_source| = 0`).
Two consequences that matter for the protocol:

1. `reset` is an exact identity on `θ` — rollback needs no bookkeeping and cannot drift.
2. All plasticity lives in 11,917 parameters, so "how far did I move" is a
   well-defined scalar (`update_magnitude`) rather than an accounting exercise.

`θ_0` may also carry a **source-pretrained** adapter (`scripts/01b_adapter_pretrain.py`,
fitted on source pairs only, L1 + 0.5·SSIM). That is still a source-only model
under §5.1; the honest "do nothing" reference is therefore `source_adapter`, and
TTA gain is measured **against it**, never against the bare backbone.

---

## The mechanism

Most TTA gates ask "did my self-supervised loss go down?". Spec §4 warns that this
is not evidence of quality, and the experiments here confirm it empirically:

- `scripts/04g_objective.py` (dev): **every** candidate no-reference objective
  *loses* PSNR when you follow its gradient; `corr(loss_decrease, ΔPSNR)` is
  **negative** for most objectives.
- `scripts/04i_statsmatch.py` (dev): every statistics-matching arm (source
  RGB/Lab prior, grey-world, gamma) loses 0.5–1.8 dB.
- `scripts/_diag_affine.py` (dev): a per-image affine fitted to GT is worth
  **+6.9 dB**, but a dataset-level affine is worth only **+0.17 dB**. So the
  missing information is *per-image* colour correction — not a global cast.

The gate's evidence vector (`src/brctta/signals.py`) has 19 channels: update
magnitude and step norm, output distance to the current and frozen models,
gradient conflict/cosine between the self-supervised objectives, clipping ratio,
high-frequency residual, saturation/contrast/colour statistics, and the
re-degradation residual in three forms (candidate-vs-input, candidate-vs-current,
candidate-vs-frozen). A calibrated logistic controller (`src/brctta/risk.py`) maps
that vector to `accept / keep / reset` under a false-accept budget chosen on dev.

**What failed, and is kept in the repository as a negative result.** The work
started from the hypothesis that the re-degradation residual is the one label-free
quantity that carries real information. Measured on dev (`scripts/04j_affine_redegrade.py`),
re-degrading a candidate clean image and comparing to the observed input is
*anti-correlated* with the candidate's true ΔPSNR (r = −0.43 / −0.36 across two
transfers), and filtering candidates by it loses 3.6–5.0 dB versus not filtering
at all. A dedicated probe (`scripts/_probe_degrader.py`) shows the source-fitted
degrader can fit its own training pairs but does not transfer the right residual
ordering. The residual survives only as one weak channel (univariate AUC ≈0.60)
inside a multi-channel model. This is why the README's original headline was
withdrawn.

---

## Repository layout

```
src/brctta/
  data.py      stream spec, ordering, manifests, whole-image resize
  cache.py     materialised uint8 stream cache (bitwise-verified vs the loader)
  model.py     ResidualAdapter, AdaptedModel (snapshot/restore/rollback)
  backbones.py MambaCue + AquaClarity loaders behind one frozen-map interface
  losses.py    self-supervised objectives (hf / rgb / lab / tv / consistency)
  degrade.py   underwater degradation model + re-degradation residual
  signals.py   per-step evidence vector
  risk.py      calibrated gate: accept / keep / reset
  adapt.py     the causal online loop, action logs, replay
  baselines.py frozen / always / random / oracle / simple-threshold / stochastic
  evaluate.py  §9.2 negative-transfer metrics
  metrics.py   PSNR / SSIM / MSE / UIQM / UCIQE (frozen; see validation)

scripts/
  00_audit.py             data audit + manifests + duplicate scan
  00b_source_stats.py     source clean-image statistics
  00c_metric_validation.py metric identity, PSNR/MSE identity, protocol replay
  00d_cache.py            build + verify the stream cache
  00e_degrade_train.py    fit the degradation model (source only)
  00f_env_lock.py         environment lock (--check to verify)
  00g_engine_checks.py    engine regression checks: candidate labels, determinism,
                          forced replay, exact rollback
  01_source_train.py      source training (if a backbone needs retraining)
  01b_adapter_pretrain.py fit θ_0's residual adapter (source only)
  02_episodes.py          dev episodes for risk learning
  03_risk_train.py        fit + calibrate the controller (dev only)
  04_probe.py             quick dev probe of one cell
  04b/04c                 adapter-budget selection + auditable reselection
  04e/04f/04g/04h/04i/04j ceiling, gradient alignment, objective, θ_0, stats, affine
  05_online_eval.py       the main causal table (test)
  06_episodic.py          episodic table (separate, never mixed)
  07_replay.py            forced replay of a logged run
  08_comparison.py        §11 comparison set
  08_acceptance.py        §11 gate -> runs/acceptance_report.json
  08b_acceptance_md.py    renders the verdict to runs/acceptance_report.md
  10_ledger.py            experiment_ledger.csv from result files
  11_plots.py             curves / histograms
  12_freeze.py            hash-freeze the protocol files (--check for drift)
  13_weights_manifest.py  SHA256 + provenance for every weight
  14_report_tables.py     result tables -> docs/main_tables.md
  gpu_run.sh              single-CUDA-slot lock wrapper (use it for every GPU job)
  run_stream.sh           one transfer end-to-end
  run_main3.sh            all three transfers + comparison + acceptance
```

---

## Running it

The pod exposes **one CUDA slot**. Two concurrent CUDA processes do not queue —
the second fails with a misleading `Expected u.is_cuda() to be true`. Every GPU
job must go through `scripts/gpu_run.sh`, which takes an flock.

```bash
# 0. environment (already provisioned on the pod)
PY=/home/user/Conda/CDF_FFT/bin/python    # torch 2.1.0+cu118, python 3.8.20

# 1. audit + cache (CPU-heavy, ~5 min)
$PY scripts/00_audit.py
$PY scripts/00d_cache.py --splits all --verify 5

# 2. metric validation -- run this before believing any number
$PY scripts/00c_metric_validation.py

# 3. θ_0: source-fitted adapter + degradation model
bash scripts/gpu_run.sh 01b_adapter_pretrain.py --source LSUI --epochs 6
bash scripts/gpu_run.sh 00e_degrade_train.py --source LSUI --epochs 8

# 4. the three locked transfers (selection -> episodes -> risk -> test -> replay)
bash scripts/run_main3.sh

# 5. gate + report
$PY scripts/08_comparison.py
$PY scripts/08_acceptance.py
$PY scripts/10_ledger.py
$PY scripts/11_plots.py
```

Every result file records the stream digest, the checkpoint hash, the adapter
hash and the code path that produced it, so a number can always be traced back.

---

## Metrics

Main table: PSNR (dB), SSIM, MSE (per-image, then mean), all on the committed
(prequential) state.

Negative-transfer reporting (spec §9.2): mean/median ΔPSNR and ΔSSIM, `NAR_0`
(fraction of images made worse), `NAR_δ` (fraction worsened by more than δ),
worst-10% mean ΔPSNR, accept/keep/reset rates, false-accept rate,
harmful-update acceptance ratio, beneficial-update false-reject rate, benefit
retention and regret against a local oracle.

**Metric honesty.** `scripts/00c_metric_validation.py` checks three things and
all three pass: our PSNR agrees with `skimage` to `0.0` and our SSIM to `1.1e-16`
(both against the *Wang* Gaussian variant — the default uniform 7×7 window is a
different definition and is reported separately), `10·log10(1/MSE)` reproduces
PSNR exactly, and replaying frozen MambaCue on its own test split reproduces the
upstream number to 0.11 dB. A real SSIM bug (scipy's default `truncate=4.0`
built a 13×13 kernel while the code cropped 5 pixels) was found and fixed by
this check, which is the reason it exists.

---

## Results (causal prequential, 256×256 whole-image, 9 arms per stream)

Full tables, including candidate-view diagnostics, are generated into
`docs/main_tables.md` by `scripts/14_report_tables.py`; the §11 verdict is
`runs/acceptance_report.md`.

| transfer (n) | frozen | always-accept | **ours** | accept / reset | worst-10 % ΔPSNR | NAR₀ | verdict |
|---|---|---|---|---|---|---|---|
| LSUI→UIEB (89) | 21.0025 | 20.9747 | **20.9361** | 0 % / 0 % | +0.0000 | 0.000 | **abstain** (dev OOF AUROC 0.458) |
| UIEB→LSUI (428) | 22.1777 | 21.4473 | **21.9886** | 18.7 % / 63.3 % | −0.0404 (always −0.0969) | 0.114 (always 0.463) | gate limits damage |
| LSUI→UFO-120 (120) | 27.7081 | 27.5165 | **27.6598** | 22.5 % / 54.2 % | −0.0271 (always −0.0827) | 0.192 (always 0.592) | gate limits damage |

Read the table this way: on two transfers unconditional adaptation is **net
harmful** (−0.73 dB and −0.19 dB against not adapting), and the gate reduces that
damage by 0.54 dB and 0.14 dB while cutting tail harm by 58 % and 67 %. It never
beats not adapting, and the rate-matched random control (`random_matched`) beats
it on tail risk on two of three streams — so the defensible claim is benefit
retention plus damage limitation, not uniform risk control.

§11 acceptance: **fail, 24 pass / 4 fail / 0 unverifiable**. All four failures are
on LSUI→UIEB — a 5.10 % PSNR shortfall (1.1251 dB) against the stronger frozen
`strong_frozen` arm (AquaClarity, source-domain checkpoint), and three claim gates
that correctly refuse to credit an abstaining method.

---


## What this work does **not** claim

- Not SOTA on PSNR. The comparison set includes strong target-supervised methods;
  the §11 gate is `g_m ≤ 0.05` (5%) against the best *deployable, same-protocol*
  arm, and the report states the verdict honestly, including `unverifiable`.
- Not "TTA always helps". The measured gateable headroom over the frozen source
  model is **+0.006 to +0.049 dB** on the three streams. That is the finding: on
  this transfer, most of the apparent headroom is per-image colour correction
  that no label-free signal recovers, and a gate can only harvest a small part.
- Not a new theoretical guarantee. The decision-theoretic view of TTT with a
  PAC-Bayes bound is prior art; this work is the empirical, protocol-locked
  measurement of what per-update accept/reject can and cannot buy.
- No oracle numbers in the main table. `oracle` reads ground truth and is
  reported separately as a diagnostic ceiling, never as a method.

See `docs/novelty_audit.md` for the 10-item threat list and the binding NOT-claim
list, and `docs/literature_matrix.csv` / `docs/sota_registry.csv` for the evidence.

---

## Reproducibility

- `configs/main.yaml` — all paths, splits, streams, budgets.
- `configs/sota_reference.json` — the pinned §11 reference values with their sources.
- `runs/manifests/` — data audit, cache report, metric validation, dataset hashes.
- `runs/experiment_ledger.csv` — every experiment, its purpose and its numbers.
- `docs/weights_manifest.json` — SHA256, size and provenance for θ₀, adapters and
  degraders (the weights themselves stay on the pod; the manifest is the contract).
- `docs/env.lock.json`, `docs/frozen_hashes.json` — environment and protocol hashes.
- `scripts/07_replay.py` — re-runs a logged stream and asserts the per-image
  PSNR/SSIM/candidate-PSNR reproduce to `1e-6` (measured float-noise floor
  `~1e-11 dB`; the pre-repair drift was `1.7e-3 dB`).

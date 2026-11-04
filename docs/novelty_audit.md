# Novelty Audit — Risk-Controlled Test-Time Adaptation for Underwater Image Enhancement

**Auditor:** literature/SOTA auditor (teammate `lit-auditor`)
**Date:** 2026-11-05
**Companion files:** `docs/literature_matrix.csv` (33 rows), `docs/sota_registry.csv` (28 rows)

**Scope of this document.** This is a threat assessment, not a defence. It exists to make the paper's
claim falsifiable and to remove any sentence we cannot support. Every claim below is tied to a row in
`docs/literature_matrix.csv` or to a primary source recorded in `docs/sota_registry.csv`.
Where I could not reach a primary page I say so and downgrade the verification status rather than
assert a detail.

**Verification legend** (same convention as the CSVs):
`verified_primary` = I fetched the paper PDF or its official publisher/OpenReview/PMLR page and read
the relevant text; `verified_secondary` = reputable index (DOI record, OpenAlex, proceedings listing)
only; `unverified` = could not confirm.

---

## 0. Summary judgement

The *problem we care about* — "after entering an unknown water body with no target reference, is this
particular parameter update good or bad?" — is **not virgin territory in 2026**. Four separate
communities have arrived at adjacent formulations within the last eighteen months:

1. **Restoration TTA** has already built the machinery of degradation-consistent self-supervision and
   already knows that naive adaptation damages pixel-level detail (DCTTA CVPR 2026; D2TTA ICASSP 2026;
   TAO ICML 2024).
2. **Decision-theoretic TTT** has already asked *when to adapt, how far, and which direction*, and has
   a PAC-Bayes guarantee for step selection (Wakayama, ICML 2026).
3. **TTA risk monitoring** has already built label-free sequential alarms on a stream (Schirmer et al.,
   NeurIPS 2025).
4. **Active/interventional test-time control** has already framed "should I intervene at all, and is the
   gain worth the evidence" and even written down that "the same intervention can rescue one batch and
   damage the next" (Zhang et al., arXiv:2610.37658, Oct 2026).

So the honest position is: **the question is taken; the object is not.** Nobody has built, for a dense
restoration task, a *signed, per-candidate-update* utility predictor that scores a proposed update
against **two different references simultaneously** (keep-current and reset-to-frozen), with physical
proxy mismatch handled explicitly, and a decision rule whose operating point is chosen offline on a dev
set and then frozen. That is the claim we can defend. Everything else in this file is about not
overreaching past it.

---

## (a) What does DCTTA's re-degradation + consistency + important-parameter selection leave unanswered?

**The paper.** "Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration",
Ni Tang, Shenghao Nie, Xiaotong Luo, Yuan Xie, Yanyun Qu; **CVPR 2026**, pp. 15476–15485
(`verified_primary`; `https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.html`;
code `https://github.com/tonia86/DCTTA`, 14 stars, last push 2026-07-09). Matrix row **L01**.

Its three components:

- **TRG** — a test-time-trained RDDM diffusion degradation generator, fitted on the residual
  `I_res = ȳ − x_in`, where `ȳ` is the teacher pseudo-label. Produces a synthetic degraded view `x_sd`.
- **DIR** — consistency between `f_θ(x_in)` and `f_θ(x_sd)` in pixel L1 plus a VGG perceptual term,
  `λ_cons = 1`, `λ_per = 0.01`; plus adaptation-consistency to `ȳ`, an 8-geometric-augmentation
  ensemble. EMA teacher, `η = 0.95`; lr `2e-4`, batch 1, RDDM 5 steps.
- **TIPS** — a *diagonal Fisher* importance, `ω(θ_i) = (1/|D|) Σ (∂L_c/∂θ_i)²`, computed **once,
  before TTA starts**, and the top `ρ = 0.4` fraction of parameters are frozen for the whole run.

**What it answers.** It answers "which parameters should move" and "what self-supervised signal keeps
the update degradation-consistent". It is a genuine advance on Tent-style full-parameter updating for
restoration.

**What it leaves unanswered — four specific gaps, each checkable in the paper text:**

1. **The masking is static and pre-committed.** TIPS is computed once from the source data and never
   revised. So the method can express *"these parameters are important in general"* but cannot express
   *"this particular update, right now, on this image, is likely harmful"*. Once `ρ` and the Fisher
   ranking are fixed, every subsequent update within the unfrozen subspace is accepted
   unconditionally. There is **no accept/reject decision anywhere in the method**.
2. **There is exactly one reference, and it is the pseudo-label.** `ȳ` comes from the model's own
   augmentation ensemble. Every consistency term is measured against a quantity produced by the very
   parameters being updated. Nothing in the objective is measured against the **frozen source model
   `θ₀`**. So DCTTA cannot detect *cumulative drift* — `d_anchor = L(θ_t+) − L(θ₀)` — at all; its
   objective is blind to it by construction. A run that slowly walks away from `θ₀` while remaining
   internally self-consistent is, to DCTTA, a success.
3. **No notion of an incremental effect.** The objective is an absolute loss on the current batch, not
   a *difference* between the candidate and the committed state. `d_local = L(θ_t+) − L(θ_t)` is not
   computed, so the method cannot report what the last update actually bought, nor can it detect the
   regime where the loss falls while the image gets worse.
4. **No rollback and no accumulation control.** There is no mechanism to undo an update after the fact.
   Harm, if it occurs, is expressed as drift in the parameters and is only visible afterwards, in the
   metric — not in the loop. DCTTA's claim (+4.57 dB over un-adapted on Rain100H) is an *end-of-run*
   number; nothing in the paper characterises the worst-case image or the distribution of per-image
   deltas.

**The residual gap, stated as a question DCTTA cannot pose:**
> Given the committed state `θ_t` and a *proposed* update producing `θ_t+`, and given only unlabeled
> input, will accepting this update reduce or increase restoration error — and is the answer different
> depending on whether we compare to `θ_t` or to `θ₀`?

That is a *decision* question about an *object* (the candidate update) that DCTTA does not construct.
This is the seam we work in.

**Important honesty note.** Re-degradation itself is **not ours and not DCTTA's**. It appears in UIE as
"Re-Degradation and Contrastive Learning for Zero-shot Underwater Image Restoration", Nisha Varghese
and A. N. Rajagopalan, **BMVC 2023**, paper #544, pp. 544–561 (matrix row **L02**;
`https://proceedings.bmvc2023.org/544/`). That venue host is DNS-unreachable from this machine and every
mirror I tried returned 404, so **row L02 is `verified_secondary`** — the title/venue/authors are
confirmed by search indices, the detailed content is not something I read first-hand. Do not quote
specific numbers from it. We must cite it as the origin of the re-degradation idea in UIE, and we must
not present re-degradation as novel. Also note that Shot-Again (IEEE Access 2026, DOI
`10.1109/access.2026.3734227`, row **L03**, `verified_secondary`) already treats the re-degradation
relation as *"a soft regularization cue rather than ... an exact physical law"* — which is our
physical-proxy-mismatch caveat, published, in this exact subfield. We must attribute that caveat, not
claim it.

---

## (b) Do existing TTA risk monitors already solve this task?

**No — but the distance is smaller than is comfortable, and one of them is close enough to require a
careful paragraph in the paper.**

### b.1 The strongest monitor: Schirmer et al., NeurIPS 2025

"Mona Schirmer, Metod Jazbec, Christian Andersson Naesseth, Eric Nalisnick — *Monitoring Risks in
Test-Time Adaptation*", **NeurIPS 2025**, main track, DOI `10.52202/085713-2705`, arXiv:2507.08721
(row **L05**; `verified_primary`; code `https://github.com/monasch/tta-monitor`, 4 stars, last push
2025-11-14, no weights).

The mechanism is **confidence sequences / sequential testing**: maintain a running, anytime-valid
bound on a performance criterion and raise an **ALERT** when the bound crosses a pre-specified
tolerance. Everything is done without test labels, while the model is being updated.

**What it solves:** stream-level risk. "Is this adaptation run as a whole going off the rails?"
It gives a *time-uniform* guarantee, which is a stronger statistical object than a fixed-sample test
and is genuinely the right tool for that question.

**Why it does not solve our task — the distinction is structural, not a matter of degree:**

| | Schirmer et al. NeurIPS 2025 | What we need |
|---|---|---|
| Unit of analysis | the **stream** (running statistic over many batches) | the **candidate update** (one step, one image) |
| Output | a **binary alarm** at a time index | a **signed** prediction of harm/benefit |
| What triggers it | the *observed* statistic crossed a bar | a *counterfactual*: what would `L(θ_t+)` be |
| Action | the caller decides (typically: stop adapting) | keep / accept / reset, chosen per update |
| Reference | one implicit reference (the run's own history) | **two**: `θ_t` (keep) and `θ₀` (reset) |
| Task | classification / segmentation | dense restoration, PSNR/SSIM |

Three consequences that matter for us:

- **It cannot distinguish "the last update was bad" from "the stream was always mediocre."** A
  confidence sequence integrates over history. By the time the bound separates, many bad updates may
  already have been committed. Our `d_local` is defined per update precisely to be *earlier* than that.
- **It has no counterfactual.** A confidence sequence is built from the loss actually incurred by the
  parameters actually used. It never evaluates `L(θ_t+)` for an update it did not take. Our whole
  apparatus — computing a candidate's utility *before* committing — is a different experiment.
- **Alarm ≠ policy.** An alarm says "stop". Our task is the harder one: keep adapting but keep the
  *beneficial* updates. A method that only ever alarms and stops will look good on harm-avoidance and
  terrible on retention, and we must report both (see the retention predictions in §6).

### b.2 The other monitors

- **AETTA** (CVPR 2024, row **L20**): label-free *accuracy estimation* for TTA. Stream-level, and
  designed for classification accuracy, not for a dense perceptual/fidelity metric. Its estimator would
  need to be re-derived for PSNR-like quantities and would still be an aggregate.
- **Estimating Model Performance Under Covariate Shift Without Labels** (Bialek, Kivimaki, Kuberski,
  Perrakis; NeurIPS 2025, DOI `10.52202/085713-5379`, row **L33**): performance *prediction* under
  shift. Same structural limitation — predicts an aggregate, not a per-update delta.
- **CoTTA** stochastic restoration (CVPR 2022, row **L11**) and **RoTTA** (CVPR 2023, row **L14**) are
  the closest things to "rollback" in the TTA literature. CoTTA's stochastic restoration randomly
  resets a *fraction of weights* to the source with a small probability, every step, unconditionally.
  It is a **regulariser, not a decision**: it does not ask whether this update was harmful, it just
  injects noise toward `θ₀` and hopes the expectation is right. This is exactly the "random-reject
  matched rate" ablation we must beat — see prediction P3.

### b.3 The verdict

Existing monitors give us **stream-level, after-the-fact, alarm-shaped** risk. We need
**update-level, before-commitment, signed** utility. The gap is real. But it is a gap in *object
granularity and sign*, not a gap in *topic*. Any framing that says "nobody has done risk monitoring for
TTA" is false and will be rejected — Schirmer et al. is a NeurIPS 2025 main-track paper.

---

## (c) Why might ordinary confidence, simple thresholds, and existing rollback be insufficient?

**This must be an experimental question, not a premise.** Below I state the three mechanisms, and for
each the *falsifiable test* that would force us to abandon it. If the tests come back the wrong way,
the corresponding part of the story dies and we say so.

### c.1 Confidence / entropy is the wrong observable for dense restoration

TTA's standard reliability signal is predictive entropy or confidence (Tent, EATA's
`S(x) = entropy × (1 − cos-sim to prototype)`, SAR's entropy-based reliable-sample selection — rows
**L15**, **L13**, **L12**). For a dense regression/restoration output there is no categorical
distribution to be confident about. Any confidence proxy must be constructed (e.g. output variance
under augmentation, or a re-degradation residual).

**The failure mode we suspect:** entropy-like signals are *monotone in image degradation severity*,
not in *update quality*. A badly over-enhanced image can be confidently wrong. If that is true, then a
confidence-gated TTA will preferentially reject updates on hard-but-improving images and accept
updates on easy-but-degrading ones — i.e. it will be *anti-correlated* with utility in the region that
matters.

**Falsifiable test T1.** On the audit stream, compute Spearman correlation between each candidate
signal and the *true* per-update utility `d_local` (available offline, since we hold the labels for
audit only). Compare: (i) output-confidence proxies, (ii) augmentation-variance proxies, (iii)
re-degradation residual, (iv) our update-utility score.
**Kill criterion:** if (i) reaches within 0.05 Spearman of (iv), the "confidence is insufficient"
premise is false and the paper's motivation must be rewritten around *cost*, not *validity*.

### c.2 A single global threshold on a scalar loss cannot separate two references

The risk event `d_local > δ` with `δ` fixed on a dev set is well-defined, but `δ` is a threshold on a
*difference of losses*, and the identity of the loss determines what it can see. A scalar loss evaluated
on the current batch confounds:
- genuine improvement in restoration, and
- the model's own drift in the direction that happens to lower that scalar (reward hacking against a
  self-supervised surrogate — the classical failure of self-supervised TTA).

**Falsifiable test T2.** Construct the *decoupled* case by construction: take an image where the
oracle update helps (true `ΔPSNR > 0`) but the surrogate loss rises, and an image where the update hurts
(true `ΔPSNR < 0`) but the surrogate loss falls. Report the fraction of updates where
`sign(d_local)` disagrees with `sign(ΔPSNR)` per method.
**Kill criterion:** if disagreement is < 5 % for the best single-scalar baseline, there is no room for a
multi-evidence utility model and we should ship the scalar.

### c.3 Rollback exists but is blind — CoTTA-style restoration is random, not diagnostic

CoTTA's stochastic restoration resets each weight to `θ₀` with small probability `p`, independently,
every step. Two properties make it structurally incapable of what we want:
- It is **memoryless and sign-blind**: it treats every parameter identically regardless of what the
  last update did.
- Because it is applied *after* the harmful update has already affected the prediction stream, it
  cannot prevent damage to the current output — only dilute it over time.

**Falsifiable test T3 (the one that matters most).** Matching the *intervention rate*: give CoTTA-style
stochastic restoration and our gate the same expected fraction of "reset" actions, and compare worst-10 %
`ΔPSNR` and `NAR_δ`.
**Kill criterion:** if random-restoration at matched rate is within noise of our gate, our decision
mechanism adds nothing and the contribution collapses to "we measured things" — which is not enough for
a methods paper. This ablation is mandatory and must appear in the main table, not the appendix.

### c.4 An honest caveat about the threshold itself

`δ` is fixed on a dev set and then frozen. That is legitimate, but it means our risk guarantee is
conditional on the dev/target similarity. We must state that the guarantee does not survive arbitrary
distribution shift of the *stream* — the same limitation that forces conformal risk control
(Angelopoulos et al., ICLR 2024, row **L21**) to assume exchangeability, and that Barber, Candès,
Ramdas and Tibshirani (Annals of Statistics 51(2):816–845, 2023, DOI `10.1214/23-aos2276`, row **L22**)
address by replacing exchangeability with bounded likelihood ratios / drift conditions. If we claim any
coverage-type statement, we inherit one of those assumptions and must name it.

---

## (d) Does dense restoration vs classification risk change the method, or only the dataset?

**The honest answer: it changes the method in at least four places, and in two of them the classification
literature gives a *misleading* default. This is where a real contribution can live — but it must be
argued mechanism-by-mechanism, not asserted.**

Four structural differences, each with a consequence for the algorithm:

**(1) The loss is not a bounded 0/1 risk.** Classification risk is a probability of error, in `[0,1]`,
with a natural empirical estimator (the misclassification rate) and a natural concentration inequality
(Hoeffding/Bernstein on bounded variables). Restoration error is PSNR/SSIM/L1 on a continuous signal:
unbounded in principle, heavy-tailed in the tail we care about (the worst 10 % of images), and
**non-additive across pixels** — a localised colour cast can dominate human judgement while moving
global PSNR by 0.1 dB. Consequence: our risk must be defined on a **per-image** statistic with an
explicit tail treatment (worst-decile), not on a stream average; and any concentration argument must be
robust to unbounded loss (sub-Gaussian assumption must be stated and tested, not assumed).

**(2) There is no label, but there *is* a physical forward model to interrogate.** Classification TTA
has no way to check a prediction without a label. Restoration has something a classifier does not: the
degradation process. We can re-apply an estimated degradation and ask whether the restoration is
*consistent*. This is a genuinely additional evidence channel — and it is exactly what
re-degradation methods (BMVC 2023, DCTTA CVPR 2026, Shot-Again 2026) already exploit. Consequence: our
utility evidence set can include re-degradation residual, colour/structure change, perturbation
response and update sensitivity — **but every one of these is a proxy with its own bias**. The
physical-proxy mismatch is the central technical hazard, not a footnote.

**(3) The "important parameters" story is different.** Classification TTA can isolate a classifier head
or a normalisation layer; the low-level analogue is a small set of modulation/fusion parameters. DCTTA's
TIPS freezes 40 % of parameters by diagonal Fisher *once*. In restoration the Fisher ranking is much
flatter (the loss is dominated by low-level texture statistics), so "which parameters matter" is a
weaker discriminator than in classification. Consequence: an update-level decision may carry more of the
safety burden than a parameter-level mask can, which is an *argument for our object* — but it also
predicts that parameter-subspace methods will underperform less than expected, and we must test that
(prediction P5).

**(4) The metric is not the loss.** In classification the monitored quantity and the training objective
are usually the same family. In restoration we optimise L1/L2 or a perceptual surrogate but are judged
on PSNR and SSIM (and humans look at colour and contrast, which PSNR barely sees). Consequence: **the
surrogate–metric gap itself is a source of harmful updates**, and a utility model trained on surrogate
residuals may be fitting the wrong target. We must include a direct test of whether the surrogate gap
predicts realised `ΔPSNR` (prediction P2).

**Verdict for (d):** dense restoration is not "classification with a different dataset". Two of the four
differences (unbounded heavy-tailed risk; proxy-based evidence with physical mismatch) change what the
decision rule must be, not merely what it is evaluated on. But (1) and (4) are reductions of the same
underlying fact — *the observable we can compute is not the quantity we are judged on* — and we should
present them as one argument, not two, or a reviewer will call it padding.

---

## (e) Given how much recent work overlaps, how do we narrow to a falsifiable new question?

### e.1 The narrowing move

We should **stop claiming a topic** ("risk-controlled TTA") and **start claiming an estimand**:

> **The signed, per-candidate-update utility `u(θ_t → θ_t+)` — a two-reference quantity that says how
> much a *proposed* update would help or hurt restoration, relative to *both* keeping the current state
> and resetting to the frozen source — estimated from unlabeled target evidence, with the physical-proxy
> bias modelled rather than assumed away, and with an operating point calibrated offline and frozen.**

Why this survives the prior art:

- It is **not** DCTTA (no per-update decision at all; single self-referential target; no anchor term).
- It is **not** Schirmer et al. (stream-level, alarm-shaped, no counterfactual, one implicit reference).
- It is **not** Wakayama ICML 2026 (asks *when/how far/which direction* with a PAC-Bayes guarantee, in
  a Gaussian-process / linear-Gaussian setting, no restoration, no physical proxy, no dense metric).
- It is **not** PORTAL / "When Should We Intervene" (requires **label feedback**, selects among fixed
  prediction rules, no parameter updates, no `d_anchor`).
- It is **not** EATA/SAR sample selection (a per-*sample* gate on a training loss, with no notion of
  what the update does and no reference to `θ₀`).
- It is **not** CoTTA stochastic restoration (memoryless random regularisation, not a decision).

### e.2 The falsifiable new question, in one sentence

> For dense underwater restoration under a causal prequential protocol with no target labels, can a
> two-reference utility estimate computed *before commitment* separate updates that improve restoration
> from those that degrade it, well enough that gating on it beats both unconditional adaptation and
> rate-matched random rollback on worst-decile `ΔPSNR` — *while still accepting a substantial fraction
> of the beneficial updates*?

Each clause is measurable and each can fail:
"separate" → AUC/Spearman of `u` vs true `d_local` (T1); "beats unconditional" → always-update baseline
(S26); "beats rate-matched random rollback" → T3; "while still accepting" → accept-rate and retention
columns, not just harm columns.

### e.3 If the overlap turns out heavier than expected at submission time

Escalation ladder, in order — each rung is a smaller but still publishable claim:

1. **Rung 1 (preferred):** the two-reference utility object plus the offline-learn / online-unlabeled
   flow, with the physical-proxy bias modelled. Full method paper.
2. **Rung 2:** if someone publishes a per-update gate first, narrow to the **two-reference
   decomposition** as the contribution: show empirically that `d_local` and `d_anchor` disagree on a
   substantial fraction of updates, and that a `d_local`-only gate (i.e. what a naive implementation
   would do) is measurably worse than the two-reference gate. This is a *finding*, not a method, and it
   survives a concurrent method paper.
3. **Rung 3:** if even that is occupied, publish the **protocol and the measurement**: the causal
   prequential UIE-TTA benchmark with locked transfers, the `NAR_δ` / worst-decile / false-accept /
   oracle-gap instrument set, and the negative results (which signals do *not* work). A well-built
   benchmark with honest negatives is a real contribution; a weak method paper is not.

We should decide the rung *before* writing the method section, because it determines how much of the
paper is method vs measurement.

---

## 2. Prior-art threat list, ranked by closeness

Ranked by how much of our claim they would pre-empt if a reviewer put them side by side. Rows refer to
`docs/literature_matrix.csv`.

**T1 — DCTTA, CVPR 2026** (row L01, `verified_primary`). Same task family (restoration TTA), same
sub-problem (which updates to trust), same decade. Has degradation-consistency, important-parameter
selection, teacher. **Missing exactly our object**: no per-update decision, no `θ₀` reference, no
rollback, no worst-case characterisation. This is the paper a reviewer will cite to reject us if we
claim "re-degradation + consistency for restoration TTA" as novel. *Our defence: the update-level
two-reference utility and the decision, not the consistency signal.*

**T2 — Wakayama, "A Decision-Theoretic View of Test-Time Training", ICML 2026** (row L24,
`verified_primary`, PMLR 306:124279–124317). Asks *when to adapt, how far, which directions*, and
**proves** that selecting update steps via prompt evidence admits a PAC-Bayes guarantee against
overfitting. This occupies the *question* with a stronger theoretical instrument than we will have.
*Our defence: the setting is Gaussian-process / linear-Gaussian, not restoration; there is no dense
perceptual metric, no physical degradation proxy, no online unlabeled UIE stream, and no two-reference
decomposition.* **We must not claim the "how far / which direction" question as novel.**

**T3 — Schirmer, Jazbec, Naesseth, Nalisnick, "Monitoring Risks in Test-Time Adaptation", NeurIPS 2025**
(row L05, `verified_primary`). Has the words "risk", "monitoring", "test-time adaptation",
"label-free", all in a main-track NeurIPS title. *Our defence: stream-level vs update-level; alarm vs
signed utility; no counterfactual; one reference vs two; classification/segmentation vs dense
restoration.* This defence is real but must be made explicitly and early, or a reviewer will assume we
did not know about it.

**T4 — PORTAL / "When Should We Intervene at Test Time?", arXiv:2610.37658, Oct 2026** (row L23,
`verified_primary`, preprint, 79 pages). The nearest *conceptual* neighbour: a controller that decides
whether to intervene, which inputs to label, and when to stop. Its own framing — "the same intervention
can rescue one batch and damage the next" — is our motivation sentence. *Our defence: it requires target
**label feedback** (active acquisition); we have none. It selects among fixed prediction rules, not
parameter updates. No signed delta against a frozen anchor.* Because it is a very recent preprint,
**it may be a concurrent work at submission** — we must cite it and differentiate in the related-work,
not hope it is missed.

**T5 — D2TTA, "Low-Level Continual Test-Time Adaptation for Image Restoration", ICASSP 2026**
(row L07, `verified_secondary`, DOI `10.1109/icassp55912.2026.11463364`). Teacher-student + SVD feature
projection + contrastive alignment + **uncertainty-guided filtering of unreliable supervision**. States
our motivation almost verbatim: existing CTTA "often disrupt pixel-level details when applied to
restoration". Its uncertainty filter is a **per-sample** gate, not a per-update utility, and it has no
`θ₀` anchor term. *Our defence: granularity (sample vs update) and the two-reference decomposition.*
Note this row is `verified_secondary` — I did not read the full paper, only the DOI/proceedings record
and abstract-level description, so **do not attribute specific numbers to it**.

**T6 — EATA / SAR reliable-sample selection, ICML 2022 / ICLR 2023** (rows L13, L12,
`verified_primary`). The canonical "filter unreliable samples before they corrupt the model" mechanism.
If a reviewer thinks our contribution is "select reliable updates", this is the citation that kills it.
*Our defence: they gate the *sample* on entropy/prototype distance and never compare `θ_t+` to `θ_t` or
`θ₀`; the gated quantity is a training-loss heuristic, not an estimated counterfactual.*

**T7 — CoTTA stochastic restoration, CVPR 2022** (row L11, `verified_primary`). The canonical
"rollback" in TTA. *Our defence: random and memoryless. It is precisely our rate-matched control, and we
should present it as such — a control we must beat, not a competitor we ignore.*

**T8 — TAO, ICML 2024** (row L06, `verified_primary`). Establishes that test-time degradation adaptation
for restoration is an existing idea. *Our defence: per-image diffusion adapter, no decision, no drift
control, no online stream.*

**T9 — Re-degradation & Contrastive Learning, BMVC 2023** (row L02, `verified_secondary`). Origin of
re-degradation for zero-shot UIE. Establishes that our evidence channel is not new. *We cite it as
prior art for the signal; our claim is about the decision built on it.*

**T10 — Conformal risk control (ICLR 2024) and conformal prediction beyond exchangeability
(AoS 2023)** (rows L21, L22, `verified_primary`). Not competitors on task, but they own the *vocabulary*
of "risk control" and the *assumptions* it drags in. If we say "risk-controlled" we inherit
exchangeability-or-drift-bounds and must say which. A reviewer from the statistics side will ask.

Lower-ranked but must appear in related work: SDUIE (CVPR 2026, row L26 — no PSNR/SSIM at all, so a
*methods* neighbour but not a table neighbour), UniUIR (TIP 2025, row L27 — strongest in-domain
supervised coordinate on our splits), WWE-UIE (WACV 2026, row L25 — strongest recent fully-supervised
efficiency story), TUDA (TIP 2023, row L04 — the transductive DA track, a *separate* track by our own
protocol design), AquaClarity (TCE 2025, row L28), Shot-Again (IEEE Access 2026, row L03), WaterLUT
(JOE 2026, row L29), TTTIR (arXiv 2609.26151, row L08), "Restoring without Forgetting" filter-level
(arXiv 2609.38591, row L09) and continual (arXiv 2608.23799, row L10), SPARNet (arXiv 2501.00818,
row L17), Self-Bootstrapping TTA (PMLR 267, row L19), DSS/CTDA (WACV 2024, row L18), Bag of Tricks
(WACV 2024, row L31), computational-time-constrained TTA (ICML 2024, row L32).

---

## 3. What we will NOT claim

This list is binding. If a draft sentence violates one of these, it gets deleted, not softened.

1. **We will not claim to introduce domain shift in UIE.** TUDA (TIP 2023) and the whole DA line
   pre-exist us by years.
2. **We will not claim unlabeled / source-free test-time adaptation.** Tent (ICLR 2021), TTT (ICML
   2020), CoTTA (CVPR 2022), SAR (ICLR 2023), EATA (ICML 2022) all pre-exist; TAO (ICML 2024) and
   DCTTA (CVPR 2026) do it for restoration.
3. **We will not claim freezing parameters / important-parameter selection as novel.** DCTTA's TIPS is
   a diagonal-Fisher mask at `ρ = 0.4`; EATA has a Fisher anti-forgetting term. We may use such a mask,
   we may not sell it.
4. **We will not claim LoRA / low-rank adaptation as novel**, in TTA or elsewhere.
5. **We will not claim teacher–student consistency as novel.** DCTTA's DIR, D2TTA, CoTTA all have it.
6. **We will not claim re-degradation as novel**, nor the observation that the physical model is
   approximate. BMVC 2023 originated the first in UIE; Shot-Again (IEEE Access 2026) already writes that
   the relation is "a soft regularization cue rather than ... an exact physical law".
7. **We will not claim confidence/entropy thresholds or reliable-sample selection as novel.** EATA, SAR.
8. **We will not claim rollback as novel**, nor stochastic restoration. CoTTA.
9. **We will not claim risk monitoring for TTA as novel.** Schirmer et al., NeurIPS 2025, is a
   main-track paper with that exact title.
10. **We will not claim the question "when to adapt / how far / which direction" as novel.** Wakayama,
    ICML 2026, owns it and has a guarantee.
11. **We will not claim "first to" anything** without a primary-source sentence we can quote that
    survives a same-week arXiv search. At the time of writing I found four 2026 works in the adjacent
    space; the prior probability that no fifth exists is low.
12. **We will not claim a SOTA number on a table whose protocol differs from ours.** In particular we
    will not compare against WWE-UIE's 8:1:1 numbers or PhaseFormer's 32.1 dB LSUI figure on a
    different split; the registry separates those tiers explicitly (`comparability` column).
13. **We will not present the fully-supervised target-domain upper bound as a competing method.** It is
    a coordinate, not a baseline (registry tier `target_supervision_reference`).
14. **We will not claim a coverage/guarantee without naming the assumption** (exchangeability, or a
    drift bound in the sense of Barber et al. 2023).

---

## 4. What we CAN claim (the positive statement, for the abstract)

Working wording, to be tightened once experiments land:

> We formulate the *utility of a candidate test-time update* for dense underwater restoration as a
> two-reference, signed quantity — the incremental effect relative to the committed state, and the
> accumulated drift relative to the frozen source — estimated before commitment from unlabeled target
> evidence whose physical-proxy bias is modelled explicitly. We show that this object is not recoverable
> from stream-level risk monitors, from per-sample reliability gates, or from rate-matched stochastic
> restoration, and we give a decision rule, calibrated offline and frozen, that retains beneficial
> updates rather than merely suppressing harmful ones.

Every clause there maps to a measurement in the spec (`NAR_0`, `NAR_δ`, worst-10 % `ΔPSNR`, accept/reset
rate, false-accept rate, oracle gap), and to a kill criterion in §6.

---

## 5. Assumptions made (explicit, per the task instruction)

- **A1.** "UIEB-R90" is read as the 90 paired UIEB images held out by the standard 800/90 split used by
  UniUIR and U-shape. This matches the registry rows S09/S12 and the U-shape `Test-U90` protocol.
- **A2.** "UFO-120 enhanced test" is read as the UFO-120 test portion in its enhanced/paired form, used
  only as a third locked transfer, not as a training set.
- **A3.** 256×256 evaluation is taken as a *unified* protocol we impose for comparability, which means
  our numbers are **not** directly comparable to published native-resolution numbers (UniUIR tests at
  native resolution). This is recorded in `metric_notes` in the registry. Any table that mixes the two
  is invalid.
- **A4.** Where a 2026 venue paper exists only as an accepted-paper page or an arXiv preprint, I say so
  in the row rather than treating acceptance as full publication. D2TTA (ICASSP 2026) and PORTAL
  (arXiv 2610.37658) are the two places this matters most.
- **A5.** I could not access `proceedings.bmvc2023.org` (DNS failure) — row L02 is `verified_secondary`
  for that reason and no number from it should be quoted.

---

## 6. What this means for our experiments

Eight falsifiable predictions. Each names the measurement that would refute it and the consequence of
refutation. These are the experiments that decide whether the contribution exists.

**P1 — The two references disagree, and often.**
Prediction: on the audit stream, `sign(d_local)` and `sign(d_anchor)` disagree on **≥ 25 %** of updates.
*Refuted if:* disagreement < 10 %. *Consequence if refuted:* the two-reference decomposition is
decoration; collapse to a single-reference gate and move to Rung 2/3 of §e.3.

**P2 — Surrogate loss is a biased proxy for realised quality.**
Prediction: the sign of the surrogate `d_local` disagrees with the sign of the true `ΔPSNR` on
**≥ 20 %** of candidate updates, and this disagreement rate is *higher* on images with strong colour
casts than on well-balanced ones.
*Refuted if:* disagreement < 8 % overall. *Consequence if refuted:* the "metric ≠ loss" argument in §d(4)
is dead, and utility estimation reduces to a well-posed regression — easier, less interesting, and we
should say so.

**P3 — A learned gate beats rate-matched random rollback.**
Prediction: at matched expected reset rate, our gate improves **worst-10 % `ΔPSNR` by ≥ 1.0 dB** over
CoTTA-style stochastic restoration, and reduces `NAR_δ` by a relative ≥ 30 %.
*Refuted if:* the gap is within the paired bootstrap CI. *Consequence if refuted:* the decision
mechanism adds nothing over a regulariser (T3 in §c.3). This is the single most dangerous prediction and
must be run early, not last.

**P4 — The gate retains benefit, it does not just suppress harm.**
Prediction: accept rate on the oracle-beneficial subset is **≥ 60 %**, while accept rate on the
oracle-harmful subset is **≤ 25 %**; and the total `ΔPSNR` versus source-only is **positive** on all
three locked transfers.
*Refuted if:* the gate is essentially "always reset" (accept rate < 20 % overall) or if total `ΔPSNR`
is negative on any transfer. *Consequence if refuted:* we have built a harm-avoidance method, not a
risk-*controlled* adaptation method, and the abstract's "retains beneficial updates" sentence must go.

**P5 — Parameter-subspace masking underperforms update-level gating.**
Prediction: DCTTA-style TIPS masking at its published `ρ = 0.4` yields a *smaller* worst-decile
improvement than our update-level gate at comparable compute, and the gap is larger on UIE than the
restoration benchmarks DCTTA reports.
*Refuted if:* TIPS matches or beats the gate. *Consequence if refuted:* the "dense restoration changes
the method" argument (§d(3)) is wrong in the direction that matters, and a cheaper parameter-level
method should replace ours.

**P6 — The physical-proxy bias is measurable and matters.**
Prediction: the re-degradation residual's correlation with true `d_local` degrades systematically with
water-type mismatch between the estimated and true attenuation/backscatter regime; a model that
*ignores* this mismatch shows inflated false-accept rate on the most-shifted transfer (LSUI→UFO).
*Refuted if:* the residual's predictive power is invariant to water type. *Consequence if refuted:*
the physical-proxy-mismatch modelling is unnecessary complexity and should be dropped from the method.

**P7 — The advantage is not a threshold artefact.**
Prediction: sweeping `δ` over a decade, our method's Pareto frontier (harm avoided vs benefit retained)
dominates the best single-signal baseline's frontier at **every** operating point, not just at the
tuned one.
*Refuted if:* the frontier crosses. *Consequence if refuted:* the gain is one lucky `δ`, which is a
tuning result, not a mechanism — and a reviewer will find it.

**P8 — Oracle gap is small enough to be worth closing.**
Prediction: an oracle that accepts exactly the beneficial updates achieves **≤ 3.0 dB** more than our
gate on the primary transfer; i.e. there is real headroom but not so much that our gate is far from the
ceiling.
*Refuted if:* the oracle gap is > 6 dB (our gate is far from the ceiling — the headroom claim is
unearned) or < 0.5 dB (the problem is too easy to need a method). *Consequence:* either way the
contribution's framing changes; this number belongs in the abstract, not the appendix.

**Cross-cutting measurement requirements** (from the spec, restated here so no experiment forgets them):
`NAR_0` (no-risk-at-all rate), `NAR_δ` (fraction of updates with `d_local > δ`), worst-10 % `ΔPSNR`,
accept/reset/reject rates, false-accept rate, oracle-gap, and ablations for (i) random-reject at matched
rate (P3), (ii) smaller-learning-rate at matched compute (does "just adapt slower" win? — if it does,
our mechanism is a cost, not a benefit), (iii) output routing vs parameter control. Every number must be
reported per transfer, not pooled, because the three locked transfers have different difficulty and
pooling would hide exactly the failure P4 is designed to catch.

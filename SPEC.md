# SPEC — Selective Conformal Regression with Conditional Coverage Under Shift

> The detailed design of record for this repository. Author: Niles Liu.
> Live orientation is in [`README.md`](README.md); the two synthesis notes are
> [`reports/p4_writeup.md`](reports/p4_writeup.md) (short) and
> [`reports/p5_archival_note.md`](reports/p5_archival_note.md) (archival).

---

## 0. One-paragraph thesis

Conformal prediction can certify that an interval has nominal coverage; selective prediction can
trade coverage for accuracy by abstaining. What neither can usually show is whether a model
**abstains in the right place** — because real tasks carry no ground truth for "which points were
ever recoverable." We construct a financial regression task that **does** carry that ground truth
(a known recoverable-vs-irreducible decomposition), and use it to build and **mechanism-validate** a
selective conformal regressor that holds **group-conditional coverage under documented distribution
shift**. The method is portable; the demonstration is cross-sectional reconstruction of a systematic
manager's portfolio *weights* from public features — a regression whose irreducible-noise floor
(~60% bespoke) is *known by construction* rather than assumed.

---

## 1. Motivation

Reconstructing a systematic manager's cross-sectional allocation from public features is a
regression with a hard, structural ceiling: public information recovers only part of the sizing, and
the remainder is a proprietary signal that output-side data cannot reveal. That **known-unknown
split is not a failure — it is the ideal testbed for selective prediction.** "Where is the recovery
trustworthy?" becomes a precise, *checkable* question: abstain where the bespoke component dominates;
commit where public features carry the weight. This repo reframes the plateau from *"how much can we
recover"* to *"where is recovery trustworthy, and can we prove the coverage"* — turning a reconstruction
task into a portable UQ method contribution that serves both a **research** interest
(conditional/shift-robust selective prediction) and a **quant** interest (financial UQ / risk calibration).

---

## 2. Where the literature is, and the gap we take

The space is active (good: publishable; constraining: pure novelty is gone). What exists:

- **Closest prior art — "When Alpha Breaks: Two-Level Uncertainty for Safe Deployment of
  Cross-Sectional Stock Rankers" (2026, arXiv 2603.13252).** Conformal + aleatoric/epistemic
  uncertainty on cross-sectional stock **rankers**, regime-conditional coverage, abstention for safe
  trading deployment. Nearest neighbor — but it is **return/ranking prediction for trading**, with
  external feeds and deployment framing.
- **Selective Conformal Risk Control (2025, arXiv 2512.12844).** Select-then-calibrate, two λ's,
  selective coverage + conditional risk — but **classification only**, generic datasets. The paper
  gestures at a regression setting but does not build it — the selective conformal **regression** lane
  §2.2 identifies stays open (see §11 P9).
- **Field movement** (sharpens positioning + seeds §11): compositional-data conformal on the simplex
  (arXiv 2511.18141, 2026) — marginal-only, no selection/shift/known-floor (→ §11 P8); fast kernel/RKHS
  **continuous** conditional coverage (SpeedCP arXiv 2509.24100; Shape-Adaptive 2603.23374; Enhanced
  Localized CP, JASA 2026) — a modern comparator beyond Gibbs–Cherian–Candès (→ §11 P10); e-value /
  anytime-valid coverage monitoring (arXiv 2503.13050; "When Your Model Stops Working" 2603.13156) —
  an optional shift-break detector (→ §11 P11).
- **Backbone:** CQR (Romano et al. 2019), weighted conformal under covariate shift (Tibshirani et
  al. 2019), Adaptive Conformal Inference (Gibbs & Candès 2021), conditional-guarantee conformal
  (Gibbs, Cherian & Candès 2023), conformal beyond exchangeability (Barber et al. 2023). Hard wall:
  the **distribution-free conditional-coverage impossibility** (Foygel Barber et al. 2021) — respect
  it; never claim exact distribution-free conditional coverage.

**The gap we take** — distinct from all of the above on several axes simultaneously:

1. **Target = allocation reconstruction**, not returns/ranking. A supervised regression whose
   irreducible-noise floor is *known* (~60%), not the efficient-market near-zero-R² setting.
2. **Selective *regression*** with conditional coverage — SCRC is classification; selective conformal
   *regression* with both selective coverage and group-conditional coverage is comparatively open.
3. **Real, documented shift** — a train-calm → test-drawdown regime shift with concrete,
   pre-registered shift axes, not a synthetic perturbation.
4. **The known-floor mechanism check** — the headline novelty: *can we show the abstention region
   coincides with the genuinely-recoverable signal, and that interval widths track the irreducible
   floor?* Most conformal papers cannot ask this. We can, because the task is built to have ground
   truth for recoverability.
5. **Pre-registration discipline** — gate-first ship/reject, shuffle nulls, clustered bootstrap,
   multi-seed bands, 4-lens adversarial verify. Rare in applied conformal work.

---

## 3. The pinned contribution

> **A pre-registered, mechanism-validated selective conformal regression framework that holds
> group-conditional coverage under documented distribution shift — and, on a task with a known
> recoverable/irreducible decomposition, demonstrates that the abstained-from region coincides with
> the irreducibly-bespoke signal while retained-set intervals stay calibrated and tight.**

- **Headline (the defensible hook):** the *known-floor evaluation* — selective UQ that is shown to
  abstain in the *right* place, not merely to abstain.
- **Portable:** method is domain-agnostic; finance is the demonstration. Reviewable by a pure-UQ
  committee with zero finance co-advisor.
- **Honest boundary:** group-conditional / approximate-and-validated coverage, never distribution-
  free exact conditional coverage (Foygel Barber 2021). No alpha, no trading claim.

---

## 4. Substrate — layered, public-primary

### Layer 1a — known-ceiling semi-synthetic (the controlled experiment)
A transparent, fully public allocator we build ourselves:
`w_i = base(cap-diversified) × momentum_tilt_i(public features) + λ · hidden_overlay_i + ε_i`,
where `hidden_overlay_i` is a seeded function the *learner is not given* (the analogue of a
proprietary score) and `ε_i` is irreducible noise. **We therefore know, per name, how recoverable
the weight is** (the recoverable fraction = variance from public-feature terms / total). This is the
only way to test G-MECHANISM. Knobs (frozen in pre-reg): overlay strength `λ`, noise scale,
cross-section size, number of periods, the shift generator. Tuned so the public-recoverable R²
lands near ~0.40 (a deliberately hard but non-trivial recoverable fraction), for face validity.

### Layer 1b — real public task (external validity; guards against a self-fulfilling synthetic)
Predict a genuinely-hard *public* target from the same PIT panel, with a real
**train-pre-drawdown → test-drawdown** split: 21-day cross-sectional forward returns — classic,
honest, real regime shift, near-zero mean-R² so the selective story is stark. 1b has no ground-truth
recoverability mask (so it can't carry G-MECHANISM), but it shows the method works on a real,
non-synthetic financial regression with real shift (G-PRIMARY / G-SHIFT / G-NULL).

### Layer 2 — real-data confirmation (private; NOT in this repo)
The *identical* pipeline is also run on a genuine real-data reconstruction task held in a **separate
private repository** (its own data, its own constraints). Only a one-line pointer appears here; no
private data, numbers, or provenance are copied into this repo. Layer-2 ports run **only when a
Layer-1a method ships** — the discipline that keeps the lineage honest.

---

## 5. Pre-registered gates (the single ship/reject)

Structures are fixed here; **numbers are frozen in `reports/p1_baselines_preregistration.md` after a
first baseline read and before the method runs.** All evaluated multi-seed (≥3) with bands; coverage
CIs via clustered/blocked bootstrap over the sector/regime cluster.

### G-PRIMARY — "the UQ layer adds something" (the one that decides ship/reject)
At a pre-specified target coverage (default **90%**), the selective conformal regressor must do
**both**, jointly:
- **(i) useful abstention** — at ≥ **X%** retention, retained-set predictive error (and/or mean
  interval width) is reduced by ≥ **Δ** versus *random* abstention at the same retention; **and**
- **(ii) conditional coverage held** — empirical coverage per group (sector / cap-tier /
  momentum-decile) stays within **±τ** of nominal on the retained set.

Beating the risk–coverage curve *while* holding conditional coverage — not one or the other — is the
bar. Failing either ⇒ **reject**.

### G-SHIFT — shift-robust conditional coverage
On the out-of-regime split (train calm → test drawdown), the shift-robust variant holds worst-group
coverage within **±τ_shift**, in a setting where **naive split-conformal demonstrably fails** (we
report the naive worst-group coverage gap first; the method must close it by ≥ a pre-committed margin).

### G-MECHANISM — abstention lands in the right place (1a only; the headline)
On the known-ceiling task: (a) confident (non-abstained) names vs. the truly-recoverable-names mask →
**AUC ≥ θ**; (b) per-name interval width vs. injected irreducible-noise magnitude → **Spearman ρ ≥ θ′**.
This is "validate the mechanism, not the P&L" ported to UQ.

### G-NULL — leakage guard
Re-run with **label-shuffled** targets: the selective advantage (G-PRIMARY (i)) must collapse to the
random-abstention baseline (within band). If it survives the shuffle, there is leakage — stop and fix.

---

## 6. Method stack & baselines

**Base learners.** Mean: GBM (LightGBM). Quantile: LightGBM quantile (or linear quantile) at the
lower/upper levels for CQR.

**Conformal layers** (each a thin module in `src/conformal.py`, composable):
1. **Naive split-conformal** (marginal coverage) — baseline #1.
2. **CQR** (Romano 2019) — heteroscedastic intervals; baseline #2 (marginal) and the body of the method.
3. **Mondrian / group-conditional** — separate calibration per group (sector / cap-tier /
   momentum-decile) for conditional coverage.
4. **Shift-robust** — weighted conformal with covariate-shift weights (Tibshirani 2019) and/or
   Adaptive Conformal Inference (Gibbs & Candès 2021) over the regime axis.
5. **Conditional-guarantee comparator** — Gibbs–Cherian–Candès (2023) as the SOTA reference point.

**Selective layer** (`src/selective.py`): abstain by interval width / normalized uncertainty;
sweep the abstention threshold → **risk–coverage curve**. Optionally cast as a two-stage
select-then-calibrate (the SCRC structure) but in **regression** — the open variant (see §11 P9).

**Baselines before models** (`src/eval.py` computes these first): naive split-conformal (marginal
only), unconditional CQR, **random abstention** (the G-PRIMARY comparator), and — on 1a only — an
**oracle abstention** that knows the noise (an upper bound on how good selection can get).

---

## 7. Phases & deliverables

| Phase | Deliverable | Gate(s) |
|---|---|---|
| **P0** setup + pre-reg | public panel built; `reports/p1_baselines_preregistration.md` with frozen numbers; 1a generator spec | — |
| **P1** baselines | naive + CQR marginal coverage on 1a & 1b; **demonstrate naive conditional failure** | establishes G-SHIFT target |
| **P2** method | group-conditional + shift-robust selective CQR; **risk–coverage curve**; conditional-coverage table (bands); 1a mechanism plots | **G-PRIMARY / G-SHIFT / G-MECHANISM / G-NULL** |
| **P2.1–P2.3** | multi-axis union → calibrated coupled shape → per-group online (the conditional-coverage lineage) | per-phase pre-reg |
| **P4** writeup | short public-safe note + README; committed figures | — |
| **P5** archival note | the single contribution sharpened; the five-method tradeoff surface + diagnosed design tension | — |
| **P6/P7** | alpha-space group-conditional adaptation; cross-axis level combination (resolve the P5 tension) | per-phase pre-reg |
| **P8–P11** | extension sprints (§11): simplex, risk control, localized comparator, e-value monitor | per-phase pre-reg |

**Interview-grade "done" (P2+P4):** a clean repo + note where the risk–coverage curve beats
baselines *while* conditional coverage holds under a real shift, and the abstention region is shown
(on 1a) to coincide with the recoverable signal. Defensible in 5 minutes, reproducible on a laptop.

---

## 8. Risks & mitigations
- **Closest-prior-art overlap ("When Alpha Breaks").** Lean on the differentiators (§2) — especially
  the **allocation target + known-floor mechanism check**, which that paper does not have. Position as
  method+evaluation, not trading deployment.
- **Conditional-coverage impossibility (Foygel Barber 2021).** State the guarantee honestly
  (group-conditional / validated), never distribution-free exact.
- **Self-fulfilling synthetic.** 1b (real public task) is the external-validity guard; G-NULL is the
  leakage guard.
- **Thin cross-sections.** Blocked/clustered resampling; report power; don't over-slice conditional
  groups beyond what the sample supports.
- **Scope creep toward a trading strategy.** Hard non-goal: no alpha, no backtest P&L. UQ method only.

---

## 9. Open design knobs (fixed in the P1 pre-reg, not before)
- G-PRIMARY numbers: target coverage (default 90%), retention X%, error/width margin Δ, tolerance τ.
- G-SHIFT: which drawdown window defines the shift; τ_shift; the naive-failure margin.
- G-MECHANISM: AUC θ, width-noise ρ θ′.
- 1a generator: overlay strength λ, noise scale, cross-section size, #periods, shift generator —
  tuned so public-recoverable R² ≈ 0.40 for face validity.

---

## 10. References (anchors)
- Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (conformal foundations).
- Romano, Patterson, Candès (2019) — **Conformalized Quantile Regression**.
- Tibshirani, Foygel Barber, Candès, Ramdas (2019) — **Conformal under covariate shift** (weighted).
- Gibbs & Candès (2021) — **Adaptive Conformal Inference under distribution shift**.
- Foygel Barber, Candès, Ramdas, Tibshirani (2021) — **Limits of distribution-free conditional inference** (impossibility).
- Gibbs, Cherian, Candès (2023) — **Conformal Prediction with Conditional Guarantees** (arXiv 2305.12616).
- Barber, Candès, Ramdas, Tibshirani (2023) — **Conformal Prediction Beyond Exchangeability**.
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction* / conformal risk control.
- El-Yaniv & Wiener — selective prediction / risk–coverage.
- "When Alpha Breaks" (2026, arXiv 2603.13252) — closest prior art (cross-sectional stock rankers).
- Selective Conformal Risk Control (2025, arXiv 2512.12844) — select-then-calibrate (classification).
- Conformal Prediction for Compositional Data (arXiv 2511.18141, 2026) — simplex-geometry conformal regions (→ P8).
- SpeedCP (arXiv 2509.24100, 2025); Shape-Adaptive (arXiv 2603.23374, 2026); Enhanced Localized CP (JASA 2026) — localized-conditional comparators (→ P10).
- E-Values Expand the Scope of Conformal Prediction (arXiv 2503.13050, 2025); "When Your Model Stops Working" (arXiv 2603.13156, 2026) — anytime-valid monitor (→ P11).

---

## 11. Extension sprints (P8–P11)

Ranked by leverage (P8 highest). Each keeps every binding constraint (public-safe by construction,
gate-first pre-registration, no alpha/P&L claim, honest group-conditional guarantee). A sprint's
ship/reject numbers are frozen in `reports/p<N>_*_preregistration.md` *after* a method-blind baseline
read and *before* the method runs — never at planning time; the descriptions below fix *structure* only.
All four were run; outcomes are in `reports/p{8,9,10,11}_*` and the archival note §13.

### P8 — Conformal on the simplex (the target is compositional) — *highest leverage*
The allocation is a **vector** (non-negative, sums to one — compositional), but a scalar-per-weight
pipeline treats each coordinate independently. Simplex-geometry conformal exists (arXiv 2511.18141)
but is **marginal-only, non-selective, no shift, no known-floor**. The intersection *simplex region ×
selective abstention × group-conditional-under-shift × known-floor mechanism* is open, and it makes
the contribution **more honest** (certify the allocation as one point on the simplex, not N unrelated
scalars). New score + region constructor (`src/simplex.py`), coverage redefined as *true weight
vector ∈ region*, and **G-MECHANISM redefined on the simplex** (does compositional difficulty track
the known irreducible σ?).

### P9 — Selective conformal *risk* control in regression (the "open variant" §6 named)
Upgrade the selective layer from coverage-only to a **monotone bounded risk** (a capped
miss-magnitude loss) with formal two-λ (select-then-calibrate) control — turning "we abstain and
coverage holds" into "we **control a risk** on the retained set with a guarantee." Mostly
`src/riskcontrol.py` (a second calibration pass + an online variant), reusing the risk–coverage
machinery in `src/eval.py`. New G-RISK; G-NULL still collapses the advantage.

### P10 — Localized / kernel-conditional comparator arm
Conditional coverage is Mondrian (discrete groups). Add a **continuous** approximate-conditional
comparator (`conformal.localized_cqr`; SpeedCP / Shape-Adaptive / Enhanced Localized CP) as a modern
SOTA reference beyond Gibbs–Cherian–Candès, and check whether it tightens intervals at equal
conditional coverage. Slots into the existing conditional-coverage table; no gate redefinition.

### P11 — E-value anytime-valid coverage-break monitor over the regime axis
An **anytime-valid detector for *when* conditional coverage breaks** in the drawdown window (e-value
conformal; testing-by-betting). A monitoring utility (`src/evalue_monitor.py`) over the walk-forward
stream; reporting-only, a diagnostic companion to the G-SHIFT story, not a change to the interval methods.

**Do not open:** training-conditional coverage bounds (arXiv 2405.16594 / 2602.16537) — theory-heavy,
not a laptop experiment; and any LLM/foundation-model abstention tie-in — off-scope.

# Conditional coverage under shift for *selective* regression:
## a calibrated-shape / online-correction tradeoff surface, validated against a known floor

**Niles Liu · 2026-06-30 · archival-grade methods note (P5)**

> **What this note is.** The P4 note (`p4_writeup.md`) is the 5-minute interview artifact: the gap,
> the headline gates, the mechanism figure. **This note is the archival version** — it sharpens the
> *single contribution* for a COPA / workshop write-up. The contribution is **not** any one estimator;
> it is the **lineage**: a family of head-to-head conditional-coverage methods that maps the
> calibrated-vs-online / union-vs-coupled / global-vs-group **and width-space-vs-alpha-space** tradeoff
> surface for *selective* CQR under distribution shift, run on a substrate with a **known** irreducible-
> noise floor. The surface first terminated in a single, well-identified design tension (§5.3), which a
> follow-up pair of pre-registered experiments (**P6/P7**, §12) then **resolved** — a mapped surface
> whose one named tension is diagnosed *and* closed, not tuned away. Four later optional sprints
> (**P8–P11**, §13) then upgraded the headline: joint certification of the allocation *on the simplex*
> with the known-floor mechanism ported to compositional geometry (P8), and online *risk* control to
> complement coverage control (P9). Public-safe by construction: no
> account data, no private numbers; the real-substrate confirmation (Layer 2) is referenced by pointer
> only. Every Layer-1 number reproduces bit-for-bit from a seeded driver.

---

## 1. Contribution, in one paragraph

We study **group-conditional coverage under distribution shift** for a *selective* conformalized
quantile regressor — a setting where two literatures meet but are rarely tested together: conditional
conformal (which holds coverage within groups) and adaptive conformal (which holds coverage through
time). We make two contributions. **(C1) A mechanism-validated abstention test.** On a task with a
*known* recoverable-vs-irreducible decomposition (the irreducible floor is injected, not assumed), we
show the selective layer abstains in the **provably right place** — the region it drops coincides with
the genuinely-irreducible signal (AUC 0.81, ρ 0.86, survives a volatility confound; `fig3`). No prior
conformal-for-finance work carries this ground truth. **(C2) A tradeoff surface, not a point claim.**
We place five conditional estimators on one axis — *static-calibrated*, *single-axis online*,
*multi-axis union*, *global-online-on-the-calibrated-shape*, and *group-online-on-the-calibrated-shape*
— and show that **moving along the natural "add structure" direction monotonically improves
worst-group coverage under shift** (0.433 → 0.599 → 0.651 on the adversarial synthetic; 0.706 → 0.740
→ 0.766 on real returns), at width that stays **under the union**. The endpoint is a **named design
tension** — a *width-space* online correction on a *single fitted* conditional shape over-covers the
base when severe-shift and base regimes interleave, where a *frozen alpha-space* per-group rule does
not — which we identify precisely rather than tune away.

The honest framing for a venue: **C1 is a clean PASS and is the headline; C2 is a mapped surface whose
one named design tension is diagnosed *and then resolved* (P6/P7, §12).** The scientific value of C2 is
the full arc — why each rung helps, where the family hits an impossibility-adjacent wall, and how the
one genuine tension (width-space over-coverage) is closed by an alpha-space reconciliation — not a SHIP
on the most adversarial synthetic (the worst-group floor there is a horizon wall common to every variant,
coupled or union).

## 2. The gap (related work)

Conformal prediction certifies that an interval has nominal **marginal** coverage; selective prediction
trades coverage for accuracy by **abstaining**. Neither, alone, can show whether a model abstains in
the *right place* — real tasks carry no ground truth for which points were ever recoverable, so
"calibrated and abstains on 30%" is unfalsifiable as a claim about *which* 30%.

- **Conditional coverage** is provably unattainable distribution-free (Foygel Barber et al. 2021); the
  achievable target is **group-conditional / Mondrian** (Vovk) or the **linear-class** relaxation of
  Gibbs, Cherian & Candès (2023) — one coupled threshold from a quantile regression of conformity
  scores on group features.
- **Shift-robust coverage** through time is the domain of **Adaptive Conformal Inference** (Gibbs &
  Candès 2021) and **weighted/covariate-shift conformal** (Tibshirani et al. 2019).
- **Conformal for finance** (closest prior art: "When Alpha Breaks", 2026) lives in the crowded
  **returns/ranking** setting, with no known-floor mechanism check.

The unoccupied intersection — and our contribution — is **the two coverage axes (group × time) on a
*selective* regressor, on an *allocation-reconstruction* target whose floor is known by construction**.
That last clause is what lets us validate the mechanism (C1); the first is what the method lineage
explores (C2).

## 3. Substrate (layered, public-primary)

- **Layer 1a — known-ceiling semi-synthetic (the controlled experiment).** A transparent public
  allocator `w = base(cap-diversified) × momentum_tilt(public features) + λ·hidden_overlay + ε`. The
  overlay is a seeded function the learner is **not** given (the analogue of a proprietary score); `ε`
  is irreducible noise. We therefore know, per name, the recoverable fraction and the irreducible
  magnitude — the only way to test the mechanism. Tuned to public-recoverable R² ≈ 0.40 (a
  deliberately hard but non-trivial recoverable fraction) for face validity against realistic
  allocation-reconstruction tasks. The shift is a deliberately **severe** train-calm → test-drawdown
  regime, interleaving base and shift sessions in the test stream.
- **Layer 1b — real public task (external validity).** 21-day cross-sectional forward returns from the
  same PIT panel, with a real calm → drawdown split. Guards against a self-fulfilling synthetic; the
  realised shift is **milder** than 1a's by design (static-conditional shift coverage 0.799 vs 0.564).
- **Layer 2 — real-data confirm (private).** The identical pipeline was also run on a genuine
  real-data reconstruction task held in a **separate private repository**, referenced by pointer only
  (§7). Layer-2 ports run **only when a Layer-1a method ships** — the discipline that kept this lineage
  honest. No private data, numbers, or provenance enter this repo.

## 4. Method family (all public-feature-only; every knob frozen before each variant ran)

Common stack: LightGBM mean + pinball quantiles → **CQR** heteroscedastic intervals (Romano 2019),
conformity score `E = max(qlo − y, y − qhi)` → conditional / online coverage layer (the variable below)
→ a **selective** layer that abstains on a per-name calibration-residual scale `u_i` (shrunk MAD of the
name's own calibration residuals — a PIT-legal proxy for its irreducible noise). The selective layer
`u_i` is **independent of the conformal interval**, so it is identical across all five variants; the
mechanism result (C1) and the leakage null are shared.

The five conditional/coverage layers — the tradeoff surface (C2):

| # | Variant (`conformal.*`) | What it adds | Phase |
|---|---|---|---|
| 1 | `aci_cqr` — **single-axis online** | ACI on one axis (momentum decile); per-group α-level update via empirical quantiles (**alpha-space**) | P2 |
| 2 | `aci_cqr_multi` — **multi-axis union** | independent ACI per axis → **union** of three intervals | P2.1 |
| 3 | `cqr_conditional` — **static calibrated shape** | one coupled threshold `t(x)=φ(x)ᵀβ` from a **pinball regression of `E` on multi-axis group indicators** (Gibbs–Cherian–Candès), additive in sector+cap+momentum | P2.2 |
| 4 | `cqr_conditional_aci` — **+ global online** | shape (3) carried online with a single **global** width-space offset `c` | P2.2 |
| 5 | `cqr_conditional_gaci` — **+ group online** | shape (3) carried online with `c` **plus** per-(axis,level) deviation offsets `d[k,ℓ]`; `Δ(x)=c+Σ_k d[k,ℓ_k(x)]` | P2.3 |

**The unification (the spine of the surface).** Variant 5 is the general object: `Δ≡0` recovers the
static calibrated shape (3); `d≡0` recovers the global hybrid (4); the per-group `d` is the shift fix
that (1)/(2) achieve via per-group empirical quantiles or a union, here applied additively to a *single
fitted* shape. `c` owns the average drift, `d[k,ℓ]` owns each group's *residual* (each nudged toward
the realised session marginal `err_t`), so the marginal is not triple-counted across the three axes.
`α=0.10`, `γ=0.05` shared, `scale = median|t|` — frozen across all five.

## 5. Results — the surface (Layer 1a, multi-seed {0,1,2}, target 90%)

Pre-registered gates were frozen **before each variant ran** (`reports/*_preregistration.md`); no knob
was tuned post-hoc, and the static `aci_cqr` / `aci_cqr_multi` / `cqr_conditional` / `cqr_conditional_aci`
were never re-fit when later variants were added.

| variant | base | shift | **worst-group \| shift** | mean width |
|---|---|---|---|---|
| single-axis ACI (P2) | 0.904 | 0.867 | 0.815 | 8.09 |
| multi-axis union (P2.1) | 0.934 | 0.915 | **0.881** | 9.08 |
| static calibrated shape (P2.2) | 0.898 | 0.564 | 0.433 | **6.18** |
| + global online (P2.2) | 0.965 | 0.702 | 0.599 | 8.38 |
| **+ group online (P2.3)** | 0.966 | 0.704 | **0.651** | 8.35 |

Three readings, each a finding:

1. **The width win.** The calibrated static shape is the **tightest** method on the board (6.18, −32%
   vs the union's 9.08) while marginally calibrated (0.898) — coupling one threshold across axes beats
   unioning three separately-calibrated intervals. *(P2.2 G-EFFICIENCY: clean PASS.)*
2. **The monotone shift signal (the contribution).** Worst-group|shift climbs **static 0.433 → global
   0.599 → group-online 0.651** — each rung adds exactly the structure pre-registered for (static→online,
   then global→group-conditional). The group term lifts worst-group **+0.052 over the global offset at
   equal width** (8.35 vs 8.38): per-group adaptation is strictly the better online correction. The
   width win survives going online — gACI stays **under the union** (8.35 < 9.08). *(P2.3 G-EFFICIENCY:
   PASS.)*
3. **The wall (the named design tension).** On this *deliberately severe* synthetic, the online family
   does **not** clear the union's worst-group (0.651 vs 0.881) and **over-covers the base** (0.966). The
   over-coverage is **not** a group-conditionality effect — the global hybrid already sits at 0.965; the
   group term adds +0.001. It is a **width-space-ACI × interleaved-regime** artifact: under severe shift
   sessions the width-space offset `c` inflates and relaxes only at rate γ·scale per session, so the
   more-numerous base sessions inherit the inflated width. The frozen **alpha-space** `aci_cqr` does
   **not** show this (base 0.904) — its per-group α-level update with empirical quantiles is
   self-correcting in a way an additive width-space offset on a *single fitted shape* is not.

## 6. Results — external validity (Layer 1b, real 21d forward return)

On the real substrate the unification behaves **as designed**, because the realised shift is milder and
the width-space offset never over-inflates:

| variant | base | shift | worst-group \| shift | width |
|---|---|---|---|---|
| single-axis ACI | 0.898 | 0.898 | 0.858 | 0.275 |
| multi-axis union | 0.933 | 0.934 | 0.896 | 0.306 |
| static calibrated shape | 0.870 | 0.799 | 0.706 | 0.247 |
| + global online | 0.906 | 0.829 | 0.740 | 0.269 |
| **+ group online (P2.3)** | **0.910** | 0.832 | **0.766** | **0.268** |

Group-online is the **best conditional method on the real board**: base near-nominal (0.910, within
±τ), worst-group **0.766** (clears the 0.704 floor, beats the global hybrid's 0.740), width 0.268 <
union 0.306. The 1a/1b contrast is itself the lesson — **the base over-coverage and floor-miss on 1a
are products of the synthetic's deliberately severe shift, not of the method**; on a realistically-mild
shift the family delivers the calibrated-and-tighter-than-union-and-group-robust combination no single
prior variant achieved at once.

## 7. Layer-2 real-data confirm — private, by pointer

The interview-grade pipeline (P2 knobs frozen) was also run on a genuine real-data reconstruction
task held in a **separate private repository**. Qualitatively, the per-name selective ranking and the
online marginal-coverage restoration transferred, and the conditional-coverage guarantee is confirmed
to be **group-conditional and substrate-dependent** rather than universal; a multi-axis union is the
diagnosed remedy for the un-conditioned-axis miss. The later coupled-shape variants (P2.2/P2.3) stayed
gated off the private confirm because neither cleanly shipped on the adversarial 1a — the discipline,
not an omission. The takeaway the lineage carries: **core uncertainty-ranking is portable;
conditional-coverage guarantees are group-conditional and substrate-dependent.** No private data,
numbers, or provenance are reproduced here by design; the mechanism *validation* lives on Layer 1a,
where ground truth exists.

## 8. The open question — posed here, then answered (P6/P7, §12)

The surface terminated cleanly on **one** well-identified tension, not a knob: a *width-space* additive
offset on a *single fitted* conditional shape over-covers the base under interleaved severe-shift/base
regimes; a *frozen alpha-space* per-group rule does not, but it seemed to require a per-group empirical-
quantile object — exactly the Mondrian per-group sparsity the coupled shape was built to avoid. The
principled open move we logged was **alpha-space group-conditional adaptation reconciled with the
single-regression shape** — adapting the *quantile level* per group rather than adding width, without
re-introducing per-group sparsity — as a **fresh pre-registered experiment**, not a retune.

**That experiment was run (P6), and its own follow-up (P7); both are archived in §12.** The short of it:
the reconciliation works — an alpha-space update read off a *fitted level family* `t_τ(x)` (one additive
pinball regression per level, all on the full calibration set → no sparsity) **removes the base
over-coverage at the lineage's lowest width**, and a most-conservative cross-axis combination fixes an
additive-cancellation artifact it surfaced. What it does **not** do is clear the worst-group floor on the
deliberately-severe synthetic — but that floor is a **horizon wall common to the whole family** (even the
always-over-covering union reaches only 0.881), not the named tension. The tension is closed; the residual
is a property of the adversarial substrate. Details, tables, verdicts: §12.

## 9. Honest boundaries

- **No distribution-free exact conditional coverage** — forbidden (Foygel Barber et al. 2021). The
  guarantee is **group-conditional / approximate-and-validated**, stated as such. The un-conditioned-axis
  miss (sector, on the retained set) is the visible edge of that boundary; on the retained subset it is a
  **selective × conditional interaction** (abstention keeps the easier low-`u` names within a sector, so
  the retained subset over-covers) — a property of the selective layer, logged not tuned.
- **The width-space over-coverage (§5.3)** is a real limitation of the coupled-shape online family under
  severe interleaved shift, reported in full, not hidden — it is why P2.3 is QUALIFIED, not SHIP, on 1a.
- **No alpha, no backtested return, no trading claim.** This is a UQ method demonstrated on an
  allocation-reconstruction regression; the mechanism is validated, not a P&L.

## 10. Reproduce

```
pip install -r requirements.txt
python src/run_p1.py     # baselines: marginal coverage holds; conditional fails under shift
python src/run_p2.py     # interview-grade gates (G-PRIMARY/G-SHIFT/G-MECHANISM/G-NULL), 1a + 1b
python src/run_p21.py    # multi-axis union
python src/run_p22.py    # calibrated coupled shape (static + global hybrid)
python src/run_p23.py    # group-online on the coupled shape — the full five-method surface
python src/figures.py    # the three headline figures
```

Each driver pairs with a frozen `reports/<phase>_preregistration.md` and a `reports/<phase>_results.md`;
every reported number reproduces bit-for-bit from the seeded run.

## 11. References (anchors)

Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (conformal foundations; Mondrian).
CQR — Romano, Patterson, Candès (2019). Weighted conformal under covariate shift — Tibshirani, Foygel
Barber, Candès, Ramdas (2019). Adaptive Conformal Inference — Gibbs & Candès (2021).
Conditional-inference impossibility — Foygel Barber, Candès, Ramdas, Tibshirani (2021). Conditional
(linear-class) conformal — Gibbs, Cherian & Candès (2023). Closest prior art — "When Alpha Breaks"
(2026, arXiv 2603.13252).

---

## 12. Update — the design tension resolved (P6/P7, 2026-06-30)

The §5.3 endpoint named **one** tension: the P2.3 *width-space* online correction on the single fitted
conditional shape over-covers the base (1a base 0.966) under interleaved severe-shift/base regimes,
where a *frozen alpha-space* rule does not. §8 logged the reconciliation as a fresh pre-registered
experiment. It was run (P6) with a follow-up (P7); both **QUALIFIED**, and together they **close the
tension**. Pre-reg+results: `reports/p6_alphaspace_*`, `reports/p7_levelcomb_*`.

**The new object.** A **fitted level family** `t_τ(x)` — one additive pinball regression of the CQR
conformity scores on the multi-axis group indicators per quantile level `τ∈{0.50,…,0.99}`, each on the
**full** calibration set (monotone-rearranged, linear-interpolated in level). It is the conditional
analogue of `aci_cqr`'s per-group empirical quantile *function*, **with no per-group sparsity** — which
is exactly what makes the alpha-space update runnable against the *single* coupled shape. Online, the
ACI update (the frozen `aci_cqr` form/rate, γ=0.05) adapts the quantile **level** per group instead of
adding width. This extends the surface to a full **(width-space vs alpha-space) × (global vs group)** 2×2
on one shape, plus a cross-axis-combination axis (P7).

**The seven-variant surface (1a, seeds {0,1,2}, target 90%):**
| variant | correction | base | shift | worst-grp\|shift | width |
|---|---|---|---|---|---|
| ACI multi-axis union (P2.1) | — | 0.934 | 0.915 | **0.881** | 9.076 |
| static calibrated shape (P2.2) | — | 0.898 | 0.564 | 0.433 | 6.178 |
| + global online (P2.2) | width | 0.965 | 0.702 | 0.599 | 8.376 |
| + group online (P2.3) | width | 0.966 | 0.704 | 0.651 | 8.348 |
| + global online (P6) | **alpha** | 0.904 | 0.797 | 0.683 | 7.605 |
| + group online (P6) | **alpha** | **0.900** | 0.679 | 0.655 | 6.715 |
| + group, cross-axis max (P7) | **alpha** | 0.905 | 0.770 | **0.687** | 7.401 |

**What P6/P7 establish:**
1. **The tension is a width-space artifact, and alpha-space removes it (P6).** Swapping only the
   correction *space* on the same fitted shape moves base **0.966 → 0.900** (dead nominal) at the
   lineage's **lowest width** (6.715, −26% vs union). Both alpha variants sit at nominal; both width
   variants over-cover. Mechanism confirmed (|base−nominal| alpha 0.000 vs width 0.066; realised α_eff
   interior, median ≈ nominal). **G-BASE-CALIB + G-EFFICIENCY pass.**
2. **A cross-axis cancellation, then fixed (P7).** P6's *additive* cross-axis level combination let a
   point easy-on-two-axes-hard-on-one lose its widening to cancellation (its worst group drifted to a
   spurious mom_decile). Combining axes by the **most-conservative** one (`min`, union-in-level-space on
   the single shape) restores group > global on worst-group and relocates the worst group back to
   sector=tech. **G-GROUP-HELPS passes** (small on 1a, +0.004; **material on real 1b, +0.037**).
3. **The residual is a horizon wall, not the tension.** No coupled-shape variant in either space clears
   the 0.704 worst-group floor on the *deliberately-severe* synthetic — the same wall the union (0.881)
   and single-axis ACI (0.815) hit — so P6/P7 are **QUALIFIED, Layer-2 gated off**. On the realistically-
   mild **real** shift (1b) the reconciled family is the **best coupled method to date**: `cond+gqACI-max`
   base 0.899 / shift 0.896 (both near-nominal) / worst-group **0.860** (approaching the union's 0.896) /
   width 0.281 < union 0.306.

**Venue takeaway.** The archival story is now stronger than "a diagnosed open tension": it is a mapped
tradeoff surface whose single genuine design tension is **diagnosed and then closed** by a principled
alpha-space reconciliation, with the only remaining gap identified as a substrate-horizon property shared
by every method. C1 (mechanism-validated abstention on the known floor) remains the headline; C2 is now a
*complete* surface. No frozen knob was tuned post-hoc anywhere in the lineage.

---

## 13. Update — extension sprints (P8–P11, 2026-07-01)

The width×alpha × global×group surface (§12) closed the method ladder. Four **optional** sprints then ran
end-to-end — each gate-first (method-blind baseline read → frozen pre-reg → method → 4-lens verify →
multi-seed), **no frozen P1–P7 knob touched** — to test whether the artifact lifts from *defensible demo*
to *publishable contribution*. Two upgrade the headline (P8, P9); two are supporting arms (P10, P11).
Pre-reg+results: `reports/p{8,9,10,11}_*`.

**P8 — conformal on the simplex (QUALIFIED; the highest-leverage upgrade to C1).** The allocation is a
*composition*, so the honest object to certify is the **whole vector on the simplex**, not N unrelated
scalar intervals. A joint Aitchison-metric region (with a bounded-TV companion and an online ACI variant)
**certifies the true allocation vector at base 0.923** [0.892, 0.964], where the independent-scalar
**product region covers it 0.0000** of the time — 93 marginal-90% intervals never all contain the truth
at once. Crucially the **known-floor mechanism ports to compositional geometry**: per-name compositional
difficulty ↔ injected irreducible σ at **Spearman 0.967** (AUC 0.861, survives the `vol_63` partial at
0.967). That is C1 — "abstain in the right place" — restated for the *allocation object itself*.
**G-COMPO-SHIFT misses**: under the deliberately-severe synthetic shift the joint region collapses
(Aitchison 0.000, bounded-TV degrades more gracefully to 0.230), because the whole-vector shift pushes
every session's compositional distance past the entire calibration support and a quantile-reading online
rule is capped by that support — a whole-vector guarantee is intrinsically more fragile than a
per-coordinate one. **QUALIFIED, Layer-2 gated off.** (A grid-refined HDR-over-simplex is infeasible at
~90 names; the scalar-score metric ball is the stated laptop realization.) 1b is forward returns, not a
composition, so P8 external validity is Layer 2 by pointer.

**P9 — selective conformal *risk* control in regression (SHIP on 1a; a new axis, not a new rung).** Every
variant above controls *coverage* (a 0/1 miss rate); P9 controls a **proper risk** — capped
miss-*magnitude* — via select-then-calibrate CRC (the Angelopoulos–Bates bound), the "open variant" SPEC
§6 named, published for classification only and here built for regression. By the pre-committed rule
(G-RISK-BASE ∧ G-RISK-SHIFT ∧ G-NULL) it **ships on 1a**, on **one** substantive leg stated without
overclaim: the **online** `scrc_aci` holds retained-pool miss-magnitude risk under the drawdown at
**0.080 vs the static 0.184** (more than halved; retained shift coverage 0.471 → 0.738), with the
risk-controlled radius genuinely adapting (λ path 0 → 3.5 under shift → 0 after), where the static radius
fails. Reported honestly as the discipline requires: **G-RISK-BASE passes trivially** (at α_risk=0.05 the
batch CRC picks λ=0 — raw CQR quantiles already meet the calm target) and **G-EFFICIENCY is a NULL** (no
radius to allocate, so "select-then-calibrate is tighter" is *not* demonstrated on this task), neither
retuned away. G-NULL clean (0.011 → 0.000). The lesson mirrors the whole project's: selection helps a
little, the shift-robust interval machinery carries the result.

**P10 — localized / kernel-conditional comparator (both gates PASS, marginal; strengthens C2).** A modern
continuous-conditional arm (kernel-reweighted split CQR; Guan 2023 / Hore–Barber 2024) is **valid** (base
0.898) but at comparable base worst-group coverage is only **−0.7% width vs Mondrian** (6.191 vs 6.235;
1b on par). Read: the discrete Mondrian groups already capture nearly all the recoverable conditional
structure, so the surface's coupled-shape methods were not leaving a continuous-conditioning win on the
table. A comparator that *confirms* the design choice, not a new ship.

**P11 — e-value coverage-break monitor (QUALIFIED diagnostic; a G-SHIFT companion).** A testing-by-betting
(Ville) monitor detects naive split-conformal's coverage break **early, in-shift, 3/3 seeds** (median
session ~17), where group-ACI's mild residual under-coverage only trips it late (2/3, median ~144) — a
useful anytime-valid break detector to pair with the shift results. Kept scoped: **G-NOFALSE** shows a
1/3 base false alarm (the frozen aGRAPA λ_cap=0.5 is a touch aggressive for exact anytime-validity; not
retuned). Not promoted to a headline.

**Updated venue takeaway.** The poster/abstract now has a **stronger, more honest headline** than the P5
framing: C1 is no longer only "abstain in the right place on scalar intervals" — with **P8** it becomes
*certify the allocation as one point on the simplex, where independent scalars certify nothing (0%), and
the known-floor mechanism holds on the compositional object* (ρ=0.967). **P9** adds an orthogonal axis —
*online risk (miss-magnitude) control under shift*, not just coverage — a clean "risk control ≠ coverage
control" story. P10/P11 are supporting arms (a continuous-conditional comparator that validates the
Mondrian choice; an anytime-valid break monitor). **None is committed to a venue**; the venue gate was
lifted only to *explore*. For a COPA 2-page abstract, the tightest telling remains **C1 (now upgraded via
P8) as the headline**, C2 (§5/§12) as the mapped-and-closed surface, and P9 as a one-line "the same
machinery extends from coverage to risk." No frozen knob was tuned post-hoc anywhere in the lineage.

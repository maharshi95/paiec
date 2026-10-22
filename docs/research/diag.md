# diag: error decomposition and opportunity sizing (research loop, round 1)

Scout report for the orchestrator. It answers *where the champion's loss is and how much each direction can win*. It
proposes no new champion.

**Protocol.**
- Champion (`configs/research/champion.yaml`) on the dev protocol, run in-process with seed 0 only. That is 5 folds,
  30 test benchmarks and 344 pairs.
- Reproduced champion: fold-macro ALC **0.1844**, pair-macro 0.1870, matching the 1-seed control. B0 .240, B1 .201,
  B3 .186, B7 .178, B15 .170, B31 .160.
- Δ values are paired over pairs (95% CI) unless marked fold-macro.
- Code is in `analysis/diag/`:
  - `fold_job.py`: oracle replays and static ceilings;
  - `bench_job.py`: the wider benchmark set;
  - `subjects_job.py`: subject transfer;
  - `order_job.py`: replay mechanics;
  - `analyze.py` and `extra.py`: tables.
- Full tables are in `analysis/diag/out/summary.md` and `out/extra.md`. Jobs run with `analysis/diag/job.slurm`.

**Two kinds of oracle.**
- *In-sample* oracles are fitted on all of a test benchmark's responses, **including the scored eval responses**, as
  the brief asked. On benchmarks with few subjects per item this memorises the answer: an in-sample K=3 factorisation
  reaches 0.051. The in-sample numbers are therefore not achievable bounds.
- *Cross-fitted* oracles (`cf`) are the honest bounds. They are fitted on everything except the eval pools of the
  evaluated pairs: all other subjects plus those pairs' full acquisition pools. This equals "every acquisition label
  of every pair, plus the non-evaluated subjects", i.e. B∞.
- Oracle parameters are plugged into the champion's own test-time adaptation (`OracleIRT` in `fold_job.py`): fixed
  values, zero variance, and the absolute-target parametrisation β' = β + mean(ŷ), d' = d_c + mean(ŷ), δ = θ − β.
  Everything not oracled is still learned from `labeled` exactly as in the champion.

## 1. Oracle decomposition

Fold-macro ALC. The gap to close is champion − B∞ Rasch (cf, all three oracled) = 0.1844 − 0.1044 = **0.080**.

| variant | ALC | Δ (pairs, ±CI) | % of gap | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|---|---|
| champion | 0.1844 | 0 | 0 | .240 | .201 | .186 | .178 | .170 | .160 |
| oracle β (cf) | 0.1755 | −0.0082 ± 0.0021 | 11% | **.193** | .190 | .183 | .176 | .168 | .160 |
| oracle δ, pair ability (cf) | 0.1707 | −0.0141 ± 0.0030 | 17% | .215 | .179 | .170 | .167 | .163 | .156 |
| **oracle d, item difficulty (cf)** | **0.1284** | **−0.0573 ± 0.0057** | **70%** | .186 | .136 | .122 | .122 | .119 | .113 |
| oracle β+δ (cf) | 0.1625 | −0.0216 ± 0.0029 | 27% | .170 | .169 | .167 | .165 | .162 | .156 |
| oracle β+d (cf) | 0.1185 | −0.0663 ± 0.0052 | 82% | .135 | .131 | .121 | .114 | .114 | .111 |
| oracle δ+d (cf) | 0.1147 | −0.0710 ± 0.0052 | 87% | .159 | .112 | .115 | .113 | .107 | .107 |
| all three (cf) = B∞ Rasch | 0.1044 | −0.0810 ± 0.0050 | 100% | .106 | | | | | |
| per-pair base rate (cf) | 0.1637 | −0.0208 ± 0.0051 | | .166 | | | | | |
| B∞ 2PL (cf) | 0.1033 | −0.0823 | +1.3% vs Rasch | | | | | | |
| B∞ 4PL, per-benchmark asymptotes (cf) | 0.1046 | −0.0807 | −0.3% | | | | | | |
| B∞ Rasch + rank-2 (cf, λ=4) | 0.1022 | −0.0832 | +2.6% | | | | | | |
| B∞ Rasch + rank-3 (cf, λ=4) | 0.1023 | −0.0830 | +2.6% | | | | | | |
| B∞ Rasch + rank-2/3 (cf, λ=1) | 0.1103 / 0.1144 | | worse (overfits) | | | | | | |
| *in-sample* oracle β / δ / d / all | 0.1755 / 0.1709 / 0.1119 / 0.0855 | | | | | | | | |
| *in-sample* 2PL / 4PL / rank-2 / rank-3 | 0.0806 / 0.0863 / 0.0596 / 0.0512 | | memorisation | | | | | | |

Readings:
- **Item difficulty is the headroom: 70% of the gap and 0.057 ALC.** It applies at every budget (B1 .201 → .136,
  B31 .160 → .113).
  - With β and δ both oracled but d unknown, ALC is still 0.1625. With d alone known it is 0.128.
  - The champion learns d from global labels only on items that other pairs happen to label. Under uniform selection,
    the probability that an eval item has ≥1 label from another evaluated pair is median **6% at B7 and 25% at B31**
    (`extra.md`).
  - The per-benchmark o_d gain does not correlate with that coverage (r = −0.04). Item knowledge must therefore come
    from amortisation (features, duplicates, kv groups) inside the benchmark, not from direct label sharing.
- **The benchmark offset β is worth 0.008, 60% of it at B0.** B0 Brier falls .240 → .193. From B3 on, β is already
  learned (Δ ≤ 0.003 per budget).
- **Pair ability δ is worth 0.014**, spread over budgets. It is the main residual at B15/B31 once d is known.
- **Non-Rasch structure is worth at most about 0.001–0.002 ALC, even with every acquisition label (B∞).**
  - 2PL: −0.001. Per-benchmark 4PL asymptotes: +0.0003, since fitted asymptotes are below 0.02 on 23/30 benchmarks. Low rank
    K=2–3: −0.002 with strong regularisation, and worse without it.
  - At test time a pair has ≤31 labels, so the realisable gain is about 0.
  - Deprioritise discrimination, 4PL and low-rank interactions (the hier and subj sub-items).

## 2. Calibration

Pooled pair-weighted reliability, in `summary.md` §2.
- The champion **under-predicts success at every budget.** Mean p vs mean y:
  - B0: .507 vs .621;
  - B1: .561;
  - B7: .590;
  - B31: .599 vs .621.
- The B0 bias is the offset problem (per-benchmark bias spans −0.39 to +0.29). It is negative on 22/30 benchmarks,
  which fits the dev eval subjects being ref6-seen frontier models.
- The model is **slightly under-confident at B1–B31.** The fitted temperature T ≈ 0.82–0.85 (T < 1 means sharpen).
  At B0, T = 1.01, so B0 is neither over- nor under-confident in shape.

| recalibration (applied to the champion's predictions) | ALC (pairs) | Δ |
|---|---|---|
| none | 0.1870 | |
| global temperature, in-sample (T = 0.852) | 0.1863 | −0.0007 |
| per-budget temperature, in-sample | 0.1862 | −0.0008 |
| per-budget Platt, in-sample | 0.1849 | −0.0021 |
| per-budget isotonic, in-sample (upper bound) | 0.1823 | **−0.0047** |
| per-budget Platt, leave-one-fold-out (realistic) | 0.1853 | **−0.0017** |
| per-budget isotonic, leave-one-fold-out | 0.1863 | −0.0007 |

Post-hoc calibration is worth about **0.002 realistically and ≤0.005 at best**. Most of it is the mean shift at B0–B1,
which may not transfer to the hosted subject mix.

**Prior widths.** On the test benchmarks:
- In-sample (cf) item spread: sd(d) median 1.87, IQR 1.72–2.25. The prior is sd 2.1–2.4.
- Pair-ability spread: sd(δ) median 1.20. The prior is σ_u ≈ 1.7–1.8.
- Both cf sds are *inflated* by estimation noise, so the priors are wider than the test benchmarks' true spreads. The
  per-fold σ_β also varies between 1.2 and 2.4.

This supports test-time per-benchmark variance estimation (hier). It conflicts with the old result that shrinking the
BOW variance hurt, so test it per benchmark rather than as a global scale.

## 3. Where the loss is

Per-benchmark table: `summary.md` §3 and `extra.md`.

| group (benchmark-macro) | n | champion | B∞ Rasch | trial floor | champion − B∞ |
|---|---|---|---|---|---|
| MCQ-like (≥50% items, regex) | 6 | 0.175 | 0.113 | 0.000 | 0.062 |
| open / other | 24 | 0.197 | 0.109 | 0.005 | 0.087 |
| extreme base rate (<.2 or >.8) | 6 | 0.116 | 0.063 | 0.011 | 0.053 |
| mid base rate | 24 | 0.211 | 0.122 | 0.002 | 0.089 |
| repeated trials (>1.2 per cell) | 8 | 0.181 | 0.114 | 0.015 | 0.067 |
| single trial | 22 | 0.196 | 0.109 | 0.000 | 0.088 |
| <300 / 300–2,000 / ≥2,000 items | 4 / 12 / 14 | .174 / .197 / .194 | .109 / .100 / .120 | | .064 / .097 / .074 |

- **Irreducible trial noise is negligible.** The within-cell variance of repeated trials on eval items averages 0.004
  Brier. It exceeds 0.001 on 7/30 benchmarks, with a maximum of 0.040 (engdesign) and 0.031 (agentdojo). The B∞ Rasch
  level of 0.104 is therefore mostly item and subject uncertainty that is unresolvable from ≤ all labels, not
  coin-flip noise.
- **The loss sits on mid-base-rate, open-ended benchmarks with 300–2,000 items.**
  - Largest gaps to B∞: ikp, icpc2_code_selector, helm_airbench, mathvista, editbench and ai2d_test, at 0.08–0.15
    each, nearly all of it item difficulty (`extra.md`).
  - Extreme-base-rate benchmarks are already near their ceiling.
- **The 14/45 benchmarks where IRT lost to 0.5 under the old 45-benchmark dev protocol** (`runs/dev_irt`; `summary.md`
  §3b):
  - Mostly benchmarks with **no learnable signal**. With every acquisition label, a cross-fitted Rasch still scores:
    - devbench .243, reliancescope .241, ceval .236, bedd_basalt .230, tumlu .225;
    - vl_rewardbench .212, mochi .207, image2struct .205, prism .193, visit_bench .188, worldcentralbanks .181.

    The pair base-rate sd is ≤0.15, so subjects and items barely differ. Many are judge, preference or reward tasks.
  - IRT loses there because its global prior variances (σ_d ≈ 2.3, σ_u ≈ 1.8) produce spread-out predictions where
    the truth is ≈ 0.5.
  - The other three (atmossci .147, mmoral .140, mmlupro .098 B∞) do have signal and were offset or calibration
    failures.
  - **Under the current protocol and champion:** only 2/30 test benchmarks lose to 0.5 (dataclawbench 0.259,
    worldcentralbanks 0.251), but 19% of pairs (67/344) do. On the overlap, atmossci went 0.254 → 0.235 and mmlupro
    0.257 → 0.174.
  - Fix direction: learn the per-benchmark item and subject spread at test time, so a benchmark whose labels show no
    item or subject structure collapses toward its base rate (hier).

## 4. Subject-transfer potential

From `subjects_job.py`: per-benchmark Rasch on all 207 universe benchmarks, keyed by `subject_identity`.
- **Identity coverage is high on this protocol.** The dev eval pairs come from 65 identities.
  - Training benchmarks per identity, over pairs: median 17; ≥1: 100%, ≥5: 90%, ≥20: 46%, ≥50: 10%.
  - Over distinct identities: median 7; ≥5: 72%, ≥20: 17%.
  - 69% of eval pairs also have their identity on another *test* benchmark in the same fold (mean 1.35).
  - This differs from the old protocol's "1.01–1.15 benchmarks per identity", because seen-dev restricts evaluation to
    ref6-seen identities. Hosted subjects may not be like this.
- **Pair abilities correlate only moderately across benchmarks.** Over 230 benchmark pairs with ≥10 shared
  identities, Pearson r has median 0.39 (IQR 0.16–0.58).
  - Variance decomposition of δ over identities on ≥2 benchmarks: total 2.39, within-identity 1.95. **Identity
    explains about 19%.**
  - The champion's fitted var_pair/var_u ≈ 2.9/3.3 says the same (≈12%).
- **Low-rank completion adds nothing over the identity mean.** Setup: 136 identities on ≥3 benchmarks, 23 held-out
  benchmarks with ≥15 of them.
  - The identity mean on other benchmarks predicts held-out abilities with R² median 0.20 (mean 0.09).
  - Revealing n = 3/5/10 subjects of the held-out benchmark and regressing on K = 0–3 factors gives R² −0.3 / −0.1 /
    +0.1 for every K.
  - With noisy abilities from k labels each (noise var 4/k), the subject's own labels alone give R² 0.30 / 0.57 /
    0.76 / 0.86 / 0.92 at k = 1/3/7/15/31. Combining them with the low-rank or identity prediction does not beat own
    labels plus a well-set prior.
- **Expected gain:** the δ oracle bounds every subject-side gain at 0.014. The history prior already carries the
  identity mean, and K > 0 adds nothing, so test-time cross-benchmark sharing or low rank is worth ≲ 0.002. The
  earlier `subject_sharing` and `mirt` results (Δ ≈ 0) agree. Identity resolution matters only insofar as it raises
  that 19%.

## 5. Within-benchmark item transfer

Setup: a random half of a test benchmark's items with known (in-sample) centred d trains a ridge (RidgeCV) on
`ItemFeaturizer` features, which then predicts the other half. Five repetitions; 30 benchmarks.

| features | R² mean / median | Spearman mean / median |
|---|---|---|
| text + length + kv (champion's set) | **0.32 / 0.39** | **0.52 / 0.55** |
| text only | 0.26 / 0.20 | 0.45 / 0.43 |
| kv only | 0.15 / 0.11 | 0.27 / 0.32 |
| length only | 0.06 / 0.05 | 0.22 / 0.22 |
| + interactors | 0.32 / 0.36 | 0.52 / 0.57 |
| champion's cross-benchmark prior ŷ on the same benchmarks | | 0.06 / 0.03 |

- **Within a benchmark, the existing cheap features rank difficulty with ρ ≈ 0.55, versus 0.06 across benchmarks.**
  This is a lower bound, since the in-sample d targets are noisy.
- Strong cases: rakuda .81, ikp .74, workbench .74, editbench .72, arcagi .69. Weak cases: benger .18,
  medhelm_head_qa .22.
- 15% of items share identical content with another item. phyblock is 100% duplicate text, distinguished only by kv,
  where kv alone gives R² 0.50.
- Taken with §1 (item difficulty = 0.057 headroom, little direct label coverage), this is the **most promising
  direction**. Fit a per-benchmark ridge or GP from item features to the benchmark's own revealed labels at test time
  (numpy only), and tie duplicate content.
- **Sizing (estimate, not measured).** Exact d on half the items gives R² ≈ 0.3–0.4. At test time the targets are
  ~S·n binary labels, about 140 at B7 and 600 at B31 with 20 pairs. A realistic capture is perhaps 10–25% of the
  o_d gain, i.e. **0.005–0.015 ALC**, concentrated at B7–B31.

## 6. Replay realism

From `order_job.py`, varying only the acquisition mechanics (Δ vs default, pairs):

| variant | ALC | Δ |
|---|---|---|
| default: pairs advanced in sorted (subject, benchmark) order, evidence live | 0.1870 | |
| reversed pair order | 0.1850 | −0.0020 ± 0.0012 |
| order reshuffled each round | 0.1859 | −0.0011 ± 0.0010 |
| round-robin, one label at a time across pairs | 0.1864 | −0.0006 ± 0.0005 |
| frozen: acquisition sees only evidence from the previous checkpoint | 0.1862 | −0.0008 ± 0.0011 |

- **Pair order within a round moves ALC by up to 0.002.** The hosted coordinator's order is unknown: the client just
  receives `acquire` events, with the live global `labeled` attached (`tools/streaming_ingestion.py`). Acquisition
  variants with gains < 0.002 therefore need confirming under ≥2 orders, e.g. default and reversed.
- **Live evidence during acquisition is not needed** (frozen ≈ default). Whether the coordinator updates `labeled`
  between acquisitions does not change conclusions.
- Other gaps that change magnitudes:
  - **Subjects per test benchmark.** Dev has a median of 12.5 evaluated pairs per benchmark (≤20; 9 of 30 benchmarks have 1–6). The
    item-label coverage and the value of global evidence scale with this. If hosted benchmarks really have ~20
    subjects, the global-evidence and item-learning gains are larger than dev shows. Benchmarks with 1–3 pairs in dev
    (mathvista, kmmlu, biggen, mmlupro, worldcentralbanks) understate it.
  - **Eval subjects are ref6-seen identities.** They are strong models with rich history, and this drives the B0
    under-prediction (§2) and the high identity coverage (§4). Hosted subjects may be weaker or new, so do not tune a
    global B0 shift on dev.
  - Eval items are capped at 100 per pair, while the platform scores all of them. This is neutral in expectation for
    pair-macro Brier.
  - One revealed trial per acquisition, with all trials scored, matches the platform.

## Ranked opportunities (expected dev-ALC headroom; ceiling → realistic estimate)

1. **Within-benchmark item amortisation and tying (item).**
   - Ceiling 0.057 (cf oracle d, 70% of the gap). Realistic **0.005–0.015**.
   - Test-time ridge from item features to the benchmark's own labels (ρ ≈ 0.55 within vs 0.06 across), plus
     duplicate-content and kv tying.
2. **Test-time per-benchmark variances / shrink no-signal benchmarks (hier).**
   - Fixes the "loses to 0.5" pairs (19% of pairs); priors are wider than test-benchmark spreads.
   - Realistic **0.002–0.005**, overlapping with calibration.
3. **Calibration (meta).**
   - Leave-fold-out per-budget Platt −0.0017; ceiling −0.0047 (in-sample isotonic).
   - Under-confident at B1–B31 (T ≈ 0.85), under-predicting at B0.
4. **Benchmark offset at B0.**
   - Ceiling 0.008, nearly all at B0/B1. No signal is known to predict it (text fails), so realistic ≲ 0.002.
5. **Subject transfer (subj).**
   - Ceiling 0.014 (oracle δ). Identity explains ~19% of pair-ability variance and low rank adds nothing over the
     identity mean, already in the history prior. Realistic **≲ 0.002**.
6. **Acquisition (acq).** Order and mechanics noise is ±0.002. Any acq gain must clear it under ≥2 replay orders.
   Concentrating labels so that more eval items get labelled is the one structural lever, since coverage is only 25%
   at B31.
7. **Non-Rasch structure (2PL, 4PL asymptotes, low-rank).** ≤0.002 even at B∞, about 0 at ≤31 labels. Deprioritise.

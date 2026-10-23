# auto: what stacks, what does not, and the best working model (2026-10-23)

Branch `exp/auto` (from `stack/r1`). Dev protocol `configs/research/dev.yaml`: 5 grouped folds, seeds 0-2, 1,032 pair-seeds.
Controls: `R_champion_dev` (main checkout, ALC 0.1830) and `item__A3_mix_eb10` (`paiec-wt/item/runs`, ALC 0.1768,
reproduced bit-for-bit here as `auto__A3`). Δ < 0 is better. "pairs ± CI" is the paired mean over pair-seeds with a
95% CI; "fm" is the fold-macro (headline) Δ; "folds" counts folds where the run beats the control. All runs are in
`paiec-wt/auto/runs/`. Hold-out not touched.

## TL;DR

| model | ALC | Δ vs champion | Δ vs A3 | fresh seeds 3-5: Δ vs A3 | status |
|---|---|---|---|---|---|
| A3 (item amortisation) | 0.1768 | −0.0066 ± 0.0009 | 0 | 0 | previous best |
| A3 + subject regression + VI (`auto__A3_lr_vi05`) | 0.1740 | −0.0090 ± 0.0012 | −0.0024 ± 0.0009, fm −0.0028, 4/5 folds | −0.0023 ± 0.0010 | structural stack, no fitted weights |
| + fitted logit gate (`auto__mixfit_lrx`) | 0.1729 | −0.0100 ± 0.0012 | −0.0035 ± 0.0010, fm −0.0039, 5/5 folds | −0.0030 ± 0.0010 | 2 weights fitted leave-one-fold-out |
| **+ subject-variance EB `eb_u_nu` 10** (`auto__mixfit_lrx_nuu10`, `configs/research/auto/best.yaml`) | **0.1726** | **−0.0105 ± 0.0012**, fm −0.0105, 5/5 | **−0.0040 ± 0.0009**, fm −0.0042, 5/5 folds, win 64% | **−0.0037 ± 0.0010**, 5/5 folds | **best** |

The best model clears the promotion rule against both controls on the dev seeds and again on seeds 3-5. No variant
was chosen on seeds 3-5. Per budget vs A3 (dev): B0 −0.0021, B1 −0.0045, B3 −0.0044, B7 −0.0050, B15 −0.0036,
B31 −0.0025. Over the structural stack (lr+VI) it gains −0.0015 ± 0.0005 on dev and −0.0013 ± 0.0005 on seeds 3-5.

What stacks, in plain words:
- **Item amortisation + VI + subject regression stack, but only all three together.** Each of the last two fixes a
  different failure, and the gains add up nearly linearly (VI −0.0015 and lr −0.0007 on top of item; −0.0024 together).
  - Subject regression alone looked like it conflicted with item at B15-B31. That was an artefact of MAP inference:
    with VI its late-budget loss disappears.
  - Stacking VI onto the *B* item config (stack/r1) showed no gain. Stacking it onto *A3* gives −0.0015.
- **A learned logit blend adds a further −0.0010.** It uses two members of the same fitted model: lr+VI, and lr with
  MAP inference. Its weights say "push further in the direction VI moves away from MAP" (2.36·VI − 1.71·MAP).
  - Running more VI inside the inference does not reproduce this: vi_frac 1.0 and 1.5 are worse.
- **What does not stack or help:**
  - the collaborator's pair-local lexical kernel (blend weight 0.00-0.03 in every fit, worse on every slice);
  - the champion-without-item, hier VI and plain lr predictors as extra blend members (they add < 0.0003);
  - per-label-count weights (worse than global weights under LOFO, because B0 weights overfit);
  - a stacker with evidence features (= the per-bin version);
  - temperature recalibration (−0.0002);
  - mpnet semantic embeddings in the online content map (+0.0008, and not hosted-deployable);
  - gating lr off at high label counts (0 change once VI is on).
- **Not shipped, and why:**
  - A B0 intercept or sharpening fitted on dev would give a further −0.0007 to −0.0013 ALC (B0 −0.006 to −0.008
    Brier). The B0 under-prediction it corrects comes from dev's ref6-seen frontier subjects (diag §2), so it may not
    transfer to hosted subjects.
- **Small but consistent: per-benchmark subject-variance EM (`eb_u_nu` 10).**
  - On lr+VI it gives −0.0003 ± 0.0003 (dev) and −0.0002 ± 0.0004 (seeds 3-5).
  - On the gate it gives −0.0005 ± 0.0003 (dev) and −0.0007 ± 0.0004 (seeds 3-5).
  - It is a free runtime flag, so it is kept in the best config.

## 1. Stacking analysis (cheap first: shared-trajectory dumps)

**Why pairs.csv alone was not enough.** Each run's acquisition follows its own predictions, so two runs see
different labels. A per-pair "oracle pick" between any two runs is dominated by trajectory noise: champion vs A3 gives a
spurious −0.0031 (`analysis/auto/pairs_compl.py`). Only B0 is trajectory-free there. Prediction-level blends need
predictions on identical evidence.

**Component dumps** (`analysis/auto/dump.py`): one driver predictor runs the normal replay (acquisition + global
evidence), and at every checkpoint *every* component scores every eval item against the same `labeled`. The driver's
Brier reproduces its run exactly (checked on fold 0: A3 dump = `item__A3_mix_eb10` to 4 decimals at every budget).
- Dump 1 (driver A3): A3, champion, A3+VI (vi_frac 1), A3+lr, hier VI (`irt_h vi nu_u 10`), subj lr (`irt_lr`) and
  lex_cosine (`ref_lex`). Output in `analysis/auto/out/dump/`.
- Dump 2 (driver A3+lr+VI .5): A3+lr+VI, A3+VI .5, A3+lr, A3 and lex. Output in `out/dump2/`.

All 5 folds × 3 seeds = 582,624 item-checkpoint rows per component.

**Blends and stackers** (`analysis/auto/blend.py`, `fit_mix.py`):
- Logit-linear models are fitted by minimising the exact ALC-weighted fold-macro Brier.
- **LOFO**: the weights used on fold f are fitted on the other four folds only.
- Bins are the pair's own-label count: 0, 1, 2-3, 4-7, 8-15, 16+, which on the replay equal the budgets.
- Outputs: `out/blend_dump1.txt`, `out/blend_dump2.txt`.

### 1a. Components on one trajectory (dump 1; Δ vs A3, per budget B0..B31)

| component | ALC | Δ vs A3 | ΔB0 | ΔB1 | ΔB3 | ΔB7 | ΔB15 | ΔB31 |
|---|---|---|---|---|---|---|---|---|
| A3 | 0.1768 | | | | | | | |
| champion (no item amortisation) | 0.1844 | +0.0079 ± 0.0008 | 0 | +.0043 | +.0066 | +.0089 | +.0114 | +.0130 |
| A3 + VI (vi_frac 1) | 0.1752 | −0.0014 ± 0.0006 | 0 | −.0034 | −.0012 | −.0012 | −.0012 | −.0018 |
| A3 + lr | 0.1761 | −0.0007 ± 0.0003 | −.0024 | −.0011 | −.0008 | −.0005 | −.0001 | +.0000 |
| hier VI (no item) | 0.1806 | +0.0042 ± 0.0010 | 0 | −.0006 | +.0028 | +.0046 | +.0075 | +.0094 |
| subj lr (no item) | 0.1838 | +0.0074 ± 0.0009 | −.0025 | +.0031 | +.0061 | +.0087 | +.0115 | +.0131 |
| lex_cosine | 0.1994 | +0.0227 ± 0.0022 | +.0097 | +.0357 | +.0239 | +.0173 | +.0186 | +.0248 |

From dump 2: A3+lr+VI(.5) 0.1740 (−0.0024 ± 0.0005) and A3+VI(.5) 0.1748 (−0.0018 ± 0.0004).

### 1b. Two-way logit blends with A3 (dump 1, LOFO global weight unless noted)

| A3 + X | in-sample best w | LOFO Δ vs A3 | LOFO per-bin Δ | reading |
|---|---|---|---|---|
| champion | −0.25 | −0.0001 ± 0.0003 | +0.0006 | redundant: A3 contains it |
| A3+VI | 0.70 (LOFO coefs −0.82·A3 + 1.44·VI) | **−0.0024 ± 0.0005** | −0.0018 | complementary; the blend *extrapolates* past VI |
| A3+lr | 1.00 | **−0.0010 ± 0.0005** | −0.0004 | complementary, mostly B0-B3 |
| hier VI | 0.25 | −0.0005 ± 0.0003 | +0.0003 | mostly redundant with A3+VI |
| subj lr | −0.10 | −0.0002 ± 0.0003 | +0.0001 | redundant with A3+lr |
| lex_cosine | **0.00** | −0.0002 ± 0.0003 | +0.0004 | **no signal beyond A3** (weight 0.00-0.03 in every fit) |
| A3 with itself (pure temperature) | | −0.0003 ± 0.0003 | +0.0004 | calibration is not the lever |

### 1c. Multi-component stackers (dump 1, LOFO)

| stacker | Δ vs A3 | ΔB0 | ΔB1 | ΔB31 |
|---|---|---|---|---|
| A3 + A3vi + A3lr, global weights | −0.0025 ± 0.0006 | +.0012 | −.0043 | −.0024 |
| same, per-bin weights | −0.0022 ± 0.0006 | +.0053 | −.0040 | −.0023 |
| same, per-bin weights + per-bin intercepts | −0.0033 ± 0.0009 | **−.0065** | −.0052 | −.0022 |
| same + evidence features (log n_bench, log n_item, interactions) | −0.0033 ± 0.0009 | −.0065 | −.0052 | −.0023 |
| + lex | −0.0025 ± 0.0007 (lex coef −0.07) | | | |
| all 7 components, global | −0.0027 ± 0.0007 | | | |
| A3 recalibration alone: temperature / per-bin / per-bin + intercept | −0.0002 / +0.0004 / −0.0013 | / / −.0067 | | |

Readings:
- Everything beyond "A3 + VI + lr" is redundant: the champion, hier, plain lr, lex and evidence features add
  ≤ 0.0003.
- Per-bin weights lose under LOFO. The B0 bin has only the A3/lr contrast to go on, and its fitted weights do not
  transfer between folds (+0.005 to +0.009 at B0).
- The intercept gain is almost entirely B0 (−0.0065 Brier): a global upward mean shift. That is the dev subject-mix
  bias diag §2 warns about, so it is not shipped.

### 1d. Gate around the structural best (dump 2, `fit_mix.py`, bin 0 fixed to the lr+VI member)

| members | LOFO Δ vs A3+lr+VI | all-fold weights (bins ≥ 1) |
|---|---|---|
| lr+VI, lr(MAP) | −0.0006 ± 0.0002 | 2.358·lr+VI − 1.705·lr |
| lr+VI, VI, lr, A3 | −0.0006 ± 0.0003 | 1.63·lrVI + 0.85·VI − 0.93·lr − 0.92·A3 |
| per-bin weights | −0.0004 ± 0.0002 | |
| + per-bin intercepts (bins ≥ 1) | −0.0006 ± 0.0004 | |
| lr+VI, VI | +0.0001 | |
| lr+VI alone (temperature) | +0.0001 | 0.97 |

In a real run (`auto__mixfit_lrx`, own trajectory, LOFO weights per dev benchmark) the 2-member gate gives −0.0010 ±
0.0005. That is larger than the offline estimate.

## 2. Architectural effects

Contrasts on one trajectory (`analysis/auto/effects.py`, output `out/effects.txt`). Values are ALC-weighted pair-mean Δ
Brier, negative = helps. Benchmark groups: base rate extreme (< .2 or > .8; 297 pair-seeds) vs mid (735); evaluated
pairs per benchmark 1-6 / 7-15 / 16-20.

| decision | total | by budget B0 / B1 / B3 / B7 / B15 / B31 | extreme / mid | 1-6 / 7-15 / 16-20 pairs | eval items labelled by others / not (B ≥ 7, per response) |
|---|---|---|---|---|---|
| online item amortisation (A3 − champion) | −0.0079 | 0 / −.0044 / −.0071 / −.0096 / −.0119 / −.0134 | −.0063 / −.0086 | −.0036 / −.0063 / −.0106 | −.0045 / **−.0138** |
| VI on top of item (A3vi − A3) | −0.0014 | 0 / −.0030 / −.0010 / −.0011 / −.0010 / −.0016 | **−.0077 / +.0012** | +.0003 / +.0006 / −.0038 | −.0018 / −.0009 |
| VI on the champion (hier VI − champion) | −0.0037 | 0 / −.0046 / −.0039 / −.0044 / −.0039 / −.0034 | −.0115 / −.0006 | −.0010 / −.0016 / −.0065 | −.0030 / −.0038 |
| subject regression on item (A3lr − A3) | −0.0007 | **−.0021** / −.0010 / −.0007 / −.0005 / −.0001 / 0 | −.0005 / −.0007 | −.0012 / −.0003 / −.0010 | −.0002 / −.0003 |
| subject regression on the champion | −0.0005 | −.0022 / −.0010 / −.0004 / −.0001 / +.0001 / +.0001 | | | |
| subject regression given VI (A3lrvi − A3vi) | −0.0006 | −.0021 / −.0006 / −.0007 / −.0005 / −.0002 / −.0001 | −.0001 / −.0008 | | |
| lex kernel instead of the model (lex − A3) | +0.0227 | +.0097 / +.0361 / +.0242 / +.0174 / +.0186 / +.0248 | +.0121 / +.0270 | | +.0344 / +.0157 |

What each decision does, and where it helps or hurts:

- **Online item amortisation** (`irt_item`: shared content map, kv groups, duplicate block, EB scales). This is the
  largest lever.
  - Its gain grows with labels on the benchmark (B1 −0.004 → B31 −0.013) and with pairs per benchmark (−0.004 for
    1-6 pairs → −0.011 for 16-20).
  - By construction it is zero at B0, because the unlabeled-item prior variance stays var_d.
  - It mostly helps items nobody has labelled (−0.0138 per response vs −0.0045 for items already labelled by
    others), i.e. it transfers labels to unlabelled items through content and groups.
  - Its parts (from `docs/research/item.md`, 1-seed sweeps):
    - kv groups give the early gain: a few labels identify a group offset.
    - The content map gives the late gain.
    - The duplicate block and the EB scales are worth ≈ 0.
    - Fractions, dimension and the degree-2 kernel are flat.
- **Mean-field VI vs joint MAP.** Joint MAP attenuates β and δ, because the item effects absorb the labels. VI uses
  the expected likelihood, so β and δ keep their scale.
  - Gain: −0.0037 on the champion but only −0.0014 to −0.0018 on top of item. About half of VI's gain overlaps with
    item amortisation: item effects are now shared, so less signal goes into per-item d.
  - What remains is **almost entirely on extreme-base-rate benchmarks** (−0.0077; mid +0.0012) and on benchmarks with
    16-20 pairs. Biggest per-benchmark wins (`auto__A3_vi07` vs A3): biggen −0.024, robust_reasoning −0.014, rakuda −0.014.
  - It loses slightly on mid-base-rate and few-pair benchmarks (editbench, agentdojo +0.008 in `auto__A3_vi07`), where
    sharper predictions hurt.
  - **vi_frac is flat between 0.3 and 1.0** (0.1749 / 0.1750 / 0.1746 / 0.1750 at .3 / .5 / .7 / 1.0 on A3). With lr,
    0.5 is better than 1.0 (+0.0005) and 1.5 (+0.0016, mostly B1).
- **Subject-descriptor regression** (`lr`: per-benchmark w_b on [u(s), history z, think, size, date, effort]). It works
  at low budgets only: B0 −0.0021 Brier, B1 −0.0010, B3 −0.0007, then 0 by B15.
  - It gives the same gain on the champion and on item, so it is **orthogonal to item amortisation**.
  - The B15-B31 loss seen in `stack__item_lr` (B config, MAP) disappears with VI. Gating lr off at ≥ 4 or ≥ 8 labels
    is either worse (`auto__mix_gate4`, +0.0004: it loses B7) or neutral (`auto__mix_gate8`, +0.0001).
  - It is the only component that moves B0 without a dev-fitted shift. `lr_wvar_mult` 1 vs 2 is flat.
- **Prior vs learned variance (EB).** It helps only marginally.
  - The item side's EB on the kv/map scales gives −0.0003 to −0.0005 (item.md).
  - Per-benchmark subject-variance EM on top of lr+VI (`eb_u_nu` 10 / 20, new here) gives −0.0003 ± 0.0003 at both
    settings, mostly B3-B7.
  - hier found the same on the champion (−0.0004). Spreads are weakly identified from ≤ 20 subjects × few labels.
- **Slope on the prior** (`prior_slope_var`, item.md) gives −0.0003 to −0.0005, at noise level. It was not re-tested
  here.
- **Static content embeddings vs online adaptation.**
  - Replacing the online map's hashed BOW+kv features with frozen mpnet sentence embeddings (`auto__emb_lrvi`) is
    **worse** (+0.0008 ± 0.0007, B1 +0.0032). It is also not hosted-deployable (it needs the encoder at runtime for
    unseen items).
  - feat's static content features were +0.003 worse.
  - The collaborator's CaimiraLite retrains are still queued for a GPU (jobs 7968333-5, Priority), so they could not
    be stacked. Their item embeddings would have the same deployment problem as mpnet.
  - The gain comes from *online* adaptation to the benchmark's own labels, not from a better static representation.
- **Pair-local lexical kernel (lex_cosine).**
  - It is worse than A3 everywhere, by budget, base rate and pair count. It is worst on items others have labelled
    (+0.034 per response), because it ignores other subjects' labels.
  - As a blend member or residual its weight is 0.00-0.03.
  - It has no information beyond the item map + kv groups, which already capture within-benchmark lexical similarity
    using *all* subjects' labels.
  - A residual kernel on the model's own-label residuals would target subject×item interaction. diag bounds that at
    ≤ 0.002 even with every label, so it was not built.
- **Acquisition policy.**
  - Uncertainty acquisition is worth +0.0034 for the item model vs random: `ref__item_rand` (B config, random) 0.1802
    vs B config 0.1772 (ΔB1 +0.0013 but ΔB15 −0.0073 vs champion).
  - lex under uncertainty acquisition (`ref__lex_unc`) 0.1949 vs 0.1965.
  - acq's VOI / coverage policies found nothing beyond pair-order noise (±0.002), so acquisition was not re-opened.
  - Trajectory sensitivity is small: on the lr+VI trajectory, A3's Brier equals its own run's (0.1768).
- **Learned stacker.**
  - Meta's inner-replay stacker (champion, 1 seed) got −0.0025 by recalibrating a MAP model, which VI now does
    in-model.
  - Over A3 alone, temperature or per-bin recalibration is worth ≤ 0.0004 (excluding the B0 shift).
  - The only stacker gain that survives LOFO is the VI-vs-MAP extrapolation (§1d).

## 3. All variants (dev, seeds 0-2, 1,032 pair-seeds; absolute fold-macro Brier per budget)

Generated by `python analysis/auto/table.py ../../paiec/runs/R_champion_dev ../item/runs/item__A3_mix_eb10 runs/auto__*`.

| run | ALC | B0 | B1 | B3 | B7 | B15 | B31 | Δ champ (pairs ± CI) | fm | folds | win% | Δ A3 (pairs ± CI) | fm | folds | win% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R_champion_dev | 0.1830 | 0.2404 | 0.1944 | 0.1821 | 0.1740 | 0.1663 | 0.1562 | +0.0000 ± 0.0000 | +0.0000 | 0/5 | 0 | +0.0066 ± 0.0009 | +0.0062 | 0/5 | 31 |
| item__A3_mix_eb10 | 0.1768 | 0.2403 | 0.1911 | 0.1767 | 0.1667 | 0.1566 | 0.1456 | -0.0066 ± 0.0009 | -0.0062 | 5/5 | 69 | +0.0000 ± 0.0000 | +0.0000 | 0/5 | 0 |
| auto__A3 | 0.1768 | 0.2403 | 0.1911 | 0.1767 | 0.1667 | 0.1566 | 0.1456 | -0.0066 ± 0.0009 | -0.0062 | 5/5 | 69 | -0.0000 ± 0.0000 | -0.0000 | 2/5 | 44 |
| auto__A3_vi | 0.1750 | 0.2403 | 0.1877 | 0.1754 | 0.1651 | 0.1546 | 0.1441 | -0.0079 ± 0.0014 | -0.0080 | 5/5 | 68 | -0.0013 ± 0.0011 | -0.0018 | 4/5 | 58 |
| auto__A3_vi03 | 0.1749 | 0.2403 | 0.1873 | 0.1744 | 0.1652 | 0.1555 | 0.1446 | -0.0082 ± 0.0011 | -0.0081 | 5/5 | 72 | -0.0017 ± 0.0007 | -0.0019 | 5/5 | 61 |
| auto__A3_vi05 | 0.1750 | 0.2403 | 0.1876 | 0.1750 | 0.1654 | 0.1551 | 0.1436 | -0.0080 ± 0.0012 | -0.0080 | 5/5 | 69 | -0.0015 ± 0.0009 | -0.0018 | 3/5 | 59 |
| auto__A3_vi07 | 0.1746 | 0.2403 | 0.1877 | 0.1741 | 0.1643 | 0.1547 | 0.1442 | -0.0084 ± 0.0013 | -0.0084 | 5/5 | 69 | -0.0018 ± 0.0010 | -0.0022 | 3/5 | 60 |
| auto__A3_lr | 0.1760 | 0.2379 | 0.1890 | 0.1754 | 0.1664 | 0.1574 | 0.1463 | -0.0072 ± 0.0011 | -0.0070 | 5/5 | 67 | -0.0007 ± 0.0007 | -0.0008 | 4/5 | 51 |
| auto__A3_lr_vi05 | 0.1740 | 0.2379 | 0.1868 | 0.1737 | 0.1641 | 0.1549 | 0.1435 | -0.0090 ± 0.0012 | -0.0090 | 5/5 | 69 | -0.0024 ± 0.0009 | -0.0028 | 4/5 | 59 |
| auto__A3_lr_vi05_w1 | 0.1742 | 0.2377 | 0.1865 | 0.1746 | 0.1641 | 0.1548 | 0.1441 | -0.0088 ± 0.0012 | -0.0088 | 5/5 | 69 | -0.0023 ± 0.0009 | -0.0026 | 4/5 | 59 |
| auto__A3_lr_vi10 | 0.1745 | 0.2379 | 0.1880 | 0.1738 | 0.1644 | 0.1551 | 0.1441 | -0.0085 ± 0.0014 | -0.0086 | 5/5 | 67 | -0.0019 ± 0.0011 | -0.0024 | 4/5 | 58 |
| auto__A3_lr_vi15 | 0.1756 | 0.2379 | 0.1906 | 0.1756 | 0.1655 | 0.1553 | 0.1446 | -0.0074 ± 0.0014 | -0.0074 | 5/5 | 66 | -0.0009 ± 0.0012 | -0.0012 | 3/5 | 56 |
| auto__lrvi_parts_bu | 0.1756 | 0.2379 | 0.1889 | 0.1751 | 0.1655 | 0.1563 | 0.1462 | -0.0076 ± 0.0011 | -0.0075 | 5/5 | 71 | -0.0010 ± 0.0008 | -0.0012 | 4/5 | 56 |
| auto__lrvi_parts_bu10 | 0.1745 | 0.2379 | 0.1871 | 0.1735 | 0.1645 | 0.1558 | 0.1455 | -0.0086 ± 0.0012 | -0.0085 | 5/5 | 70 | -0.0021 ± 0.0009 | -0.0023 | 4/5 | 59 |
| auto__lrvi_parts_d | 0.1746 | 0.2379 | 0.1888 | 0.1731 | 0.1643 | 0.1556 | 0.1444 | -0.0085 ± 0.0012 | -0.0084 | 5/5 | 69 | -0.0019 ± 0.0009 | -0.0022 | 4/5 | 59 |
| auto__lrvi_nuu10 | 0.1737 | 0.2379 | 0.1866 | 0.1731 | 0.1635 | 0.1544 | 0.1436 | -0.0093 ± 0.0012 | -0.0094 | 5/5 | 71 | -0.0028 ± 0.0009 | -0.0031 | 5/5 | 61 |
| auto__lrvi_nuu20 | 0.1738 | 0.2379 | 0.1868 | 0.1731 | 0.1633 | 0.1549 | 0.1437 | -0.0093 ± 0.0012 | -0.0093 | 5/5 | 71 | -0.0027 ± 0.0009 | -0.0030 | 5/5 | 60 |
| auto__mix_gate4 | 0.1744 | 0.2379 | 0.1868 | 0.1737 | 0.1651 | 0.1555 | 0.1441 | -0.0086 ± 0.0012 | -0.0086 | 5/5 | 69 | -0.0021 ± 0.0009 | -0.0024 | 4/5 | 58 |
| auto__mix_gate8 | 0.1741 | 0.2379 | 0.1868 | 0.1737 | 0.1641 | 0.1552 | 0.1438 | -0.0089 ± 0.0012 | -0.0089 | 5/5 | 69 | -0.0024 ± 0.0009 | -0.0027 | 4/5 | 58 |
| auto__mix1_repro | 0.1740 | 0.2379 | 0.1868 | 0.1737 | 0.1641 | 0.1549 | 0.1435 | -0.0090 ± 0.0012 | -0.0090 | 5/5 | 69 | -0.0024 ± 0.0009 | -0.0028 | 4/5 | 59 |
| auto__emb_lrvi | 0.1749 | 0.2379 | 0.1900 | 0.1746 | 0.1637 | 0.1549 | 0.1442 | -0.0082 ± 0.0011 | -0.0082 | 5/5 | 67 | -0.0017 ± 0.0009 | -0.0020 | 3/5 | 56 |
| auto__mixfit_lrx | 0.1729 | 0.2379 | 0.1857 | 0.1721 | 0.1626 | 0.1537 | 0.1432 | -0.0100 ± 0.0012 | -0.0101 | 5/5 | 72 | -0.0035 ± 0.0010 | -0.0039 | 5/5 | 63 |
| auto__mixfit_g4 | 0.1730 | 0.2379 | 0.1857 | 0.1722 | 0.1629 | 0.1534 | 0.1435 | -0.0100 ± 0.0012 | -0.0100 | 5/5 | 72 | -0.0034 ± 0.0010 | -0.0038 | 5/5 | 63 |
| auto__mixfit_lrx_nuu10 | 0.1726 | 0.2379 | 0.1857 | 0.1719 | 0.1616 | 0.1531 | 0.1431 | -0.0105 ± 0.0012 | -0.0105 | 5/5 | 73 | -0.0040 ± 0.0009 | -0.0042 | 5/5 | 64 |

Notes:
- `auto__A3` reproduces `item__A3_mix_eb10` exactly.
- `auto__mix1_repro` (`irt_item_mix` with one member) reproduces `auto__A3_lr_vi05` exactly, which validates the mixer
  path.
- `vi_parts` restricts which posterior variances enter the VI expectation: `bu` = offset + pair only, `d` = item only.
  Both are worse than full VI. The item term carries the late-budget benefit (`bu`: B31 +0.0026), the β/δ terms the
  B1 benefit (`d`: B1 +0.0018).
- `auto__mixfit_g4` (4 members, 4 adaptations per benchmark) equals the 2-member gate `auto__mixfit_lrx`, at twice the
  cost.
- `auto__mixfit_lrx_nuu10` adds `eb_u_nu: 10` to both gate members, keeping the weights fitted without it.

Fresh seeds 3-5 (not used for any choice; paired vs their own controls on the same seeds):

| run | ALC | B0 | B1 | B3 | B7 | B15 | B31 | Δ champ (pairs ± CI) | fm | folds | win% | Δ A3 (pairs ± CI) | fm | folds | win% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto__champ_s345 | 0.1829 | 0.2404 | 0.1958 | 0.1822 | 0.1727 | 0.1649 | 0.1569 | +0.0000 ± 0.0000 | +0.0000 | 0/5 | 0 | +0.0067 ± 0.0009 | +0.0063 | 0/5 | 30 |
| auto__A3_s345 | 0.1765 | 0.2403 | 0.1904 | 0.1773 | 0.1661 | 0.1558 | 0.1460 | -0.0067 ± 0.0009 | -0.0063 | 5/5 | 70 | +0.0000 ± 0.0000 | +0.0000 | 0/5 | 0 |
| auto__A3_vi05_s345 | 0.1749 | 0.2403 | 0.1876 | 0.1754 | 0.1646 | 0.1547 | 0.1444 | -0.0082 ± 0.0011 | -0.0079 | 5/5 | 70 | -0.0015 ± 0.0008 | -0.0016 | 5/5 | 58 |
| auto__A3_lr_vi05_s345 | 0.1741 | 0.2379 | 0.1862 | 0.1740 | 0.1644 | 0.1547 | 0.1445 | -0.0090 ± 0.0012 | -0.0088 | 5/5 | 70 | -0.0023 ± 0.0010 | -0.0024 | 4/5 | 60 |
| auto__lrvi_nuu10_s345 | 0.1738 | 0.2379 | 0.1860 | 0.1747 | 0.1635 | 0.1539 | 0.1441 | -0.0092 ± 0.0012 | -0.0090 | 5/5 | 71 | -0.0026 ± 0.0010 | -0.0027 | 4/5 | 62 |
| auto__mixfit_lrx_s345 | 0.1734 | 0.2379 | 0.1847 | 0.1732 | 0.1638 | 0.1544 | 0.1443 | -0.0096 ± 0.0012 | -0.0095 | 5/5 | 72 | -0.0030 ± 0.0010 | -0.0031 | 4/5 | 61 |
| auto__mixfit_lrx_nuu10_s345 | 0.1727 | 0.2379 | 0.1847 | 0.1730 | 0.1626 | 0.1527 | 0.1436 | -0.0103 ± 0.0012 | -0.0101 | 5/5 | 74 | -0.0037 ± 0.0010 | -0.0038 | 5/5 | 62 |

Offline-only variants (dumps, LOFO; no run): §1b-1d above.

## 4. Best model

`configs/research/auto/best.yaml` (run `auto__mixfit_lrx_nuu10`, ALC 0.1726) = `configs/research/auto/mixfit_lrx.yaml` + `eb_u_nu: 10`. Class `irt_item_mix` (`src/paiec/predictors/irt_mix.py`):
- **Shared fit**: one `irt_item_lr` fit (IRT + per-benchmark subject regression). Members are runtime configurations
  of that one fit.
- **Members**:
  1. A3 item amortisation + lr + VI (vi_frac .5);
  2. the same with MAP inference.
- **Gate**: logit p = 2.358·L1 − 1.705·L2 for pairs with ≥ 1 own label; L1 at 0 labels.
- **Weights**:
  - `mix_fold_w` holds, for each dev benchmark, the weights fitted on the other four folds (honest dev number).
  - `mix_w` is the all-fold fit used on hosted (anonymous benchmarks).
  - The LOFO weights are stable: 2.19-2.33 / −1.55 to −1.65.
- **Runtime**: numpy + stdlib. Each benchmark is adapted twice. `time_predict.py` on the 2-member gate (mmlupro, 20 subjects): 1.03 ms per acquisition call and 0.68 ms per eval call, or 4.2 min for 100× formative (0.88% of 8 h). `eb_u_nu` adds one scalar update per Newton iteration. The fit is the cached `irt_item_lr` fit (`FIT_AS`); no torch at runtime.

The structural alternative without fitted weights is `configs/research/auto/lrvi.yaml` (`auto__A3_lr_vi05`, 0.1740):
`irt_item_lr` with the A3 item settings, `lr_wvar_mult` 2, `vi: true`, `vi_frac: 0.5`. It costs 0.96 ms per
acquisition call and 0.66 ms per eval call (`time_predict.py`, mmlupro, 20 subjects), or 4.1 min for 100× formative
(0.86% of 8 h).

## 5. Next steps (ranked)

1. **Promote `best.yaml`** (gate + `eb_u_nu` 10; −0.0040 vs A3 on dev, −0.0037 on fresh seeds).
   - Fallback without dev-fitted weights: `lrvi.yaml` + `eb_u_nu: 10` (−0.0028 / −0.0026).
   - Before promotion, refit the gate weights on the final champion's trajectory if anything upstream changes
     (`analysis/auto/dump.py` then `fit_mix.py`, about 30 min).
2. **Hosted B0.**
   - The remaining big B0 lever is a mean shift (dev subjects under-predicted by ≈ 0.1 at B0).
   - Decide from a hosted formative submission, not dev, whether a B0 intercept transfers. Offline it is worth
     −0.0007 to −0.0013 ALC.
3. **More pairs per benchmark on hosted favour item amortisation and VI** (both grow with 16-20 pairs). If hosted
   benchmarks have more subjects, expect larger gains than dev shows.
4. **Understand the VI-vs-MAP extrapolation** (2.4× the VI correction helps, but more VI inside inference hurts).
   - A plausible in-model analogue: VI for the β/δ scale but MAP-like item terms.
   - Expected ≤ 0.001.
5. **Deprioritise:** lex/pair-local kernels, static embeddings in the online map, per-budget weight schedules,
   further vi_frac / lr_wvar / EB tuning (all flat or negative here).

## 6. Code and reproduction

- `src/paiec/predictors/irt_mix.py`: `irt_item_mix`, the label-count-gated logit blend of runtime members of one
  `irt_item_lr` fit.
  - `FIT_AS = IRTItemLR`, so it shares the fit cache.
  - Parameters: `mix_members`, `mix_w`, `mix_c`, and `mix_fold_w` / `mix_fold_c` for LOFO weights per dev benchmark.
  - `emb_online` is a DEV-ONLY mpnet probe.
- `src/paiec/predictors/irt_item.py`: two runtime flags, both default off or identical to before.
  - `eb_u_nu`: online EB for the per-benchmark pair-ability variance (as irt_h `nu_u`).
  - `vi_parts`: which variances enter the VI expectation; default `bud` = unchanged.
- `analysis/auto/`:
  - `dump.py` / `dump.slurm`: shared-trajectory component dumps.
  - `blend.py`: blends and stackers, LOFO.
  - `fit_mix.py`: fits gate weights and writes a run config.
  - `effects.py`: architectural contrasts.
  - `pairs_compl.py`: pairs.csv complementarity, and why it is not enough.
  - `table.py`: tables in this doc.
  - `run_fits.sh`.
- Reproduce the gate weights:
  1. `sbatch`-equivalent `hpc-submit analysis/auto/dump.slurm --export=ALL,FOLD=k,ARGS="--comps A3lrvi05 A3vi05 A3lr A3 lex --out analysis/auto/out/dump2"`
     for k = 0..4;
  2. `PYTHONPATH=src python analysis/auto/fit_mix.py --pattern 'analysis/auto/out/dump2/f*_s*.npz' --comps A3lrvi05 A3lr --kind gate --b0fixed A3lrvi05 --name auto__mixfit_lrx`.
- Tests: `python -m pytest -q` gives 59 passed with `CUDA_VISIBLE_DEVICES=` on klingon. The three `ref_caimira`
  tests fail only when torch sees the GTX 1080 Ti ("no kernel image"), a pre-existing environment issue of the aims26
  torch build.

# Research direction: item (online within-benchmark item amortisation)

Branch `exp/item`. Code: `src/paiec/predictors/irt_item.py` (registry `irt_item`), `src/paiec/fitcache.py` (`FIT_AS`).
Scripts and launchers: `analysis/item_online/`. Control: `runs/R_champion_dev` (main checkout, 3 seeds, ALC 0.1830).

## Model
At test time, per benchmark b, the item difficulty is

    d_i = yhat_i + z_i . w_b + sum_k a_ik g_{b,k} + eps_i

- `yhat_i`: the champion's cross-benchmark content prior (unchanged fit).
- `z_i . w_b` (**content map**, `rp_frac`): z_i is a seeded random projection (`rp_dim`) of the champion's own item
  features (ItemFeaturizer: hashed BOW, kv tokens, length), centred on the benchmark's labeled items and normalised to
  unit length, i.e. a cosine kernel. w_b ~ N(0, rp_frac·var_d·I) is learned jointly with β, δ and ε in the Newton loop
  (non-centred parametrisation; one k×k Newton step per iteration). Unlabeled items get mean z·w and variance zᵀΣ_w z
  (Σ_w: Laplace posterior with ε marginalised).
- `g_{b,k}` (**kv groups**, `kv_frac`): one random effect per `item_features` key=value token (numeric values
  skipped), weight a = sqrt(kv_frac·var_d / n_tokens(i)); Gauss–Seidel over keys (values of a key are disjoint).
  `dup_frac` adds one more block keyed by identical `item_content` (in practice: shared templates that differ in kv).
- `eps_i ~ N(0, (1 − Σ fracs)·var_d)`: the item's own effect, as in the champion.
- The total prior variance of an unlabeled item stays var_d, so **B0 is unchanged** (ΔB0 −0.0001, from the slope term
  only).
- Options: `eb_nu` (online EM for per-benchmark scales of each kv key and of the content map, pseudo-count toward the
  configured fraction), `eb_eps_nu` (same for ε), `rp2_frac` (degree-2 tensor-sketch kernel block),
  `prior_slope_var` (per-benchmark slope on yhat: d += c_b (yhat − mean)), `kv_numeric`.
- All new params are `RUNTIME_PARAMS`, and `FIT_AS = IRT` makes the class share the champion's cached fits (the fit is
  identical; `fitcache` re-classes the loaded object and stores fits as the parent class). With all fractions 0 the
  model reproduces the champion exactly (tested, and the 1-seed control `item__ctrl1` = 0.1844 = champion seed 0).
- Feature interface: `IRTItem.online_features(items, X)` returns the vectors for the content map (default: the IRT
  prior's own features X). The `feat` direction can swap its representation in there (or via `item_fields`/`item_dim`).

## Results (dev, paired vs control; Δ < 0 is better)
3 seeds (n = 1,032 pair-seeds) vs `R_champion_dev` (0.1830):

| run | config (over champion) | ALC | Δ fold-macro | Δ pairs ± CI | win % | ΔB0 | ΔB1 | ΔB3 | ΔB7 | ΔB15 | ΔB31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **item__A3_mix_eb10** | rp .3 (d64), kv .2, dup .2, eb_nu 10 | **0.1768** | **−0.0062** | **−0.0066 ± 0.0009** | 69 | −.0001 | −.0036 | −.0058 | −.0078 | −.0101 | −.0109 |
| item__B3_d32kv30 | rp .4 (d32), kv .3 | 0.1772 | −0.0059 | −0.0061 ± 0.0011 | 64 | −.0001 | −.0031 | −.0065 | −.0079 | −.0087 | −.0086 |
| item__D3_B_slope25 | B + eb_nu 10 + prior_slope_var .25 | see below | | | | | | | | | |

1 seed (n = 344) vs the 1-seed control `item__ctrl1` (0.1844, identical to champion seed 0):

| run | config | ALC | Δ fold-macro |
|---|---|---|---|
| rp50 | rp .5 | 0.1788 | −0.0056 |
| rp70 | rp .7 | 0.1786 | −0.0058 |
| rp50d32 / d128 | rp .5, k 32 / 128 | 0.1781 / 0.1779 | −0.0063 / −0.0065 |
| kv30 / kv50 | kv .3 / .5 | 0.1799 / 0.1796 | −0.0045 / −0.0048 |
| dup30 | dup .3 | 0.1843 | −0.0001 |
| mix | rp .3, kv .2, dup .2 | 0.1777 | −0.0067 |
| mix_it30 | mix, 30 Newton iterations | 0.1778 | −0.0066 |
| rp50kv30, rp40kv40 | | 0.1781, 0.1782 | −0.0063, −0.0062 |
| d32kv30 (= B) | rp .4 (d32), kv .3 | 0.1776 | −0.0068 |
| mix_eb3 / mix_eb10 | mix + eb_nu 3 / 10 | 0.1776 / 0.1774 | −0.0068 / −0.0070 |
| mixe_eps50 / eps200 | mix_eb10 + eb_eps_nu | 0.1778 / 0.1775 | −0.0066 / −0.0069 |
| mixe_vs12 | mix_eb10 + var_scale 1.2 | 0.1777 | −0.0067 |
| mixe_r25 | mix_eb10 + uncertainty radius .25 | 0.1782 | −0.0062 |
| mixe_rand | mix_eb10 + official_random acquisition | 0.1819 | −0.0025 |
| B_eb10 | B + eb_nu 10 | 0.1773 | −0.0071 |
| B_dup10 | B + dup .1 | 0.1777 | −0.0067 |
| B_rp2_15 / _d128 / rp2_25 | degree-2 block | 0.1783 / 0.1775 / 0.1776 | ≈ 0 vs B |
| B_kvnum | B + numeric kv values | 0.1779 | −0.0065 |
| B_slope10 / 25 / 1 | B_eb10 + prior_slope_var | 0.1769 / 0.1768 / 0.1779 | −0.0075 / −0.0076 / −0.0065 |
| M_slope25 | mix_eb10 + prior_slope_var .25 | 0.1775 | −0.0069 |

Scaling with subjects per benchmark (1 seed, `cv.max_subjects_per_benchmark=50`, own control `S50_ctrl` 0.1784):
mix_eb10 gives **−0.0083 fold-macro, −0.0090 ± 0.0016 pairs** (ΔB1 −0.0072), against −0.0067 at ≤20 subjects. In the
3-seed run, benchmarks with 1–6 / 7–15 / 16–20 evaluated pairs gain −0.0024 / −0.0056 / −0.0088 (benchmark means).

## Insights
- **Online amortisation works; it is the first large item-side gain.** About −0.0065 ALC on dev (3.6%), 69% of pairs
  better, consistent over folds. It grows with budget (labels on the benchmark) and with the number of pairs per
  benchmark, so hosted benchmarks with more subjects should gain more than dev.
- **The two components are complementary.** kv groups help early (ΔB1 −0.005 alone: a few labels identify a group
  offset); the content map helps late (ΔB15 −0.011). Content-duplicate groups add nothing beyond them (true duplicates
  already share `item_key`; template groups are covered by the map and kv tokens).
- **Hyper-parameters are flat.** Fractions 0.5–0.8 shared, k 32–128, 15 or 30 iterations, EB on or off, the degree-2
  kernel, numeric kv: all within ±0.0005. The limit is information (labels per benchmark and feature quality), not the
  online model's flexibility. EB (eb_nu 10) and the prior slope give ≈ −0.0003 to −0.0005, at noise level.
- Biggest wins: ai2d_test, phyblock (kv-only distinguished items), arcagi, ikp, scale_mrt (−0.015 to −0.026).
  Small losses: editbench, afrimedqa, kmmlu, icpc2 (+0.002 to +0.004), mostly benchmarks with few pairs or no structure.
- **Acquisition interaction (for acq):** with item amortisation, random acquisition loses most of the gain
  (0.1819 vs 0.1774), and a wider uncertainty band (r .25) is slightly worse. Labels that inform many eval items via
  groups or the content map are worth more now; an acquisition score that values "labels on under-covered groups /
  regions of z" is the natural next step.
- **For feat:** the online model consumes any per-item vector through `online_features`. Within-benchmark feature
  quality is now the main lever; the map is a cosine kernel on centred features, so representations should be compared
  by within-benchmark ridge R² on noisy targets.
- **For hier:** per-benchmark spread is partly handled here (EB on the kv/map scales and ε); the total item variance
  still uses the global var_d.

## Runtime
`scripts/time_predict.py` (mmlupro, 20 subjects, hosted call pattern): champion 0.49 ms/acq call, 0.48 ms/eval call;
irt_item (B config) 0.79 ms/acq call, 0.64 ms/eval call. 100× formative: 3.9 min (0.8% of 8 h). numpy + stdlib only.

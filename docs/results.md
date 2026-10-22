# Results log

Protocol for every row unless noted:
- Data revision `54074453`, matharena and mmdocrag binarised at grade ≥ 0.5.
- LOBO over 5 test benchmarks; swe_rebench (1 subject) is train-only.
- Seeds {0, 1, 2}, `split_seed` 0, ≤200 eval items per pair, global evidence.
- ALC is the mean over folds of the pair-macro ALC. Lower is better.

Regenerate this table with `python scripts/leaderboard.py --folds`.

## 2026-10-22: baseline grid (`grid1__*`)
| predictor | acquisition | ALC | B0 | B1 | B3 | B7 | B15 | B31 | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|
| irt | accept_all | **0.1832** | 0.267 | 0.203 | 0.188 | 0.172 | 0.153 | 0.133 | 0.155 |
| irt | official_random | 0.1835 | 0.267 | 0.204 | 0.188 | 0.173 | 0.153 | 0.132 | 0.156 |
| beta_shrink | accept_all | 0.2043 | 0.327 | 0.205 | 0.195 | 0.187 | 0.182 | 0.178 | 0.132 |
| empirical_mean | official_random (= official baseline) | 0.2328 | 0.250 | 0.347 | 0.225 | 0.196 | 0.183 | 0.177 | 0.177 |
| constant 0.5 | any | 0.2500 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.249 |

Per-fold IRT ALC:
- matharena 0.190
- mmdocrag 0.184
- multi_swebench 0.169
- real_webagents 0.180
- researchcodebench 0.193

Readings:
- **IRT beats the official baseline by 0.05 ALC (21%)** on every fold. Most of the gain is at low budgets (B1 0.20 vs 0.35), where global evidence lets the model learn the benchmark offset and item difficulty from other subjects' labels.
- **B0 is the weakest point.** IRT scores 0.267 and beta_shrink 0.327, both worse than a constant 0.5. Base rates differ a lot across benchmarks (0.15–0.67), and the cold-start prior is confidently wrong on low-base-rate benchmarks such as multi_swebench. B0 carries only 10% of the weight, but B1 is affected too.
- accept_all and official_random are equivalent within noise, as expected: both choose uniform subsets because candidate order is random.

## 2026-10-22: IRT acquisition, evidence scope and calibration (`irt__*`)
All rows are paired against `irt + accept_all` (ALC 0.1832) on identical pools, orders and trials. Δ is the mean per-pair ALC difference with its 95% CI (n = 864 pair-seeds).

| variant | ALC | Δ vs accept_all | B1 | B31 | notes |
|---|---|---|---|---|---|
| acquisition=uncertainty (r=0.15) | **0.1794** | −0.0039 ± 0.0008 | 0.200 | 0.128 | better on all 5 folds, 67% of pairs |
| acquisition=information (γ=1) | 0.1804 | −0.0034 ± 0.0010 | 0.200 | 0.129 | |
| acquisition=information (γ=3) | 0.1814 | n/a | 0.201 | 0.130 | |
| var_scale=2 | 0.1840 | n/a | 0.204 | 0.134 | B0 improves (0.260) but the rest is worse |
| var_scale=0.5 | 0.1851 | n/a | 0.203 | 0.135 | |
| **evidence_scope=pair** | 0.2135 | **+0.0315 ± 0.0015** | 0.239 | 0.185 | worse on all folds; still beats the official baseline (0.2328) |

Readings:
- **Whether `labeled` is global is the single largest factor (0.03 ALC).** It is an *unverified* assumption from the collaborator's replay. Confirm it with the starting kit or the organizers before relying on it.
- Selective acquisition helps a little (−0.004) once the predictor models item difficulty. Uncertainty sampling is the best simple rule so far, and it is now the default in `configs/irt.yaml`. Runtime cost is negligible: at about 4 ms per acquisition call, hosted time stays under 2% of the 8 h limit even at 100× the formative workload.
- The prior-variance scale is already about right at 1.0. The remaining B0 gap needs a better cold-start prior, not a rescaled one.

## 2026-10-22: B0 prior and acquisition rules (rounds 2–3)
Control: `irt + uncertainty(0.15)` (ALC 0.1794). Δ is the paired per-pair ALC difference with its 95% CI (n = 864).

**B0 diagnosis** (`analysis/b0/b0_prior.py`, B0 Brier, mean over folds):
- current IRT 0.267
- offset prior variance ×64: 0.253
- IRT with the oracle benchmark offset: **0.204**
- per-pair base rate (oracle): 0.170

The cold-start loss is almost entirely the unknown benchmark offset (mmdocrag, base rate 0.78, is worst). Widening the offset variance only hedges toward 0.25.

| variant | ALC | Δ vs control | B0 | B1 | B31 |
|---|---|---|---|---|---|
| **absolute item target + beta_var_mult=4** (new default) | **0.1772** | **−0.0032 ± 0.0010** | 0.255 | 0.197 | 0.127 |
| absolute + beta_var_mult=16 | 0.1773 | −0.0030 ± 0.0010 | 0.250 | 0.197 | 0.128 |
| beta_var_mult=16 | 0.1778 | −0.0025 ± 0.0009 | 0.258 | 0.197 | 0.129 |
| absolute | 0.1781 | −0.0023 ± 0.0009 | 0.258 | 0.199 | 0.128 |
| beta_var_mult=4 | 0.1789 | −0.0009 ± 0.0009 | 0.263 | 0.197 | 0.128 |
| absolute + balanced benchmarks (± ×4) | 0.179 | ≈0 | 0.26 | 0.197 | 0.128 |
| beta_var_mult=64 | 0.1794 | −0.0009 ± 0.0013 | 0.253 | 0.198 | 0.131 |
| uncertainty radius 0.1 / 0.05 / 0.25 | 0.1792 / 0.1809 / 0.1809 | 0.0000 / +0.0013 / +0.0011 | | | |
| information γ=0.5 / 1 / 3 | 0.1808 / 0.1804 / 0.1814 | +0.0011 / +0.0006 / +0.0019 | | | |
| quantile (slack 1 / 1.5) | 0.1811 / 0.1808 | +0.0015 / +0.0011 | | | |

Readings:
- **Best B0 prior:** let item text predict absolute difficulty (offset included), and widen the offset prior (×4–16). B0 falls from 0.267 to 0.250–0.255, and B1 improves slightly because the offset is learned faster from the first labels. That is still far from the oracle-offset 0.204. Closing the gap needs better benchmark-level signals (see the subagent plan in docs/item-difficulty-plan.md).
- Balancing benchmarks in the item ridge removes the gain. Large benchmarks seem to carry the useful cross-benchmark signal, but with 5 folds this is weak evidence.
- **Acquisition:** uncertainty sampling with r ≈ 0.1–0.15 remains best. The model-aware `quantile` rule and rate-adaptive `information` rules are slightly worse. A likely reason: rejecting early is free while the pool is large, and a fixed tight band picks more informative items than a rule calibrated to fill exactly. Acquisition gains look saturated at about 0.004 ALC over random. The prior and features are the bigger levers.

## 2026-10-22: item-difficulty exploration (subagent; details in item-difficulty-plan.md)
Protocol: ref6-LOBO. Paired against the default config (ALC 0.1772).
| variant | Δ ALC |
|---|---|
| no item features | +0.0055 ± 0.0013 (so the BOW prior is worth about 0.005) |
| BOW + Qwen3-8B difficulty rating | −0.0003 ± 0.0007 (not significant) |
| LLM rating alone / hand-crafted (z per benchmark) / MiniLM ridge | +0.0041 / +0.0042 / +0.0049–0.0123 |
| BOW item-variance ×0.4 / 0.7 / 1.5 | +0.0035 / +0.0011 / +0.0007 |

LOBO difficulty prediction (Spearman, held-out benchmark):
- every text, embedding or hand-crafted model: −0.06 to 0.10;
- Qwen3-8B zero-shot rating: 0.19;
- Qwen3-32B: 0.33 on a 600-item pilot.

## 2026-10-22: training-set selection on the 57-benchmark panel (protocol: panel-5fold, see validation.md)
Default IRT config, 1 seed, ≤20 subjects and ≤100 eval items per pair, 921 pair evaluations. Runs: `sel_pool`, `sel_panel`, `sel_ref6`.
| training set | ALC | B0 | B1 | B31 | paired Δ |
|---|---|---|---|---|---|
| **pool** (207 benchmarks, capped 100K responses each) | **0.2076** | 0.258 | 0.220 | 0.180 | vs ref6: −0.0081 ± 0.0023, better on 37/57 benchmarks |
| panel only (other 4 folds, ~46 benchmarks) | 0.2091 | 0.253 | 0.222 | 0.181 | vs ref6: −0.0066 ± 0.0021; pool vs panel: −0.0015 ± 0.0016 (n.s.) |
| ref6 only | 0.2157 | 0.260 | 0.224 | 0.198 | |

Readings:
- **More, and more diverse, training benchmarks help.** The gain is mostly at high budgets (B31 0.198 → 0.180), i.e. better item and subject priors and variance estimates. Domain gains vs ref6 are largest for education, language understanding, multilingual, law and safety (−0.02 to −0.04). A few domains get slightly worse (cybersecurity, navigation, long-term memory, preference: +0.01 to +0.02), which suggests similarity weighting could help.
- **Panel ALC (~0.21) is not comparable to ref6-LOBO (~0.177).** The benchmark mixes differ. Compare only within a protocol.
- Still to run on the panel: the constant and empirical-mean baselines, for context.

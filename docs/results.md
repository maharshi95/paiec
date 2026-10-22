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

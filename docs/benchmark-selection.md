# Benchmark selection (train pool and internal-eval panel)

Data: all 278 public measurement-db benchmarks, core tables only (15 GB), in `/fs/clip-scratch/mgor/measurement-db`.

Pipeline:
1. `scripts/download_data.py --all`
2. `scripts/mdb_inventory.py`
3. `scripts/benchmark_report.py`
4. `scripts/make_selection.py`

## Tiers (`benchmark_report.csv`; binarised with the scale-aware rule at 0.5)
| tier | meaning | benchmarks | eligible pairs (≥80 items) | responses |
|---|---|---|---|---|
| A | test-like: binarisable, ≥3 eligible pairs | 210 | 18,049 | 84.5M |
| B | train-only: ≥2 subjects, <3 eligible pairs | 31 | 23 | 1.9M |
| C | one subject only | 19 | 17 | 2.2M |
| X | not binarisable (unordered/unbounded scale) or withheld | 18 | 0 | 3.8M |

Quality problems found:
- 11 tier-A benchmarks have degenerate base rates. One example is tabarena, where the 0.5 cut on a tabular metric gives a rate of 4e-6.
- 22 have no subject spread (per-subject accuracy sd < 0.03).
- Sizes are very skewed. monkey_power_laws alone has 28.8M responses (repeated sampling). The median benchmark has 32K.

## Selections (`configs/sel/`, regenerate with `scripts/make_selection.py`)
**Train pool: 207 benchmarks.** Tier A/B, base rate in [0.03, 0.97], subject sd ≥ 0.03. Each benchmark is capped at 100K responses by keeping a seeded random subset of subjects, giving 10.5M responses, 2.4M items and 20K subjects.

**Internal-eval panel: 57 benchmarks** in 5 domain-mixed folds (`configs/sel/panel.csv`). The panel takes pool tier-A benchmarks that meet all of:
- 8–150 eligible pairs;
- ≤600K responses;
- subject sd ≥ 0.05 and item sd ≥ 0.15;
- at most 3 per primary domain, preferring binary-graded and larger benchmarks.

The 29 primary domains include agents, medicine, safety, math, SWE, law, multilingual and vision.

## Experiment: which training benchmarks help? (done 2026-10-22; numbers in results.md)
Result: pool < panel-only < ref6 in ALC (0.2076 / 0.2091 / 0.2157). Pool vs ref6 is −0.0081 ± 0.0023, and pool vs panel is not significant. Use the pool as the default training set.

Setup:
All three variants are evaluated on the same panel folds with the default IRT config, 1 seed, ≤20 subjects and ≤100 eval items per pair:
- `sel_ref6`: train only on the six original benchmarks;
- `sel_panel`: train on the other panel folds;
- `sel_pool`: train on the whole capped pool minus the held-out fold.

`metrics.json → benchmarks` gives per-benchmark ALC, which shows which kinds of benchmarks gain from more training data.

Follow-ups:
- greedy or domain-level inclusion tests;
- weighting training benchmarks by similarity to the test benchmark;
- whether `lower_is_better` binary benchmarks (`binarize.orient`) should be flipped for cross-benchmark subject priors.

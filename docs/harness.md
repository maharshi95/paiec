# Using the research harness

First time here? Start with [getting-started.md](getting-started.md) and the `configs/aims/` protocol.

The design rationale is in [harness-plan.md](harness-plan.md). This page covers day-to-day use.

## Run, sweep, compare
```bash
export PYTHONPATH=src
python scripts/run.py --config configs/irt.yaml                         # 5 LOBO folds x 3 seeds
python scripts/run.py --config configs/irt.yaml --set predictor.var_scale=1.5 --name irt_vs1.5
python scripts/run.py --config configs/irt.yaml --set 'cv.seeds=[0]' --set 'cv.test_benchmarks=[multi_swebench]'  # quick
python scripts/sweep.py --config configs/irt.yaml --grid predictor.var_scale=0.5,1,2 --grid acquisition.name=accept_all,information --jobs 6
python scripts/leaderboard.py --folds             # every run under runs/, sorted by ALC
python scripts/time_predict.py --config configs/irt.yaml   # hosted-style latency vs the 8 h limit
python scripts/build_submission.py --config configs/irt.yaml   # dist/irt.zip
```
Quote `--set` values that contain brackets (`'cv.seeds=[0]'`), because zsh globs them.
On the cluster or workstations, use `launch/run_experiment.slurm` (see the header of that file).

Each run writes `runs/<name>/`:
- `config.yaml`: the fully resolved config.
- `metrics.json`: `alc` (mean over folds), `alc_pairs` (mean over pairs), `by_budget`, `ece_alc`, per-fold detail, timing, data revision and git sha.
- `pairs.csv`: one row per (subject, benchmark, seed), with B0..B31, ECE, ALC and acquisition stats.

Use `pairs.csv` for paired comparisons. Runs with the same `cv.split_seed` and `cv.seeds` see identical pools, candidate orders and revealed trials.

## Config keys (defaults in `src/paiec/config.py`)
| Key | Meaning |
|---|---|
| `data.root` | a measurement-db-layout directory, or a list of them (e.g. `[data/aims/train, data/aims/val]`) |
| `data.benchmarks` | benchmarks loaded (train + test universe); null = everything under `data.root` |
| `data.binarize` | `{mode: exclude}` drops non-binary benchmarks; `{mode: threshold, threshold: t}` uses y = grade ≥ t; `per_benchmark: {mmdocrag: 0.8, matharena: exclude}` |
| `cv.test_benchmarks` | folds: a benchmark name (LOBO) or a list held out together (default: every loaded benchmark) |
| `cv.train_benchmarks` | training pool per fold, minus the fold: a list, a data directory (every benchmark in it), or null (all loaded) |
| `cv.eval_seen_from` | score only subjects whose model identity appears in these benchmarks |
| `cv.seeds`, `cv.split_seed` | candidate order and trial draws; fixed pools |
| `cv.max_subjects_per_benchmark` | mimic platform subject sampling, or speed up |
| `cv.max_eval_items_per_pair` | speed. Base config uses 200; set `null` for a final number |
| `sim.evidence_scope` | `global` (all pairs' labels visible; confirmed hosted behaviour) or `pair` |
| `predictor.*` | `name` plus any class attribute of that predictor |
| `acquisition.*` | `name` plus any class attribute of that policy |

## Main knobs worth sweeping
| Key | Values | Effect |
|---|---|---|
| `predictor.item_target` | `centered` / `absolute` | whether item text may predict the benchmark offset (cold start) |
| `predictor.balance_benchmarks` | bool | equal benchmark weight in the item ridge |
| `predictor.beta_var_mult` | ≥1 | widens the benchmark-offset prior (hedges B0, learns the offset faster from evidence) |
| `predictor.var_scale` | ~1 | scales all prior variances |
| `predictor.item_fields` / `subject_fields` | lists | field ablations |
| `acquisition.name` | `uncertainty` (radius), `information` (gamma), `quantile` (slack), `accept_all`, `official_random` | streaming selection. `quantile` uses the bound predictor plus global evidence |

## Extending
- **New predictor.** Subclass `paiec.predictors.base.Predictor`:
  - Implement `fit(train)`, `score(state, subject, item)` and, for speed, `score_batch`.
  - Optionally override `adapt` / `extend` if the state is more than evidence counts.
  - Every class attribute is a config parameter.
  - Register it in `predictors/__init__.py`.
  - Runtime methods may use only numpy + stdlib. State must serialise via `paiec.runtime` (numpy arrays, dicts, tuples, paiec objects).
  - `tests/test_harness.py` automatically checks batch = single, incremental = fresh, and the serialisation round trip for every registered predictor.
- **New acquisition policy.** Subclass `acquisition.Policy` and implement `decide(input, p, context, rate, labeled)`. `self._predictor` is available when bound: the replay binds it, and so does the submission's `labeling.py`. The base class enforces the fill rule. Register it in `POLICIES`.
- **New feature or field.** Add a named extractor to `features/item.py` or `features/subject.py` and list it in `ALL_FIELDS`. It can then be ablated with `--set predictor.item_fields=[text,kv]`.

## Things the harness does to keep results honest
- Benchmark ids in inputs are always anonymised. Training benchmarks never include the test fold.
- Prior variances in `irt` come from cross-benchmark residuals, not in-sample fit.
- Each checkpoint clears predictor caches, as the hosted worker restart would. `predict_batch` is used only for speed and is tested equal to `predict`.
- Revealed outcomes are one recorded trial per accepted item. Evaluation scores every trial of every evaluation item.

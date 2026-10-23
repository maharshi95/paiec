# Getting started: from a fresh clone to a scored model

This page takes you from `git clone` to a model of your own that is scored on the shared val split
and compared with the champion. The README has the short version. For the research history behind
the defaults, see `results.md` and `model.md`.

## 1. Environment

Python ≥ 3.10. The core dependencies are numpy, pandas, pyarrow, pyyaml and huggingface_hub.
```bash
git clone https://github.com/maharshi95/paiec && cd paiec
uv venv && source .venv/bin/activate          # or: python -m venv .venv / conda create -n paiec python=3.12
uv pip install -e ".[dev]"                    # + ".[ml]" for torch / scikit-learn / lightgbm, ".[report]" for plots
export PYTHONPATH=src                         # every command below runs from the repo root
python -m pytest -q                           # ~30 s, synthetic data only
```
Before running many jobs in parallel, set `OMP_NUM_THREADS=1` (`scripts/sweep.py` does this for
you). Otherwise BLAS oversubscription can make runs about 50× slower.

## 2. Data

```bash
hf auth login                                  # token with read access to the two (private) datasets
python scripts/download_data.py --split        # both halves -> data/aims/train, data/aims/val (2.3 GB)
```
The split datasets are private, because upstream measurement-db is gated with manual review. Ask the
maintainer (@mgor on HF) for read access. Alternatively, with approved access to
`aims-foundations/measurement-db`, rebuild byte-identical files locally:
```bash
hf auth login                                  # with approved access to aims-foundations/measurement-db
python scripts/download_data.py --config configs/split_seen/dev_base.yaml --root data/mdb \
    --revision 54074453da7b9911be0f48e4c560be547bc84d21    # 196 benchmarks, ~7 GB, ~2 min
python scripts/export_split.py --root data/mdb --out data/aims   # ~5 min -> data/aims/{train,val}, identical files
```
| | HF dataset | benchmarks | responses |
|---|---|---|---|
| train | `mgor/aims-measurement-db-train` | 166 (includes the 6 reference benchmarks) | 12.8M |
| val | `mgor/aims-measurement-db-val` | 30, in 5 groups of 6 | 1.7M |

Each benchmark directory has the upstream measurement-db layout and schema
(`{benchmark}/formatted_tables/{benchmarks,subjects,items,responses}.parquet`; columns are listed in
`dataset.md`). Each split's `split.json` records the benchmark list, the val `folds`, the
preprocessing and per-benchmark stats. `scripts/export_split.py` builds both splits from the full
measurement-db (revision `54074453`), keeping:
- responses the harness can binarise, with raw values kept. Graded scales are normalised to [0, 1]
  and thresholded at 0.5 at load time.
- about 100K responses per benchmark at most, by seeded subject subsampling that keeps models seen in
  the reference benchmarks first.
- the items that have a kept response, and the full subjects table.

Loading the export with `configs/aims/base.yaml` gives exactly the same `Dataset` as loading the full
measurement-db with `configs/split_seen/dev_base.yaml`. So numbers on the two are comparable, and
`configs/aims/irt_cv.yaml` reproduces the research loop's `R_champion_dev`.

**Why benchmark-level splits.** The hosted test asks about benchmarks that were never released. A
random row split would leak each benchmark's base rate and item difficulties, and local numbers would
come out far too optimistic. The val benchmarks also contain models seen in the reference benchmarks,
are diverse in domain, and are checked for duplicate items shared with train (`benchmark-selection.md`).
12 more benchmarks are a locked hold-out, kept out of both splits for milestone checks.

The full upstream dataset (281 benchmarks, gated) is available through
`python scripts/download_data.py --all --root <dir>`. You need it only to rebuild this split (above) or to build
new ones (`scripts/make_selection.py`).

## 3. What a run does

`python scripts/run.py --config configs/aims/<method>.yaml` performs these steps:
1. **Load.** It loads all benchmarks under `data.root` into a `Dataset`, which is cached as a pickle in
   `data/processed/` after the first load (that first load takes several minutes).
2. **Fit.** For each val group, it fits the predictor on `cv.train_benchmarks`, i.e. all of
   `data/aims/train`. Fits are content-addressed and cached in `data/fit_cache/` (or
   `$PAIEC_FIT_CACHE`), so a change that only affects test time skips the refit.
3. **Replay the hosted protocol** on each val benchmark of the group (`sim/replay.py`, `evaluation.md`):
   - every (subject, benchmark) pair with ≥80 distinct items is split 50/50 into an *acquisition* pool and
     an *evaluation* pool;
   - acquisition candidates stream in random order, and for each one your `acquisition_function`
     decides whether to reveal its label, up to 31 labels;
   - at budgets n ∈ {0, 1, 3, 7, 15, 31}, every evaluation item is scored with `predict(input, labeled)`.
     `labeled` is **global**: it holds the revealed labels of *every* pair, so a model can learn the
     unseen benchmark's base rate and item difficulties from other subjects.
4. **Score.** It computes the Brier score at each budget and
   `ALC = 0.1·B0 + 0.2·(B1 + B3 + B7 + B15) + 0.1·B31` per pair. Pairs are averaged, then seeds,
   then groups.

Benchmark ids are anonymised in every input (`bm_<hash>`). Never key a model on benchmark names.

**Outputs** (`runs/<name>/`):
- `metrics.json`: `alc` (headline: mean over the 5 groups), `alc_pairs` (mean over all pairs),
  `by_budget`, `ece_alc`, per-group `folds`, per-benchmark `benchmarks`, timing, data revision and git sha.
- `pairs.csv`: one row per (subject, benchmark, seed), with B0…B31, ECE, ALC and acquisition stats.
  Runs with the same seeds see identical pools, candidate orders and revealed trials, so
  `scripts/compare.py` can compute *paired* differences. Paired differences are much tighter than the
  gap between two headline numbers.
- `config.yaml`: the fully resolved config. Re-run any result with `--config runs/<name>/config.yaml`.

## 4. Protocol configs

| config | trains on | evaluates | use |
|---|---|---|---|
| `configs/aims/base.yaml` (+ method files) | train (166) | val, 5 groups × 3 seeds | **standard**: compare models here |
| `configs/aims/irt_cv.yaml` | train + the other 4 val groups | val, 5 groups × 3 seeds | the research loop's dev protocol (more training data) |
| `configs/aims/quick.yaml` | the 6 reference benchmarks | val group 1, 1 seed | smoke test: does a new model run? |

Every value can be overridden on the command line. Quote values that contain brackets, because zsh
globs them:
```bash
python scripts/run.py --config configs/aims/irt.yaml --set 'cv.seeds=[0]' --name aims_irt_s0
python scripts/sweep.py --config configs/aims/irt.yaml --grid predictor.beta_var_mult=2,4,8 --grid acquisition.radius=0.1,0.2 --jobs 4
```
All config keys are listed in `harness.md`.

**Resources.** A full-split run holds about 14M responses and the item text in memory, which takes
about 40–60 GB of RAM. The champion IRT fit takes about 10 min on 8 CPUs, and the replay another 5–10.
`quick.yaml` runs in seconds in under 1 GB. On a Slurm cluster, use `launch/research.slurm`; its header has usage.

## 5. Write a predictor

The contract is in `src/paiec/predictors/base.py`:
```
fit(train) -> self                 offline, on training benchmarks only; any library
adapt(labeled) -> state            test time: summarise the revealed labels (default: Evidence counts)
extend(state, new) -> state        optional incremental version of adapt
score(state, subject, item) -> p   test time: one probability
score_batch(state, inputs) -> p[]  optional vectorised score (must agree with score)
```
The official `predict(input, labeled)` is derived from these. Runtime methods (`adapt`, `extend`,
`score`) may use only numpy and the standard library, because the fitted object is pickled into the
submission ZIP and the hosted worker has no extra packages. Hosted workers also restart at every
checkpoint. The harness simulates this by clearing caches, so test-time state must be a function of
`labeled` alone.

**What `fit` sees.** `train` is a `paiec.data.dataset.Dataset`:
- `subjects[k]`, `items[k]`: the exact dicts `predict` receives, with all values as strings;
- `resp_s`, `resp_i`, `resp_y`: one entry per response, i.e. subject index, item index and binary label;
- `item_bench` / `resp_b` with `bench_names`: benchmark membership (fit time only);
- `restrict(benchmarks)`: a view on a subset of benchmarks.

Subject dict keys are `normalized_name`, `provider`, `release_date`, `access_date`, `harness`,
`harness_version`, `reasoning_effort` and `subject_features_extra`. Item dict keys are `item_content`,
`item_features`, `interactors` and `benchmark_id` (anonymised). Model names are inconsistent across
benchmarks, so use `features.subject.subject_identity` to match the same model across them.

**What `score` sees.** `state` is by default an `Evidence` object with O(1) lookups:
- `pair_counts(subject, item)` → (successes, n) for this subject on this benchmark;
- `bench_counts(item)` → (successes, n) over all subjects on this benchmark;
- `by_bench[benchmark_id]` → the list of revealed (subject_key, item_key, y, subject, item).

**Steps:**
1. Copy `src/paiec/predictors/starter.py` to `src/paiec/predictors/mine.py`, and rename the class and its
   `name = "mine"`. Every public class attribute is a hyperparameter that a config or `--set` can override.
2. Register the class in `src/paiec/predictors/__init__.py` (the `PREDICTORS` tuple).
3. Run `python -m pytest -q`. For every registered predictor, the tests check that
   `predict_batch == predict`, that incremental and fresh states agree, and that serialisation
   round-trips.
4. Run `python scripts/run.py --config configs/aims/quick.yaml --set predictor.name=mine` to see a number
   in seconds.
5. Create `configs/aims/mine.yaml` with `base: configs/aims/base.yaml`, `predictor: {name: mine, ...}`
   and an acquisition policy, then run it.
6. Compare: `python scripts/compare.py runs/aims_irt runs/aims_mine --budgets --folds`. A negative Δ
   means better. Trust a Δ whose 95% CI excludes 0 and that wins on most benchmarks.

To reuse the champion's pieces, `irt.py` contains item text features
(`features/item.py`: hashed tokens, length, key-value features), subject features (`features/subject.py`:
provider, family, size, "thinking", date), a cross-benchmark item-difficulty prior and a test-time MAP
update. `model.md` explains the formulation.

## 6. Write an acquisition policy

Policies live in `src/paiec/acquisition/__init__.py`. Subclass `Policy`, implement
`decide(input, p, context, rate, labeled) -> bool`, and add the class to `POLICIES`.
- `p` is the model's current prediction for the candidate.
- `rate = labels_remaining / items_remaining`.
- The base class already enforces the fill rule: it accepts when the remaining candidates are needed to
  reach 31 labels, and rejects once 31 are taken.
- When the policy is bound to a predictor (the replay and `labeling.py` bind it), `self._predictor` is
  available, so a policy can score reference items (see `Quantile`).

Select a policy with `acquisition: {name: ..., <params>}`.

## 7. Submit

```bash
python scripts/build_submission.py --config configs/aims/mine.yaml   # fits on all loaded benchmarks -> dist/aims_mine.zip
python scripts/check_zip.py dist/aims_mine.zip                       # quick local check
python scripts/time_predict.py --config configs/aims/mine.yaml       # latency vs the 8 h hosted limit
```
For the final ZIP, train on everything you are allowed to use. Then run the organizers' starting-kit
checks (`submission-format.md`) and upload to [Codabench](https://www.codabench.org/competitions/17828/).

## 8. Conventions for shared work
- Keep the val split fixed when comparing. Report paired Δ against a control run with the same seeds,
  and log batches in `results.md`.
- Never tune on the locked hold-out (`configs/research/holdout.yaml`; milestones only).
- Commit subjects follow Conventional Commits with a scope, e.g. `feat(irt): ...`.

# Research harness plan

Goal: iterate quickly on (predictor × acquisition × feature fields × data policy). Every run should produce
comparable, reproducible Brier-ALC scores under a local replay that mirrors the hosted protocol as closely as
we know it. The best configuration should package into a Codabench ZIP with one command.

## 1. Facts the design rests on

Sources: the competition page, plus the collaborator's `profiles/neurips/{replay,contract,metrics}.py` in
`../cs321m-predictive-eval`, which were written against the Codabench starting kit (SHA `a870970a…`).
Items marked (unverified) are their inference, not something the organizers state.

| Fact | Consequence |
|---|---|
| Acquisition is **streaming**. Candidates from a pair's acquisition pool arrive one at a time in random order, and each is accepted or skipped forever. `labels_remaining` = 31 − acquired, `items_remaining` includes the current candidate. | The policy is a stateless function of `(prediction, context)`. It can skip at most `\|acq pool\| − 31` items, and must accept everything once `items_remaining ≤ labels_remaining`. |
| Default policy: accept iff `u < labels_remaining / items_remaining`, with `u` a hash of (context, input). | This is selection sampling: a uniform random subset. "Accept all" also gives a uniform subset, because candidate order is random. |
| Checkpoints are nested. All pairs are advanced to budget n, then every evaluation input is scored. A **fresh predictor process** runs at each checkpoint. | `predict` must rebuild its state from `labeled` alone. Cache the adapted state, keyed by the evidence. |
| (unverified) `labeled` is **global**: it holds the revealed labels of every pair, not only the current (subject, benchmark). The baseline's filter by subject and benchmark only makes sense if so. | This is the biggest modeling lever. At budget n, an unseen test benchmark has about n × #subjects labels across subjects, and the same item can be labeled for one subject and scored for another. Benchmark offsets and item difficulties can be learned at test time. The simulator supports `evidence_scope: global | pair` so we can measure how much this matters. |
| Every trial is a scored target. Identical visible inputs get one prediction. One random trial's outcome is revealed per accepted item. | Brier is computed per response. Repeated trials give an irreducible noise floor. |
| Pairs need ≥ 80 distinct items. Pairs are weighted equally. Formative evaluation caps at 1,000 subject–item pairs. | Small pairs count as much as large ones. Skip room varies a lot by pair. |
| Hosted limit: 8 h total. No pip installs by default. Python 3.13 + torch/transformers stack (collaborator's notes, unverified). | Runtime code is stdlib + numpy only. Item featurization must be cheap or precomputed per unique item. |

Data facts (revision `54074453`):
- 6 public benchmarks: 4 binary, matharena (mixed), mmdocrag (fraction).
- 34 models appear in more than one benchmark by `normalized_name`, so a cross-benchmark subject prior is possible.
- `interactors` is always empty. `test_condition` appears only in mmdocrag.
- researchcodebench item text has a median of about 96K characters, so truncate.
- Fields often empty: `harness` 74%, `reasoning_effort` 90%, `normalized_name` 16%. Identity must also come from `subject_features_extra`.

## 2. Architecture

```
src/paiec/
  data/      dataset.py   tables -> Dataset (competition-format subject/item dicts + response arrays), binarisation policy, cache
  features/  subject.py, item.py, hashing.py  -- field-level extractors, each toggled by config ("which fields to use")
  predictors/ base.py (Predictor: fit / adapt / score; predict() and predict_batch() derived), registry
             constant, empirical_mean, beta_shrink (EB with pluggable prior), irt (amortised logistic IRT, global-evidence MAP)
  acquisition/ policies.py: official_random, accept_all, uncertainty, information (rate-adaptive), registry
  sim/       replay.py (streaming protocol), metrics.py (response-level Brier, pair-macro ALC, ECE)
  experiment.py  config -> LOBO folds -> fit -> replay -> runs/<name>/{config.yaml, metrics.json, pairs.csv}
  config.py      YAML + dotted overrides (--set predictor.alpha=8)
scripts/  run.py, sweep.py, leaderboard.py, time_predict.py, build_submission.py, download/profile
submission_template/model.py, labeling.py  -- thin shims that load state.pkl + the vendored paiec runtime
```

### Contracts
- **Predictor**
  - `fit(train: Dataset)` sees only training benchmarks.
  - `adapt(labeled) -> state` is pure and cached by an evidence key.
  - `score(state, subject, item) -> p`.
  - `predict(input, labeled)` is the official signature. `predict_batch` speeds up simulator evaluation and must agree with `predict`, which a test checks.
  - `state_dict()` / `from_state()` support packaging.
- **Acquisition policy**: `__call__(input, prediction, labeled, context) -> bool`, built from config. It must guarantee the fill rule.
- **Features**: `SubjectFeaturizer(fields=[...])` and `ItemFeaturizer(fields=[...])` return hashed sparse vectors. Every field is one named extractor, so an ablation is a config edit.
- **Anonymisation**: input dicts always carry a hashed `benchmark_id`. Predictors never see benchmark names, and true names exist only in fit-time metadata.

### Evaluation protocol (local)
- **Leave-one-benchmark-out** over the included benchmarks. Fit on the rest, replay on the held-out one.
- The headline is the mean of per-fold ALC. Also report a pair-weighted mean, per-budget Brier, ECE, and per-fold ALC.
- Seeds: `split_seed` fixes the pools, and `seed` sets candidate order and trial draws. Use 3 seeds by default.
- **Paired comparisons**: same seeds, same pairs, and Δ ALC per pair against a control run.
- Options:
  - `evidence_scope`: global or pair.
  - `max_subjects_per_benchmark`: mimics the platform's subject sampling.
  - `max_eval_items_per_pair`: for speed.
  - `binarize`: per-benchmark threshold, or exclude.

## 3. Model ladder
1. `constant`, `empirical_mean`: reproduce the baselines (sanity check: constant = 0.25).
2. `beta_shrink`: p = (k + αμ)/(n + α). The prior μ comes from a global constant, the subject's history on training benchmarks, or the benchmark's pooled global evidence. α is tuned.
3. `irt`: logit p = θ_pair − d_item.
   - d_item = ridge(item features), fitted to Rasch difficulties on training benchmarks (REEval-style amortisation). It is updated from global evidence on the same item.
   - θ_pair = ridge(subject features) + subject history + a benchmark offset β_b learned from global evidence. It is updated by MAP from pair labels.
   - Prior variances are estimated from training residuals (empirical Bayes).
4. Later: text embeddings via an encoder in `models.txt`, multidimensional IRT, and calibration on top of LOBO predictions.

## 4. Acquisition and runtime strategy
- Under random candidate order, *which* items to label only matters relative to the predictor. Uncertainty-based picks bias any predictor that ignores item difficulty (empirical mean, beta_shrink). Pair informative acquisition only with `irt`.
- Policies:
  - `accept_all`: the first n candidates. This is the cheapest option and fills budget n as early as possible.
  - `official_random`: the platform default.
  - `uncertainty(r)`: accept if |p − ½| ≤ r, with the fill guarantee.
  - `information(γ)`: accept if p(1−p)/¼ ≥ (1 − rate)^γ, where rate = labels_remaining / items_remaining. This loosens as skip room runs out.
- Runtime: `scripts/time_predict.py` measures per-call latency of `predict` and of the acquisition step, then projects hosted time as (#acq calls + 6 × #eval inputs) × latency. Rules:
  - Keep adaptation O(|labeled|) with a cache.
  - Precompute training-side artifacts.
  - Featurize each unique item once per process.

### Findings so far (docs/results.md)
- Under global evidence, uncertainty sampling (r = 0.15) beats accept_all by 0.004 ALC (paired, all folds). The information rule is similar. It is the IRT default now.
- Global versus pair evidence is worth 0.03 ALC, far more than any acquisition rule. Verify the hosted behaviour first.
- Next on acquisition: tune r per budget phase; use expected reduction in evaluation-pool Brier (needs item-prior variance); and, under global evidence, favour items that other subjects have labeled or will be scored on.

## 5. Order of work
1. Dataset + features + sim/replay + metrics + experiment runner + leaderboard. Tests: replay invariants, `predict_batch` = `predict`, acquisition fill rule.
2. Baselines → beta_shrink → irt. Acquisition sweep on the best predictor.
3. `build_submission.py` (fit on all benchmarks, vendor the runtime, check the ZIP), then the starting-kit checks.
4. Field ablations, `evidence_scope` sensitivity, write-up material (curves, per-benchmark tables).

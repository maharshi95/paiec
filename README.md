# paiec: a research harness for the Predictive AI Evaluation Competition (NeurIPS 2026)

[Competition](https://aims.stanford.edu/competition) ·
[Codabench](https://www.codabench.org/competitions/17828/) ·
data: [train](https://huggingface.co/datasets/mgor/aims-measurement-db-train) /
[val](https://huggingface.co/datasets/mgor/aims-measurement-db-val) ·
[getting started](docs/getting-started.md)

**The task.** Predict P(correct) for an AI model (a *subject*) on each item of a benchmark it has never been
seen on, from the subject's and item's visible attributes plus ≤31 labels that your own streaming
acquisition policy chooses to reveal. The score is **Brier ALC**: the mean Brier score over label budgets
0, 1, 3, 7, 15 and 31, weighted as an area under the learning curve. Lower is better, and a constant 0.5
scores 0.25. Deadline **2026-10-30 AoE** (code + report).

This repo replays the hosted protocol locally on held-out benchmarks. You can train a model, score it
on the same footing as everyone else's, compare it with the baselines, and package it as a Codabench
ZIP.

## Quickstart (about 10 minutes plus the download)

```bash
# 1. Environment (Python >= 3.10). Any venv / conda works; uv shown.
git clone https://github.com/maharshi95/paiec && cd paiec
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Data: the benchmark-level train/val split (2.3 GB parquet) -> data/aims/{train,val}
hf auth login                              # once; needs read access to the two datasets
python scripts/download_data.py --split

# 3. Smoke test: fits on 6 benchmarks, scores 6 val benchmarks (~10 s, <1 GB RAM)
export PYTHONPATH=src
python -m pytest -q
python scripts/run.py --config configs/aims/quick.yaml

# 4. Score the baseline and the current best model on the full val split
python scripts/run.py --config configs/aims/empirical_mean.yaml   # official baseline
python scripts/run.py --config configs/aims/irt.yaml              # champion
python scripts/compare.py runs/aims_empirical_mean runs/aims_irt  # paired Δ, 95% CI
```

A full-split run loads all 196 benchmarks. It needs about 40–60 GB of RAM and takes 15–30 min on 8 CPUs
(most of that is the first data load and the model fit, which are cached). Each run writes
`runs/<name>/{config.yaml, metrics.json, pairs.csv}`.

## The data split

| | HF dataset | benchmarks | responses | role |
|---|---|---|---|---|
| train | [`mgor/aims-measurement-db-train`](https://huggingface.co/datasets/mgor/aims-measurement-db-train) | 166 | 12.8M | fit your model here; includes the 6 reference benchmarks |
| val | [`mgor/aims-measurement-db-val`](https://huggingface.co/datasets/mgor/aims-measurement-db-val) | 30 | 1.7M | scored by the replay, in 5 groups of 6 (`split.json` → `folds`) |

The two splits share no benchmark, because the hosted test uses benchmarks never released for training.
Both are subsets of
[`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db)
(revision `54074453`) in the same layout and schema (`{benchmark}/formatted_tables/{benchmarks,subjects,items,responses}.parquet`).
They were produced by `scripts/export_split.py`. The val benchmarks are diverse, have ≥80 items per pair,
are guarded against item leakage, and contain models that also appear in the reference benchmarks
([docs/benchmark-selection.md](docs/benchmark-selection.md)).
12 further benchmarks are a locked hold-out kept for milestone checks and are in neither split.
Data licence: CC-BY-SA-4.0; individual benchmarks keep their own licences. Cite
*Truong, Truong, Koyejo. The AI Measurement Data Bank, 2026.*

## Results on the val split

`configs/aims/*.yaml` (3 seeds; ≤20 subjects per benchmark, ≤100 eval items per pair, ref6-seen models).
Lower is better.

RESULTS_TABLE

## Make your own model

1. Copy [`src/paiec/predictors/starter.py`](src/paiec/predictors/starter.py). It is a 40-line,
   commented predictor with `fit(train)` (offline, any library) and `score(evidence, subject, item)`
   (test time, numpy + stdlib).
2. Register the class in `src/paiec/predictors/__init__.py`. `python -m pytest -q` then checks the
   contract automatically: batch equals single, incremental equals fresh, and pickling round-trips.
3. Add a config, e.g. `configs/aims/mine.yaml`:
   ```yaml
   name: aims_mine
   base: configs/aims/base.yaml
   predictor: {name: mine, alpha: 4.0}     # class attributes are hyperparameters
   acquisition: {name: accept_all}         # or uncertainty / information / quantile / official_random
   ```
4. Score it and compare it with a control on matched pairs:
   ```bash
   python scripts/run.py --config configs/aims/quick.yaml --set predictor.name=mine   # does it run?
   python scripts/run.py --config configs/aims/mine.yaml
   python scripts/run.py --config configs/aims/mine.yaml --set predictor.alpha=8 --name aims_mine_a8
   python scripts/compare.py runs/aims_irt runs/aims_mine runs/aims_mine_a8 --budgets --folds
   python scripts/leaderboard.py --filter aims_
   ```
5. Package it for Codabench: `python scripts/build_submission.py --config configs/aims/mine.yaml`,
   which writes `dist/aims_mine.zip`.

The full walkthrough, covering the protocol, the `Dataset` object, the predictor and acquisition
contracts, reading `metrics.json`, sweeps, and cluster use, is in
**[docs/getting-started.md](docs/getting-started.md)**.

## Layout
    configs/aims/   protocol on the published split (base.yaml) + one file per method   <- start here
    configs/        older protocols: ref6 LOBO (base.yaml), split_seen/ (same split on the full db), research/
    src/paiec/
      data/         dataset.py (parquet tables -> competition-format dicts, binarisation, cache), synthetic.py
      features/     item.py, subject.py, identity.py: named field extractors, canonical model identity
      predictors/   base.py (contract + evidence cache), simple.py, starter.py, irt.py (champion), mirt.py
      acquisition/  streaming policies: official_random, accept_all, uncertainty, information, quantile
      sim/replay.py hosted-protocol replay (pools, streaming acquisition, nested checkpoints, scoring)
      experiment.py folds x seeds -> runs/<name>/{config.yaml, metrics.json, pairs.csv}
      runtime.py    JSON (de)serialisation + safe predict for the ZIP
    scripts/        run, sweep, compare, leaderboard, build_submission, check_zip, time_predict,
                    download_data, export_split, make_selection, reports
    submission_template/  model.py / labeling.py shims copied to the ZIP root
    launch/         Slurm / workstation job scripts
    data/ runs/ dist/ logs/   gitignored

## Docs
- [docs/getting-started.md](docs/getting-started.md): onboarding walkthrough (start here)
- [docs/README.md](docs/README.md): index of the research docs
- [docs/evaluation.md](docs/evaluation.md), [docs/submission-format.md](docs/submission-format.md): hosted protocol and ZIP interface
- [docs/model.md](docs/model.md): the champion model; [docs/results.md](docs/results.md): results log
- [docs/dataset.md](docs/dataset.md), [docs/validation.md](docs/validation.md), [docs/benchmark-selection.md](docs/benchmark-selection.md): data and splits

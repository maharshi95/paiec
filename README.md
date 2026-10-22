# PAIEC experiments: Predictive AI Evaluation Competition (NeurIPS 2026)

This is a research harness for <https://aims.stanford.edu/competition>. The task is to predict P(correct) for an AI
subject on an item from an unseen benchmark, given subject and item attributes and ≤31 labels revealed by a streaming
acquisition policy. The metric is **Brier ALC** over budgets 0, 1, 3, 7, 15 and 31; lower is better. The deadline is **30 Oct 2026 AoE** (code + report).

## Quickstart
```bash
hf auth login                                  # dataset is gated: accept its terms on HF first
python scripts/download_data.py                # ~130 MB, pinned revision written to data/raw/REVISION
export PYTHONPATH=src
python -m pytest -q
python scripts/run.py --config configs/irt.yaml          # leave-one-benchmark-out replay -> runs/irt/
python scripts/sweep.py --config configs/irt.yaml --grid predictor.var_scale=0.5,1,2 --jobs 4
python scripts/leaderboard.py --folds
python scripts/build_submission.py --config configs/irt.yaml   # -> dist/irt.zip
```

## Layout
    src/paiec/
      data/         dataset.py (tables -> competition-format dicts, binarisation, cache), synthetic.py
      features/     item.py, subject.py: one named extractor per field (ablate via config)
      predictors/   base.py (contract + evidence cache), simple.py (constant, empirical_mean, beta_shrink), irt.py
      acquisition/  streaming policies: official_random, accept_all, uncertainty, information
      sim/replay.py hosted-protocol replay (pools, streaming acquisition, nested checkpoints, per-trial scoring)
      experiment.py LOBO folds x seeds -> runs/<name>/{config.yaml, metrics.json, pairs.csv}
      runtime.py    JSON (de)serialisation + safe predict for the ZIP
    configs/        base.yaml + one file per method (configs inherit via `base:`)
    scripts/        run, sweep, leaderboard, time_predict, build_submission, download/profile, check_zip
    submission_template/  model.py / labeling.py shims copied to the ZIP root
    launch/         Slurm / workstation job script
    data/ runs/ dist/ logs/   gitignored

## Docs
- [docs/README.md](docs/README.md): docs index (start here)
- [docs/harness.md](docs/harness.md): how to run, compare and extend
- [docs/harness-plan.md](docs/harness-plan.md): design, protocol facts, model ladder, acquisition and runtime strategy
- [docs/results.md](docs/results.md): results log
- [docs/competition.md](docs/competition.md), [docs/evaluation.md](docs/evaluation.md), [docs/submission-format.md](docs/submission-format.md)
- [docs/dataset.md](docs/dataset.md), [docs/baselines.md](docs/baselines.md), [docs/literature.md](docs/literature.md)
- [docs/experiment-plan.md](docs/experiment-plan.md), [docs/open-questions.md](docs/open-questions.md)

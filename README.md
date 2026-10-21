# PAIEC experiments — Predictive AI Evaluation Competition (NeurIPS 2026)

Fast-experimentation workspace for <https://aims.stanford.edu/competition>: predict P(correct) of an AI
subject on an unseen benchmark item, given subject/item attributes and ≤31 revealed labels. Metric: **Brier ALC**
(budgets 0,1,3,7,15,31). Deadline **30 Oct 2026 AoE** (code + report).

## Docs (start here)
- [docs/competition.md](docs/competition.md) — task, rules, deadlines, links
- [docs/evaluation.md](docs/evaluation.md) — pools, budgets, Brier-ALC, caveats
- [docs/submission-format.md](docs/submission-format.md) — ZIP contents, `predict`/`acquisition_function` I/O, failure codes
- [docs/dataset.md](docs/dataset.md) — measurement-db schema, the 6 public benchmarks, gotchas
- [docs/baselines.md](docs/baselines.md), [docs/literature.md](docs/literature.md)
- [docs/experiment-plan.md](docs/experiment-plan.md), [docs/open-questions.md](docs/open-questions.md)

## Layout
    src/paiec/          metrics.py, simulate.py (local evaluator), data/{schema,load}.py, models/
    submission_template/ model.py + labeling.py that go to the ZIP root
    scripts/            download_data, profile_data, train (stub), build_zip, check_zip
    launch/             Slurm/workstation job scripts (hpc-submit / ws-submit)
    configs/            experiment configs;  data/ runs/ dist/ logs/ are gitignored

## Quickstart
    hf auth login                      # accept dataset terms on HF first (gated)
    python scripts/download_data.py    # ~130MB, no traces
    python scripts/profile_data.py
    PYTHONPATH=src python -m pytest
    make zip && make check             # builds dist/submission.zip from submission_template/

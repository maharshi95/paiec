# paiec: Predictive AI Evaluation Competition (NeurIPS 2026) research harness

Start with `docs/README.md` (the docs index), `docs/experiment-plan.md` (status and priorities) and `docs/results.md` (latest numbers).

## Key facts (sources in docs)
- **Task:** predict P(correct) for an (anonymous) unseen benchmark's items with `predict(input, labeled)`. Labels are revealed by a streaming `acquisition_function`. Metric: Brier ALC over budgets 0, 1, 3, 7, 15, 31; lower is better. Deadline **2026-10-30 AoE**.
- **`labeled` is global** (every pair's revealed labels), confirmed from the organizers' client. See `docs/submission-format.md`.
- **Data:** measurement-db has **281 benchmarks** (278 public), not 6. Core tables for all of them are in `/fs/clip-scratch/mgor/measurement-db` (15 GB, revision 54074453), with `inventory.csv` and `benchmark_report.csv`. `data/raw` holds only the 6 reference benchmarks.
- **Onboarding / published split:** `docs/getting-started.md`, `configs/aims/` (HF `mgor/aims-measurement-db-{train,val}`, built by `scripts/export_split.py`). Same data as the seen-dev split.
- **Default validation protocol:** `configs/split_seen/{dev,holdout}_*.yaml`. These are eval benchmarks with ref6-seen models, diverse and leakage-guarded (`docs/validation.md`). The hold-out is locked: milestones only.
- **Champion (2026-10-23):** `irt_item_lr` (`configs/research/champion.yaml`, `configs/aims/irt_item_lr.yaml`): the IRT below plus online within-benchmark item amortisation (`docs/research/item.md`) and per-benchmark subject-descriptor regression. Dev control run: `runs/stack__item_lr`.
- **Base model:** `configs/irt.yaml`, i.e. unidimensional amortised IRT with content-based item priors, global-evidence MAP and uncertainty acquisition (`docs/model.md`).

## Working conventions
- Run from the repo root with `export PYTHONPATH=src`. Tests: `python -m pytest -q`. Lint: `ruff check .`.
- Parallel runs: set `OMP_NUM_THREADS=1` (`scripts/sweep.py` does), or BLAS oversubscription makes everything ~50× slower.
- Never use `pkill -f <pattern>` from a shell whose own command line contains the pattern; it kills itself. Kill by PID.
- Large runs go on Slurm: `hpc-submit launch/run_cpu.slurm --job-name=X --export=ALL,CONFIG=configs/...`. Use the `clip` account and `qos=medium` (≤8 CPUs, ≤64 GB; `default` QoS caps at 4 CPUs / 32 GB).
- Record every experiment batch in `docs/results.md` with paired Δ vs a control (same seeds give matched pools).
- Commit subjects: Conventional Commits with scope, e.g. `feat(irt): ...`.

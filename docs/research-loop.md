# Research loop: climbing the headline metric

Goal: lower hosted Brier ALC (`evaluation.md`) before 2026-10-30. This is metric climbing, not a paper. Keep a variant if it
improves the headline by a clear margin; ablations can be reconstructed later from the runs.

## Roles
- **Orchestrator** (main session): owns `main`, the champion, the hold-out and this doc. It launches direction agents each round,
  merges winners, re-measures stacked combinations, and promotes the champion.
- **Direction agents**: one per research direction per round. Each works in its own git worktree and branch, runs experiments on
  Slurm, and returns a report (format below). They never touch `main`, the hold-out, or another agent's worktree.

## Headline protocol
- **Dev:** `configs/research/dev.yaml`. This is the seen-dev split (`validation.md`): 30 benchmarks in 5 grouped folds, ≤20
  ref6-seen subjects per benchmark, ≤100 eval items per pair, **seeds 0, 1, 2** (≈1,030 pair-seeds), global evidence.
- **Champion:** `configs/research/champion.yaml`, with its dev control run `runs/R_champion_dev` (in the main checkout).
- **Compare** (paired: same seeds give identical pools, orders and draws):
  `python scripts/compare.py /fs/clip-quiz/mgor/aims-neurips-2026/paiec/runs/R_champion_dev runs/<variant> --budgets --folds`
- **Promotion rule:** paired Δ ALC < 0 with the 95% CI excluding 0, fold-macro Δ < 0, and no hosted-runtime problem.
  A gain under 0.0005 is not worth complexity unless it is free.
- **Hold-out** (`configs/research/holdout.yaml`, `champion_holdout.yaml`): milestone check by the orchestrator only, at promotion.
  Never tune on it.

## Hosted constraints (hard)
- Runtime code paths (`adapt`, `extend`, `score*`, acquisition) use numpy and stdlib only. The fitted object must pickle
  (`paiec.runtime`). `tests/test_harness.py` checks batch = single, incremental = fresh, and serialisation for every registered
  predictor.
- 8 h total for the hosted run, which has up to about 1,000 pairs per formative run. Check latency with
  `python scripts/time_predict.py --config <cfg>`.
- `predict` must be a pure function of `labeled`, because workers restart at each checkpoint. Caches are fine only if keyed by
  evidence.
- Fit-time code may use anything installed (torch, sklearn, …). Only its outputs ship.

## Mechanics
- **Worktree:**
  `git -C /fs/clip-quiz/mgor/aims-neurips-2026/paiec worktree add /fs/clip-quiz/mgor/aims-neurips-2026/paiec-wt/<slug> -b exp/<slug> <base>`
- **Submit a job** from the worktree:
  `hpc-submit launch/research.slurm --job-name=<slug>__<v> --export=ALL,REPO=$PWD,CONFIG=<cfg>,ARGS="--set k=v --name <slug>__<v>"`
  - `CONFIG` may list several configs, which run serially.
  - Logs go to `paiec/logs/research/`.
  - Use the `clip` account, `qos=medium`, 8 CPUs.
  - Keep ≤8 of your own jobs in flight.
- **Run names** are `<slug>__<variant>`. Runs land in the worktree's `runs/`.
- **Fit cache** (`src/paiec/fitcache.py`): fitted predictors are cached per fold, keyed by fit params, fit code and the training
  set.
  - A variant that changes only test-time params (`RUNTIME_PARAMS`) or runtime methods skips the ~5 min/fold fit.
  - Declare new test-time-only params in your class's `RUNTIME_PARAMS`.
  - If your fit depends on code the key does not hash, bump a `FIT_VERSION` class attribute.
- **Quick checks:**
  - fewer seeds: `--set 'cv.seeds=[0]'`;
  - one fold: `--set 'cv.test_benchmarks=[[agentdojo,agi_elo,benger,medhelm_head_qa,phyblock,atmossci_bench]]'`.

  Headline claims need the full dev protocol.
- **Code organisation:** implement as a new predictor or policy class, or as optional flags that default off. Register it in the
  `__init__`. This keeps merges between directions trivial. Commit to your branch with Conventional Commits (`feat(scope): …`).

## Direction report (each agent's final message)
1. **Best variant:**
   - config (YAML overrides) and run name;
   - branch and commit;
   - paired Δ vs `R_champion_dev` (pairs Δ ± CI, fold-macro Δ, win %, per-budget Δ).
2. **Table of every variant tried**, with the same columns, including negative results.
3. **Insights:** mechanism, where the gain or loss comes from (budgets, benchmarks), and what the other directions should know.
4. **Next steps** ranked by expected gain.
5. **Runtime cost** (`time_predict.py`) if runtime code changed.

Agents also write their findings into `docs/research/<slug>.md` on their branch.

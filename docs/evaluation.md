# Evaluation protocol & metrics

## Label budgets and pools
- Each (subject, benchmark) pair with ≥80 distinct items is split **50/50 into acquisition and evaluation pools** (fixed across submissions and budgets).
- Budgets n ∈ {0, 1, 3, 7, 15, 31}. Labels are **nested** across budgets (31 labels ⊃ 15 ⊃ …). At most 31 labels per pair.
- Acquisition-pool outcomes are never scored; evaluation-pool outcomes are never revealed.
- Which acquisition items get revealed: your `acquisition_function` (called item by item; True = reveal, False = skip forever) else random.
- Every budget is scored on the **same evaluation items**.

## Primary metric: Brier ALC (lower better)
Brier score `B_n = mean (p - y)^2` at each budget, then area under the learning curve with x = log2(n+1) (equally spaced 0..5),
trapezoid normalised:

    ALC = 0.1·B0 + 0.2·B1 + 0.2·B3 + 0.2·B7 + 0.2·B15 + 0.1·B31

Averaged equally over subject–benchmark pairs. Perfect = 0; constant 0.5 = 0.25. Implemented in `src/paiec/metrics.py`.
Implication: B1…B15 carry 80% of the weight → **how fast you adapt from a few labels matters more than B0 or B31**; calibration of the
shrinkage from prior (feature-based) to empirical (labels) is the core modeling problem. Always output probabilities, avoid 0/1 extremes (Brier punishes confident errors).

Diagnostic (not ranking): ECE, 10 equal-width bins, count-weighted.

## Streaming acquisition protocol (from the starting kit, via the collaborator's replay; unverified by us)
- Per pair, acquisition candidates arrive one at a time in random order. For each candidate the platform calls `predict` to get `prediction`, then `acquisition_function`. True reveals one recorded trial's outcome; False skips the item for good.
- `context.labels_remaining` = 31 − acquired, regardless of the next checkpoint. `items_remaining` includes the current candidate.
- The default policy (no `labeling.py`) is hash-uniform `u < labels_remaining / items_remaining`, i.e. selection sampling.
- Checkpoints are nested. Every pair is advanced to budget n, then every evaluation input is scored. Workers restart at each checkpoint, so `predict` must rebuild from `labeled`.
- `labeled` is **global** (confirmed 2026-10-22 from the organizers' shipped client; see docs/submission-format.md §2). It contains every pair's revealed labels, so the model can learn the unseen benchmark's offset and item difficulties at test time. The harness still supports `sim.evidence_scope: pair` for ablation.
- Each repeated trial is a separate scored target, and identical visible inputs get one prediction.

## Caveats for local simulation
`src/paiec/sim/replay.py` approximates the hosted evaluator. Unknown: the exact seeds and odd-size rounding, how trials are drawn, which subjects are chosen, the 1,000-pair formative subsampling (not simulated), and the test benchmark distribution.
Treat local numbers as relative, and validate them against Codabench formative feedback.
**Cold start by benchmark**: validate with leave-benchmark-out CV, never random row splits. See `docs/harness.md`.

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

## Caveats for local simulation
`src/paiec/simulate.py` approximates the hosted evaluator. Unknown: exact random seed logic, how the acquisition function sequence is driven,
which subjects are chosen, test benchmark distribution. Treat local numbers as relative, and validate against Codabench formative feedback.
**Cold start by benchmark**: validate with leave-benchmark-out CV (never random row splits) — see `docs/experiment-plan.md`.

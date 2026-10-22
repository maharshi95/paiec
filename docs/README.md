# Docs index (one topic per doc; update the doc that owns the topic)

| doc | owns | update when |
|---|---|---|
| `competition.md` | task, rules, deadlines, links | the organizers change something |
| `evaluation.md` | hosted metric (Brier ALC), pools, budgets, streaming acquisition protocol | protocol facts change |
| `submission-format.md` | ZIP layout, `predict` / `acquisition_function` I/O, `labeled` / `context` semantics, hosted call flow (sourced) | interface facts are learned |
| `dataset.md` | measurement-db: layout, schema, local copies, binarisation, data gotchas | data facts are learned |
| `benchmark-selection.md` | tiers, quality filters, train pool, eval panel, selection experiments | selection criteria or results change |
| `validation.md` | local train/eval split configuration (LOBO, grouped folds, train sets, hold-out) | split design changes |
| `model.md` | current model formulation, limitations, model roadmap | the default model changes |
| `harness.md` | how to run, sweep, compare, extend; config keys | the harness interface changes |
| `harness-plan.md` | harness design rationale (architecture, contracts) | the architecture changes |
| `results.md` | **chronological results log** with paired Δs; the source of truth for numbers | after every experiment batch |
| `item-difficulty-plan.md` | item-difficulty exploration: findings, experiments, LLM-feature spec | item or text-feature work |
| `experiment-plan.md` | current status, priorities, deliverables checklist | priorities shift |
| `open-questions.md` | unresolved questions and resolved ones with sources | something is learned |
| `baselines.md`, `literature.md` | organizer baselines; related work | as needed |

Conventions:
- Every number in a doc cites a run name in `runs/` or a script.
- Use dates in absolute form (YYYY-MM-DD).
- Mark facts by source ([page], [client], [tests]) or as *inferred*.

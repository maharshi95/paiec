# Status and priorities (update as priorities shift)

Deadline: **2026-10-30 AoE** (code + 4–8 page report on OpenReview, plus the chosen Codabench submission ID).

## Current state (2026-10-22)
- Default model: `configs/irt.yaml` (`model.md`). Built ZIP: `dist/irt.zip`, which passes the organizers' validator. **Not yet submitted to Codabench.**
- Best numbers (`results.md`):
  - ref6-LOBO: ALC 0.1772 vs the official baseline 0.2328;
  - **ref6-seen split (default): dev 0.1844 vs 0.2269, hold-out 0.2113 vs 0.2689;**
  - random-model panel: dev 0.2129, hold-out 0.1966.
  - MIRT and subject sharing: no measurable gain (dev Δ within ±0.0006).
- Data: all 278 public benchmarks are local, with a train pool (207) and eval panel (57) (`benchmark-selection.md`).

## Research loop (started 2026-10-22; `research-loop.md`)
- Headline protocol: `configs/research/dev.yaml` (seen-dev, 3 seeds); champion `configs/research/champion.yaml`, control `runs/R_champion_dev`.
- Round 1 directions (worktrees `../paiec-wt/<slug>`, branches `exp/<slug>`): diag, hier, subj, item, acq, meta.
- After each round: merge winners, measure the stacked combination, promote, check the hold-out, and launch the next round.

## Priorities
1. **Diagnose the 14 dev benchmarks where IRT is worse than a constant 0.5** (offset/calibration). Improve model-identity resolution (it helps history priors, and would let sharing and MIRT matter).
2. **Submit to Codabench** with the pool-trained IRT, to get formative feedback and calibrate local-vs-hosted numbers.
3. **Model:**
   - kv-group and duplicate tying, plus robustness guards (`model.md` roadmap 1–2);
   - a multidimensional subject model now that there are many training benchmarks (roadmap 4);
   - training-set weighting by benchmark similarity.
4. **Item difficulty:** test Qwen3-32B difficulty scores end to end, then decide on a test-time LLM feature (`item-difficulty-plan.md` §9).
5. **Report:** method, ablations (`results.md`), data sources and disclosures.

## Deliverables checklist (per rules)
- [ ] Final ZIP chosen on Codabench, submission ID recorded
- [ ] Code released (no keys, no private data), pinned dataset revision, rerun instructions
- [ ] Technical report 4–8 pages (NeurIPS template): data sources, pretrained models, LLM-assistant disclosure, limitations, analysis
- [ ] Runs within 8 h, with no runtime downloads or network except allowed APIs

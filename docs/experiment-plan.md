# Experiment plan

Time left: ~9 days to the 30 Oct deadline (report + code). Prioritise a strong, simple, reproducible method over breadth.

## Validation protocol (local)
1. `scripts/download_data.py` → `data/raw` (pin revision).
2. Build records (`paiec.data.load.build_records`), keep binary benchmarks (+ decide binarisation for matharena/mmdocrag).
3. **Leave-one-benchmark-out**: train on others, evaluate on held-out benchmark with `paiec.simulate.evaluate_pair` per (subject, benchmark) pair with ≥80 items, several seeds for the acq/eval split. Report mean ± over benchmarks; also per-budget Brier and ECE.
4. Also leave-subject-out within benchmark (new model, known benchmark) as a secondary check — the test has *new benchmarks* but subjects are likely seen/new mix (unknown).

## Model ladder (stubs in `src/paiec/models/`)
0. `empirical_mean` (reproduce baseline; sanity check for simulator, expect ALC ≈ >0.2).
1. Smoothed empirical-Bayes: p = (k + a·μ_prior)/(n + a), prior μ from subject features (+ cross-benchmark mean for similar subjects), a tuned for ALC. Clip to [ε,1−ε].
2. Subject-level prior: parse model family/size/"think"/date from `subject_features_extra`; regress benchmark-agnostic skill; item difficulty from text embeddings (REEval-style) → Rasch/2PL logit: `p = σ(θ_subject − b_item)` with θ updated from acquired labels (MAP, Gaussian prior).
3. Multi-dim IRT / CAIMIRA-lite with text-derived item loadings; GBDT on (subject feats × item feats) as alternative prior.
4. Acquisition: uncertainty / expected-Brier-reduction vs random; check against `labeling.py` baseline (|p−0.5|<0.15).
5. Calibration layer (temperature/isotonic) fit on LOBO predictions at each budget.
6. (Optional) LLM-in-the-loop prior (API allowed, no browsing) — costly and slow vs 8 h limit; only if time.

## Deliverables checklist (per rules)
- [ ] Final ZIP chosen on Codabench, submission ID recorded
- [ ] Code released (no keys, no private data), pinned dataset revision, rerun instructions
- [ ] Technical report 4–8 pages, NeurIPS template: data sources, pretrained models, LLM-assistant disclosure, limitations, analysis
- [ ] Everything runs within 8 h, no runtime downloads/network except allowed APIs

## Run bookkeeping
Each run writes `runs/<name>/{config.json,metrics.json,git_sha,data_revision}`. Launch scripts: `launch/*.slurm` (Slurm via `hpc-submit`) and workstation `ws-submit` use the same scripts.

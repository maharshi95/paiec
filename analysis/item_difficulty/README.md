# Item-difficulty analysis

Scripts behind `docs/item-difficulty-plan.md`. Run them from this directory with the `aims26` env and
`OMP_NUM_THREADS=4`. Outputs go to `data/analysis/item_difficulty/` (gitignored).

| script | what it does | time |
|---|---|---|
| `common.py` | Fits Rasch targets with `paiec.predictors.irt.rasch`, the same construction as `IRT.fit`. `d` is centred per benchmark. Also writes the solve rate, response and subject counts, `d_se`, strong/weak-half difficulties and a 2PL discrimination, plus hidden metadata. Output: `targets.pkl`, `pairs.pkl`. | 15 s |
| `features.py` | Hand-crafted features. `visible_features()` uses only test-time fields (stdlib, `re` and numpy, so it is runtime-safe). `hidden_features()` reads the gold patch, the tests and the grading rule, for analysis only. Output: `features.pkl`. | 80 s |
| `explore.py` | Per-benchmark Spearman of each feature with d, η² for kv groups, prompt-template groups and the easiest/hardest items. Output: `feature_corr.csv`, `kv_groups.csv`, `extremes.txt`. | 10 s |
| `views.py` | Truncation views: head, tail, head+tail, chunks, and `focus`, a transductive boilerplate stripper based on within-benchmark line frequency. | – |
| `embed.py MODEL VIEW...` | Sentence embeddings on CPU, saved to `emb/<model>__<view>.npy`. The local 1080 Ti (sm_61) is not supported by the env's torch cu130. | 4–9 min per view at 14 threads |
| `lobo.py [--within] [--only K] [--tag T]` | Leave-one-benchmark-out difficulty prediction for every variant. Hyper-parameters and a shrink factor are chosen by an inner LOBO. Output: `<tag>.csv`, plus `_within.csv` with `--within`. | 5–40 min |
| `transductive.py [EMB]` | Test-time sharing within one benchmark (kv groups, duplicate content, kNN, ridge) from a fraction of noisily observed items. Output: `transductive.csv`. | 1 min |
| `e2e.py export FEATSET` / `e2e.py run NAME --set ...` | End-to-end harness replay with an `irt_lookup` predictor, registered at runtime without touching `src/`. Output: `runs/NAME`. | 6–10 min per run |
| `llm_pilot.py prepare / run / score`, `llm_pilot.slurm` | LLM feature-extraction pilot on 600 stratified items, using local Qwen3-8B under vLLM on one cluster GPU (`hpc-submit llm_pilot.slurm`). | ~10 min GPU |

Typical order: `python common.py && python features.py && python explore.py`, then
`python embed.py minilm head focus` (and other models), then `python lobo.py --within`, `python transductive.py`,
and `python e2e.py ...`.

Packages: `sentence-transformers` 6.2.0 (with transformers 5.20.2) was pip-installed into `aims26` for `embed.py`.
The LLM pilot uses the existing `llm-infer` env (vLLM 0.31).

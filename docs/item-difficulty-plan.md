# Item difficulty from text and metadata: findings, experiments, plan

Status 2026-10-22. Data revision `54074453`. matharena and mmdocrag are binarised at grade ≥ 0.5, as in
`configs/base.yaml`. Scripts are in `analysis/item_difficulty/` (see its README). Outputs are in
`data/analysis/item_difficulty/`. Everything in §1–§3 was **run and measured**. §4–§8 are **proposals**, except
where a number is quoted from a run.

## TL;DR
1. **Text features learned from other benchmarks barely transfer to a new benchmark.** Across ~45 LOBO variants, the
   mean held-out Spearman with the fitted d is between −0.06 and +0.10. These variants cover hashed BOW, TF-IDF,
   42 hand-crafted features, 5 sentence encoders × 5 truncation views, kNN and GBDT. All but one have negative
   centred-RMSE skill (the prior is worse than predicting the benchmark mean). The same features reach Spearman
   0.3–0.45 *within* a benchmark (5-fold CV). The signal exists, but it is benchmark-specific.
2. **Zero-shot LLM judgments are the only clearly transferable signal.** Qwen3-8B, without thinking, rated
   "difficulty 1–10" for every item. LOBO Spearman is **0.19** (all items), or **0.24** on a stratified 600-item
   pilot. Qwen3-32B-FP8 on the same pilot reaches **0.33**, and 0.58–0.60 on matharena and researchcodebench.
   Larger models help, so an API frontier model should do better still. Neither the LLM nor any text model predicts
   the **benchmark offset** (base rate): the LLM says "p(frontier solves) ≈ 0.8–0.9" for every benchmark.
3. **End-to-end, the current hashed-BOW item prior is worth about 0.005 ALC, and nothing tried beats it
   significantly.** The comparison uses the current default config (`item_target: absolute`, `beta_var_mult: 4`):
   ALC is 0.1772 with BOW and 0.1814 without item features; paired Δ = +0.0055 ± 0.0013 per pair.
   - BOW + the LLM-8B difficulty score: 0.1767, Δ −0.0003 ± 0.0007, not significant.
   - LLM-8B score alone: Δ +0.0041. hand_bz: +0.0042. BOW mean ×0.5: +0.0030. MiniLM-focus ridge: +0.0123.
   - Raw hand-crafted features with a global z-score were catastrophic (+0.105): features that are out of range on a
     new benchmark extrapolate wildly.

   **Robustness matters more than signal.** Standardise features within the benchmark, regularise strongly, and never
   let a single feature extrapolate.
4. **The pooled item-prior variance is already about right on average.** Scaling `var_d` by 0.4, 0.7 or 1.5 changes
   ALC by +0.0035, +0.0011 and +0.0007 (all worse or neutral). The per-benchmark direction matches the true spreads,
   though: mmdocrag (sd(d) = 1.3) prefers a smaller variance, researchcodebench (sd(d) = 2.65) a larger one
   (Δ −0.0018 at ×1.5). A per-benchmark τ_b learned at test time is cheap and principled, but the measured headroom
   is small (≈ −0.0005).
   - Caution: an earlier comparison mixed in a run made before the default config changed
     (`irt_baseline`, 0.1794). It made `var_d × 0.4` look like a big win on mmdocrag. That gain came from the config
     change, not from the variance.
5. **Within a benchmark, `item_features` groups and exact-duplicate content carry real difficulty structure.** The
   share of variance explained by groups (η²) is 0.45 for the matharena competition, 0.38 for the researchcodebench
   paper and 0.22 for the real_webagents website. Duplicated matharena problems (49% of items!) have d correlated at
   0.97 across copies. Kangaroo items are the exception: 336 items share one template text, and the problem is
   only in the image. These structures can be exploited at test time from global evidence: shrink each item toward
   the mean of its kv group, and tie duplicates whose text is not a template.

## 1. Targets (run: `common.py`)
Rasch is fitted with the harness's `rasch` (pairs = (subject, benchmark), cells aggregated over trials, prior var 9,
d centred per benchmark).

| benchmark | items | mean solve rate | sd(d) | median responses / item | median subjects / item | never solved | always solved | items with <5 subjects | median se(d) | Spearman(d_strong, d_weak) |
|---|---|---|---|---|---|---|---|---|---|---|
| matharena | 1755 | 0.70 | 2.79 | 36 | 9 | 8.5% | 35% | **28.5%** | 0.91 | 0.68 |
| mmdocrag | 2000 | 0.79 | 1.31 | 173 | 68 | 0% | 0% | 0% | 0.25 | 0.82 |
| multi_swebench | 2127 | 0.14 | 2.02 | 27 | 27 | **52%** | 0.1% | 0% | 1.41 | **0.18** |
| real_webagents | 233 | 0.45 | 2.21 | 12 | 12 | 12% | 12% | 4% | 0.82 | 0.67 |
| researchcodebench | 212 | 0.36 | 2.65 | 32 | 31 | 21% | 0% | 0% | 0.51 | 0.85 |
| swe_rebench | 6306 | 0.48 | 2.88 | 10 | **1** | 40% | 33% | 100% | 1.45 | n/a |

What can be trusted:
- **matharena** uses a sparse design: different competitions were run by different model sets. The raw solve rate
  is confounded with which models were tested. For example, aime_2025_I has a mean rate of 0.51 but d = −2.0
  (easy), because weak models were run on it. Use d, not the rate. 28.5% of matharena items have fewer than 5
  subjects. The analysis drops them; the "trusted" set is n_subj ≥ 5, plus all of swe_rebench.
- **multi_swebench**: 52% of items were never solved, so their d is set by the prior (se 1.4). Weak models solve
  almost nothing (d_strong vs d_weak ρ = 0.18), so for weak models the difficulty ranking is mostly noise.
- **swe_rebench**: one subject with about 10 trials per item, so d is single-model difficulty. It is useful as
  training data, not as a population difficulty.
- **real_webagents**: d correlates +0.19 with n_subj. The hard tasks (e.g. fly-unified) were run by more subjects.
  This is a mild design confound.
- 2PL discrimination: the median a is 1.05–1.3. On mmdocrag, discrimination falls with difficulty (ρ(a, d) = −0.46):
  hard mmdocrag items are noisy judge-graded items, not hard ones.

## 2. What makes items hard (run: `explore.py`; full table in `feature_corr.csv`)
Spearman(feature, d) on trusted items. The strongest visible signals per benchmark:

| benchmark | strongest visible signals (ρ with d) | hidden (training-only) signals |
|---|---|---|
| matharena | prompt template / competition (η² = 0.45); log length +0.43; n_sentences +0.44; "prove/show that/determine all" +0.36; mentions table/page +0.33; digit fraction −0.32 | proof-criterion item +0.36; reference answer length −0.36 (short integer answers are AIME-style) |
| mmdocrag | nothing above 0.17: n_count ("how many / sum of / total") +0.17, n_numbers +0.16, digit fraction +0.15 | reference answer length −0.21 |
| multi_swebench | nothing above 0.10; lang η² = 0.02 (java −0.98 easier, c/go +0.2) | **gold patch size +0.40**, hunks +0.37, test patch +0.31, files +0.30 |
| real_webagents | website (η² = 0.22; fly-unified +2.5, staynb +1.3, dashdish/omnizon −1.0); log length +0.38; n_dates +0.32; n_and +0.25 | – |
| researchcodebench | paper (η² = 0.38); **"Approximately N line(s) of code" in the TODO block, +0.54**; the paper/prompt-length features are *negative* (−0.1 to −0.3), because they encode which paper it is | reference code lines +0.53 |
| swe_rebench | nothing above 0.15 | patch lines +0.38, FAIL_TO_PASS count +0.29 |

Examples (d is in logits, + = harder):
- matharena, easiest: AIME 2025 #1, "sum of all integer bases b>9 for which 17_b divides 97_b" (d = −7.8, solved by
  43/43). Hardest: apex_2025 #12, the 2025×2025 tiling problem (d = +8.4); arxivmath "find the final answer"
  research problems (d ≈ +8). Mean d by template: "Put your final answer within \boxed{} … integer 0–999" −2.3;
  "Please reason step by step" −0.9; Kangaroo image template −0.1; "You are given a difficult question" +2.7;
  "Give a thorough answer … graded by human judges" +3.2; "write a proof" +3.5.
- real_webagents, easiest: "Omnizon: Buy me a silverware set", "NetworkIn: Connect with Olivia Martinez" (d ≈ −4).
  Hardest: multi-step create/reschedule tasks with exact times ("Create … July 19, 2024, from 2 PM to 2:30 PM …
  location"), drag-and-drop, compare-and-rank-all-listings (d ≈ +4).
- researchcodebench, easiest: one-line blocks in DyT and DiffusionDPO (d ≈ −5). Hardest: SISS importance-sampling
  weights, Tanh-Init (d ≈ +3.8).
- multi_swebench, hardest: a cluster of `sphinx-doc` docstring-rendering bugs (d ≈ +3.5), which look like ordinary
  issues. The text gives little signal; the gold patch size does.
- mmdocrag, hardest: questions that aggregate across a document ("Which projects had a duration of 3 years, and
  what were their allocated and spent amounts?", d = +4.1). Easiest: single-figure descriptive questions.

Transferable vs benchmark-specific:
- No visible feature has a consistent sign with |ρ| ≥ 0.1 across 4 or more of the 6 benchmarks. Length is positive
  on matharena and real_webagents, roughly zero on SWE and mmdocrag, and negative on researchcodebench.
- Signals that probably transfer *within a task family*:
  - solution size: gold-patch size on both SWE benchmarks (+0.4); "Approximately N lines" on researchcodebench;
  - the number of steps and constraints in agent tasks;
  - proof vs final answer in math.

  The text model cannot see most of these, but an LLM can estimate them (§3.3).
- Within-benchmark, transductive structure: kv groups (η² up to 0.45), prompt templates, and duplicated contents
  (matharena: 127 duplicate groups covering 695 items, d correlation 0.97 between copies once the 336-item Kangaroo
  template is excluded; multi_swebench: 44 groups, 0.66).

## 3. Experiments

### 3.1 LOBO difficulty prediction (run: `lobo.py`)
Protocol:
- Fit on the trusted items of 5 benchmarks and predict the 6th. The headline is the mean over the 5 harness test
  folds; swe_rebench is shown as an extra column.
- ridge α (and k for kNN) is chosen by inner LOBO on the training benchmarks. Training benchmarks are weighted
  equally unless marked "unweighted".
- **skill** = 1 − MSE/Var(d) after centring the predictions within the held-out benchmark. 0 means "predict the
  mean"; negative means harmful as a prior mean.
- **skill_shr** applies an inner-LOBO shrink factor c ∈ [0, 1] to the centred prediction.
- **offset** = RMSE of the implied benchmark-offset error over the 5 folds. The baseline is 2.08 logits.
- **within** = mean within-benchmark 5-fold CV Spearman (items grouped by content).

| variant | ρ mean5 | skill | skill_shr (c) | ρ matharena | ρ mmdocrag | ρ multi_swe | ρ real_web | ρ rcb | ρ swe_rebench | offset | within ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero (benchmark mean) | 0 | 0 | – | 0 | 0 | 0 | 0 | 0 | 0 | 2.08 | – |
| **bow_irt, ridge λ=10 unweighted (= current IRT)** | 0.060 | −0.052 | −0.018 (0.53) | 0.25 | −0.07 | 0.09 | 0.14 | −0.12 | 0.11 | **1.92** | – |
| bow_irt, ridge | 0.067 | −0.239 | 0.002 (0.14) | 0.14 | 0.00 | 0.04 | 0.08 | 0.09 | 0.06 | 2.25 | 0.29 |
| bow text only | 0.048 | −0.308 | 0.003 | 0.06 | 0.01 | 0.06 | 0.10 | 0.00 | 0.04 | 2.13 | 0.39 |
| length (4 stats) | −0.011 | −0.022 | −0.011 | 0.03 | −0.02 | 0.01 | 0.08 | −0.16 | 0.14 | 2.09 | 0.09 |
| hand-crafted (42), ridge | −0.019 | −0.132 | −0.039 | 0.06 | 0.01 | 0.07 | 0.01 | −0.24 | 0.03 | 2.70 | 0.28 |
| hand-crafted, unweighted | **0.100** | −0.084 | −0.027 | 0.24 | −0.04 | 0.10 | **0.35** | −0.15 | 0.12 | 2.38 | 0.28 |
| hand_bz (z-scored within benchmark) | 0.048 | −0.147 | −0.022 | 0.25 | −0.06 | 0.05 | 0.29 | −0.28 | 0.08 | 2.10 | 0.28 |
| hand_bz, z-scored target | 0.060 | −0.144 | −0.022 | 0.28 | −0.06 | 0.05 | 0.28 | −0.24 | 0.09 | 2.10 | 0.28 |
| hand, GBDT | 0.077 | −0.086 | −0.012 | 0.28 | 0.01 | 0.06 | 0.18 | −0.14 | 0.10 | 1.93 | – |
| hand_bz, target = d_weak | 0.074 | −0.012 | −0.006 | 0.32 | −0.04 | 0.02 | 0.18 | −0.12 | 0.03 | 2.09 | – |
| hand_bz, target = d_strong | 0.050 | −0.086 | −0.008 | 0.28 | −0.04 | 0.05 | 0.24 | −0.28 | 0.02 | 2.11 | – |
| bow_irt, absolute target (offset in target) | 0.074 | −0.081 | −0.002 | 0.24 | −0.02 | 0.02 | 0.10 | 0.03 | 0.07 | 2.08 | 0.39 |
| hand, GBDT, absolute target | −0.046 | −0.125 | −0.002 | 0.01 | 0.04 | 0.06 | −0.07 | −0.27 | 0.01 | 2.43 | – |
| TF-IDF word 1–2g, head 4k | −0.055 | −0.080 | 0.000 | −0.04 | −0.09 | 0.11 | −0.10 | −0.16 | 0.05 | 2.00 | ~0* |
| TF-IDF char 3–5g, head 4k | 0.007 | −0.475 | −0.023 | 0.10 | −0.06 | 0.11 | 0.04 | −0.15 | 0.06 | 1.99 | ~0* |
| TF-IDF word, head+tail 4k | −0.056 | −0.446 | −0.015 | −0.01 | −0.09 | 0.11 | −0.03 | −0.26 | 0.05 | 2.09 | ~0* |
| TF-IDF word, focus view 4k | −0.048 | −0.014 | 0.000 | −0.07 | −0.07 | 0.10 | −0.08 | −0.12 | 0.06 | 2.03 | ~0* |
| MiniLM-L6, head 2k, ridge | −0.021 | −0.485 | −0.005 | −0.06 | −0.01 | 0.07 | −0.03 | −0.09 | 0.03 | 2.44 | 0.31 |
| MiniLM, tail 2k | 0.052 | −0.416 | −0.004 | 0.00 | 0.01 | 0.06 | 0.04 | 0.15 | 0.04 | 2.08 | – |
| MiniLM, head+tail | −0.006 | −0.476 | −0.006 | −0.01 | 0.00 | 0.06 | −0.03 | −0.06 | 0.03 | 2.44 | – |
| MiniLM, chunk-mean (8×2k) | −0.005 | −0.447 | −0.004 | −0.01 | 0.02 | 0.07 | −0.04 | −0.06 | 0.01 | – | – |
| MiniLM, focus | −0.021 | −0.442 | −0.001 | −0.02 | 0.01 | 0.00 | 0.04 | −0.13 | 0.00 | 1.99 | 0.34 |
| MiniLM focus, kNN | 0.020 | −0.226 | −0.004 | −0.04 | −0.02 | 0.02 | 0.08 | 0.07 | 0.03 | 2.25 | 0.15 |
| bge-small, focus | −0.041 | −0.420 | −0.005 | −0.14 | −0.03 | 0.05 | −0.02 | −0.06 | 0.04 | 2.20 | 0.35 |
| bge-small, head, kNN | 0.033 | −0.365 | −0.002 | −0.10 | −0.04 | 0.06 | 0.06 | 0.17 | 0.03 | 2.23 | 0.27 |
| e5-small, focus | 0.048 | −0.489 | −0.003 | 0.14 | 0.00 | 0.03 | 0.08 | −0.01 | 0.03 | 2.28 | 0.33 |
| e5-small, focus, kNN | 0.083 | −0.099 | 0.004 | 0.12 | −0.03 | 0.06 | 0.17 | 0.09 | 0.05 | 2.24 | 0.33 |
| e5-small focus + hand_bz | 0.053 | −0.494 | −0.007 | 0.22 | 0.01 | 0.06 | 0.09 | −0.11 | 0.05 | 2.17 | 0.36 |
| bge-base (768-d), focus | 0.022 | −0.419 | −0.009 | 0.19 | −0.05 | 0.06 | −0.03 | −0.05 | 0.07 | – | – |
| bge-base focus + hand_bz | 0.044 | −0.534 | −0.012 | 0.27 | −0.07 | 0.09 | 0.04 | −0.12 | 0.08 | – | – |
| *oracle* hand_bz + hidden (patch size etc.) | 0.032 | −0.245 | −0.098 | 0.21 | −0.16 | **0.36** | 0.25 | −0.50 | **0.35** | 2.11 | 0.46 |
| **LLM: Qwen3-8B `difficulty_1_10` only** | **0.190** | **+0.015** | 0.000 (0.53) | 0.27 | −0.04 | 0.06 | 0.24 | **0.42** | 0.03 | 2.09 | – |
| LLM: Qwen3-8B, 10 extracted features | 0.171 | −0.037 | −0.021 | 0.20 | −0.12 | 0.10 | 0.23 | 0.45 | 0.10 | 2.09 | 0.18 |
| LLM 8B features + hand_bz | 0.068 | −0.111 | −0.034 | 0.32 | −0.08 | 0.08 | 0.18 | −0.16 | 0.10 | 2.13 | 0.36 |

\* The within-benchmark TF-IDF numbers reuse the LOBO-selected α, which is very large, so they are not meaningful.

On the 600-item stratified pilot (100 trusted items per benchmark, de-duplicated), Spearman with d:

| LLM signal (pilot) | matharena | mmdocrag | multi_swe | real_web | rcb | swe_rebench | mean5 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B `difficulty_1_10` (raw, zero-shot) | 0.51 | −0.01 | 0.01 | 0.16 | 0.50 | 0.19 | 0.24 |
| Qwen3-8B, 10 features, LOBO ridge | 0.43 | −0.14 | −0.01 | 0.19 | 0.52 | 0.23 | 0.20 |
| **Qwen3-32B-FP8 `difficulty_1_10`** | **0.58** | 0.09 | 0.09 | 0.29 | **0.60** | 0.25 | **0.33** |
| Qwen3-32B, 10 features, LOBO ridge | 0.53 | 0.10 | 0.11 | 0.31 | 0.60 | 0.23 | 0.33 |
| Qwen3-32B `expert_minutes` | 0.62 | 0.00 | 0.02 | 0.17 | 0.56 | 0.20 | 0.27 |
| Qwen3-32B `needs_unseen_inputs` | 0.05 | **0.48** | 0.13 | 0.25 | 0.01 | 0.12 | – |
| hand_bz, LOBO ridge (same items) | 0.01 | 0.08 | −0.07 | 0.27 | −0.34 | 0.10 | −0.01 |

The benchmark-level mean of `p_solve_frontier` (32B) against the true mean rate gives a Spearman of −0.2 over the
6 benchmarks. The LLM thinks multi_swebench is solved 80% of the time; the true rate is 15%.

Readings:
- **Truncation and views do not matter cross-benchmark**: head, tail, head+tail, chunk-mean and focus are all ≈ 0.
  Within a benchmark it depends:
  - matharena: the transductive **focus** view (lines rare within the benchmark ±10 lines of context) wins, 0.58 vs
    0.38 for head, because it strips the shared instruction templates;
  - researchcodebench: head + hand features is better (0.64 vs 0.34–0.47 for focus). The focus view does isolate
    the TODO block: a 100 KB paper prompt reduces to the ~600-char masked function plus "# Approximately N line(s)
    of code", which is what an LLM should read. But it drops the paper identity, which carries η² = 0.38 within rcb.
- **Bigger encoders do not help cross-benchmark** (bge-base ≈ bge-small ≈ MiniLM). This is not a capacity problem:
  with 5 training benchmarks, the map from text to difficulty does not generalise.
- **Code-aware signals**: patch size, the real code-complexity signal, transfers between the two SWE benchmarks
  (oracle ρ 0.36–0.42) but hurts elsewhere. Visible proxies (paths, identifiers, stack traces, code fences) carry
  nothing (|ρ| ≤ 0.1). An LLM estimate of `expected_solution_size` reached ρ 0.25–0.26 on swe_rebench and 0.58–0.62
  on rcb, but only 0.05 on multi_swebench.
- **Difficulty-aware targets**: training on d_weak (the weak half's failure logit) gives slightly better LOBO than d
  (skill −0.01 vs −0.15, ρ 0.07 vs 0.05). That is not material.
- **Equal benchmark weighting** (swe_rebench has 6.3K of the 12.6K items) hurts ρ for hand-crafted features (0.10
  unweighted vs −0.02). Without it, the model learns swe_rebench-specific patterns; with it, it learns
  researchcodebench-specific ones. Neither generalises.

### 3.2 Within-benchmark transductive sharing (run: `transductive.py`)
Simulation: a fraction f of a benchmark's items have an observed d + N(0, σ²). Predict the rest. The metric is
skill = 1 − MSE/Var. Rows show σ = 1.5 (about a few labels per item); σ = 0 and σ = 3 are in `transductive.csv`.

| f | method | matharena | mmdocrag | multi_swe | real_web | rcb |
|---|---|---|---|---|---|---|
| 0.05 | kv-group mean (shrunk) | **0.19** | 0.00 | −0.07 | −0.03 | −0.10 |
| 0.05 | same content, else kv group | 0.20 | 0.00 | −0.07 | −0.03 | −0.10 |
| 0.05 | kNN-10 on MiniLM focus | 0.08 | −0.03 | −0.07 | −0.04 | −0.21 |
| 0.05 | ridge on embeddings | 0.07 | 0.04 | −0.01 | −0.04 | −0.13 |
| 0.20 | kv-group mean | **0.32** | −0.01 | −0.00 | 0.05 | 0.09 |
| 0.20 | same content | **0.35** | −0.01 | −0.00 | 0.05 | 0.09 |
| 0.20 | kNN-10 | 0.18 | −0.00 | −0.01 | 0.03 | −0.00 |
| 0.20 | ridge on embeddings | 0.23 | **0.07** | 0.03 | 0.04 | 0.01 |

The shrinkage in this simulation is crude (fixed pseudo-count 3, no noise-aware weighting), so the small negative
values show over-trusting noisy neighbours, not true harm. At test time, under global evidence, an unseen
benchmark with S subjects has roughly 1 − (1 − n/(N/2))^S of its items labeled at budget n. For multi_swebench
(S = 82, N = 2127) that is ≈ 7% at budget 1 and ≈ 42% at budget 7, with about 1 binary label each (σ ≫ 1.5).
Sharing therefore matters mainly where groups are strong (matharena competitions, rcb papers, webagents sites) and
from budget 3 onward.

### 3.3 End-to-end replay (run: `e2e.py`; 5 folds × 3 seeds, ≤200 eval items per pair, uncertainty(0.15) acquisition)
`irt_lookup` is IRT with its item featurizer swapped for a precomputed table. `item_shrink` scales the ridge
weights; `var_d_mult` scales the item prior variance. These are runtime registrations in `analysis/`; `src/` is
not modified. As a check, `lk_bow_raw` (the BOW table through the lookup) reproduces stock `irt` exactly
(Δ = 0.0000).

**Config.** Every row below uses the *current* `configs/irt.yaml` defaults: `item_target: absolute`,
`beta_var_mult: 4`, commit 3b601aa, which landed while this analysis was running. Δ is the paired mean over
(subject, benchmark, seed) pairs against `abs_irt` (n = 864), with a 95% CI; negative is better. ALC is the
mean over folds. The paired summary is in `data/analysis/item_difficulty/e2e_paired.txt`.

| run | ALC | Δ vs BOW (paired) | B0 | B1 | matharena | mmdocrag | multi_swe | real_web | rcb |
|---|---|---|---|---|---|---|---|---|---|
| **irt, hashed BOW prior (`abs_irt`)** | **0.1772** | 0 | 0.255 | 0.197 | 0.1852 | 0.1688 | 0.1621 | 0.1777 | 0.1921 |
| no item features | 0.1814 | +0.0055 ± 0.0013 | 0.266 | 0.201 | 0.1935 | 0.1708 | 0.1709 | 0.1800 | 0.1919 |
| **BOW + LLM-8B `difficulty`** (1029 cols, λ=10) | **0.1767** | −0.0003 ± 0.0007 | 0.254 | 0.196 | 0.1840 | 0.1698 | 0.1621 | 0.1777 | 0.1900 |
| BOW + LLM-8B, mean ×0.5 | 0.1773 | +0.0010 ± 0.0009 | 0.257 | 0.197 | 0.1877 | 0.1691 | 0.1648 | 0.1774 | 0.1872 |
| BOW + LLM-8B, var_d ×0.6 | 0.1773 | +0.0006 ± 0.0007 | 0.256 | 0.195 | 0.1863 | 0.1701 | 0.1627 | 0.1752 | 0.1922 |
| BOW, mean ×0.5 | 0.1797 | +0.0030 ± 0.0010 | 0.273 | 0.200 | 0.1881 | 0.1633 | 0.1727 | 0.1822 | 0.1922 |
| BOW, var_d ×0.4 | 0.1814 | +0.0035 ± 0.0010 | 0.256 | 0.200 | 0.1895 | 0.1689 | 0.1649 | 0.1809 | 0.2031 |
| BOW, var_d ×0.7 | 0.1784 | +0.0011 ± 0.0007 | 0.255 | – | 0.1864 | 0.1691 | 0.1633 | 0.1767 | 0.1963 |
| BOW, var_d ×1.5 | 0.1775 | +0.0007 ± 0.0008 | 0.253 | – | 0.1853 | 0.1710 | 0.1632 | 0.1777 | 0.1903 |
| LLM-8B `difficulty` only | 0.1799 | +0.0041 ± 0.0012 | 0.265 | 0.200 | 0.1924 | 0.1694 | 0.1698 | 0.1784 | 0.1897 |
| LLM-8B `difficulty`, mean ×0.5 | 0.1804 | +0.0046 ± 0.0012 | 0.265 | 0.202 | 0.1919 | 0.1710 | 0.1700 | 0.1798 | 0.1895 |
| LLM-8B `difficulty`, var_d ×0.6 | 0.1800 | +0.0040 ± 0.0013 | 0.266 | 0.200 | 0.1941 | 0.1677 | 0.1682 | 0.1784 | 0.1918 |
| LLM-8B, 10 features, λ=100 | 0.1801 | +0.0048 ± 0.0012 | 0.265 | 0.200 | 0.1946 | 0.1729 | 0.1678 | 0.1772 | 0.1881 |
| hand_bz (42, z-scored per benchmark), λ=10 | 0.1814 | +0.0042 ± 0.0009 | 0.267 | 0.202 | 0.1886 | 0.1717 | 0.1676 | 0.1764 | 0.2028 |
| hand_bz, λ=1000 | 0.1810 | +0.0038 ± 0.0009 | 0.267 | 0.201 | 0.1890 | 0.1701 | 0.1669 | 0.1766 | 0.2027 |
| hand (global z-score), λ=10 | 0.3039 | +0.1050 ± 0.0119 | 0.293 | 0.318 | 0.3992 | 0.1624 | 0.1647 | 0.1788 | 0.6146 |
| MiniLM focus (384-d), λ=10 | 0.1887 | +0.0123 ± 0.0016 | 0.269 | 0.213 | 0.2041 | 0.1929 | 0.1618 | 0.1922 | 0.1926 |
| MiniLM focus, λ=3000 | 0.1818 | +0.0049 ± 0.0012 | 0.262 | 0.202 | 0.1934 | 0.1771 | 0.1626 | 0.1856 | 0.1901 |
| *old default (`item_target: centered`, `beta_var_mult: 1`)*: irt BOW | 0.1794 | +0.0032 ± 0.0010 | 0.267 | 0.200 | 0.1849 | 0.1797 | 0.1658 | 0.1774 | 0.1889 |
| *old default*: no item features | 0.1818 | +0.0064 ± 0.0013 | 0.274 | 0.204 | 0.1946 | 0.1774 | 0.1686 | 0.1767 | 0.1915 |

Readings:
- **BOW earns its place.** Under both configs it beats "no item features": by 0.0055 under the new default and by
  0.0024 in fold-mean ALC (0.1794 vs 0.1818) under the old one. That is despite its weak LOBO ρ (0.06). Most of the gain is on
  matharena and multi_swebench.
  - Part of BOW's edge is a mean shift that acts like an offset correction. On rcb, the BOW prior's mean is shifted
    by +1.5 logits relative to the centred baseline, which cuts the implied offset error from −1.27 to +0.27 (LOBO,
    centred target). With 5 training benchmarks we cannot tell whether such shifts generalise.
- **The LLM-8B difficulty score adds a little on top of BOW** (−0.0003 ± 0.0007, rcb −0.0021, matharena −0.0012),
  but the result is not significant. Alone, the LLM score recovers about a quarter of BOW's value (0.0014 of 0.0055).
  The 32B score (pilot ρ 0.33 vs 0.24) has not been tested end to end yet; job 7956248 is running.
- **Scaling the item prior mean or variance uniformly does not help.** Mean ×0.5 is +0.0030. var_d ×0.4/0.7/1.5 is
  +0.0035/+0.0011/+0.0007. The pooled LOBO var_d is well calibrated on average. Per benchmark, the sign follows the
  true spread (mmdocrag prefers smaller, rcb larger), so a per-benchmark τ_b could gain about 0.0005.
- **Dense, weakly regularised priors hurt**: MiniLM at λ=10 (+0.012), and hand-crafted features with a global
  z-score (+0.105, from extrapolation on matharena and rcb). Any new item prior needs within-benchmark
  standardisation, strong ridge, and clipping of the prior mean (e.g. to ±2 sd of the training d).
- Fold-to-fold differences (±0.01) are larger than most effects. Use paired Δ with CIs, never single-fold ALC.

## 4. Recommendations, ranked
Expected impact is a guess anchored on the numbers above (Δ ALC, negative = better). Cost is hosted runtime.

| # | change | evidence | expected Δ ALC | runtime cost |
|---|---|---|---|---|
| 1 | **Keep the hashed-BOW prior; add a z-scored, clipped LLM difficulty column** (`difficulty_1_10` from the strongest affordable model), fitted jointly with BOW | BOW = −0.0055 vs no prior; BOW+LLM-8B −0.0003 n.s.; LLM is the only signal that transfers (LOBO ρ 0.19 for 8B, 0.33 for 32B on the pilot) | −0.0003 measured (8B); −0.001 to −0.003 hoped for with 32B / Claude (untested) | API: ≈ $1.2 per 10K items with Haiku 5.5; local 8B ≈ 8 items/s on an A6000 |
| 2 | **kv-group and duplicate tying within the test benchmark** (d_i = prior_i + g_{b,kv(i)} + e_i, with group effects learned from evidence and shrunk; tie identical `item_content` unless it is shared by >5 items, i.e. a template) | η² 0.45 / 0.38 / 0.22; duplicate corr 0.97; transductive skill 0.2–0.35 on matharena | −0.001 to −0.004 (mostly matharena, rcb, real_web; B3+), untested | negligible |
| 3 | **Robustness guards for any item prior**: standardise dense features within the benchmark pool, clip prior means to the training range, ridge λ chosen by LOBO | global-z hand features +0.105; MiniLM λ=10 +0.012 | protects against catastrophic folds on unseen benchmark types | none |
| 4 | **Per-benchmark item spread τ_b** (d_i ~ N(prior_i, τ_b²), log τ_b ~ N(log √var_d, 0.5²), MAP from evidence; marginalise with τ_b) | uniform scaling is neutral to worse, but the per-benchmark direction matches sd(d) (mmdocrag 1.3, rcb 2.65) | ≈ −0.0005 | negligible |
| 5 | Stage-2 transductive text model: within-benchmark ridge/kNN on BOW or focus-view embeddings, fitted on test-time evidence and gated by evidence count | within-benchmark ρ 0.3–0.6 with clean labels; unclear with budget-1 noise | 0 to −0.002 at B15/B31 | small (numpy ridge on ≤ 2K items) |
| 6 | Discrimination / 2PL | a ∈ [1.0, 1.3]; on mmdocrag a falls with d (−0.46) | small; skip unless (1)–(2) are done | – |

Not recommended:
- Sentence embeddings or hand-crafted features as cross-benchmark item-prior features: ≈ 0 in LOBO and +0.004 to
  +0.012 worse end to end. Embeddings stay useful only within a benchmark (stage 2, #5).
- TF-IDF variants, chunk-mean, larger encoders (bge-base) and GBDT on hand features: all ≈ 0 or worse in LOBO.
- Uniformly shrinking the BOW prior mean (×0.5: +0.0030) or rescaling var_d.
- Predicting d_strong / d_weak separately as the difficulty target: no gain.
- Benchmark-offset prediction from item text: no variant beat the 2.08-logit baseline meaningfully. The offset has
  to come from labels (budget ≥ 1, global evidence) or from a calibrated hedge at B0 (`beta_var_mult`, already in
  `irt.py`).

## 5. Implementation spec

### 5.1 `IRT` changes (src/paiec/predictors/irt.py; for the main session to implement)
New class attributes (config keys `predictor.*`):

Ordered by priority (§4). #1 needs no IRT change beyond an extra feature column (§5.2 `llm`).

| key | default | meaning |
|---|---|---|
| `item_clip` | 2.0 | clip the prior mean to ± `item_clip` × sd(d_train) around the training mean (robustness guard, #3) |
| `item_tau` | `"pooled"` | `"pooled"` (current) or `"bench"`: learn log τ_b per benchmark in `_fit_bench` |
| `tau_prior_sd` | 0.5 | sd of log τ_b around log √var_d (LOBO); fitted on training benchmarks by LOBO |
| `item_shrink` | 1.0 | multiply the centred prior-mean deviation (Xw − mean over the benchmark pool) by c; choose c by inner LOBO in `fit` (code: `lobo.inner_select`) |
| `item_group` | `[]` | kv keys whose values define within-benchmark groups (e.g. all keys except numeric ones; default: every key whose value is non-numeric) |
| `group_var` | fitted | prior variance of group effects, from the η² on training benchmarks (≈ 0.2–0.45 × var_d) |
| `tie_duplicates` | true | items with identical `item_content` share a difficulty unless the content occurs in > `template_max` = 5 items |
| `item_std` | `"bench"` | standardise dense item features within each benchmark pool (transductive) instead of globally |

Changes in `_fit_bench`:
- d_i ← prior_i + g_{kv(i)} + e_i, where e_i ~ N(0, τ_b²) and g ~ N(0, σ_g²).
- One extra Newton step per iteration for log τ_b. It is the EB marginal given the item posteriors; use
  τ_b² ≈ (Σ_i (d_i − prior_i − g)² + Σ v_i + κ τ_0²) / (n_items + κ), with κ pseudo-items.
- g is fitted like `delta` (bincount over the group index).
- `score_batch` uses g_{kv} and τ_b for *unseen* items of a seen benchmark. This changes B1+ for all eval items,
  not only labeled ones.

Duplicates: change `item_key` used in IRT evidence to `(benchmark_id, content_hash)` when `tie_duplicates`, with a
template guard, i.e. a per-benchmark content counter built from the evidence.

### 5.2 Item feature extractors (src/paiec/features/item.py)
- `focus` view (port `analysis/item_difficulty/views.py:focus`, stdlib only). It needs the benchmark pool, so build
  the line-df from the items in `labeled` plus the item itself (this is the transductive pool) and cache the df by
  evidence fingerprint. At B0 the pool is empty; fall back to head.
- `todo_lines` / `solution_size_hint`: the regex `Approximately (\d+) line` and generic "N lines/steps" cues.
  ρ = 0.54 on rcb, cheap. It is a narrow benchmark-family feature, so add it only behind `item_std="bench"`.
- `llm`: lookup of LLM-extracted features (§6) by `sha256(item_features + "\0" + item_content)[:16]`. When the
  cache misses, the feature is 0 (= the benchmark mean after z-scoring), so prediction degrades gracefully.

### 5.3 Artifacts to precompute (training time)
- `llm_item_cache.jsonl` for all training items. Produced by `llm_pilot.py prepare all && run` (vLLM), or with an
  API extractor. It is needed to fit the ridge weight on the LLM feature and its LOBO shrink.
- Ridge weights for the BOW + LLM columns (one ridge, λ=10 as now; that is exactly the `lk_bow_llmdiff8b` run),
  τ_0 and κ for the τ_b prior, and σ_g for groups. All are stored in `predictor.json` as now.
- No embedding artifacts are needed unless stage 2 (#5) is adopted.

### 5.4 models.txt
- Recommended path (API LLM): **empty**.
- No-API fallback: `Qwen/Qwen3-8B` (16 GB bf16; measured 8 items/s with vLLM on an A6000, ~600 prompt tokens per
  item). Or `Qwen/Qwen3-32B-FP8`, which is better (ρ 0.33 vs 0.24) but slower (~1.4 items/s on an A6000; faster on
  an H100). This only works if vLLM or a recent transformers is present in the hosted image; unverified.
- An embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 90 MB) only for transductive kNN. Low priority.

## 6. LLM feature-extraction spec
**Prompt.** The system prompt and schema in `analysis/item_difficulty/llm_pilot.py` were used in the pilot.
- Benchmark-blind: no benchmark names; `item_features` is passed as `[metadata: …]`.
- The text is the full content when ≤ 6000 chars; otherwise the transductive focus view, capped at 6000 chars.
- The model returns a single JSON object:

```json
{"task_type": "math_answer|math_proof|code_fix_repo|code_implement|doc_qa|web_gui_agent|knowledge_qa|other",
 "domain": "str", "needs_tools_or_environment": true, "needs_unseen_inputs": false,
 "n_reasoning_steps": 0, "n_constraints": 0, "expected_solution_size": 0, "expert_minutes": 0.0,
 "ambiguity": 0, "difficulty_1_10": 0, "p_solve_frontier": 0.0, "p_solve_small": 0.0}
```

**Production version.**
- Use structured outputs (`output_config.format` with this JSON schema) on Claude, or the equivalent on other
  providers. 100% of pilot outputs parsed even without it.
- Temperature defaults, effort `low`, `max_tokens` 300.
- Features used downstream: `difficulty_1_10`, logit(`p_solve_frontier`), logit(`p_solve_small`),
  log1p(`expert_minutes`), `n_reasoning_steps`, `n_constraints`, log1p(`expected_solution_size`), `ambiguity`, and
  the two booleans. Each is **z-scored within the benchmark pool**; raw levels are not comparable across
  benchmarks (§3.1).
- `difficulty_1_10` alone is as good as all of them (LOBO).

**Model choice.**
- Pilot scaling: 8B gives ρ 0.24 and 32B gives 0.33 on the pilot.
- Next: test `claude-haiku-5-5` (cheapest) and `claude-sonnet-5-5` on the same 600 pilot items. That costs
  < $0.10 and ≈ $2 respectively. Include a "thinking" variant only if it buys > 0.05 ρ.

**Caching.**
- Key = sha256(prompt text). Persist to a JSONL next to the predictor (training) and to a writable temp dir at test
  time.
- Workers restart at each checkpoint; whether the filesystem persists across checkpoints is unverified. The cache
  must be an optimisation, not a dependency.

**Cost / latency at test time.**
- Measured: ~680 prompt tokens + ~110 output tokens per item (Qwen tokenizer; similar for Claude).
- Haiku 5.5 ($0.10 / $0.50 per MTok): ≈ $0.00012 per item, so $0.12 for the 1,000-pair formative run and ≈ $1.2
  per 10K items.
- Sonnet 5.5 ($2 / $10): ≈ $0.0025 per item, i.e. $25 per 10K items.
- Latency is 1–3 s per call, so items must be prefetched concurrently, not called inline per `predict`:
  - on each `predict` call, collect the unseen item hashes from `labeled` plus the current item;
  - submit them to a thread pool of 16–32 requests;
  - return immediately with the non-LLM prior for items whose feature is not ready.

  At budget 0 nothing is in `labeled`, so B0 uses the fallback prior unless the per-call budget allows a blocking
  call (≤ 3 s, configurable).
- Total wall-clock for 10K unique items at 16× concurrency is ≈ 15–30 min, within 8 h.
- Note: `predict` must stay a pure function of (input, labeled) for evidence caching. Results then depend on
  network timing. Make the LLM prior affect only the *prior mean*, and log hits and misses.

**Fallback.**
- On timeout, HTTP error or parse failure, set the feature to the benchmark-pool mean (z = 0). This is identical to
  "no LLM feature", which is the current model.
- After 20 consecutive failures, open a circuit breaker for the run.
- If no API key is configured, skip entirely.
- The local-model alternative (models.txt) removes the network dependency, but needs a GPU inference stack in the
  hosted image.

**Rules / risks to confirm.**
- An API key would ship inside the ZIP (or env), and an organiser rule on that is needed.
- Model knowledge of public benchmarks: an LLM may recognise AIME/IMO problems. That inflates our matharena numbers
  relative to truly unseen benchmarks.

## 7. Beyond plain regression (assessment)
- **Benchmark offset β_b at cold start.**
  - Text does not predict it. The LOBO offset RMSE is 2.08 logits for the mean-of-others and 1.92–2.4 for every
    text model. The LLM's mean "p(frontier solves)" is uncorrelated with base rates (ρ = −0.2 over 6 benchmarks).
  - With only 5 training offsets, any learned offset model is a 5-point regression.
  - Best levers:
    - hedge at B0 (`beta_var_mult`, already shipped by the main session);
    - learn β fast from global evidence at B1;
    - optionally, a *task-family* prior (agentic/code vs QA/math) keyed on the LLM `task_type` distribution of the
      pool. It is plausible (SWE/web/rcb base rates 0.14–0.45 vs QA/math 0.7–0.8), but it rests on 5 points, so
      treat it as a weak prior with large variance.
- **Transductive use of the visible test pool.**
  - The pool is only what appears in `labeled` plus the current input. The full test pool is never passed.
  - Use it for focus-view boilerplate stripping, per-benchmark feature standardisation, kv-group and duplicate
    tying (§5.1), and τ_b estimation.
  - A within-benchmark text→d ridge fitted on test-time evidence (stage 2) has within-benchmark ρ 0.3–0.6 under
    clean labels. Under budget-1 label noise its value is unclear; test it after (1)–(2).
- **Multidimensional IRT** (loadings from text, e.g. CAIMIRA-style).
  - The difficulty ordering is consistent between the strong and weak halves on matharena, mmdocrag, real_webagents
    and rcb (ρ 0.67–0.85). It is not consistent on multi_swebench (0.18), where weak models are at the floor.
  - A second dimension (agentic/code skill vs reasoning) mostly helps *subject* modelling across benchmark types.
    The item side would need the LLM `task_type`.
  - Lower priority than (1)–(3).
- **Discrimination (2PL).**
  - Most items have a ≈ 1.0–1.3. mmdocrag's hard items have low a (noisy judge).
  - A per-benchmark common discrimination (a scale on the logit) is cheap and could help calibration where grades
    are noisy. A per-item a needs labels that we do not have at test time.
- **Interaction with acquisition.**
  - The uncertainty policy prefers items with p ≈ 0.5, which depends on d.
  - Under global evidence, items that other subjects have labeled have a sharper d. Items in large kv groups or
    duplicates propagate information to many eval items.
  - Proposal: add a term to the information score for "expected variance reduction on same-group eval items", i.e.
    favour items from under-sampled groups early. This only pays once (2) exists.

## 8. Risks and open questions
- **5 folds.** Fold-level noise (±0.01 ALC) swamps item-prior effects (±0.002). Use paired per-pair Δ with CIs
  (`pairs.csv`) and more seeds before adopting anything.
- **Unseen benchmark types.** The hidden test benchmarks may be unlike these 6 (≈ 300 inventory candidates). That
  argues for zero-shot LLM judgments and test-time learning, and against text→d regressors fit on 6 benchmarks.
- **LLM familiarity with public items** (AIME/IMO and SWE-bench repos) may overstate the LLM signal.
- **Whether `item_features` keys in the test pool carry structure** (e.g. a `competition=`-like key) is unknown.
  The group prior must degrade to nothing when keys are absent or all distinct.
- Hosted constraints: per-call timeout for `predict` (failure 41), filesystem persistence across checkpoints,
  network access from workers, the API-key policy and the hosted inference stack for a local LLM are all
  unverified.
- matharena's sparse subject design makes its d noisy for 28% of items. mmdocrag's binarisation threshold (0.5)
  changes its d spread, and the test-pool binarisation rule is unknown.

## 9. Experiment queue (until 30 Oct)
| day | experiment | success criterion |
|---|---|---|
| 22–23 Oct | When job 7956248 finishes: `llm_pilot.py table Qwen3-32B-FP8_input_all`, export `bow_irt&llmdiff_<32B>`, then e2e paired vs `abs_irt` (more seeds: `cv.seeds=[0..5]`) | Δ < −0.001 with CI excluding 0 |
| 23 Oct | Kv-group effects + duplicate tying with template guard (§5.1); e2e | matharena / rcb / real_web improve at B3–B15; Δ < −0.001 |
| 23 Oct | Robustness guards: prior-mean clipping, within-benchmark standardisation for every dense feature | no regression; protects against `lk_hand`-type blowups |
| 24 Oct | Claude pilot: Haiku 5.5 vs Sonnet 5.5 on the 600 pilot items (< $3 total; needs a key) | pilot ρ ≥ 0.33 mean5 → run all items and e2e |
| 24 Oct | Per-benchmark τ_b (MAP) | Δ ≤ −0.0005; mmdocrag and rcb both non-worse |
| 24–25 Oct | Hosted feasibility: API reachability and latency from a Codabench worker (formative submission with a stub that calls the API once and logs timing via the returned probability pattern), per-call timeout, writable cache dir | API usable → wire in §6 with prefetch |
| 25–26 Oct | Transductive within-benchmark ridge on focus-view embeddings or BOW, fitted on test-time evidence (stage-2 prior), gated by evidence count | positive Δ at B15/B31 without hurting B1 |
| 26–27 Oct | Acquisition: group-aware information bonus | Δ ALC < −0.001 |
| 28–30 Oct | Freeze; full-pool runs (`max_eval_items_per_pair: null`), more seeds; write-up figures (feature-correlation table, the LOBO vs within-benchmark gap, LLM scaling 8B → 32B → API) | – |

## Appendix: what was run, installed and spent
- **Ran:** everything in §1–§3, on 5 workstations (CPU) and on the UMIACS cluster (RTX A6000) for 3 vLLM jobs:
  - Qwen3-8B, 600-item pilot;
  - Qwen3-8B, all 12,633 items (29.5 min);
  - Qwen3-32B-FP8, pilot (10 min).

  A fourth job (32B on all items, 7956248) was submitted and may still be running. Its output will appear as
  `data/analysis/item_difficulty/llm/output_Qwen3-32B-FP8_input_all.jsonl`. Then run
  `python llm_pilot.py table Qwen3-32B-FP8_input_all`.
- **Installed:** `sentence-transformers==6.2.0` (pulled `transformers==5.20.2`, `tokenizers`, `safetensors`) into
  the `aims26` env. torch was unchanged; its cu130 build does not run on the local GTX 1080 Ti, so embeddings ran on
  CPU.
- **Cost:** $0. No API keys were present; no paid API was called.
- **Not run:** Claude or other API extraction, multidimensional IRT, any 2PL predictor, and the §5.1 τ_b / group /
  duplicate model (proposed only; the transductive simulation in §3.2 is a proxy, not the model).
- **Config drift during the analysis:** `configs/irt.yaml` changed (3b601aa) mid-analysis. `irt_baseline` and
  `no_item_prior` ran on the old default; every other e2e run ran on the new one. §3.3 compares like with like.

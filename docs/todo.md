# Known issues and things to look at

Items found in review that are **deliberately not fixed yet**, with the evidence and the options. Fixing any of them
changes labels or numbers, so each needs a re-score of the affected protocol. Date each entry; delete it when resolved
and point to the commit or the `results.md` entry.

## 1. Mixed-scale benchmarks are binarised against the whole benchmark's range (open, 2026-10-23)

**What happens.** `load_dataset` / `_binarize` (`src/paiec/data/dataset.py`) turns a graded benchmark into 0/1 by
normalising `response` to [0, 1] with the scale's declared bounds and thresholding at 0.5. When the benchmark declares no
bounds (`response_scale = {"kind": "mixed"}`), the bounds are the **observed min/max of the whole benchmark**. But a `mixed` benchmark holds several sub-scales side by side, so one range does not fit them all.

**Confirmed cases** (checked against the raw tables, 2026-10-23):
- **`vhelm`** (train). 92% of responses are exact-match grades, 0 or 1. The other 8% (`metric=prometheus_vision` in
  `test_condition`: scenarios `flickr30k`, `crossmodal_3600`, `bingo`, `mementos`) are 1–5 judge ratings. The loader sees
  the range 0..5, so a correct answer, `1.0`, normalises to 0.2 and becomes **y = 0**. The benchmark's mean y is 0.046;
  keeping the 0/1 part as is (and thresholding the ratings at ≥ 3) gives 0.546.
- **`scale_mrt`** (was val). `item_features` has `rating_scale`: `one_to_five` (51.9% of responses, values 1–5) and
  `zero_to_hundred` (48.1%, values 0–100). Against the range 0..100 every 1–5 rating is below 0.5, so that half is
  y = 0 throughout (mean y 0.000); the 0–100 half has mean y 0.347.
- **`prism`** (train). `item_features` has `dimension`: `if_chosen` (0/1, 50% of responses) and `score` (1–100). Against
  0..100, `if_chosen = 1` normalises to 0.01 and becomes y = 0.

**Suspected, not verified.** Benchmarks without a declared range whose observed range is above 1 where more than 1% of responses
sit on items that never exceed 1: wonderbread (95% of responses), goodai_ltm_benchmark (89%), psychosis_bench (78%),
stanford_orb (67%), donotanswer (67%), llm4ir (65%), chipbench (41%), lambda_fp_course (24%), refgrader (21%),
dpai (14%), image2struct (100%, but the range is only −0.2..1, so just the negative tail is affected), kris_bench (1%).
None of these has a sub-scale key in `item_features` / `test_condition`, so the heuristic may be flagging legitimate
low scores (a `1.0` on a 0..5 scale is a low rating, and y = 0 is then right). Each needs a look at what the values mean.
Benchmarks with a declared range or value set (fraction, likert, ordinal types) and matharena (observed 0..1) are fine.

**Why it was left alone.** Every number in `results.md`, the research loop and the published split uses these labels,
so changing them breaks comparability with the whole history. Models trained on them still beat the baselines, and the
hosted platform's own binarisation is unknown (`open-questions.md`), so the "right" labels are not even established.
Decision (2026-10-23): leave as is, document, and exclude `scale_mrt`, the confirmed case that is scored (item 2).

**Options if revisited.**
1. *Exclude* the heterogeneous mixed benchmarks (cleanest; loses ~15 training benchmarks and changes the eval mix).
2. *Per-sub-scale normalisation*: group by an `item_features` / `test_condition` key (`metric=`, `rating_scale=`,
   `dimension=`), normalise each group by its own range, and treat groups whose values are all in {0, 1} as binary.
   Needs a per-benchmark check; `scale_mrt` has `rating_scale` and `vhelm` has `metric` in `test_condition`.
3. *Use the platform's rule* once it is known (the organisers' binarisation of non-binary benchmarks).

Whichever is chosen: re-export the split (`scripts/export_split.py`), re-push to HF, and re-run every `configs/aims/*`
and `configs/research/*` control, because train labels and the fit cache key change.

## 2. `scale_mrt` is excluded from the published split (done, 2026-10-23)

`scale_mrt` was the one confirmed case that is **scored**: it was in val fold 4, and 52% of its responses use the
`rating_scale=one_to_five` sub-scale, which the loader maps to **y = 0 for every response** (mean y 0.000), so half of its
evaluation labels were a constant readable from `item_features`. It was never in the published train split. (`vhelm` and
`prism` are confirmed too but are training benchmarks, so they only distort what the models learn.)

- `configs/aims/export.yaml` and `configs/aims/base.yaml` set `data.exclude: [scale_mrt]`; `load_dataset` skips excluded
  benchmarks even if they are on disk (a stale download keeps working).
- HF `mgor/aims-measurement-db-val` no longer has `scale_mrt` (30 → 29 benchmarks, fold 4 has 5); train is unchanged.
  `split.json` lists it under `excluded`.
- Val: 29 benchmarks, 1,622,445 responses (was 30 and 1,672,906).
- **Not changed:** `configs/split_seen/*` (the full-db protocol used by the research loop and every `R_*` run) still
  contains `scale_mrt` in dev fold 4. Numbers on `split_seen` and on `configs/aims` therefore differ by this one benchmark
  and are not directly comparable; `scale_mrt` is also in the `holdout` universe as a *training* benchmark and in
  `configs/sel/pool.csv`. If the research loop moves to the published split, retire `split_seen` for dev.

## 3. Other deferred findings from the 2026-10-23 review

Found by four independent code reviews (model core, protocol and submission path, data and caching, other predictors)
plus an adversarial-evidence fuzz of every predictor. The ones that were fixed are in `results.md` (2026-10-23, "review
fixes"). These were left open on purpose.

**Changes numbers or labels, so needs a re-score of the history**
- **Replay pools are seeded by a benchmark's position in the loaded list, not by its name** (`sim/replay.py`: the
  `_rng("subjects"/"split"/"order", ..., b)` calls). Loading one extra benchmark first reshuffles every later
  benchmark's subject sample, acquisition/evaluation split and candidate order (measured: 0 of 12 pairs kept the same
  pool). `compare.py` matches rows on (subject, benchmark, seed) and so can report a "paired" Δ over different pools.
  `compare.py` now warns when the runs' data/cv settings differ; the root fix is to seed with `ds.bench_names[b]`,
  which re-draws every pool and breaks pairing with every existing `R_*` control, so do it at the next protocol reset.
- **Item variance for items without kv tokens** (`irt_item.py` `_var_eps`, `_groups`): the `kv_frac * var_d` share is
  taken out of the item's own effect but only added back when the item has a non-numeric `item_features` token. Items
  with empty or numeric-only features therefore get `(1 − kv_frac) · var_d` instead of `var_d`, so B0 differs slightly
  from IRT (0.5101 vs 0.5099 at the reviewer's example) and the item posterior is a little over-confident. Fix: put the
  share back when an item has no kv tokens (in both `_adapt_bench` and `_item_post`).
- **Lower-is-better orientation is applied inconsistently** (`dataset.py` `_binarize`): graded lower-is-better scales are
  always flipped (`oasst`), binary ones only with `binarize.orient` (default off), so `harmbench`, `jailbreakbench`,
  `advprompter`, `crow` and `emoji_attack` keep 1 = attack success. Related to §1; what the platform does is unknown.
- **The fitted-gate model `irt_item_mix` (`configs/research/auto/best.yaml`) is a promotion candidate, not the champion.**
  Honest dev ALC after the fold-weight lookup fix is about 0.1730 (the recorded 0.1726 was in-sample), against 0.1766
  for the champion. Its `mix_w` are fitted on the seen-dev folds, so (a) running it on the published val split is
  in-sample, and (b) promoting it needs a locked hold-out run with weights not fitted on the evaluated benchmarks.

**Latent or informational**
- **Evidence fingerprint samples at most ~48 entries** (`predictors/base.py` `fingerprint`). A different `labeled` list of
  the same length that agrees on the sampled entries would reuse a stale cached state. The hosted client's list is
  append-only, so this does not trigger there; it can in `sim.evidence_scope: pair` ablations or if the list were
  re-ordered. (`_entry_fp` now covers every visible field.)
- **Warm-started incremental state differs from a fresh adapt**: typically ≤ 4e-3 on realistic evidence at the default `adapt_iters` 15, up to
  ~0.2 for degenerate evidence (every label from one subject). The hosted acquisition calls use the incremental state,
  evaluation rebuilds fresh, so it only perturbs which items get acquired.
- **`time_predict.py` underestimates the hosted runtime.** It scales per-call time measured at small evidence
  linearly, but every accepted label triggers a warm refit of the whole benchmark whose cost grows with the labels
  (one-label extend: 32 ms at 100 labels, 103 ms at 6,200; cold adapt 51 → 197 ms, per mix member). With ~1,000 pairs
  on one benchmark (31k accepts) the acquisition phase alone is plausibly 1–2 h: under the 8 h limit, but not the
  margin the script prints.
- **Identical visible subject dicts** (108 subject–benchmark pairs, e.g. on `theagentcompany`) are merged by
  `subject_key`, which double-counts a pair's own labels. The hosted inputs cannot tell them apart either, so this is
  not a replay/hosted mismatch.
- **Fit-cache key** hashes `fit`, `_fit*`, `FIT_HELPERS` methods, module-level functions of the classes' own modules and
  `features/` and `data/`. Functions imported from another module (e.g. `simple.sigmoid` in `IRT.rasch`) are not hashed.
- **`eb_eps_nu` (item-effect variance EM) is unbounded**, unlike the other EB scales. Tested on all-correct and
  separable tiny benchmarks it stayed within 0.55–2.7× its prior; clip it to `eb_bound` for symmetry if it ever matters.
- **Dev-only reference model `ref_caimira`**: `b_map` is keyed by raw benchmark names but looked up by anonymised ids
  (so the per-benchmark biases are never used, which only matters in-sample), and `temperature` does nothing for
  `kind=lite|irtm`. It cannot be packaged (`build_submission` refuses it).
- **Hosted Python version is undocumented.** The shipped code now avoids `itertools.pairwise` (3.10+), but nothing
  else has been checked against an older interpreter.

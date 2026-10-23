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

(Filled in when the bug-fix pass finishes; see `results.md` for what was fixed.)

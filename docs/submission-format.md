# Submission format and runtime protocol (tutorial)

Sources:
- the competition page <https://aims.stanford.edu/competition>;
- the organizers' shipped client `tools/streaming_ingestion.py` in
  <https://github.com/aims-foundations/paiec_baseline>, read 2026-10-22: "Stdlib streaming client, shipped in the bundle";
- the baseline tests (`tools/test_mean_baseline.py`, `tools/smoke_test.py`).

Statements are marked **[page]**, **[client]** or **[tests]**. Anything unmarked is our inference.

## 1. ZIP layout
The ZIP **root** must contain:

| file | required | role |
|---|---|---|
| `model.py` | yes | defines `predict(input, labeled) -> float` |
| `labeling.py` | no | defines `acquisition_function(input, prediction, labeled, context) -> bool`. If absent, labels are picked at random |
| `models.txt` | no | HF repos to prefetch (≤5, ≤300B params each, ≤1,024 GB total). No runtime downloads |
| `requirements.txt` | no | gated by organizer policy; assume it is ignored |
| anything else | no | fitted state, embeddings, code. Ours: `paiec/`, `predictor.json`, `acquisition.json` |

Build ours with `python scripts/build_submission.py --config configs/irt.yaml`, which writes `dist/irt.zip` and the unpacked `dist/irt/`.

## 2. The data structures

### `input = [subject, item]`
Every value is a **string**, and a missing value is `""` **[page]**.

**subject** (`input[0]`). The dict *as a whole* is the subject's identity. The same model under another harness counts as a different subject **[page, tests]**.

| field | from `subjects` table | example (matharena, "Kimi K2 Thinking") | notes |
|---|---|---|---|
| `normalized_name` | `normalized_name` | `Moonshot Kimi K2` | empty for 16% of subjects |
| `provider` | `provider` | `Moonshot` | empty for 16% |
| `release_date` | `release_date` | `2025-07-11` | |
| `access_date` | `access_date` | `""` | empty for 74% |
| `harness` | `harness` | `""` | e.g. `OpenHands`, `MSWE-agent` on SWE benchmarks |
| `harness_version` | `harness_version` | `""` | essentially always empty |
| `reasoning_effort` | `reasoning_effort` | `""` | `low` / `medium` / `high` / `xhigh` when present |
| `subject_features_extra` | `subject_features_extra` | `model_config=moonshot/k2-thinking;source_model_name=Kimi K2 Thinking` | `k=v;k=v`; often the only usable model identity |

Not passed: `subject_id` and `display_name`. The private `subject_id` appears only in the acquisition `context`.

**item** (`input[1]`):

| field | source | example (real_webagents) | notes |
|---|---|---|---|
| `item_content` | `items.content` | `Show me places in Delhi, India for 2 guests on the dates Sept 7th to 9th.` | may be huge (researchcodebench median is ~96K chars) |
| `item_features` | `items.item_features` | `website=staynb` | `k=v;…`, e.g. `lang=go`, `competition=aime_2025;problem_idx=2` |
| `interactors` | `responses.interactors` | `""` | tools or agents; empty in all public data |
| `benchmark_id` | `items.benchmark_id` | anonymous id | **anonymous** in the test pool. It is a grouping key, not a feature |

Not passed: `item_id`, `raw_item_id`, `content_hash`, `grading_criterion` (the reference answer), `verifier`, `asset_manifest` (images), the trial number and `test_condition`.

### `labeled`
`labeled = [ [[subject, item], label], ... ]` with `label ∈ {0, 1}`. It is empty at budget 0 **[page]**.

**`labeled` is global.** It holds every label revealed so far for *all* sampled (subject, benchmark) pairs, not just the pair being predicted. The evidence:
- **[client]** Each coordinator event carries one `labeled` list (`labeled = event["labeled"]`). An `evaluate` event passes that same list to every prediction for all targets of that budget (`prediction_stream(model_path, unique, labeled, ...)`, logged as `Budget {b}: {count} evaluation calls`).
- **[page]** The acquisition hook's `labeled` "contains `[[subject, item], label]` entries across sampled subject–benchmark pairs".
- **[tests]** The empirical-mean baseline's test `test_other_benchmarks_and_subject_settings_do_not_change_prediction` feeds labels from other subjects and benchmarks, and the baseline must filter them out. The BLE smoke test mixes in another subject's label in the same way.

Consequence: at budget n, a test benchmark carries about n labels for *each* sampled subject. Our IRT uses these to learn the unknown benchmark's offset and the difficulty of shared items. That is worth 0.03 ALC locally (`docs/results.md`).

### `context` (acquisition only)
`{subject_id, benchmark_id, labels_acquired, labels_remaining, max_labels: 31, items_remaining}`. The counts are per pair **[page]**. `labels_remaining` is the allowance left out of 31, not the distance to the next checkpoint **[page]**. `items_remaining` includes the current candidate.

## 3. The two functions

```python
def predict(input, labeled) -> float:
    subject, item = input          # dicts of str
    ...                            # any adaptation must come from `labeled` alone
    return p                       # finite Python float in [0, 1]

def acquisition_function(input, prediction, labeled, context) -> bool:
    # input:      the candidate [subject, item] from this pair's acquisition pool
    # prediction: YOUR predict(input, labeled) for it, computed by the client just before
    # return True to reveal one recorded response of it; False skips it permanently
```
- **Statelessness.** "Evaluation workers are recreated with frozen evidence at each checkpoint" **[client]**. Inputs are deep-copied into each call. Caches are fine only if they are pure functions of their inputs.
- **Repeated trials.** Identical inputs are predicted once, and the prediction is reused for every recorded trial. Each trial is scored separately **[page, client]**.
- **No `labeling.py`.** The client never calls `predict` during acquisition. It accepts a candidate iff `sha256(json([context, input]))` as a uniform is below `labels_remaining / items_remaining`, i.e. reproducible selection sampling **[client]**.
- **With `labeling.py`.** If the hook takes `prediction`, the client first calls `predict(candidate, labeled)` under the per-call timeout, then the hook under `acquisition_timeout` **[client]**. The hook must return an actual `bool`, not `np.bool_`.
- **Concurrency.** Evaluation calls at a checkpoint run in up to `concurrency` workers. The value comes from an optional root `prediction_config.json` according to the collaborator's notes, and the platform may cap it.

## 4. End-to-end flow of one hosted run
1. For each sampled pair with ≥80 items, the platform splits items 50/50 into an acquisition and an evaluation pool. The split is fixed across submissions **[page]**.
2. Budget 0: an `evaluate` event arrives with all evaluation targets and `labeled = []`, and `predict` is called once per distinct input.
3. To reach budget n: `acquire` events stream each pair's candidates in a fixed random order until the pair has n labels **[page, client]**.
4. An `evaluate` event arrives with the frozen global `labeled`. This repeats for n = 1, 3, 7, 15 and 31, with nested labels.
5. The score is the Brier ALC per pair, averaged equally over pairs **[page]**.

## 5. Output validation and failure codes
`predict` must return a finite float in [0, 1]. NaN, inf, out-of-range or uncastable values are rejected.

Failure codes:
- 10/11 invalid `model.py`
- 12 invalid `labeling.py`
- 40 prediction exception
- 41 call timeout
- 42 invalid probability
- 50 total timeout (8 h including setup)

Hosted logs hide stdout/stderr. Our `paiec.runtime.safe_predict` therefore falls back to 0.5 rather than failing the run, and `labeling.py` falls back to accepting while budget remains.

## 6. Local checks
```bash
python scripts/build_submission.py --config configs/irt.yaml
python scripts/check_zip.py dist/irt.zip                     # ours: imports model.py, two calls
python <paiec_baseline>/check_submission_zip.py dist/irt.zip  # organizers' validator
```
The Codabench starting kit also ships `tools/run_smoke_test.py`. Per the collaborator's notes, the kit's zero-label smoke input omits `benchmark_id`. Our predictors tolerate that.

## 7. Submitting
Upload at <https://www.codabench.org/competitions/17828/>, or use `submit_api.py` from the baseline repo (`--login`, `<zip>`, `--list`, `--status ID`).
For the final submission, put the code, the report and the chosen Codabench submission ID on OpenReview.

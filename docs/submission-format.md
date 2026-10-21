# Submission format (Codabench)

ZIP **root** must contain:
- `model.py` (required): `predict(input, labeled) -> float`
- `labeling.py` (optional): `acquisition_function(input, prediction, labeled, context) -> bool`
- `models.txt` (optional): HF repos to prefetch (≤5; no runtime download)
- `requirements.txt` (optional; gated by organizer policy)
- any artifacts (fitted state, templates, embeddings…)

## `input = [subject, item]` — all values are **strings**, missing = `""`
- subject: `normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra`
- item: `item_content, item_features, interactors, benchmark_id` (`benchmark_id` is **anonymous** for test benchmarks)

`labeled`: list of `[[subject, item], label]`, label ∈ {0,1}; empty at budget 0. Predict is stateless per call (collaborator's
notes: workers restart each checkpoint; the full evidence is passed each time). Do not rely on global mutable state.

Table → input mapping: `items.content → item_content`, `items.item_features → item_features`, `responses.interactors → interactors`,
`subjects.*` as named. `subject_features_extra` is a `k=v;k=v` string (e.g. `model_config=...;source_model_name=...`). **Note**: many subjects have null
`normalized_name`/`provider` in the dataset, so the string `source_model_name`/`model_config` inside extra features is often the only identity.

## Output
Finite float in [0,1] (native Python float preferred). NaN/inf/out-of-range/uncastable ⇒ rejected.

## `acquisition_function` context
`subject_id, benchmark_id, labels_acquired, labels_remaining, max_labels (31), items_remaining`.

## Local checks (from starting kit — download from Codabench)
    python tools/check_submission_zip.py my_submission.zip
    python tools/run_smoke_test.py my_submission/
(Collaborator notes: the kit's smoke input omits `benchmark_id` at zero labels; kit HF templates bypass real model loading in local-smoke mode — neither proves hosted behaviour.)
Our own: `make zip && make check` (stubs: `scripts/build_zip.py`, `scripts/check_zip.py` TODO once kit is available).

## Failure codes
10/11 invalid model.py · 12 invalid labeling.py · 40 prediction exception · 41 call timeout · 42 invalid probability · 50 total timeout. Hosted logs hide raw stdout/stderr → test locally; wrap risky code and fall back to a safe prior inside `predict`.
Time limit: 8 h total per run including setup (per-call timeout value not published — keep predict fast; precompute embeddings in artifacts).

## Submitting
Upload ZIP at Codabench (<https://www.codabench.org/competitions/17828/>) or via the baseline's `submit_api.py` (`--login`, `<zip>`, `--list`, `--status ID`).
Final: OpenReview submission with code + report + chosen Codabench submission ID.

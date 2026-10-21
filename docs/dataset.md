# measurement-db (training data)

HF: `aims-foundations/measurement-db` (gated, CC-BY-SA-4.0; individual benchmarks keep their own licences — check `benchmarks.license`).
Site claims 270 benchmarks / 8.4M items / 781 subjects / 92.1M responses; **the public HF release currently exposes only 6 benchmarks**
(~1.1M rows excl. traces) at revision `54074453da7b9911be0f48e4c560be547bc84d21` (2026-10-07). ImageNet-Hard, VisualMemory, SWE-Together
are `release: withheld`. The inventory sheet lists ~300 candidate benchmarks (curation_status mostly blank) — **only inventory sources are
permitted for training**; the set may grow (participant contributions). Pin a revision; re-run `scripts/download_data.py` to refresh.

Cite: Truong, Truong, Koyejo. *The AI Measurement Data Bank*, 2026. <https://aimslab.stanford.edu/measurement-db>

## Layout
`{benchmark_id}/formatted_tables/{benchmarks,subjects,items,responses}.parquet` (+ `traces.parquet`, `assets.parquet` when available).
Join all tables from the **same revision**. IDs are hashed 16-hex strings; `item_id`/`subject_id` are per-benchmark-derived (subjects repeat across benchmarks only if ids match — verify before assuming cross-benchmark identity).

## Tables (columns verified from HF preview)
- **benchmarks** (25): benchmark_id, name, version, license, source_url, description, one_line_description, modality[], domain[], multi_single_turn, response_type, response_scale (JSON: kind/values/direction/meanings), categorical, paper_url, release_date, granularity, release, n_response_values, benchmark_features, n_subjects, n_items, n_responses, max_trial, coverage, has_reference_answer
- **subjects** (10): subject_id, display_name, normalized_name, provider, release_date, access_date, harness, reasoning_effort, harness_version, subject_features_extra (`k=v;k=v`, e.g. `model_config=anthropic/claude-sonnet-45;source_model_name=Claude-Sonnet-4.5 (Think)`). Many rows have null normalized_name/provider.
- **items** (9): item_id, benchmark_id, raw_item_id, content (text), asset_manifest, grading_criterion (JSON: reference_answer/rule), verifier (JSON: class/spec), content_hash, item_features (`k=v`, e.g. `website=staynb`)
- **responses** (8): subject_id, item_id, response_id, benchmark_id, trial (int), test_condition, interactors, response (float64; null = ungraded attempt)
- **traces** (8, large ~1GB): model outputs, joined on response_id. Not in competition input; may be used for training features only.

## Public benchmarks at the pinned revision
| benchmark | scale | subjects | items | responses | max trial | mean resp | pairs ≥80 items |
|---|---|---|---|---|---|---|---|
| matharena | **mixed** (15 values: final-answer + proof criterion grades) | 104 | 1,755 | 92,944 | 200 | 0.654 | 81/104 |
| mmdocrag | **fraction** (26 values 0..1 in 0.04 steps, judge rating) | 68 | 2,000 | 342,695 | 1 | 0.668 | 68/68 |
| multi_swebench | binary | 82 | 2,127 | 57,975 | 2 | 0.148 | 82/82 |
| real_webagents | binary | 33 | 233 | 4,449 | 5 | 0.358 | 26/33 |
| researchcodebench | binary | 31 | 212 | 6,784 | 2 | 0.361 | 31/31 |
| swe_rebench | binary | 1 | 6,306 | 67,074 | 34 | 0.479 | 1/1 |

(Run `python scripts/profile_data.py` to regenerate.) Reminder: the **test targets are binary**. Non-binary benchmarks must be binarised
(threshold, or treat as Bernoulli soft label) — the test-pool conversion rule is unknown → see `docs/open-questions.md`.

## Gotchas
- Few labelled benchmarks ⇒ **very few distinct benchmarks to validate cold-start generalisation** (6; effectively 4 binary). Use leave-one-benchmark-out and report variance.
- Subject identities are noisy (null normalized_name). Parse `subject_features_extra` (model_config/source_model_name) for model family, size, "Think", date.
- `trial` > 1 means repeated runs of same (subject,item): each trial is a separate target, labels at test time are per (subject,item) pair.
- `benchmark_id` is anonymised in the test pool: **do not key on benchmark names**; condition on benchmark-level statistics inferred from `item_content`/`item_features`/labels.
- Keep `traces` out of the repo and package.

# Open questions

## Open
1. **Per-call and acquisition timeouts.** The client enforces `per_call_timeout` and `acquisition_timeout`, but their values are not published. Keep `predict` fast (currently ~1–4 ms per call).
2. **How the test pool turns non-binary grades into 0/1**, and whether `lower_is_better` binary benchmarks (e.g. "1 = attack success") are scored as recorded or flipped. We currently binarise at 0.5 on the normalised scale and keep binary values as recorded (`binarize.orient: false`).
3. **Which subjects appear at test time** (seen in measurement-db vs new) and how many per benchmark.
4. **Is `interactors` populated in test inputs?** It is empty in the 6 reference benchmarks. Check the 278-benchmark pool.
5. **LLM API use at test time.** Allowed providers are listed in `competition.md`. Is an API key in the hosted environment acceptable, and does a disk cache survive worker restarts? (`item-difficulty-plan.md` §6)
6. **Codabench limits:** submission quota per day and queue time.
7. **Hosted Python environment** (the collaborator's notes say Python 3.13, torch 2.10, transformers 5.5). Is numpy guaranteed? Our runtime assumes stdlib + numpy.

## Resolved
- 2026-10-22: **`labeled` is global** (all pairs' revealed labels). Source: the organizers' client `paiec_baseline/tools/streaming_ingestion.py`; see `submission-format.md` §2.
- 2026-10-22: **How many benchmarks are public?** 278 of 281 (3 withheld). The HF viewer lists only 6. See `dataset.md`.
- 2026-10-22: The starting-kit validator equivalent is `paiec_baseline/check_submission_zip.py`. Our `dist/irt.zip` passes it.
- 2026-10-21: The dataset id is `aims-foundations/measurement-db` (with an s).
- 2026-10-21: The deadline is 30 Oct 2026 AoE (the competition page). A 20 Nov date seen elsewhere is outdated.

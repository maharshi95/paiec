# Open questions / things to verify

1. **Starting kit** (Codabench ZIP) not downloadable without login: get `check_submission_zip.py`, `run_smoke_test.py`, examples; confirm per-call timeout and the exact acquisition protocol.
2. How are non-binary benchmarks (fractions, mixed) converted to 0/1 in the test pool? Threshold? Check curation guidelines doc / Discord.
3. Which subjects appear at test time (seen in measurement-db vs new)? Are test subjects' other-benchmark responses available at all? (Probably not — only features + ≤31 labels.)
4. Is `interactors` populated in test inputs (agent benchmarks)? Format?
5. Is the benchmark pool inventory (≈300 slugs in the sheet) mostly *unreleased* on HF? Only 6 are public ⇒ small training set; ask on Discord whether more will be released.
6. Codabench submission quota/day, queue time.
7. Dataset id typo: user-provided `aims-foundation/measurement-db` does not exist; correct is `aims-foundations/measurement-db`.
8. Conflicting deadline text (20 Nov vs 30 Oct) — organizers' page says 30 Oct AoE.
9. Allowed pip packages in the hosted env (requirements.txt gated); stay stdlib/numpy-light or ship pure-python/numpy weights.

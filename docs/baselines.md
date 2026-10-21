# Reference baselines (github.com/aims-foundations/paiec_baseline)

| Baseline | Predictor | Acquisition | Needs |
|---|---|---|---|
| Empirical Mean | mean acquired label for (subject, benchmark); 0.5 if none (unsmoothed, can output exact 0/1) | random | stdlib |
| Empirical Mean + BLE acq. | same | uncertainty sampling from a BLE estimate | data payload + API keys |
| BLE (Bayesian Linguistic Evaluator) | LLM agent (default `openai/gpt-5.6-luna`, ≤10 turns, 5 in competition example, 240 s timeout) retrieves *public data payload* evidence with 7 keyword/table tools (+ embedding search, `text-embedding-3-small`) and revises a belief; probabilities clipped [0.02,0.98] | request label if |p−0.5|<0.15 or budget forces | payload from `tools/prepare_data.py`, API keys |

Repo layout: `ble/`, `empirical_mean/`, `empirical_mean_ble_acquisition/`, `tools/` (`build_mean_zip.py`, `prepare_data.py`, tests), `check_submission_zip.py`, `submit_api.py` (Codabench CLI; token in `~/.config/paiec/codabench-token`).
Not yet inspected in detail — TODO clone locally into `external/` (gitignored) and read `ble/src/agent/prompts.py` and the packaging scripts.

Other prior code (collaborator's `cs321m-predictive-eval`, sibling dir): CAIMIRA + empirical-Bayes hybrid (`submission/{model,labeling,train,caimira_lite,_eb_core,anchor_calibration}.py`),
`profiles/neurips/` importer for HF → normalised JSONL, replay/pair-macro metrics, packaging with `prediction_config.json` concurrency. Review later and copy selectively.

export PYTHONPATH := src
CONFIG ?= configs/aims/irt.yaml
.PHONY: data test lint run board zip time
data:  ; python scripts/download_data.py --split
test:  ; python -m pytest -q
lint:  ; ruff check .
run:   ; python scripts/run.py --config $(CONFIG)
board: ; python scripts/leaderboard.py --folds
zip:   ; python scripts/build_submission.py --config $(CONFIG)
time:  ; python scripts/time_predict.py --config $(CONFIG)

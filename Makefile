.PHONY: data test lint zip check
data:        ; python scripts/download_data.py --out data/raw
test:        ; python -m pytest -q
lint:        ; ruff check .
zip:         ; python scripts/build_zip.py --dir $(or $(DIR),submission_template) --out dist/submission.zip
check:       ; python scripts/check_zip.py dist/submission.zip

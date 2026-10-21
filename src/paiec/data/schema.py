"""Column names of measurement-db formatted tables and the competition input contract."""

SUBJECT_COLS = ["subject_id", "display_name", "normalized_name", "provider", "release_date",
                "access_date", "harness", "reasoning_effort", "harness_version",
                "subject_features_extra"]
ITEM_COLS = ["item_id", "benchmark_id", "raw_item_id", "content", "asset_manifest",
             "grading_criterion", "verifier", "content_hash", "item_features"]
RESPONSE_COLS = ["subject_id", "item_id", "response_id", "benchmark_id", "trial",
                 "test_condition", "interactors", "response"]

# input[0]: subject, input[1]: item. ALL values are strings; missing -> "".
INPUT_SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                        "harness_version", "reasoning_effort", "subject_features_extra"]
INPUT_ITEM_FIELDS = ["item_content", "item_features", "interactors", "benchmark_id"]
# Mapping from table columns -> competition input fields
ITEM_FIELD_MAP = {"content": "item_content", "item_features": "item_features",
                  "benchmark_id": "benchmark_id"}  # interactors comes from responses table

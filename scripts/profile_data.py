"""Print per-benchmark stats: scale, sizes, label rate, and eligible pairs (>=80 distinct items)."""
import sys

sys.path.insert(0, "src")
from paiec.data.load import load_table  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
b = load_table(root, "benchmarks")
r = load_table(root, "responses")
cols = ["benchmark_id", "release", "response_type", "n_response_values", "n_subjects", "n_items",
        "n_responses", "max_trial", "coverage"]
print(b[cols].to_string(index=False))
for bid, g in r.groupby("benchmark_id"):
    per = g.groupby("subject_id").item_id.nunique()
    print(f"{bid}: mean={g.response.mean():.3f} null={g.response.isna().mean():.3f} "
          f"values={sorted(g.response.dropna().unique())[:6]} "
          f"pairs>=80items: {(per >= 80).sum()}/{len(per)}")

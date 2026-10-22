"""Print per-benchmark stats: scale, sizes, label rate, and eligible pairs (>=80 distinct items)."""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
for d in sorted(p for p in root.iterdir() if (p / "formatted_tables").is_dir()):
    b = pd.read_parquet(d / "formatted_tables/benchmarks.parquet").iloc[0]
    r = pd.read_parquet(d / "formatted_tables/responses.parquet")
    per = r.groupby("subject_id").item_id.nunique()
    print(f"{d.name}: {b.response_type} ({b.n_response_values} values) | {r.subject_id.nunique()} subjects, "
          f"{r.item_id.nunique()} items, {len(r)} responses, mean {r.response.mean():.3f}, "
          f"pairs>=80 items {(per >= 80).sum()}/{len(per)}")

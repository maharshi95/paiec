"""Cross-benchmark item overlap by items.content_hash (leakage check for eval selection).

Writes <root>/item_overlap.csv: benchmark, n_items, shared_frac (share of its distinct items that also
occur in another benchmark), partners (top overlapping benchmarks with counts).

    python analysis/subjects/item_overlap.py
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/fs/clip-scratch/mgor/measurement-db")


def main():
    hashes = {}
    owner = defaultdict(set)
    for f in sorted(ROOT.glob("*/formatted_tables/items.parquet")):
        b = f.parts[-3]
        hs = {h for h in pq.read_table(f, columns=["content_hash"]).column(0).to_pylist() if h}
        hashes[b] = hs
        for h in hs:
            owner[h].add(b)
    rows = []
    for b, hs in hashes.items():
        partners = defaultdict(int)
        shared = 0
        for h in hs:
            if len(owner[h]) > 1:
                shared += 1
                for o in owner[h] - {b}:
                    partners[o] += 1
        top = sorted(partners.items(), key=lambda x: -x[1])[:5]
        rows.append({"benchmark": b, "n_items": len(hs), "shared_frac": shared / max(len(hs), 1),
                     "partners": ";".join(f"{o}:{c}" for o, c in top)})
    df = pd.DataFrame(rows).sort_values("shared_frac", ascending=False)
    df.to_csv(ROOT / "item_overlap.csv", index=False)
    print(f"{(df.shared_frac > 0.01).sum()} of {len(df)} benchmarks share >1% of items with another benchmark")
    print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()

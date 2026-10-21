"""Load downloaded measurement-db tables and build (subject, item, label) records."""
from pathlib import Path
import pandas as pd

TABLES = ("benchmarks", "subjects", "items", "responses")


def load_table(root, table: str, benchmarks=None) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for d in sorted(p for p in root.iterdir() if (p / "formatted_tables").is_dir()):
        if benchmarks and d.name not in benchmarks:
            continue
        f = d / "formatted_tables" / f"{table}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df["_benchmark"] = d.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_records(root, benchmarks=None) -> pd.DataFrame:
    """One row per response (trial): subject fields + item fields + binary label.

    Only binary-scale benchmarks are directly usable as 0/1 targets; check
    benchmarks.response_scale before including others (see docs/dataset.md).
    """
    r = load_table(root, "responses", benchmarks)
    s = load_table(root, "subjects", benchmarks).drop(columns=["_benchmark"])
    i = load_table(root, "items", benchmarks).drop(columns=["_benchmark", "benchmark_id"])
    s = s.drop_duplicates("subject_id")
    i = i.drop_duplicates("item_id")
    return r.merge(s, on="subject_id", how="left").merge(i, on="item_id", how="left")

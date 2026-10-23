"""Inventory every measurement-db benchmark without bulk-downloading it.

For each top-level benchmark folder: exact sizes of formatted_tables/*.parquet (via the Hub tree
API) and its one-row catalog (benchmarks.parquet: release, response_type, n_subjects, n_items, ...).
Writes <out>/inventory.csv, used by scripts/download_data.py --select to pick benchmarks.

    python scripts/mdb_inventory.py --root /fs/clip-scratch/mgor/measurement-db
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.paths import MDB_ROOT  # noqa: E402

REPO = "aims-foundations/measurement-db"
TABLES = ("benchmarks", "subjects", "items", "responses", "traces", "assets")


def one(api, sha, root, slug):
    row = {"benchmark": slug}
    try:
        for e in api.list_repo_tree(REPO, path_in_repo=f"{slug}/formatted_tables", repo_type="dataset", revision=sha):
            name = e.path.rsplit("/", 1)[-1].removesuffix(".parquet")
            if name in TABLES:
                row[f"mb_{name}"] = round(getattr(e, "size", 0) / 1e6, 3)
        p = hf_hub_download(REPO, f"{slug}/formatted_tables/benchmarks.parquet", repo_type="dataset",
                            revision=sha, local_dir=root)
        cat = pd.read_parquet(p).iloc[0].to_dict()
        for k in ("name", "release", "response_type", "response_scale", "n_response_values", "n_subjects", "n_items",
                  "n_responses", "max_trial", "coverage", "granularity", "multi_single_turn", "license", "release_date"):
            v = cat.get(k)
            row[k] = v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v
        for k in ("modality", "domain"):
            v = cat.get(k)
            row[k] = ",".join(map(str, list(v))) if v is not None and not isinstance(v, str) else (v or "")
    except Exception as e:  # noqa: BLE001 - record and keep going
        row["error"] = repr(e)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=MDB_ROOT)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    api = HfApi()
    sha = api.dataset_info(REPO, revision=a.revision).sha
    slugs = sorted(e.path for e in api.list_repo_tree(REPO, repo_type="dataset", revision=sha)
                   if e.__class__.__name__ == "RepoFolder" and not e.path.startswith("."))
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda s: one(api, sha, a.root, s), slugs))
    df = pd.DataFrame(rows)
    mb = [c for c in df if c.startswith("mb_")]
    df["mb_tables_no_traces"] = df[[c for c in mb if c not in ("mb_traces", "mb_assets")]].sum(axis=1)
    Path(a.root).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.root) / "inventory.csv", index=False)
    (Path(a.root) / "REVISION").write_text(sha + "\n")
    print(f"{len(df)} benchmarks @ {sha[:8]}; errors: {df.get('error', pd.Series()).notna().sum()}")
    print(f"formatted tables w/o traces: {df.mb_tables_no_traces.sum() / 1e3:.1f} GB; "
          f"traces: {df.get('mb_traces', pd.Series(0)).sum() / 1e3:.1f} GB")


if __name__ == "__main__":
    main()

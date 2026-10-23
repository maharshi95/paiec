"""Download data: the ready-made train/val split (start here), or raw measurement-db tables.

    python scripts/download_data.py --split                 # -> data/aims/{train,val}
    python scripts/download_data.py --split val             # one half only

--split fetches the harness's benchmark-level split from the HF datasets mgor/aims-measurement-db-train
and mgor/aims-measurement-db-val (built by scripts/export_split.py; same layout as measurement-db), which
configs/aims/*.yaml read. Everything below is for working with the full upstream dataset instead.

Download measurement-db formatted tables (gated: needs `hf auth login` with accepted terms).

Hub layout: {benchmark}/formatted_tables/{benchmarks,subjects,items,responses,traces,assets}.parquet
(+ raw/ source archives, which we never fetch). 281 benchmark folders at revision 54074453:
core tables total ~14.8 GB, traces ~108 GB (skipped unless --traces).

    python scripts/download_data.py --all                                  # every public benchmark
    python scripts/download_data.py --benchmarks matharena mmdocrag        # a list
    python scripts/download_data.py --config configs/aims/export.yaml --root data/mdb \
        --revision 54074453da7b9911be0f48e4c560be547bc84d21               # a config's universe (rebuild the split)
    python scripts/download_data.py --query "n_subjects >= 10 and response_type == 'binary'"
    python scripts/download_data.py --all --max-mb 1000                    # skip benchmarks > 1 GB

--query/--all/--max-mb read <root>/inventory.csv (scripts/mdb_inventory.py). Files already present
are skipped (HF cache semantics), so reruns resume. The pinned revision goes to <root>/REVISION.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.paths import MDB_ROOT  # noqa: E402

REPO = "aims-foundations/measurement-db"
DEFAULT_ROOT = MDB_ROOT
CORE = ("benchmarks", "subjects", "items", "responses")
SPLIT_REPOS = {"train": "mgor/aims-measurement-db-train", "val": "mgor/aims-measurement-db-val"}


def download_split(which, out, revision=None, workers=8):
    from huggingface_hub import HfApi, snapshot_download
    for name in SPLIT_REPOS if which == "all" else [which]:
        path = snapshot_download(SPLIT_REPOS[name], repo_type="dataset", revision=revision,
                                 local_dir=Path(out) / name, max_workers=workers)
        local = {p.name for p in Path(path).iterdir() if (p / "formatted_tables").is_dir()}
        remote = {f.split("/")[0] for f in HfApi().list_repo_files(SPLIT_REPOS[name], repo_type="dataset", revision=revision)
                  if "/formatted_tables/" in f}
        print(f"{SPLIT_REPOS[name]} -> {path} ({len(remote)} benchmarks)")
        if local - remote:   # snapshot_download never deletes: benchmarks removed from the split stay on disk
            print(f"  warning: stale local benchmarks no longer in the split: {sorted(local - remote)}; "
                  f"delete {path}/<name> (configs/aims/base.yaml ignores the excluded ones)")


def select(root, benchmarks, query, all_, max_mb):
    if benchmarks:
        return list(benchmarks)
    import pandas as pd
    inv = pd.read_csv(Path(root) / "inventory.csv")
    inv = inv[inv.release == "public"]
    if query:
        inv = inv.query(query)
    elif not all_:
        raise SystemExit("pass --benchmarks, --query or --all")
    if max_mb:
        inv = inv[inv.mb_tables_no_traces <= max_mb]
    return inv.benchmark.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="?", const="all", choices=["all", "train", "val"],
                    help="download the ready-made train/val split (default: both) to --root (default data/aims)")
    ap.add_argument("--root", "--out", dest="root", default=None)
    ap.add_argument("--revision", default=None, help="commit sha to pin (default: latest)")
    ap.add_argument("--benchmarks", nargs="*")
    ap.add_argument("--config", help="download the benchmarks a config loads (its data.benchmarks)")
    ap.add_argument("--query", help="pandas query over inventory.csv")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-mb", type=float)
    ap.add_argument("--traces", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.split:
        return download_split(a.split, a.root or "data/aims", a.revision, a.workers)
    a.root = a.root or DEFAULT_ROOT
    sha = HfApi().dataset_info(REPO, revision=a.revision).sha
    if a.config:
        import sys
        from paiec.config import load_config
        d = load_config(a.config)["data"]
        a.benchmarks = [b for b in d["benchmarks"] if b not in set(d.get("exclude") or [])]
    slugs = select(a.root, a.benchmarks, a.query, a.all, a.max_mb)
    tables = CORE + (("traces",) if a.traces else ())
    jobs = [(s, t) for s in slugs for t in tables]
    print(f"{len(slugs)} benchmarks, {len(jobs)} files -> {a.root} @ {sha[:8]}", flush=True)
    t0, done, failed = time.time(), 0, []

    def get(job):
        s, t = job
        return hf_hub_download(REPO, f"{s}/formatted_tables/{t}.parquet", repo_type="dataset",
                               revision=sha, local_dir=a.root)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(get, j): j for j in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - missing table (e.g. no traces) / transient; listed at end
                failed.append((*futs[f], type(e).__name__))
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
    Path(a.root, "REVISION").write_text(sha + "\n")
    print(f"done in {time.time() - t0:.0f}s; {len(failed)} failed: {failed[:20]}")


if __name__ == "__main__":
    main()

"""Download measurement-db tables (gated: needs `hf auth login` with accepted terms).

Layout on the Hub: {benchmark_id}/formatted_tables/{benchmarks,subjects,items,responses}.parquet
(traces.parquet is large, ~1GB total, and skipped unless --traces).
Pins one dataset revision and records it in <out>/REVISION; cite it in every run.
"""
import argparse

from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
# Public benchmarks as of revision 54074453 (2026-10-07). Pass --benchmarks to override.
DEFAULT_BENCHMARKS = ["matharena", "mmdocrag", "multi_swebench", "real_webagents",
                      "researchcodebench", "swe_rebench"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--revision", default=None, help="commit sha to pin (default: latest)")
    ap.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS)
    ap.add_argument("--traces", action="store_true")
    a = ap.parse_args()
    sha = HfApi().dataset_info(REPO, revision=a.revision).sha
    tables = ["benchmarks", "subjects", "items", "responses"] + (["traces"] if a.traces else [])
    for slug in a.benchmarks:
        for t in tables:
            try:
                hf_hub_download(REPO, f"{slug}/formatted_tables/{t}.parquet", repo_type="dataset",
                                revision=sha, local_dir=a.out)
            except Exception as e:  # noqa: BLE001 - e.g. withheld benchmark / missing table
                print("skip", slug, t, type(e).__name__)
    with open(f"{a.out}/REVISION", "w") as f:
        f.write(sha + "\n")
    print("revision", sha)


if __name__ == "__main__":
    main()

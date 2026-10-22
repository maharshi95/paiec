"""Cartesian sweep over dotted config keys; each point is a named run under runs/.

    python scripts/sweep.py --config configs/beta_shrink.yaml \
        --grid predictor.alpha=1,4,16 --grid acquisition.name=accept_all,official_random --jobs 4
"""
import argparse
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

# one BLAS thread per run: parallel runs otherwise oversubscribe the CPU badly
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.config import load_config, set_dotted
from paiec.experiment import run_experiment


def _run(cfg):
    try:
        return run_experiment(cfg, verbose=False)
    except Exception as e:  # noqa: BLE001 - keep the sweep going; report at the end
        return {"name": cfg["name"], "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,...")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    base = load_config(a.config, a.set)
    axes = [(g.split("=", 1)[0], g.split("=", 1)[1].split(",")) for g in a.grid]
    cfgs = []
    for combo in itertools.product(*[vals for _, vals in axes]):
        cfg = load_config(a.config, a.set)
        tag = []
        for (key, _), val in zip(axes, combo):
            set_dotted(cfg, f"{key}={val}")
            parts = key.split(".")
            tag.append(f"{parts[-2] if parts[-1] == 'name' and len(parts) > 1 else parts[-1]}={val}")
        cfg["name"] = base["name"] + ("__" + ",".join(tag) if tag else "")
        cfgs.append(cfg)
    print(f"{len(cfgs)} runs")
    if a.jobs > 1:
        # load the dataset once so workers hit the pickle cache instead of parquet
        from paiec.data.dataset import load_dataset
        d = base["data"]
        load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"],
                          max_responses=d.get("max_responses"), max_item_chars=d.get("max_item_chars"),
                          prefer_seen_from=d.get("prefer_seen_from"))
        with Pool(a.jobs) as pool:
            results = pool.map(_run, cfgs)
    else:
        results = [_run(c) for c in cfgs]
    for r in sorted(results, key=lambda r: r.get("alc", 9)):
        print(f"{r.get('alc', float('nan')):.4f}  {r['name']}  {r.get('error', '')}")


if __name__ == "__main__":
    main()

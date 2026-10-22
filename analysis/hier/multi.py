"""Run several variants serially in one process (dataset loaded once).

    python analysis/hier/multi.py --config configs/research/champion.yaml --set 'cv.seeds=[0]' \
        --var 'hier__a:predictor.name=irt_h;predictor.nu_u=20' --var 'hier__b:...'
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from paiec.config import load_config, set_dotted
from paiec.data.dataset import load_dataset
from paiec.experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--var", action="append", default=[])
    a = ap.parse_args()
    base = load_config(a.config, a.set)
    d = base["data"]
    ds = load_dataset(d["root"], d["benchmarks"], d["binarize"], d["cache_dir"], max_responses=d.get("max_responses"),
                      max_item_chars=d.get("max_item_chars"), prefer_seen_from=d.get("prefer_seen_from"))
    for v in a.var:
        name, _, sets = v.partition(":")
        cfg = load_config(a.config, a.set)
        for s in filter(None, sets.split(";")):
            set_dotted(cfg, s)
        cfg["name"] = name
        print(f"=== {name}: {sets}", flush=True)
        run_experiment(cfg, ds=ds)


if __name__ == "__main__":
    main()

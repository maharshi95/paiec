"""Run one experiment.

    python scripts/run.py --config configs/irt.yaml
    python scripts/run.py --config configs/irt.yaml --set predictor.var_scale=1.5 --name irt_vs1.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.config import load_config
from paiec.experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. cv.seeds=[0]")
    ap.add_argument("--name")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.name:
        cfg["name"] = a.name
    run_experiment(cfg)


if __name__ == "__main__":
    main()

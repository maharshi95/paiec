"""Measure hosted-style runtime (one predict() call per input, cold caches each checkpoint)
and project it to hosted workloads, against the 8 h limit.

    python scripts/time_predict.py --config configs/irt.yaml --benchmark multi_swebench --subjects 20

Hosted cost ~= acq_calls * t_acq + 6 * eval_inputs * t_eval  (+ setup / model loading).
The formative feedback cap is 1,000 subject-item pairs across both pools, so a formative run
is roughly 500 eval inputs (x6 checkpoints) + up to 500 acquisition calls; the final hidden
evaluation may be much larger, so we also print a 100x scenario.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.acquisition import build_policy
from paiec.config import load_config
from paiec.data.dataset import load_from_config
from paiec.predictors import build_predictor
from paiec.sim.replay import build_pairs, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--benchmark", default="multi_swebench")
    ap.add_argument("--subjects", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d = cfg["data"]
    ds = load_from_config(d)
    benches = [b for b in (d["benchmarks"] or ds.benchmarks()) if b in ds.benchmarks()]
    t0 = time.perf_counter()
    pred = build_predictor(cfg["predictor"]).fit(ds.restrict([b for b in benches if b != a.benchmark]))
    fit_s = time.perf_counter() - t0
    pairs = build_pairs(ds, [a.benchmark], max_subjects=a.subjects, max_eval_items=cfg["cv"]["max_eval_items_per_pair"])
    _, t = replay(pred, build_policy(cfg["acquisition"]), ds, pairs, cfg["sim"]["evidence_scope"], True, per_call_eval=True)
    t_acq = t["acq_s"] / max(t["acq_calls"], 1)
    t_eval = t["eval_s"] / max(t["eval_inputs"], 1)
    print(f"fit {fit_s:.1f}s | {len(pairs)} pairs | acq {t['acq_calls']} calls, {1e3 * t_acq:.2f} ms/call | "
          f"eval {t['eval_inputs']} calls, {1e3 * t_eval:.2f} ms/call")
    for label, n_eval, n_acq in (("formative (~1k pairs)", 500, 500), ("100x formative", 50_000, 50_000)):
        hours = (n_acq * t_acq + 6 * n_eval * t_eval) / 3600
        print(f"  {label:>22}: ~{hours * 60:.1f} min  ({100 * hours / 8:.2f}% of 8 h)")


if __name__ == "__main__":
    main()

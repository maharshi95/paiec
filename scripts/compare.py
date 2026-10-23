"""Paired comparison of runs against a control (same split_seed/seeds -> identical pools and draws).

    python scripts/compare.py runs/seen_dev_irt runs/variant_a runs/variant_b
    python scripts/compare.py --control runs/seen_dev_irt runs/variant_*  [--folds] [--budgets]

Δ = mean over matched (subject, benchmark, seed) pairs of variant ALC − control ALC, with a 95% CI
(1.96 × SE over pairs) and the share of pairs where the variant is better. Fold-macro Δ (the
headline `alc` difference) is also reported. Negative Δ = better.

The pair-level CI treats every (pair, seed) row as independent, but seeds reuse the same pools and pairs of one
benchmark share a fitted fold and the benchmark offset, so it is too narrow (about 2x on the research dev set).
`Δ bench-macro` averages within each benchmark first and puts the CI on the spread across benchmarks (normal
approximation, ~30 clusters); prefer it for promotion decisions.

Pools are only identical when both runs loaded the same benchmarks (they are seeded by load-order index), so the
script warns when the runs' data / cv settings differ.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

BUDGETS = ("B0", "B1", "B3", "B7", "B15", "B31")


POOL_KEYS = (("data", "benchmarks"), ("data", "exclude"), ("data", "max_responses"), ("cv", "split_seed"), ("cv", "seeds"),
             ("cv", "min_items"), ("cv", "max_subjects_per_benchmark"), ("cv", "max_eval_items_per_pair"),
             ("cv", "eval_seen_from"), ("cv", "test_benchmarks"))


def pool_cfg(run):
    f = Path(run) / "config.yaml"
    if not f.exists():
        return None
    import yaml
    cfg = yaml.safe_load(f.read_text())
    return {f"{a}.{b}": (sorted(v) if isinstance(v := cfg.get(a, {}).get(b), list) and all(isinstance(x, str) for x in v) else v)
            for a, b in POOL_KEYS}


def load(run):
    run = Path(run)
    with open(run / "pairs.csv") as f:
        rows = {(r["subject_id"], r["benchmark"], r["seed"]): r for r in csv.DictReader(f)}
    return rows, json.loads((run / "metrics.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--control", help="default: first positional run")
    ap.add_argument("--budgets", action="store_true", help="paired Δ per budget")
    ap.add_argument("--folds", action="store_true", help="paired Δ per benchmark (worst/best 5)")
    a = ap.parse_args()
    runs = list(a.runs)
    ctrl = a.control or runs.pop(0)
    c_rows, c_m = load(ctrl)
    print(f"control {Path(ctrl).name}: ALC {c_m['alc']:.4f} (pairs {c_m['alc_pairs']:.4f})")
    c_cfg = pool_cfg(ctrl)
    hdr = "| run | ALC | Δ fold-macro | Δ pairs ± 95% CI | Δ bench-macro ± 95% CI | win % | n |"
    if a.budgets:
        hdr += " " + " | ".join(f"Δ{b}" for b in BUDGETS) + " |"
    print(hdr + "\n|" + "---|" * (hdr.count("|") - 1))
    for run in runs:
        if Path(run).resolve() == Path(ctrl).resolve() or not (Path(run) / "pairs.csv").exists():
            continue
        rows, m = load(run)
        keys = sorted(set(rows) & set(c_rows))
        if not keys:
            print(f"| {Path(run).name} | no matched pairs |")
            continue
        d = np.array([float(rows[k]["alc"]) - float(c_rows[k]["alc"]) for k in keys])
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        by_b = {}
        for k, x in zip(keys, d):
            by_b.setdefault(k[1], []).append(x)
        bm = np.array([np.mean(v) for v in by_b.values()])
        ci_b = 1.96 * bm.std(ddof=1) / np.sqrt(len(bm)) if len(bm) > 1 else float("nan")
        line = (f"| {Path(run).name} | {m['alc']:.4f} | {m['alc'] - c_m['alc']:+.4f} | {d.mean():+.4f} ± {ci:.4f} | "
                f"{bm.mean():+.4f} ± {ci_b:.4f} | {100 * np.mean(d < 0):.0f} | {len(d)} |")
        if a.budgets:
            line += " " + " | ".join(f"{np.mean([float(rows[k][b]) - float(c_rows[k][b]) for k in keys]):+.4f}"
                                     for b in BUDGETS) + " |"
        print(line)
        r_cfg = pool_cfg(run)
        if c_cfg and r_cfg and c_cfg != r_cfg:
            diff = sorted(k for k in c_cfg if c_cfg[k] != r_cfg[k])
            print(f"  warning: control and run differ in {diff}: pools may not match, so the Δ is not paired")
        if len(keys) < len(c_rows):
            print(f"  warning: only {len(keys)}/{len(c_rows)} control pairs matched")
        if a.folds:
            by = {}
            for k, x in zip(keys, d):
                by.setdefault(k[1], []).append(x)
            ranked = sorted((float(np.mean(v)), b, len(v)) for b, v in by.items())
            show = ranked if len(ranked) <= 10 else ranked[:5] + [None] + ranked[-5:]
            for r in show:
                print("    ..." if r is None else f"    {r[1]:>32}: {r[0]:+.4f} (n={r[2]})")


if __name__ == "__main__":
    main()

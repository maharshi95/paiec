"""Markdown table of dev runs: ALC, absolute per-budget Brier (fold-macro, from metrics.json), and paired Δ vs the
champion control and vs item__A3_mix_eb10 (pairs mean ± 95% CI, fold-macro Δ, folds better / 5, win %).

    python analysis/auto/table.py runs/auto__A3 runs/auto__A3_vi05 ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from compare import BUDGETS, load  # noqa: E402

ROOT = Path("/fs/clip-quiz/mgor/aims-neurips-2026")
CTRL = {"champ": ROOT / "paiec/runs/R_champion_dev", "A3": ROOT / "paiec-wt/item/runs/item__A3_mix_eb10"}


def delta(rows, crows, m, cm):
    keys = sorted(set(rows) & set(crows))
    d = np.array([float(rows[k]["alc"]) - float(crows[k]["alc"]) for k in keys])
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    fb = sum(m["folds"][f]["alc"] < cm["folds"][f]["alc"] for f in m["folds"])
    return f"{d.mean():+.4f} ± {ci:.4f} | {m['alc'] - cm['alc']:+.4f} | {fb}/5 | {100 * np.mean(d < 0):.0f}"


def main():
    args = sys.argv[1:]
    for k in ("champ", "A3"):   # optional control overrides: --champ <run> / --A3 <run>
        if f"--{k}" in args:
            i = args.index(f"--{k}")
            CTRL[k] = Path(args[i + 1])
            del args[i:i + 2]
    ctrl = {k: load(v) for k, v in CTRL.items()}
    print("| run | ALC | " + " | ".join(BUDGETS) + " | Δ champ (pairs ± CI) | fm | folds | win% | Δ A3 (pairs ± CI) | fm | folds | win% |")
    print("|" + "---|" * 15)
    for r in args:
        if not (Path(r) / "metrics.json").exists():
            continue
        rows, m = load(r)
        bb = " | ".join(f"{m['by_budget'][b[1:]]:.4f}" for b in BUDGETS)
        print(f"| {Path(r).name} | {m['alc']:.4f} | {bb} | {delta(rows, ctrl['champ'][0], m, ctrl['champ'][1])} | "
              f"{delta(rows, ctrl['A3'][0], m, ctrl['A3'][1])} |")


if __name__ == "__main__":
    main()

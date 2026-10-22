"""Table of all runs (runs/*/metrics.json), sorted by headline ALC.

    python scripts/leaderboard.py [--filter irt] [--folds] [--csv out.csv]
"""
import argparse
import csv
import json
from pathlib import Path

BUDGETS = ("0", "1", "3", "7", "15", "31")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--filter", default="")
    ap.add_argument("--folds", action="store_true", help="add per-benchmark ALC columns")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if a.filter not in m["name"]:
            continue
        row = {"name": m["name"], "alc": m["alc"], "alc_pairs": m["alc_pairs"], "ece_alc": m["ece_alc"],
               **{f"B{b}": m["by_budget"][b] for b in BUDGETS},
               "scope": m["sim"]["evidence_scope"], "min": round((m["timing"]["acq_s"] + m["timing"]["eval_s"]) / 60, 1)}
        if a.folds:
            row.update({k: v["alc"] for k, v in m["folds"].items()})
        rows.append(row)
    rows.sort(key=lambda r: r["alc"])
    if not rows:
        print("no runs")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

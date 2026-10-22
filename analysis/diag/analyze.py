"""Summarise out/fold*.pkl (+ bench.pkl) into markdown tables printed to stdout.

    python analysis/diag/analyze.py > analysis/diag/out/summary.md
"""
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
BUD = (0, 1, 3, 7, 15, 31)
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def load_folds():
    fs = []
    for k in range(5):
        p = OUT / f"fold{k}.pkl"
        if p.exists():
            f = pickle.load(open(p, "rb"))
            pc = OUT / f"fold{k}_cf.pkl"
            if pc.exists() and "replays" in f:
                g = pickle.load(open(pc, "rb"))
                for v, rr in g.get("replays", {}).items():
                    if v != "champion":
                        f["replays"]["cf_" + v] = rr
                f["oracle_vals_cf"] = g["oracle_vals"]
            fs.append(f)
    return fs


def fmt(x, d=4):
    return f"{x:.{d}f}"


def oracle_table(fs):
    print("## 1. Oracle decomposition (seed 0, dev folds: %s)\n" % [f["fold"] for f in fs])
    rows = {}
    for f in fs:
        for v, rr in f.get("replays", {}).items():
            B = np.array([[r[f"B{n}"] for n in BUD] for r in rr])
            rows.setdefault(v, []).append(B)
        npairs = len(f["pairs"])
        for m, vals in f["static"].items():
            B = np.repeat(np.array(vals)[:, None], 6, 1)
            rows.setdefault("static:" + m, []).append(B)
        assert all(len(v) == npairs for v in f["static"].values())
    ch = rows.get("champion")
    if ch is None:
        return
    ch_pairs = np.concatenate(ch)
    ch_alc = ch_pairs @ W
    print("| variant | ALC (pairs) | fold-macro | Δ vs champion ± 95% CI | B0 | B1 | B3 | B7 | B15 | B31 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    out = {}
    for v, Bs in rows.items():
        if len(Bs) != len(ch):
            continue
        P = np.concatenate(Bs)
        alc = P @ W
        dlt = alc - ch_alc
        ci = 1.96 * dlt.std() / np.sqrt(len(dlt))
        fm = np.mean([b @ W for b in Bs], axis=1).mean() if False else np.mean([np.mean(b @ W) for b in Bs])
        out[v] = (alc.mean(), P.mean(0))
        print(f"| {v} | {fmt(alc.mean())} | {fmt(fm)} | {dlt.mean():+.4f} ± {ci:.4f} | "
              + " | ".join(fmt(x, 3) for x in P.mean(0)) + " |")
    print(f"\n(n = {len(ch_alc)} pairs)\n")
    return out


def calib(fs):
    print("## 2. Calibration of the champion (pooled, pair-weighted)\n")
    data = []   # per budget: list of (p, y, w, fold)
    for f in fs:
        rr = f.get("replays", {}).get("champion")
        if not rr:
            continue
        for r in rr:
            ys = r["ys"]
            nr = sum(len(y) for y in ys)
            for n in BUD:
                p = r["probs"][n]
                for pk, y in zip(p, ys):
                    data.append((n, float(pk), float(y.sum()), len(y), 1.0 / nr, f["fold"], r["benchmark"]))
    if not data:
        return
    a = np.array([d[:6] for d in data], float)
    n_, p_, ysum, cnt, w, fold = a.T
    bench = np.array([d[6] for d in data])
    npairs = sum(len(f["replays"]["champion"]) for f in fs if "replays" in f)

    def brier_w(p, m):   # pair-macro Brier for budget mask m: sum w*(sum over trials (p-y)^2) / npairs
        se = cnt[m] * p ** 2 - 2 * p * ysum[m] + ysum[m]
        return float((w[m] * se).sum() / npairs)

    print("Reliability (10 equal-width bins; trial-weighted within pair, pairs equal):\n")
    print("| budget | Brier | mean p | mean y | " + " | ".join(f"bin{k}" for k in range(10)) + " |")
    print("|---|---|---|---|" + "---|" * 10)
    for n in BUD:
        m = n_ == n
        wt = w[m] * cnt[m]
        yb = ysum[m] / cnt[m]
        k = np.minimum((p_[m] * 10).astype(int), 9)
        cells = []
        for b in range(10):
            mb = k == b
            if wt[mb].sum() < 1e-9:
                cells.append("–")
                continue
            cells.append(f"{np.average(p_[m][mb], weights=wt[mb]):.2f}/{np.average(yb[mb], weights=wt[mb]):.2f} ({wt[mb].sum() / wt.sum() * 100:.0f}%)")
        print(f"| B{n} | {brier_w(p_[m], m):.4f} | {np.average(p_[m], weights=wt):.3f} | {np.average(yb, weights=wt):.3f} | " + " | ".join(cells) + " |")
    print("\nCell = mean predicted / observed (share of weight).\n")
    from scipy.optimize import minimize
    from sklearn.isotonic import IsotonicRegression
    lg = np.log(np.clip(p_, 1e-6, 1 - 1e-6) / np.clip(1 - p_, 1e-6, 1))

    def sig(x):
        return 1 / (1 + np.exp(-x))

    def fit_platt(m, temp_only=False):
        def obj(t):
            a_, b_ = (t[0], 0.0) if temp_only else t
            return brier_w(sig(a_ * lg[m] + b_), m)
        r = minimize(obj, [1.0] if temp_only else [1.0, 0.0], method="Nelder-Mead")
        return (r.x[0], 0.0) if temp_only else tuple(r.x)

    base = {n: brier_w(p_[n_ == n], n_ == n) for n in BUD}
    res = {"none": base}
    allm = np.ones(len(p_), bool)
    a_g, _ = fit_platt(allm, True)
    res["global T (in-sample)"] = {n: brier_w(sig(a_g * lg[n_ == n]), n_ == n) for n in BUD}
    res["per-budget T (in-sample)"], res["per-budget Platt (in-sample)"], res["per-budget isotonic (in-sample)"] = {}, {}, {}
    res["per-budget Platt (leave-fold-out)"], res["per-budget isotonic (leave-fold-out)"] = {}, {}
    temps = {}
    for n in BUD:
        m = n_ == n
        a1, _ = fit_platt(m, True)
        temps[n] = 1 / a1
        res["per-budget T (in-sample)"][n] = brier_w(sig(a1 * lg[m]), m)
        a2, b2 = fit_platt(m)
        res["per-budget Platt (in-sample)"][n] = brier_w(sig(a2 * lg[m] + b2), m)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_[m], ysum[m] / cnt[m], sample_weight=w[m] * cnt[m])
        res["per-budget isotonic (in-sample)"][n] = brier_w(iso.predict(p_[m]), m)
        # leave-one-fold-out
        pl, pi = np.empty(m.sum()), np.empty(m.sum())
        fm = fold[m]
        for fo in np.unique(fold):
            tr = m & (fold != fo)
            if not tr.any():
                pl[fm == fo], pi[fm == fo] = p_[m][fm == fo], p_[m][fm == fo]
                continue
            te = fm == fo
            a3, b3 = fit_platt(tr)
            pl[te] = sig(a3 * lg[m][te] + b3)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_[tr], ysum[tr] / cnt[tr], sample_weight=w[tr] * cnt[tr])
            pi[te] = iso.predict(p_[m][te])
        res["per-budget Platt (leave-fold-out)"][n] = brier_w(pl, m)
        res["per-budget isotonic (leave-fold-out)"][n] = brier_w(pi, m)
    print(f"Global temperature T = {1 / a_g:.3f} (T > 1: over-confident). Per-budget T: "
          + ", ".join(f"B{n} {temps[n]:.2f}" for n in BUD) + "\n")
    print("| recalibration | ALC | ΔALC | " + " | ".join(f"B{n}" for n in BUD) + " |")
    print("|---|---|---|" + "---|" * 6)
    a0 = np.dot(W, [base[n] for n in BUD])
    for k, v in res.items():
        alc = np.dot(W, [v[n] for n in BUD])
        print(f"| {k} | {alc:.4f} | {alc - a0:+.4f} | " + " | ".join(f"{v[n]:.4f}" for n in BUD) + " |")
    # per-benchmark bias (calibration-in-the-large)
    print("\nPer-benchmark calibration-in-the-large (mean p − mean y), selected budgets:\n")
    print("| benchmark | " + " | ".join(f"B{n}" for n in BUD) + " |")
    print("|---|" + "---|" * 6)
    for b in sorted(set(bench)):
        cells = []
        for n in BUD:
            m = (n_ == n) & (bench == b)
            wt = w[m] * cnt[m]
            cells.append(f"{np.average(p_[m], weights=wt) - np.average(ysum[m] / cnt[m], weights=wt):+.3f}")
        print(f"| {b} | " + " | ".join(cells) + " |")
    print()


def where(fs):
    print("## 3. Where the loss is (per test benchmark, seed 0)\n")
    print("| benchmark | pairs | items | subj | trials/cell | base | MCQ | champion ALC | B0 | B1 | B31 | rasch_cf (B∞) | 4pl_cf | lr2_cf | floor | rasch_full | o_beta ALC | "
          "within-bm item R² (all feats) | ρ | prior ρ |")
    print("|" + "---|" * 20)
    allrows = []
    for f in fs:
        if "replays" not in f:
            continue
        ch = f["replays"]["champion"]
        ob = f["replays"].get("o_beta")
        names = [b for _, b in f["pairs"]]
        for b, info in f["bench"].items():
            idx = [k for k, x in enumerate(names) if x == b]
            r = [ch[k] for k in idx]
            st = {m: np.mean([f["static"][m][k] for k in idx]) for m in f["static"]}
            alc = np.mean([x["alc"] for x in r])
            it = f["items"][b]["text+length+kv"]
            row = dict(b=b, n=len(idx), items=info["n_items"], subj=info["n_subjects"], tpc=info["trials_per_cell"],
                       base=info["base_rate"], mcq=info["mcq_frac"], alc=alc, B0=np.mean([x["B0"] for x in r]),
                       B1=np.mean([x["B1"] for x in r]), B31=np.mean([x["B31"] for x in r]), floor=np.mean(info["pair_floor"]),
                       ob=np.mean([ob[k]["alc"] for k in idx]) if ob else np.nan, **st, r2=it[0], rho=it[1],
                       prho=f["items"][b]["prior_yh_spearman"])
            allrows.append(row)
    for r in sorted(allrows, key=lambda r: -r["alc"]):
        print(f"| {r['b']} | {r['n']} | {r['items']} | {r['subj']} | {r['tpc']:.1f} | {r['base']:.2f} | {r['mcq']:.2f} | {r['alc']:.3f} | "
              f"{r['B0']:.3f} | {r['B1']:.3f} | {r['B31']:.3f} | {r['rasch_cf']:.3f} | {r['4pl_cf']:.3f} | {r['lr2_cf']:.3f} | {r['floor']:.3f} | "
              f"{r['rasch_full']:.3f} | {r['ob']:.3f} | {r['r2']:.2f} | {r['rho']:.2f} | {r['prho']:.2f} |")
    print()
    return allrows


def groups(allrows):
    if not allrows:
        return
    print("Group means (benchmark-macro): champion ALC, B∞ cross-fit Rasch, gap\n")
    print("| group | n bm | champion ALC | rasch_cf | floor | ALC − rasch_cf |")
    print("|---|---|---|---|---|---|")
    G = [("MCQ-like (≥50% items)", lambda r: r["mcq"] >= 0.5), ("open / other", lambda r: r["mcq"] < 0.5),
         ("extreme base rate (<0.2 or >0.8)", lambda r: r["base"] < 0.2 or r["base"] > 0.8),
         ("mid base rate", lambda r: 0.2 <= r["base"] <= 0.8),
         ("repeated trials (>1.2/cell)", lambda r: r["tpc"] > 1.2), ("single trial", lambda r: r["tpc"] <= 1.2),
         ("<300 items", lambda r: r["items"] < 300), ("300–2000 items", lambda r: 300 <= r["items"] < 2000),
         (">=2000 items", lambda r: r["items"] >= 2000)]
    for name, fn in G:
        rr = [r for r in allrows if fn(r)]
        if rr:
            print(f"| {name} | {len(rr)} | {np.mean([r['alc'] for r in rr]):.3f} | {np.mean([r['rasch_cf'] for r in rr]):.3f} | "
                  f"{np.mean([r['floor'] for r in rr]):.3f} | {np.mean([r['alc'] - r['rasch_cf'] for r in rr]):+.3f} |")
    print()


def items(fs):
    print("## 5. Within-benchmark item transfer (random half → other half; 5 reps; target = in-sample centred d)\n")
    keys = None
    acc = {}
    for f in fs:
        for b, tr in f["items"].items():
            for k, v in tr.items():
                acc.setdefault(k, []).append(v)
    print("| features | mean R² | median R² | mean Spearman | median Spearman |")
    print("|---|---|---|---|---|")
    for k, v in acc.items():
        if isinstance(v[0], tuple):
            r2 = np.array([x[0] for x in v]); sp = np.array([x[1] for x in v])
            print(f"| {k} | {r2.mean():.3f} | {np.median(r2):.3f} | {np.nanmean(sp):.3f} | {np.nanmedian(sp):.3f} |")
    print(f"\nCross-benchmark prior ŷ (champion item ridge) Spearman with in-sample d on the same benchmarks: "
          f"mean {np.mean(acc['prior_yh_spearman']):.3f}, median {np.median(acc['prior_yh_spearman']):.3f}. "
          f"Duplicate-content item fraction: mean {np.mean(acc['dup_content_frac']):.3f}.\n")


def bench_wide():
    p = OUT / "bench.pkl"
    if not p.exists():
        return
    B = pickle.load(open(p, "rb"))
    import pandas as pd
    old = pd.read_csv("/fs/clip-quiz/mgor/aims-neurips-2026/paiec/runs/dev_irt/pairs.csv").groupby("benchmark").alc.mean().to_dict()
    print("## 3b. Static ceilings on old-dev (45) ∪ current-dev (30) benchmarks (seed 0 eval pools)\n")
    print("| benchmark | old-dev IRT ALC | cur dev | pairs | items | base | pair-base sd | MCQ | const | base_cf | rasch_cf | 4pl_cf (lo/hi) | rasch_full | floor |")
    print("|" + "---|" * 14)
    for name, r in sorted(B.items(), key=lambda kv: np.mean(kv[1]["rasch_cf"]) - 0.25, reverse=True):
        print(f"| {name} | {old.get(name, float('nan')):.3f}{' **(lost)**' if old.get(name, 0) > 0.25 else ''} | {'y' if r['in_current_dev'] else ''} | {r['n_pairs']} | {r['n_items']} | "
              f"{r['base_rate']:.2f} | {r['pair_base_sd']:.2f} | {r['mcq_frac']:.2f} | {np.mean(r['const']):.3f} | {np.mean(r['base_cf']):.3f} | "
              f"{np.mean(r['rasch_cf']):.3f} | {np.mean(r['4pl_cf']):.3f} ({r['asym'][0]:.2f}/{r['asym'][1]:.2f}) | {np.mean(r['rasch_full']):.3f} | "
              f"{np.mean(r['pair_floor']):.3f} |")
    print()


if __name__ == "__main__":
    fs = load_folds()
    oracle_table(fs)
    calib(fs)
    rows = where(fs)
    groups(rows)
    items(fs)
    bench_wide()

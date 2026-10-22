"""Extra per-benchmark numbers: oracle gains by benchmark, label coverage of eval items, test-benchmark spreads."""
import pickle
import numpy as np
from analyze import load_folds, W
fs = load_folds()
rows = []
for f in fs:
    names = [b for _, b in f["pairs"]]
    R = f["replays"]
    for b, info in f["bench"].items():
        idx = [k for k, x in enumerate(names) if x == b]
        a = {v: np.mean([R[v][k]["alc"] for k in idx]) for v in ("champion", "cf_o_d", "cf_o_beta", "cf_o_delta", "cf_o_all")}
        S, N = len(idx), info["n_items"]
        cov = {n: 1 - (1 - min(n / N, 1)) ** max(S - 1, 0) for n in (1, 7, 31)}
        ov, oc = f["oracle_vals"][b], f["oracle_vals_cf"][b]
        rows.append((b, S, N, a, cov, ov["sd_d"], oc["sd_d"], oc["sd_delta"], oc["beta_prime"]))
print("| benchmark | pairs | items | champion | Δ cf_o_d | Δ cf_o_beta | Δ cf_o_delta | B∞ | P(eval item labelled by another pair) B7 / B31 | sd d (cf) | sd δ (cf) | β' (cf) |")
print("|" + "---|" * 12)
for b, S, N, a, cov, sdf, sdc, sdl, bp in sorted(rows, key=lambda r: r[3]["cf_o_d"] - r[3]["champion"]):
    c = a["champion"]
    print(f"| {b} | {S} | {N} | {c:.3f} | {a['cf_o_d'] - c:+.3f} | {a['cf_o_beta'] - c:+.3f} | {a['cf_o_delta'] - c:+.3f} | {a['cf_o_all']:.3f} | {cov[7]:.2f} / {cov[31]:.2f} | {sdc:.2f} | {sdl:.2f} | {bp:+.2f} |")
cov31 = np.array([r[4][31] for r in rows]); g = np.array([r[3]["cf_o_d"] - r[3]["champion"] for r in rows])
print(f"\ncorr(coverage B31, o_d gain) = {np.corrcoef(cov31, g)[0,1]:.2f}; median coverage B7 {np.median([r[4][7] for r in rows]):.2f}, B31 {np.median(cov31):.2f}")
print("sd_d (cf) median %.2f, IQR %.2f-%.2f; sd_delta (cf) median %.2f; |beta'| median %.2f" % (np.median([r[6] for r in rows]), *np.percentile([r[6] for r in rows], [25, 75]), np.median([r[7] for r in rows]), np.median([abs(r[8]) for r in rows])))
print("prior sds: " + ", ".join(f"fold {f['fold']}: d {np.sqrt(f['vars']['var_d']):.2f} u {np.sqrt(f['vars']['var_u']):.2f} beta {np.sqrt(f['vars']['var_beta']):.2f}" for f in fs))
ch = np.concatenate([[r["alc"] for r in f["replays"]["champion"]] for f in fs])
print(f"champion pairs with ALC > 0.25: {np.mean(ch > 0.25)*100:.0f}% ({(ch > 0.25).sum()}/{len(ch)}); benchmarks with mean ALC > 0.25: {sum(r[3]['champion'] > 0.25 for r in rows)}/{len(rows)}")

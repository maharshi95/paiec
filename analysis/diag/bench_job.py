"""Q3: static ceilings and characteristics for a wider benchmark set (no replay).

For each benchmark: eval pairs as in the dev protocol (seed 0), and per-pair Brier on eval items of
  const 0.5, in-sample / cross-fitted per-pair base rate, in-sample / cross-fitted Rasch, cross-fitted 4PL,
plus the trial-noise floor and benchmark characteristics. Writes out/bench.pkl.
"""
import pickle
import sys
import time

import numpy as np
from common import OUT, BenchCells, allowed_subjects, fit_model, load

import fold_job
from fold_job import MCQ, pair_brier, static_eval
from paiec.features.subject import subject_key
from paiec.sim.replay import build_pairs

OLD_DEV = ['wonderbread', 'stanford_orb', 'financebench', 'devbench', 'llms_lean_formalization', 'kernelbench', 'prism',
           'vl_rewardbench', 'multi_swebench', 'agentdojo', 'nyu_ctf_bench', 'agi_elo', 'mtbench', 'moe_cap', 'ceval',
           'visit_bench', 'crow', 'swebench_live', 'real_webagents', 'reliancescope', 'chatgpt_drift', 'benger', 'statqa',
           'tumlu', 'lingoly', 'harmbench', 'theagentcompany', 'proteininvbench', 'sketchjudge', 'mmlupro', 'legalbench',
           'icpc2_code_selector', 'worldcentralbanks', 'planbench', 'atmossci_bench', 'mochi', 'dpai', 'dataclawbench',
           'truthfulqa_mc', 'goodai_ltm_benchmark', 'mmoral_bench', 'bedd_basalt', 'mj_bench', 'engdesign', 'image2struct']


def main():
    t0 = time.time()
    cfg, ds = load()
    fold_job.DS = ds
    allowed = allowed_subjects(cfg, ds)
    cur = [b for g in cfg["cv"]["test_benchmarks"] for b in g]
    names = sorted(set(OLD_DEV) | set(cur))
    out = {}
    for name in names:
        if name not in ds.bench_names:
            continue
        b = ds.bench_names.index(name)
        prs = build_pairs(ds, [name], 0, 0, 80, 20, 100, allowed if name in cur else None)   # old dev had no seen filter
        if not prs:
            continue
        full = BenchCells(ds, b)
        excl = np.zeros(len(ds.resp_s), bool)
        acq_src = {pr.subject: {ds.item_ids[i].split("|", 1)[0] for i in pr.acq} for pr in prs}
        for k in np.nonzero(ds.resp_b == b)[0]:
            s = int(ds.resp_s[k])
            if s in acq_src and ds.item_ids[ds.resp_i[k]].split("|", 1)[0] not in acq_src[s]:
                excl[k] = True
        cf = BenchCells(ds, b, ~excl)

        def base(pr, c, smooth):
            m = c.P == c.skeys[subject_key(ds.subjects[pr.subject])]
            return (c.ysum[m].sum() + smooth) / (c.n[m].sum() + 2 * smooth)
        r = {"const": [pair_brier(np.full(len(pr.evl), 0.5), pr.evl_ys) for pr in prs],
             "base_full": [pair_brier(np.full(len(pr.evl), base(pr, full, 0)), pr.evl_ys) for pr in prs],
             "base_cf": [pair_brier(np.full(len(pr.evl), base(pr, cf, 1)), pr.evl_ys) for pr in prs]}
        for tag, cells, sm in (("full", full, 0), ("cf", cf, 1)):
            for kind in ("rasch", "4pl"):
                if tag == "full" and kind == "4pl":
                    continue
                f = fit_model(cells, kind)
                r[f"{kind}_{tag}"] = static_eval(prs, cells, f, lambda pr, c=cells, s=sm: base(pr, c, s))
                if kind == "4pl":
                    r["asym"] = f["asym"].tolist()
        mbr = ds.resp_b == b
        items_b = [ds.items[full.items_ds[j]] for j in range(full.nI)]
        r["pair_floor"] = [sum(float(((y - y.mean()) ** 2).sum()) for y in pr.evl_ys) / sum(len(y) for y in pr.evl_ys) for pr in prs]
        r.update({"n_items": full.nI, "n_subjects": full.nS, "trials_per_cell": float(full.n.mean()),
                  "base_rate": float(ds.resp_y[mbr].mean()), "n_pairs": len(prs),
                  "pair_base_sd": float(np.std([base(pr, full, 0) for pr in prs])),
                  "mcq_frac": float(np.mean([bool(MCQ.search(it["item_content"][:6000] + "\n" + it["item_features"])) for it in items_b])),
                  "sample": items_b[0]["item_content"][:200], "in_current_dev": name in cur, "in_old_dev": name in OLD_DEV})
        out[name] = r
        print(f"{name:>28} pairs {len(prs):>3} const {np.mean(r['const']):.3f} base_cf {np.mean(r['base_cf']):.3f} "
              f"rasch_cf {np.mean(r['rasch_cf']):.3f} 4pl_cf {np.mean(r['4pl_cf']):.3f} rasch_full {np.mean(r['rasch_full']):.3f} "
              f"floor {np.mean(r['pair_floor']):.3f} {time.time() - t0:.0f}s", flush=True)
        sys.stdout.flush()
    with open(OUT / "bench.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()

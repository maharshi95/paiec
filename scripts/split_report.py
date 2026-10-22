"""Sanity-check and describe a benchmark split (train pool / dev / hold-out) -> report folder.

    python scripts/split_report.py --split configs/split_seen --out reports/split_seen

Inputs (all under --root, produced by scripts/benchmark_report.py and analysis/subjects/*):
  benchmark_report.csv, ref6_overlap.csv, item_overlap.csv, all_subjects.parquet
plus the split's dev_base.yaml / holdout_base.yaml / eval_benchmarks.csv and, if present, a
replay run (runs/<prefix>dev_constant/pairs.csv) to verify evaluated pairs.

Outputs: report.md (sanity checks + key tables), figures/*.png, tables/*.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paiec.data.dataset import _binarize, parse_scale  # noqa: E402
from paiec.features.identity import model_identity  # noqa: E402

REF6 = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
# reference palette (dataviz skill): first three categorical slots validate all-pairs (CVD) in light mode
C = {"train": "#2a78d6", "dev": "#eb6834", "holdout": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"train": "train pool", "dev": "dev (eval)", "holdout": "hold-out (eval)"}
PROVIDERS = [("OpenAI", r"\bgpt|\bo[1345]\b|o[1345] mini|openai|chatgpt|davinci"), ("Anthropic", r"claude"),
             ("Google", r"gemini|gemma|palm|bard"), ("Meta", r"llama"), ("Alibaba", r"qwen|qwq"),
             ("DeepSeek", r"deepseek"), ("Mistral", r"mistral|mixtral|codestral|ministral"),
             ("xAI", r"grok"), ("Moonshot", r"kimi"), ("Microsoft", r"\bphi\b|phi \d"), ("Zhipu", r"\bglm"),
             ("Amazon", r"nova|titan")]
SIZE = re.compile(r"(\d+(?:\.\d+)?)b\b")
THINK = re.compile(r"think|reason|\br1\b|\bo[134]\b|xhigh|qwq")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 130})


def provider_of(row):
    p = str(row.get("provider") or "")
    if p and p.lower() not in ("none", "unknown") and not p.startswith("("):
        return p
    text = " ".join(str(row.get(k) or "") for k in ("normalized_name", "display_name", "subject_features_extra")).lower()
    for name, pat in PROVIDERS:
        if re.search(pat, text):
            return name
    return "Other / unknown"


def subject_props(S):
    S = S.copy()
    recs = [{k: ("" if pd.isna(v) else str(v)) for k, v in r.items()} for r in S.to_dict("records")]
    S["provider2"] = [provider_of(r) for r in recs]
    S["identity"] = [model_identity(r, include_display=True) for r in recs]
    text = (S.normalized_name.fillna("") + " " + S.display_name.fillna("")).str.lower()
    sizes = text.map(lambda t: max([float(x) for x in SIZE.findall(t)], default=np.nan))
    S["size_b"] = sizes
    S["think"] = text.str.contains(THINK) | S.reasoning_effort.fillna("").isin(["medium", "high", "xhigh"])
    S["release_year"] = pd.to_numeric(S.release_date.fillna("").str[:4], errors="coerce")
    S["has_harness"] = S.harness.fillna("").str.len() > 0
    return S


def load_split(split):
    dev = yaml.safe_load(open(split / "dev_base.yaml"))
    ho = yaml.safe_load(open(split / "holdout_base.yaml"))
    folds = dev["cv"]["test_benchmarks"]
    holdout = [b for f in ho["cv"]["test_benchmarks"] for b in f]
    dev_list = [b for f in folds for b in f]
    train = [b for b in ho["data"]["benchmarks"] if b not in set(dev_list) | set(holdout)]
    return dev, ho, folds, dev_list, holdout, train


def item_accuracy(root, benches, seen_ids, only_seen):
    """Per benchmark: item mean accuracy (seen models only for eval benchmarks) + per-subject accuracy."""
    out_items, out_subj = [], []
    for b in benches:
        t = root / b / "formatted_tables"
        cat = pd.read_parquet(t / "benchmarks.parquet").iloc[0]
        r = pd.read_parquet(t / "responses.parquet", columns=["subject_id", "item_id", "response"])
        y, keep, _ = _binarize(b, r.response.values, {"mode": "threshold", "threshold": 0.5},
                               parse_scale(cat.get("response_scale")))
        r = r[keep].assign(y=y[keep])
        if only_seen:
            r = r[r.subject_id.isin(seen_ids.get(b, set()))]
        if r.empty:
            continue
        ia = r.groupby("item_id").y.mean()
        sa = r.groupby("subject_id").y.mean()
        out_items.append(pd.DataFrame({"benchmark": b, "item_acc": ia.values}))
        out_subj.append(pd.DataFrame({"benchmark": b, "subject_acc": sa.values}))
    return pd.concat(out_items), pd.concat(out_subj)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="configs/split_seen")
    ap.add_argument("--root", default="/fs/clip-scratch/mgor/measurement-db")
    ap.add_argument("--out", default="reports/split_seen")
    ap.add_argument("--run-prefix", default="seen_")
    a = ap.parse_args()
    split, root, out = Path(a.split), Path(a.root), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    style()
    dev_cfg, ho_cfg, folds, dev_list, holdout, train = load_split(split)
    rep = pd.read_csv(root / "benchmark_report.csv")
    ov = pd.read_csv(root / "ref6_overlap.csv")
    io = pd.read_csv(root / "item_overlap.csv").set_index("benchmark")
    S = subject_props(pd.read_parquet(root / "all_subjects.parquet"))
    group = {**{b: "train" for b in train}, **{b: "dev" for b in dev_list}, **{b: "holdout" for b in holdout}}
    B = ov[ov.benchmark.isin(group)].copy()
    B["group"] = B.benchmark.map(group)
    B["dom1"] = B.domain.fillna("other").str.split(",").str[0]
    B["modality1"] = B.modality.fillna("text").str.split(",").str[0]
    B["shared_frac"] = B.benchmark.map(io.shared_frac).fillna(0)
    B.to_csv(out / "tables" / "benchmarks.csv", index=False)

    # ---------------- sanity checks ----------------
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    dev_universe = set(dev_cfg["data"]["benchmarks"])
    check("Hold-out benchmarks absent from the dev universe (never trained on during dev)",
          not (set(holdout) & dev_universe), f"{len(set(holdout) & dev_universe)} overlap")
    check("Dev folds are disjoint", len(dev_list) == len(set(dev_list)), f"{len(dev_list)} benchmarks in {len(folds)} folds")
    check("Reference benchmarks are training-only", not (set(REF6) & (set(dev_list) | set(holdout))))
    worst = B[B.group != "train"].shared_frac.max()
    check("Eval benchmarks share <= 1% of items with any other benchmark", worst <= 0.01, f"max {worst:.2%}")
    ev = B[B.group != "train"]
    check(">= 6 distinct ref6 models with >= 80-item pairs per eval benchmark",
          (ev.seen_eval_identities >= 6).all(), f"min {int(ev.seen_eval_identities.min())}")
    check("Eval base rates within [0.05, 0.95]", ev.base_rate.between(0.05, 0.95).all(),
          f"range {ev.base_rate.min():.2f}-{ev.base_rate.max():.2f}")
    check("Every benchmark has a binarisation rule (not excluded)", (B.rule != "excluded").all())
    check("Eval domain cap (<= 4 per primary domain)", ev.dom1.value_counts().max() <= 4,
          f"max {ev.dom1.value_counts().max()}")
    runp = Path("runs") / f"{a.run_prefix}dev_constant" / "pairs.csv"
    if runp.exists():
        P = pd.read_csv(runp)
        check("Replay: every evaluated pair has >= 80 distinct items", (P.n_acq >= 40).all(), f"{len(P)} pairs")
        check("Replay: every evaluated pair reveals 31 labels", (P.n_labels == 31).all())
    else:
        check("Replay pairs file present", False, f"{runp} missing (run the split first)")

    # ---------------- subjects ----------------
    seen_ids = {}
    S_in = S[S.benchmark.isin(group)].copy()
    S_in["group"] = S_in.benchmark.map(group)
    for b in dev_list + holdout:
        seen_ids[b] = set(S_in[(S_in.benchmark == b) & S_in.ref6_seen].subject_id)
    S_eval = S_in[(S_in.group != "train") & S_in.ref6_seen].copy()
    S_train = S_in[S_in.group == "train"].copy()
    for d in (S_eval, S_train):   # each benchmark contributes equally (scale_mrt alone has 626 seen configs)
        d["w"] = 1.0 / d.groupby("benchmark").benchmark.transform("size")

    # ---------------- figures ----------------
    figs = {}
    # 1. domains
    dom = B.groupby(["dom1", "group"]).size().unstack(fill_value=0).reindex(columns=["train", "dev", "holdout"], fill_value=0)
    dom = dom.loc[dom.sum(1).sort_values().index].tail(18)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(dom))
    left = np.zeros(len(dom))
    for g in ["train", "dev", "holdout"]:
        ax.barh(y, dom[g], left=left, color=C[g], label=LABEL[g], height=0.7, edgecolor=SURF, linewidth=1.5)
        left += dom[g].values
    for yy, tot in zip(y, left):
        ax.text(tot + 0.4, yy, f"{int(tot)}", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, dom.index)
    ax.set_xlabel("benchmarks")
    ax.set_title("Benchmarks per primary domain")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    figs["domains"] = save(fig, out, "domains")

    # 2. response type & modality shares
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in ((axes[0], "rule", "Label type (binarisation rule)"), (axes[1], "modality1", "Primary modality")):
        tab = pd.crosstab(B.group, B[col], normalize="index").reindex(["train", "dev", "holdout"])
        cats = tab.sum().sort_values(ascending=False).index[:4]
        tab = pd.concat([tab[cats], tab.drop(columns=cats).sum(1).rename("other")], axis=1)
        tab = tab.loc[:, tab.sum() > 0]
        left = np.zeros(3)
        shades = ["#1f5fae", "#5f97df", "#9cbfec", "#cfdff6", "#e9eff9"]
        for k, c in enumerate(tab.columns):
            ax.barh(range(3), tab[c], left=left, color=shades[k % len(shades)], edgecolor=SURF, linewidth=1.5, label=c)
            for yy, (l, w) in enumerate(zip(left, tab[c])):
                if w > 0.08:
                    ax.text(l + w / 2, yy, f"{w:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if k < 2 else INK)
            left += tab[c].values
        ax.set_yticks(range(3), [LABEL[g] for g in tab.index])
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", visible=False)
    figs["types"] = save(fig, out, "types")

    # 3. benchmark size and base rate
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    rng = np.random.default_rng(0)
    for ax, col, title, log in ((axes[0], "n_subjects", "Subjects per benchmark", True),
                                (axes[1], "n_items", "Items per benchmark", True),
                                (axes[2], "base_rate", "Base rate (mean label)", False)):
        for k, g in enumerate(["train", "dev", "holdout"]):
            v = B[B.group == g][col].dropna()
            ax.scatter(v, k + rng.uniform(-0.18, 0.18, len(v)), s=16, color=C[g], alpha=0.75, edgecolor=SURF, linewidth=0.5)
            ax.plot([v.median()] * 2, [k - 0.3, k + 0.3], color=INK, lw=2)
        ax.set_yticks(range(3), [LABEL[g] for g in ["train", "dev", "holdout"]])
        if log:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].text(0.0, -0.22, "black tick = median", transform=axes[0].transAxes, color=INK2, fontsize=8)
    figs["sizes"] = save(fig, out, "sizes")

    # 4. item-level accuracy across salient benchmarks
    sal = ["ref6"] * 0
    ref_items, ref_subj = item_accuracy(root, REF6, {}, False)
    ev_items, ev_subj = item_accuracy(root, dev_list + holdout, seen_ids, True)
    items = pd.concat([ref_items.assign(group="train"), ev_items.assign(group=ev_items.benchmark.map(group))])
    subj = pd.concat([ref_subj.assign(group="train"), ev_subj.assign(group=ev_subj.benchmark.map(group))])
    items.groupby(["benchmark", "group"]).item_acc.describe().to_csv(out / "tables" / "item_accuracy.csv")
    order = items.groupby("benchmark").item_acc.median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, max(6, 0.24 * len(order))))
    for k, b in enumerate(order):
        v = items[items.benchmark == b].item_acc.values
        g = group.get(b, "train")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([np.percentile(v, 5), np.percentile(v, 95)], [k, k], color=C[g], lw=1, alpha=0.6)
        ax.plot([q1, q3], [k, k], color=C[g], lw=5, solid_capstyle="round")
        ax.scatter([med], [k], s=22, color=SURF, edgecolor=INK, zorder=3, linewidth=1)
    ax.set_yticks(range(len(order)), [f"{b}{' (ref6)' if b in REF6 else ''}" for b in order], fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("per-item accuracy (eval benchmarks: ref6-seen models only)")
    ax.set_title("Item difficulty spread: 5-95% (thin), interquartile (thick), median (dot)")
    handles = [plt.Line2D([], [], color=C[g], lw=5, label=LABEL[g] if g != "train" else "reference (train)")
               for g in ["train", "dev", "holdout"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    figs["item_acc"] = save(fig, out, "item_accuracy")

    # 5. subject accuracy spread per eval benchmark (signal for subject modelling)
    sd = subj.groupby(["benchmark", "group"]).subject_acc.agg(["std", "count", "mean"]).reset_index()
    sd.to_csv(out / "tables" / "subject_accuracy.csv", index=False)

    # 6. subject properties: train pool vs eval (ref6-seen subjects in eval benchmarks)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    grp = {"train": S_train, "eval": S_eval}
    gcol = {"train": C["train"], "eval": C["dev"]}
    gl = {"train": "train-pool subjects", "eval": "eval subjects (ref6-seen)"}
    prov = pd.concat({g: d.groupby("provider2").w.sum() / d.w.sum() for g, d in grp.items()}, axis=1).fillna(0)
    prov = prov.loc[prov.max(1).sort_values().index].tail(12)
    yy = np.arange(len(prov))
    for k, g in enumerate(grp):
        axes[0, 0].barh(yy + (k - 0.5) * 0.38, prov[g], height=0.36, color=gcol[g], label=gl[g])
    axes[0, 0].set_yticks(yy, prov.index, fontsize=8)
    axes[0, 0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0, 0].set_title("Provider share (benchmarks weighted equally)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].grid(axis="y", visible=False)
    bins = np.arange(2018.5, 2027.5)
    for g, d in grp.items():
        m = d.release_year.notna()
        axes[0, 1].hist(d.release_year[m], bins=bins, weights=d.w[m], density=True, histtype="step", lw=2, color=gcol[g], label=gl[g])
    axes[0, 1].set_title("Release year (where known)")
    axes[0, 1].legend(fontsize=8)
    for g, d in grp.items():
        m = d.size_b > 0
        axes[1, 0].hist(np.log2(d.size_b[m]), bins=np.arange(-1, 11, 0.5), weights=d.w[m], density=True, histtype="step",
                        lw=2, color=gcol[g], label=gl[g])
    axes[1, 0].set_xticks(range(0, 11, 2), [f"{2 ** k}B" for k in range(0, 11, 2)])
    axes[1, 0].set_title("Parameter count (where in the name)")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    wm = lambda d, x: float((d.w * x.astype(float)).sum() / d.w.sum())  # noqa: E731
    props = pd.DataFrame({g: {"reasoning / think": wm(d, d.think), "harness set": wm(d, d.has_harness),
                              "size in name": wm(d, d.size_b.notna()), "release date known": wm(d, d.release_year.notna()),
                              "normalized_name set": wm(d, d.normalized_name.notna())} for g, d in grp.items()})
    yy = np.arange(len(props))
    for k, g in enumerate(grp):
        axes[1, 1].barh(yy + (k - 0.5) * 0.38, props[g], height=0.36, color=gcol[g], label=gl[g])
        for y_, v in zip(yy, props[g]):
            axes[1, 1].text(v + 0.01, y_ + (k - 0.5) * 0.38, f"{v:.0%}", va="center", fontsize=7, color=INK2)
    axes[1, 1].set_yticks(yy, props.index, fontsize=8)
    axes[1, 1].set_xlim(0, 1.1)
    axes[1, 1].set_title("Subject attributes (benchmark-weighted share)")
    axes[1, 1].grid(axis="y", visible=False)
    figs["subjects"] = save(fig, out, "subjects")

    # 7. model coverage: top ref6-seen models x eval benchmarks
    nice = (S_eval.assign(nm=S_eval.normalized_name.fillna(S_eval.display_name))
            .groupby("identity").nm.agg(lambda x: x.value_counts().index[0]))
    cov = S_eval.groupby(["identity", "benchmark"]).size().unstack(fill_value=0).clip(upper=1)
    cov = cov.loc[cov.sum(1).sort_values(ascending=False).index[:30]]
    cov = cov[[b for b in dev_list + holdout if b in cov.columns]]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cov.values, aspect="auto", cmap=matplotlib.colors.ListedColormap([SURF, C["train"]]), interpolation="none")
    ax.set_yticks(range(len(cov)), [str(nice.get(i, i))[:38] for i in cov.index], fontsize=7)
    ax.set_xticks(range(len(cov.columns)), cov.columns, rotation=75, fontsize=7, ha="right")
    for k, b in enumerate(cov.columns):
        ax.get_xticklabels()[k].set_color(C["holdout"] if b in holdout else INK2)
    ax.axvline(len([b for b in dev_list if b in cov.columns]) - 0.5, color=INK, lw=1)
    ax.set_title("Top-30 ref6-seen models (rows) present in each eval benchmark (dev | hold-out)")
    ax.grid(False)
    figs["coverage"] = save(fig, out, "model_coverage")
    cov.to_csv(out / "tables" / "model_coverage.csv")

    # ---------------- report ----------------
    def grp_row(g, d):
        return {"group": LABEL[g], "benchmarks": len(d), "domains": d.dom1.nunique(), "subjects": int(d.n_subj.sum()),
                "eligible pairs": int(d.eval_pairs.fillna(0).sum()), "ref6-seen eligible pairs": int(d.seen_eval_pairs.sum()),
                "responses (M, uncapped)": round(d.n_responses.sum() / 1e6, 1), "median base rate": round(d.base_rate.median(), 3)}
    summary = pd.DataFrame([grp_row(g, B[B.group == g]) for g in ["train", "dev", "holdout"]])
    summary.to_csv(out / "tables" / "summary.csv", index=False)
    evt = B[B.group != "train"].sort_values(["group", "dom1", "benchmark"])[
        ["benchmark", "group", "dom1", "rule", "n_subj", "seen_eval_identities", "seen_eval_pairs", "n_items", "base_rate", "label_sd", "item_sd"]]
    top_models = S_eval.groupby("identity").benchmark.nunique().sort_values(ascending=False).head(12)
    top_models.index = [nice.get(i, i) for i in top_models.index]
    lines = [f"# Split quality report: `{split}`", "",
             f"Generated by `scripts/split_report.py`. Data: `{root}` (revision {(root / 'REVISION').read_text().strip()[:8]}).", "",
             "## Sanity checks", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    lines += ["", "## Composition", "", summary.to_markdown(index=False), "",
              f"Eval subjects (ref6-seen, in eval benchmarks): {len(S_eval)} subject rows, {S_eval.identity.nunique()} distinct models. "
              f"Train-pool subjects: {len(S_train)} rows, {S_train.identity.nunique()} distinct models.", "",
              "Models present in the most eval benchmarks: " + ", ".join(f"{m} ({c})" for m, c in top_models.items()), "",
              "## Figures", ""]
    for k, f in figs.items():
        lines += [f"![{k}]({f})", ""]
    lines += ["## Eval benchmarks", "", evt.to_markdown(index=False, floatfmt=".3f"), ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'report.md'}; checks: {sum(ok for _, ok, _ in checks)}/{len(checks)} pass")
    for n, ok, d in checks:
        print(("PASS " if ok else "FAIL ") + n + (f" ({d})" if d else ""))


if __name__ == "__main__":
    main()

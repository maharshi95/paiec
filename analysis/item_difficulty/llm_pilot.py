"""Step 4 pilot: LLM-extracted difficulty features, with a local open model standing in for an API model.

No API keys are available in this environment, so this uses Qwen3-8B (non-thinking) through vLLM on one cluster
GPU. Same prompt and schema as the proposed API extractor (docs/item-difficulty-plan.md, LLM spec).

  python llm_pilot.py prepare            # (aims26 env) sample items -> data/analysis/item_difficulty/llm/input.jsonl
  python llm_pilot.py run [MODEL]        # (llm-infer env, GPU) -> llm/output_<model>.jsonl  (see llm_pilot.slurm)
  python llm_pilot.py score [MODEL]      # (aims26 env) correlations with fitted d, per benchmark + LOBO combo
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = ROOT / "data" / "analysis" / "item_difficulty" / "llm"
N_PER_BENCH = 100
MAX_CHARS = 6000

SYSTEM = """You are an expert evaluator who estimates how hard benchmark tasks are for AI systems.
You will see one task exactly as it is given to the AI system being evaluated (possibly truncated; images and
attached files are not shown). Do not solve the task. Assess it and reply with ONE JSON object only."""

SCHEMA = """{
 "task_type": one of ["math_answer","math_proof","code_fix_repo","code_implement","doc_qa","web_gui_agent","knowledge_qa","other"],
 "domain": short string,
 "needs_tools_or_environment": true/false,   // must act in a repo / browser / GUI, run code
 "needs_unseen_inputs": true/false,           // depends on images, documents or files not shown in the text
 "n_reasoning_steps": integer,                // distinct non-trivial steps an expert would need
 "n_constraints": integer,                    // explicit conditions/requirements the answer must satisfy
 "expected_solution_size": integer,           // lines of code, or lines of derivation/answer
 "expert_minutes": number,                    // time for a skilled human expert
 "ambiguity": integer 0-3,                    // 0 = fully specified, 3 = underspecified / grader-dependent
 "difficulty_1_10": integer,                  // 1 trivial ... 10 among the hardest tasks of its kind
 "p_solve_frontier": number 0-1,              // probability a 2025 frontier model (agent if needed) gets it graded correct
 "p_solve_small": number 0-1                  // same for a small 7B open model
}"""


def prompt_for(text):
    return (f"TASK (as given to the AI system):\n<<<\n{text}\n>>>\n\n"
            f"Reply with a JSON object with exactly these keys:\n{SCHEMA}")


def chash(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def prepare(which="pilot"):
    """which="pilot": 100 trusted items per benchmark stratified by d; "all": every item (for LOBO / e2e)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    from common import targets
    from views import focus_views

    df = targets().reset_index(drop=True)
    trusted = ((df.n_subj >= 5) | (df.bench == "swe_rebench")).values
    # long items: transductive focus view (keeps the task-specific region, drops shared boilerplate);
    # short items: full text
    fv = focus_views(df.content.tolist(), df.bench.tolist(), MAX_CHARS)
    rng = np.random.default_rng(0)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    if which == "all":
        with open(LLM_DIR / "input_all.jsonl", "w") as f:
            for j in range(len(df)):
                c = df.content[j]
                text = c if len(c) <= MAX_CHARS else fv[j]
                if df.item_features[j]:
                    text = f"[metadata: {df.item_features[j]}]\n" + text
                f.write(json.dumps({"idx": int(df.idx[j]), "bench": df.bench[j], "hash": chash(text),
                                    "prompt": prompt_for(text)}) + "\n")
        print("wrote", len(df), "items")
        return
    with open(LLM_DIR / "input.jsonl", "w") as f:
        for b, g in df[trusted].groupby("bench"):
            g = g.drop_duplicates("content")
            # stratify on d deciles so the pilot covers the whole difficulty range
            g = g.assign(q=pd_qcut(g.d.values, 10))
            take = []
            for _, gg in g.groupby("q"):
                take += list(rng.choice(gg.index.values, size=min(len(gg), N_PER_BENCH // 10), replace=False))
            for j in take:
                c = df.content[j]
                text = c if len(c) <= MAX_CHARS else fv[j]
                kv = df.item_features[j]
                if kv:
                    text = f"[metadata: {kv}]\n" + text
                f.write(json.dumps({"idx": int(df.idx[j]), "bench": b, "d": float(df.d[j]), "rate": float(df.rate[j]),
                                    "hash": chash(text), "prompt": prompt_for(text)}) + "\n")
                n += 1
    print("wrote", n, "items")


def pd_qcut(v, q):
    import pandas as pd
    return pd.qcut(pd.Series(v).rank(method="first"), q, labels=False).values


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = re.sub(r"//[^\n]*", "", m.group(0))
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def run(model="Qwen/Qwen3-8B", inp="input"):
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(LLM_DIR / f"{inp}.jsonl")]
    # cache per content hash: identical prompts are generated once
    uniq = {}
    for r in rows:
        uniq.setdefault(r["hash"], r)
    print(f"{len(rows)} rows, {len(uniq)} unique prompts")
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    keys = list(uniq)
    prompts = [prompts[[r["hash"] for r in rows].index(k)] for k in keys] if len(keys) < len(rows) else prompts
    outs = dict(zip(keys, llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))))
    tag = model.split("/")[-1] + ("" if inp == "input" else "_" + inp)
    ok = 0
    with open(LLM_DIR / f"output_{tag}.jsonl", "w") as f:
        for r in rows:
            o = outs[r["hash"]]
            txt = o.outputs[0].text
            j = parse(txt)
            ok += j is not None
            f.write(json.dumps({"idx": r["idx"], "bench": r["bench"], "hash": r["hash"], "raw": txt, "json": j,
                                "n_prompt_tokens": len(o.prompt_token_ids), "n_out_tokens": len(o.outputs[0].token_ids)}) + "\n")
    print(f"parsed {ok}/{len(rows)}")


NUM_KEYS = ["n_reasoning_steps", "n_constraints", "expected_solution_size", "expert_minutes", "ambiguity",
            "difficulty_1_10", "p_solve_frontier", "p_solve_small", "needs_tools_or_environment", "needs_unseen_inputs"]


def score(model="Qwen3-8B"):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    inp = pd.DataFrame([json.loads(l) for l in open(LLM_DIR / "input.jsonl")])
    out = pd.DataFrame([json.loads(l) for l in open(LLM_DIR / f"output_{model}.jsonl")])
    df = inp.drop(columns=["prompt"]).merge(out[["idx", "json", "n_prompt_tokens", "n_out_tokens"]], on="idx")
    print(f"parsed {df.json.notna().mean():.1%}; mean prompt tokens {df.n_prompt_tokens.mean():.0f}, "
          f"out tokens {df.n_out_tokens.mean():.0f}")
    df = df[df.json.notna()].reset_index(drop=True)

    def num(v):
        if isinstance(v, bool):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    for k in NUM_KEYS:
        df[k] = [num(j.get(k)) for j in df.json]
    df["logit_pf"] = np.log((df.p_solve_frontier.clip(0.01, 0.99)) / (1 - df.p_solve_frontier.clip(0.01, 0.99)))
    df["log_minutes"] = np.log1p(df.expert_minutes.clip(0, 1e4))
    df["task_type"] = [j.get("task_type") for j in df.json]
    keys = NUM_KEYS + ["logit_pf", "log_minutes"]
    rows = []
    for b, g in df.groupby("bench"):
        r = {"bench": b, "n": len(g)}
        for k in keys:
            r[k] = spearmanr(g[k], g.d, nan_policy="omit")[0] if g[k].nunique() > 1 else np.nan
        rows.append(r)
    R = pd.DataFrame(rows).set_index("bench")
    pd.set_option("display.width", 250)
    print("Spearman(LLM feature, fitted d) per benchmark (positive = feature rises with difficulty):")
    print(R.round(2).to_string())
    R.round(3).to_csv(LLM_DIR / f"spearman_{model}.csv")
    # benchmark-level: mean predicted p_solve vs observed mean rate (6 points)
    g = df.groupby("bench").agg(mean_p_frontier=("p_solve_frontier", "mean"), mean_p_small=("p_solve_small", "mean"),
                                mean_rate=("rate", "mean"), mean_diff=("difficulty_1_10", "mean"))
    print("\nBenchmark level (offset signal):")
    print(g.round(3).to_string())
    print("Spearman over benchmarks, mean_p_frontier vs mean_rate:", round(spearmanr(g.mean_p_frontier, g.mean_rate)[0], 2))
    print("\ntask_type by benchmark:\n", pd.crosstab(df.bench, df.task_type).to_string())
    df.drop(columns=["json"]).to_pickle(LLM_DIR / f"scored_{model}.pkl")
    lobo_llm(df)


def lobo_llm(df):
    """LOBO on the 600 pilot items: ridge on per-benchmark z-scored LLM features (+ hand-crafted), vs single scores."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    sys.path.insert(0, str(Path(__file__).parent))
    from features import build

    df = df.copy()
    df["log_size"] = np.log1p(df.expected_solution_size.clip(0, 1e5))
    llm_cols = ["difficulty_1_10", "logit_pf", "log_minutes", "n_reasoning_steps", "n_constraints", "log_size",
                "ambiguity", "needs_tools_or_environment", "needs_unseen_inputs", "p_solve_small"]
    feats = build()
    hand_cols = [c for c in feats.columns if c not in ("idx", "bench") and not c.startswith("h_")]
    df = df.merge(feats[["idx"] + hand_cols], on="idx", how="left")

    def bz(X, g):
        X = X.astype(float).copy()
        for b in np.unique(g):
            m = g == b
            X[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-6)
        return np.nan_to_num(X)
    g = df.bench.values
    sets = {"llm_all": bz(df[llm_cols].values, g), "hand_bz": bz(df[hand_cols].values, g)}
    sets["llm_all+hand_bz"] = np.hstack([sets["llm_all"], sets["hand_bz"]])
    rows = []
    for name, X in sets.items():
        for b in np.unique(g):
            tr, te = g != b, g == b
            w = 1.0 / pd.Series(g[tr]).map(pd.Series(g[tr]).value_counts()).values
            p = Ridge(alpha=30.0).fit(X[tr], df.d.values[tr], sample_weight=w).predict(X[te])
            rows.append({"variant": name, "bench": b, "spearman": spearmanr(p, df.d.values[te])[0]})
    for k in ("difficulty_1_10", "log_size", "log_minutes"):
        for b in np.unique(g):
            m = g == b
            rows.append({"variant": "raw " + k, "bench": b, "spearman": spearmanr(df[k].values[m], df.d.values[m], nan_policy="omit")[0]})
    R = pd.DataFrame(rows).pivot(index="variant", columns="bench", values="spearman")
    R["mean5"] = R[[c for c in R.columns if c != "swe_rebench"]].mean(1)
    print("\nLOBO on the pilot items (ridge alpha=30, equal benchmark weight): Spearman with fitted d")
    print(R.round(2).to_string())
    R.round(3).to_csv(LLM_DIR / "lobo_llm.csv")


def table(tag="Qwen3-8B_input_all"):
    """All-item LLM features -> llm/feat_<tag>.npy aligned with targets() rows (per-benchmark z-scored, transductive)
    and tables/llm_<tag>.npz for e2e.py."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from common import OUT, targets
    from e2e import ikey

    df = targets().reset_index(drop=True)
    out = {}
    for l in open(LLM_DIR / f"output_{tag}.jsonl"):
        r = json.loads(l)
        out[r["idx"]] = r["json"] or {}

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    cols = []
    rows = []
    for i in df.idx:
        j = out.get(int(i), {})
        pf = min(max(num(j.get("p_solve_frontier")), 0.01), 0.99) if j.get("p_solve_frontier") is not None else np.nan
        ps = min(max(num(j.get("p_solve_small")), 0.01), 0.99) if j.get("p_solve_small") is not None else np.nan
        rows.append([num(j.get("difficulty_1_10")), np.log(pf / (1 - pf)) if pf == pf else np.nan,
                     np.log(ps / (1 - ps)) if ps == ps else np.nan,
                     np.log1p(min(max(num(j.get("expert_minutes")), 0), 1e4)), num(j.get("n_reasoning_steps")),
                     num(j.get("n_constraints")), np.log1p(min(max(num(j.get("expected_solution_size")), 0), 1e5)),
                     num(j.get("ambiguity")), num(j.get("needs_tools_or_environment")), num(j.get("needs_unseen_inputs"))])
    cols = ["difficulty", "logit_pf", "logit_ps", "log_minutes", "steps", "constraints", "log_size", "ambiguity",
            "tools", "unseen"]
    X = np.array(rows, float)
    g = df.bench.values
    for b in np.unique(g):
        m = g == b
        mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-6
        X[m] = (X[m] - mu) / sd
    X = np.nan_to_num(X)
    np.save(LLM_DIR / f"feat_{tag}.npy", X)
    keys = np.array([ikey({"item_content": c, "item_features": f}) for c, f in zip(df.content, df.item_features)])
    (OUT / "tables").mkdir(exist_ok=True)
    np.savez(OUT / "tables" / f"llm_{tag}.npz", X=X.astype(np.float32), keys=keys)
    np.savez(OUT / "tables" / f"llmdiff_{tag}.npz", X=X[:, :1].astype(np.float32), keys=keys)
    print("features", cols, X.shape)


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = sys.argv[2:]
    {"prepare": prepare, "run": run, "score": score, "table": table}[cmd](*arg)

import json
import zipfile

import numpy as np
import pytest

from paiec.acquisition import POLICIES, build_policy
from paiec.config import DEFAULTS, merge
from paiec.data.synthetic import make_synthetic
from paiec.experiment import run_experiment
from paiec.metrics import BUDGETS
from paiec.predictors import PREDICTORS, build_predictor
from paiec.runtime import load_predictor, save_predictor
from paiec.sim.replay import build_pairs, replay


@pytest.fixture(scope="module")
def ds():
    return make_synthetic(n_subjects=12, n_benchmarks=3, n_items=100, trials=2, seed=1)


def _evidence(ds, n=40):
    return [[[ds.subjects[s], ds.items[i]], int(y)] for s, i, y in zip(ds.resp_s[:n], ds.resp_i[:n], ds.resp_y[:n])]


@pytest.mark.parametrize("name", sorted(PREDICTORS))
def test_predict_batch_matches_predict(ds, name):
    pred = build_predictor({"name": name}).fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::97], ds.resp_i[::97])]
    batch = pred.predict_batch(inputs, ev)
    single = [pred.predict(list(x), ev) for x in inputs]
    assert np.allclose(batch, single) and all(0 <= p <= 1 for p in single)


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policy_fill_rule(name):
    pol = build_policy({"name": name})
    inp = [{"normalized_name": "m"}, {"item_content": "x", "benchmark_id": "b"}]
    ctx = {"subject_id": "s", "benchmark_id": "b", "labels_acquired": 31, "labels_remaining": 0,
           "max_labels": 31, "items_remaining": 5}
    assert pol(inp, 0.5, [], ctx) is False
    ctx.update(labels_acquired=28, labels_remaining=3, items_remaining=3)
    assert pol(inp, 0.99, [], ctx) is True


def test_replay_protocol(ds):
    pairs = build_pairs(ds, ["bench2"], seed=0)
    assert pairs and all(len(p.acq) == 50 and len(p.evl) == 50 for p in pairs)
    acq_sets = [set(p.acq) for p in pairs]
    pred = build_predictor({"name": "constant"})
    rows, _ = replay(pred, build_policy({"name": "accept_all"}), ds, pairs)
    for p, acq in zip(pairs, acq_sets):
        assert len(p.labeled) == 31
        assert {ds.items.index(e[0][1]) for e in p.labeled} <= acq  # only acquisition items revealed
        assert not acq & set(p.evl)
    assert all(abs(r["alc"] - 0.25) < 1e-12 for r in rows)  # constant 0.5 on binary labels


def test_pair_scope_equals_global_for_pair_only_predictor(ds):
    out = []
    for scope in ("global", "pair"):
        pairs = build_pairs(ds, ["bench2"], seed=3)
        rows, _ = replay(build_predictor({"name": "empirical_mean"}), build_policy({"name": "official_random"}),
                         ds, pairs, evidence_scope=scope)
        out.append([r["alc"] for r in rows])
    assert np.allclose(*out)


def test_experiment_and_irt_beats_constant(ds, tmp_path):
    res = {}
    for name in ("constant", "irt"):
        cfg = merge(DEFAULTS, {"name": name, "out_root": str(tmp_path), "predictor": {"name": name},
                               "acquisition": {"name": "accept_all"}, "cv": {"seeds": [0]}})
        res[name] = run_experiment(cfg, ds=ds, verbose=False)
    assert res["irt"]["alc"] < res["constant"]["alc"] - 0.01
    m = json.loads((tmp_path / "irt" / "metrics.json").read_text())
    assert set(m["by_budget"]) == {str(b) for b in BUDGETS}


@pytest.mark.parametrize("name", sorted(PREDICTORS))
def test_serialisation_roundtrip(ds, tmp_path, name):
    pred = build_predictor({"name": name}).fit(ds.restrict(["bench0", "bench1"]))
    save_predictor(pred, tmp_path / "predictor.json")
    back = load_predictor(tmp_path)
    ev = _evidence(ds)
    inp = [ds.subjects[0], ds.items[250]]
    assert back.predict(inp, ev) == pytest.approx(pred.predict(inp, ev))


def test_build_submission(ds, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_submission import build
    cfg = merge(DEFAULTS, {"name": "t", "predictor": {"name": "irt"}, "acquisition": {"name": "information"}})
    z = build(cfg, tmp_path / "t.zip", ds=ds)
    names = zipfile.ZipFile(z).namelist()
    assert {"model.py", "labeling.py", "predictor.json", "acquisition.json", "paiec/runtime.py"} <= set(names)


@pytest.mark.parametrize("name", sorted(PREDICTORS))
def test_incremental_state_matches_fresh(ds, name):
    """extend() on growing evidence must give the same predictions as adapt() from scratch."""
    pred = build_predictor({"name": name, **({"adapt_iters": 200} if name in ("irt", "mirt") else {})})
    pred.fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds, 120)
    inp = [ds.subjects[1], ds.items[260]]
    for n in (10, 60, 120):
        pred.predict(inp, ev[:n])  # grows the cached state incrementally
    inc = pred.predict(inp, ev)
    pred.reset_runtime()
    assert inc == pytest.approx(pred.predict(inp, ev), abs=1e-6)


def test_build_policy_switch_drops_foreign_params():
    assert build_policy({"name": "quantile", "radius": 0.1}).name == "quantile"
    with pytest.raises(ValueError):
        build_policy({"name": "quantile", "radiuss": 0.1})


def test_mirt_learns_interaction():
    """Two item types and two subject groups with opposite skills: MIRT should beat IRT given labels."""
    from paiec.data.dataset import SUBJECT_FIELDS, Dataset, anon_benchmark_id
    rng = np.random.default_rng(0)
    subjects, items, ib, rs, ri, ry = [], [], [], [], [], []
    for s in range(24):
        d = {f: "" for f in SUBJECT_FIELDS}
        d["normalized_name"] = f"{'coder' if s % 2 else 'poet'} model {s}"
        subjects.append(d)
    for b in range(4):
        for j in range(120):
            typ = j % 2
            items.append({"item_content": ("write code function compile" if typ else "write poem verse rhyme") + f" {j}",
                          "item_features": "", "interactors": "", "benchmark_id": anon_benchmark_id(f"b{b}")})
            ib.append(b)
            for s in range(24):
                p = 1 / (1 + np.exp(-(2.0 if (s % 2) == typ else -2.0)))
                rs.append(s)
                ri.append(len(items) - 1)
                ry.append(float(rng.random() < p))
    ds = Dataset(subjects, items, np.array(ib), [f"b{b}" for b in range(4)], [f"s{s}" for s in range(24)],
                 [f"i{k}" for k in range(len(items))], np.array(rs), np.array(ri), np.array(ry), "t")
    out = {}
    for name in ("irt", "mirt"):
        cfg = merge(DEFAULTS, {"name": name, "predictor": {"name": name, **({"ndim": 2} if name == "mirt" else {})},
                               "acquisition": {"name": "accept_all"}, "cv": {"seeds": [0], "test_benchmarks": ["b3"]}})
        out[name] = run_experiment(cfg, ds=ds, write=False, verbose=False)["alc"]
    assert out["mirt"] < out["irt"] - 0.02, out


@pytest.mark.parametrize("name", ["irt", "mirt"])
def test_subject_sharing_contracts(ds, tmp_path, name):
    cfg = {"name": name, "subject_sharing": True, "adapt_iters": 200, **({"ndim": 2} if name == "mirt" else {})}
    pred = build_predictor(cfg).fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds, 120)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::97], ds.resp_i[::97])]
    assert np.allclose(pred.predict_batch(inputs, ev), [pred.predict(list(x), ev) for x in inputs])
    inp = [ds.subjects[1], ds.items[260]]
    for n in (10, 60):
        pred.predict(inp, ev[:n])
    inc = pred.predict(inp, ev)
    pred.reset_runtime()
    assert inc == pytest.approx(pred.predict(inp, ev), abs=1e-6)
    save_predictor(pred, tmp_path / "predictor.json")
    assert load_predictor(tmp_path).predict(inp, ev) == pytest.approx(pred.predict(inp, ev))


def test_subject_sharing_uses_other_benchmarks(ds):
    """With sharing, a subject's labels on bench1 move its prediction on bench2; without, they do not."""
    s = ds.subjects[0]
    other = [[[s, ds.items[i]], 1] for i in range(100, 130)]          # bench1 items, all solved
    target = [s, ds.items[250]]                                        # bench2 item
    moved = {}
    for share in (False, True):
        p = build_predictor({"name": "irt", "subject_sharing": share}).fit(ds.restrict(["bench0"]))
        moved[share] = p.predict(target, other) - p.predict(target, [])
    assert abs(moved[False]) < 1e-9 and moved[True] > 0.01


def test_build_predictor_switch_drops_foreign_params():
    assert build_predictor({"name": "constant", "item_target": "absolute"}).name == "constant"
    with pytest.raises(ValueError):
        build_predictor({"name": "constant", "item_targett": "absolute"})


def test_build_pairs_counts_source_items_and_keeps_variants_together(ds):
    from dataclasses import replace
    # duplicate every bench2 item as an interactor variant "<id>|v": visible items double, source items do not
    b2 = [i for i in range(len(ds.items)) if ds.bench_names[ds.item_bench[i]] == "bench2"]
    items = list(ds.items) + [dict(ds.items[i], interactors="tool") for i in b2]
    ids = list(ds.item_ids) + [ds.item_ids[i] + "|v" for i in b2]
    off = {i: len(ds.items) + k for k, i in enumerate(b2)}
    m = np.isin(ds.resp_i, b2)
    d2 = replace(ds, items=items, item_ids=ids, item_bench=np.concatenate([ds.item_bench, ds.item_bench[b2]]),
                 resp_s=np.concatenate([ds.resp_s, ds.resp_s[m]]),
                 resp_i=np.concatenate([ds.resp_i, [off[i] for i in ds.resp_i[m]]]),
                 resp_y=np.concatenate([ds.resp_y, ds.resp_y[m]]))
    assert not build_pairs(d2, ["bench2"], min_items=101)   # 100 source items < 101 although 200 variants
    for p in build_pairs(d2, ["bench2"]):
        src = lambda i: d2.item_ids[i].split("|")[0]
        assert len(p.acq) == 50 and not {src(i) for i in p.acq} & {src(i) for i in p.evl}


ITEM_CFG = {"name": "irt_item", "rp_frac": 0.3, "rp_dim": 16, "kv_frac": 0.2, "dup_frac": 0.2, "eb_nu": 2.0, "eb_eps_nu": 20.0, "rp2_frac": 0.1, "rp2_dim": 8, "prior_slope_var": 0.2, "vi": True}


def test_irt_item_contracts(ds, tmp_path):
    """Online item amortisation: batch = single, incremental = fresh, serialisation, zero fracs = IRT."""
    pred = build_predictor({**ITEM_CFG, "adapt_iters": 300}).fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds, 120)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::97], ds.resp_i[::97])]
    assert np.allclose(pred.predict_batch(inputs, ev), [pred.predict(list(x), ev) for x in inputs])
    inp = [ds.subjects[1], ds.items[260]]
    for n in (10, 60):
        pred.predict(inp, ev[:n])
    inc = pred.predict(inp, ev)
    pred.reset_runtime()
    assert inc == pytest.approx(pred.predict(inp, ev), abs=1e-5)
    save_predictor(pred, tmp_path / "predictor.json")
    assert load_predictor(tmp_path).predict(inp, ev) == pytest.approx(pred.predict(inp, ev))
    base = build_predictor({"name": "irt"}).fit(ds.restrict(["bench0", "bench1"]))
    zero = build_predictor({"name": "irt_item"}).fit(ds.restrict(["bench0", "bench1"]))
    assert np.allclose(base.predict_batch(inputs, ev), zero.predict_batch(inputs, ev), atol=1e-12)


def test_irt_item_isolates_failing_benchmark(ds, monkeypatch):
    """A benchmark whose adaptation fails falls back to the priors; the other benchmarks keep their fits."""
    pred = build_predictor({**ITEM_CFG, "name": "irt_item_lr"}).fit(ds.restrict(["bench0", "bench1"]))
    idx = np.concatenate([np.nonzero(ds.resp_b == b)[0][::5][:100] for b in (1, 2)])
    ev = [[[ds.subjects[ds.resp_s[k]], ds.items[ds.resp_i[k]]], int(ds.resp_y[k])] for k in idx]
    bad = ev[0][0][1]["benchmark_id"]
    assert len({e[0][1]["benchmark_id"] for e in ev}) > 1
    orig = type(pred)._adapt_bench

    def flaky(self, obs, warm=None):
        if obs[0][4]["benchmark_id"] == bad:
            raise np.linalg.LinAlgError("Singular matrix")
        return orig(self, obs, warm)

    monkeypatch.setattr(type(pred), "_adapt_bench", flaky)
    state = pred.adapt(ev)
    assert state["fits"][bad] is None
    assert all(st is not None for b, st in state["fits"].items() if b != bad)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::53], ds.resp_i[::53])]
    p = pred.predict_batch(inputs, ev)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def _write_bench(root, slug):
    """Tiny measurement-db-layout benchmark: 3 subjects x 4 items, binary responses."""
    import pandas as pd
    t = root / slug / "formatted_tables"
    t.mkdir(parents=True)
    pd.DataFrame([{"benchmark_id": slug, "release": "public", "response_scale": '{"kind":"binary"}'}]).to_parquet(t / "benchmarks.parquet")
    pd.DataFrame([{"subject_id": f"s{k}", "normalized_name": f"m{k}"} for k in range(3)]).to_parquet(t / "subjects.parquet")
    pd.DataFrame([{"item_id": f"i{j}", "content": f"item {j}", "item_features": ""} for j in range(4)]).to_parquet(t / "items.parquet")
    pd.DataFrame([{"subject_id": f"s{k}", "item_id": f"i{j}", "interactors": "", "response": float((k + j) % 2)}
                  for k in range(3) for j in range(4)]).to_parquet(t / "responses.parquet")


def test_load_dataset_multi_root_and_exclude(tmp_path):
    """A list of roots is one universe; `exclude` drops benchmarks even when they are on disk (stale downloads)."""
    from paiec.data.dataset import find_benchmarks, load_dataset
    a, b = tmp_path / "train", tmp_path / "val"
    for root, slug in ((a, "bench_a"), (a, "bench_b"), (b, "bench_c")):
        _write_bench(root, slug)
    assert sorted(find_benchmarks([a, b])) == ["bench_a", "bench_b", "bench_c"]
    kw = {"binarize": {"mode": "exclude"}, "cache_dir": str(tmp_path / "cache")}
    full = load_dataset([a, b], **kw)
    assert sorted(full.bench_names) == ["bench_a", "bench_b", "bench_c"] and len(full.resp_y) == 36
    cut = load_dataset([a, b], exclude=["bench_c"], **kw)
    assert sorted(cut.bench_names) == ["bench_a", "bench_b"] and len(cut.resp_y) == 24   # not served from the full pickle
    assert load_dataset([a, b], benchmarks=["bench_a", "bench_c"], exclude=["bench_c"], **kw).bench_names == ["bench_a"]


def test_fit_as_shares_parent_fit_cache(ds, tmp_path, monkeypatch):
    from paiec.fitcache import cache_key, fit_cached
    monkeypatch.setenv("PAIEC_FIT_CACHE", str(tmp_path))
    a, b = build_predictor({"name": "irt"}), build_predictor(ITEM_CFG)
    assert cache_key(a, ["bench0"], {}, "r") == cache_key(b, ["bench0"], {}, "r")
    fit_cached(b, ds.restrict(["bench0"]), ["bench0"], {}, "r")
    assert type(b).__name__ == "IRTItem"
    loaded_a = fit_cached(build_predictor({"name": "irt"}), ds.restrict(["bench0"]), ["bench0"], {}, "r")
    assert type(loaded_a).__name__ == "IRT"
    loaded_b = fit_cached(build_predictor(ITEM_CFG), ds.restrict(["bench0"]), ["bench0"], {}, "r")
    assert type(loaded_b).__name__ == "IRTItem" and loaded_b.rp_frac == 0.3


def test_irt_h_hyper_contracts(ds):
    """irt_h with every hierarchical part on: batch = single, incremental = fresh, and off = irt."""
    hyp = {"nu_u": 10.0, "nu_d": 10.0, "tau_u": 0.5, "tau_d": 0.5, "vi": True}
    pred = build_predictor({"name": "irt_h", "adapt_iters": 300, **hyp}).fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds, 120)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::97], ds.resp_i[::97])]
    batch = pred.predict_batch(inputs, ev)
    assert np.allclose(batch, [pred.predict(list(x), ev) for x in inputs])
    inp = [ds.subjects[1], ds.items[260]]
    pred.reset_runtime()
    for n in (10, 60, 120):
        pred.predict(inp, ev[:n])
    inc = pred.predict(inp, ev)
    pred.reset_runtime()
    assert inc == pytest.approx(pred.predict(inp, ev), abs=1e-5)
    off = build_predictor({"name": "irt_h"}).fit(ds.restrict(["bench0", "bench1"]))
    ref = build_predictor({"name": "irt"}).fit(ds.restrict(["bench0", "bench1"]))
    assert np.allclose(off.predict_batch(inputs, ev), ref.predict_batch(inputs, ev))


def test_irt_item_lr_contracts(ds, tmp_path):
    """Item amortisation + subject regression: batch = single, incremental = fresh, serialisation."""
    pred = build_predictor({**ITEM_CFG, "name": "irt_item_lr", "vi": False, "adapt_iters": 300}).fit(ds.restrict(["bench0", "bench1"]))
    ev = _evidence(ds, 120)
    inputs = [(ds.subjects[s], ds.items[i]) for s, i in zip(ds.resp_s[::97], ds.resp_i[::97])]
    assert np.allclose(pred.predict_batch(inputs, ev), [pred.predict(list(x), ev) for x in inputs])
    inp = [ds.subjects[1], ds.items[260]]
    for n in (10, 60):
        pred.predict(inp, ev[:n])
    inc = pred.predict(inp, ev)
    pred.reset_runtime()
    assert inc == pytest.approx(pred.predict(inp, ev), abs=1e-5)
    save_predictor(pred, tmp_path / "predictor.json")
    assert load_predictor(tmp_path).predict(inp, ev) == pytest.approx(pred.predict(inp, ev))

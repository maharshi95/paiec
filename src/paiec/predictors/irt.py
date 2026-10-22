"""Amortised logistic IRT with test-time MAP adaptation on (global) evidence.

    logit p(s, i) = beta_b + delta_{s,b} - d_i

fit (training benchmarks):
  1. Rasch fit per benchmark: pair ability theta_{s,b}, item difficulty d_i (items centred per benchmark).
  2. d_i ~ ridge(item features); prior var sigma_d^2 = cross-benchmark residual (cold-start realistic).
  3. beta_b = mean pair ability on b; prior N(mu_beta, sigma_beta^2) across training benchmarks.
  4. delta = theta - beta ~ ridge(subject features) blended with the subject's history on other
     benchmarks; prior var sigma_u^2 = cross-benchmark residual.
adapt (test time): per anonymous benchmark, block-coordinate Newton MAP of beta, delta (seen pairs)
  and d (seen items, if learn_items) under those priors, using every revealed label on the benchmark.
score: plug-in posterior means; with `marginalize`, the probit approximation over the summed
  posterior variances (wider uncertainty -> predictions pulled toward 0.5).
"""
import numpy as np

from ..features.item import ItemFeaturizer
from ..features.subject import SubjectFeaturizer, subject_identity, subject_key
from .base import Evidence, Predictor, item_key
from .simple import sigmoid


def ridge(X, y, lam, weights=None):
    X = np.asarray(X, float)
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float) / np.mean(weights)
    xm, ym = (sw @ X) / sw.sum(), float(sw @ y / sw.sum())
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ (Xc * sw[:, None]) + lam * np.eye(X.shape[1]), Xc.T @ (sw * (y - ym)))
    return w, float(ym - xm @ w)


def rasch(P, J, ysum, n, nP, nJ, var, iters):
    theta, d = np.zeros(nP), np.zeros(nJ)
    for _ in range(iters):
        mu = sigmoid(theta[P] - d[J])
        theta += (np.bincount(P, ysum - n * mu, nP) - theta / var) / (np.bincount(P, n * mu * (1 - mu), nP) + 1 / var)
        mu = sigmoid(theta[P] - d[J])
        d += (-np.bincount(J, ysum - n * mu, nJ) - d / var) / (np.bincount(J, n * mu * (1 - mu), nJ) + 1 / var)
    return theta, d


class IRT(Predictor):
    name = "irt"
    item_fields = ("text", "length", "kv")
    item_dim = 1024
    max_chars = 4000
    subject_fields = ("provider", "name", "think", "size", "date", "effort", "harness")
    subject_dim = 256
    ridge_item = 10.0
    ridge_subject = 10.0
    use_item_features = True
    use_subject_features = True
    use_history = True
    history_strength = 1.0
    learn_bench = True
    learn_items = True
    marginalize = True
    fit_iters = 40
    adapt_iters = 15
    fit_prior_var = 9.0
    var_floor = 0.05
    var_scale = 1.0          # multiplies all prior variances (calibration knob)
    beta_var_mult = 1.0      # multiplies only the benchmark-offset prior variance (cold-start hedge)
    # "centered": items predict within-benchmark difficulty; offset prior = mean of training offsets.
    # "absolute": items predict difficulty *including* the benchmark offset (text can carry "this kind
    #   of benchmark is easy"); offset prior N(0, cross-benchmark error of that prediction).
    item_target = "centered"
    balance_benchmarks = False   # weight items 1/n_items(benchmark) in the item ridge
    cv_groups = 10               # cross-benchmark residuals: benchmarks hashed into <= this many folds
    max_ridge_items = 200_000    # item-ridge fit on a seeded sample (memory); Rasch uses all items

    # ---------------- fit ----------------
    def fit(self, train):
        self.ifeat = ItemFeaturizer(self.item_fields, self.item_dim, self.max_chars)
        self.sfeat = SubjectFeaturizer(self.subject_fields, self.subject_dim)
        nb, ni = len(train.bench_names), len(train.items)
        pk, P_resp = np.unique(train.resp_s * nb + train.resp_b, return_inverse=True)
        cells, inv = np.unique(P_resp * ni + train.resp_i, return_inverse=True)
        ysum, n = np.bincount(inv, train.resp_y), np.bincount(inv).astype(float)
        P, I = cells // ni, cells % ni
        items_used, J = np.unique(I, return_inverse=True)
        theta, d = rasch(P, J, ysum, n, len(pk), len(items_used), self.fit_prior_var, self.fit_iters)

        pair_b, pair_s = pk % nb, pk // nb
        item_b = train.item_bench[items_used]
        for b in np.unique(item_b):
            c = d[item_b == b].mean()
            d[item_b == b] -= c
            theta[pair_b == b] -= c

        # benchmark offset prior
        benches = np.unique(pair_b)
        beta = np.array([theta[pair_b == b].mean() for b in benches])
        self.mu_beta = float(beta.mean())
        self.var_beta = max(float(beta.var()) if len(beta) > 1 else 1.0, 0.25)

        # item difficulty prior (ridge on a bounded, seeded sample of items)
        if len(items_used) > self.max_ridge_items:
            sel = np.sort(np.random.default_rng(0).choice(len(items_used), self.max_ridge_items, replace=False))
            items_used, d, item_b = items_used[sel], d[sel], item_b[sel]
        X = self.ifeat.transform([train.items[i] for i in items_used])
        wts = None
        if self.balance_benchmarks:
            cnt = np.bincount(item_b)
            wts = 1.0 / cnt[item_b]
        if self.item_target == "absolute":
            target = d - beta[np.searchsorted(benches, item_b)]   # logit(avg subject) = -target
            self.w_i, self.b_i, self.var_d, var_off = self._fit_prior(X, target, item_b, self.ridge_item,
                                                                       self.use_item_features, wts, split_offset=True)
            self.mu_beta, self.var_beta = 0.0, max(var_off, self.var_floor)
        else:
            self.w_i, self.b_i, self.var_d = self._fit_prior(X, d, item_b, self.ridge_item, self.use_item_features, wts)
        self.var_beta *= self.beta_var_mult

        # subject residual prior: features + history
        r = theta - beta[np.searchsorted(benches, pair_b)]
        Z = self.sfeat.transform([train.subjects[s] for s in pair_s])
        ident = [subject_identity(train.subjects[s]) for s in pair_s]
        resid = []
        pair_g = self._fold_of(pair_b)
        for g in np.unique(pair_g) if len(benches) > 1 else []:
            out, inn = pair_g == g, pair_g != g
            pred = self._fit_subject(Z[inn], r[inn], [ident[k] for k in np.nonzero(inn)[0]])
            resid.append(r[out] - np.array([pred(Z[k], ident[k]) for k in np.nonzero(out)[0]]))
        full = self._fit_subject(Z, r, ident, keep=True)
        if resid:
            self.var_u = max(float(np.mean(np.concatenate(resid) ** 2)), self.var_floor)
        else:
            self.var_u = max(float(np.mean((r - np.array([full(Z[k], ident[k]) for k in range(len(r))])) ** 2)) * 1.5, self.var_floor)
        return self

    def _fold_of(self, bench):
        """Map benchmark indices to <= cv_groups folds (each benchmark stays whole)."""
        u = np.unique(bench)
        if len(u) <= self.cv_groups:
            return bench
        return (np.searchsorted(u, bench) % self.cv_groups)

    def _fit_prior(self, X, y, groups, lam, enabled, weights=None, split_offset=False):
        """Ridge prior + cross-group (leave-one-benchmark-out) residual variance.

        split_offset: also return the variance of the per-benchmark *mean* residual (how wrong the
        predicted benchmark offset is) and report the item variance net of it.
        """
        wsub = (lambda m: None) if weights is None else (lambda m: weights[m])
        if not enabled:
            w, b = np.zeros(X.shape[1]), float(np.mean(y))
        res, offs = [], []
        groups = self._fold_of(groups)
        for g in np.unique(groups) if len(np.unique(groups)) > 1 else []:
            m = groups != g
            wg, bg = ridge(X[m], y[m], lam, wsub(m)) if enabled else (w, float(np.mean(y[m])))
            r = y[~m] - (X[~m] @ wg + bg)
            offs.append(r.mean())
            res.append(r - r.mean() if split_offset else r)
        if enabled:
            w, b = ridge(X, y, lam, weights)
        var = np.mean(np.concatenate(res) ** 2) if res else np.mean((y - X @ w - b) ** 2) * 1.5
        out = (w, b, max(float(var), self.var_floor))
        if split_offset:
            out += (float(np.mean(np.square(offs))) if offs else 1.0,)
        return out

    def _fit_subject(self, Z, r, ident, keep=False):
        w, b = ridge(Z, r, self.ridge_subject) if self.use_subject_features else (np.zeros(Z.shape[1]), float(r.mean()))
        hist = {}
        if self.use_history:
            acc = {}
            for k, v in zip(ident, r):
                acc.setdefault(k, []).append(v)
            hist = {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}
        if keep:
            self.w_s, self.b_s, self.hist = w, b, hist
        hs = self.history_strength

        def pred(z, idn):
            base = float(z @ w + b)
            h, c = hist.get(idn, (0.0, 0))
            return (c * h + hs * base) / (c + hs)
        return pred

    # ---------------- runtime priors (cached, pure) ----------------
    def _caches(self):
        if not hasattr(self, "_cache_u"):
            self._cache_u, self._cache_d = {}, {}
        return self._cache_u, self._cache_d

    def subject_prior(self, subject) -> float:
        cu, _ = self._caches()
        k = subject_key(subject)
        if k not in cu:
            base = float(self.sfeat.transform_one(subject) @ self.w_s + self.b_s)
            h, c = self.hist.get(subject_identity(subject), (0.0, 0))
            cu[k] = (c * h + self.history_strength * base) / (c + self.history_strength)
        return cu[k]

    def item_priors(self, items) -> np.ndarray:
        _, cd = self._caches()
        keys = [item_key(i) for i in items]
        todo = [(k, i) for k, i in zip(keys, items) if k not in cd]
        if todo:
            vals = self.ifeat.transform([i for _, i in todo]) @ self.w_i + self.b_i
            cd.update({k: float(v) for (k, _), v in zip(todo, vals)})
        return np.array([cd[k] for k in keys], float)

    # ---------------- adapt / score ----------------
    def adapt(self, labeled):
        state = {"ev": Evidence(), "fits": {}}
        return self.extend(state, labeled)

    def extend(self, state, new):
        for b in state["ev"].add(new):
            state["fits"][b] = self._fit_bench(state["ev"].by_bench[b], state["fits"].get(b))
        return state

    def _fit_bench(self, obs, warm=None):
        """MAP of beta, delta (seen pairs), d (seen items) for one benchmark's evidence."""
        vs = self.var_scale
        var_b, var_u, var_d = self.var_beta * vs, self.var_u * vs, self.var_d * vs
        pidx, iidx, subj, items = {}, {}, [], []
        for sk, ik, _, s, i in obs:
            if sk not in pidx:
                pidx[sk] = len(subj)
                subj.append(s)
            if ik not in iidx:
                iidx[ik] = len(items)
                items.append(i)
        P = np.array([pidx[o[0]] for o in obs])
        J = np.array([iidx[o[1]] for o in obs])
        y = np.array([o[2] for o in obs])
        u = np.array([self.subject_prior(s) for s in subj])
        yh = self.item_priors(items)
        nP, nJ = len(u), len(yh)
        beta, delta, d = self.mu_beta, u.copy(), yh.copy()
        if warm:  # warm start from the previous (smaller) evidence's solution
            beta = warm["beta"][0]
            for sk, k in pidx.items():
                if sk in warm["delta"]:
                    delta[k] = warm["delta"][sk][0]
            for ik, k in iidx.items():
                if ik in warm["d"]:
                    d[k] = warm["d"][ik][0]
        for _ in range(self.adapt_iters):
            if self.learn_bench:
                mu = sigmoid(beta + delta[P] - d[J])
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / ((mu * (1 - mu)).sum() + 1 / var_b)
            mu = sigmoid(beta + delta[P] - d[J])
            delta += (np.bincount(P, y - mu, nP) - (delta - u) / var_u) / (np.bincount(P, mu * (1 - mu), nP) + 1 / var_u)
            if self.learn_items:
                mu = sigmoid(beta + delta[P] - d[J])
                d += (-np.bincount(J, y - mu, nJ) - (d - yh) / var_d) / (np.bincount(J, mu * (1 - mu), nJ) + 1 / var_d)
        w = sigmoid(beta + delta[P] - d[J])
        w = w * (1 - w)
        vb = 1 / (w.sum() + 1 / var_b) if self.learn_bench else var_b
        vdel = 1 / (np.bincount(P, w, nP) + 1 / var_u)
        vd = 1 / (np.bincount(J, w, nJ) + 1 / var_d) if self.learn_items else np.full(nJ, var_d)
        return {"beta": (float(beta), float(vb)),
                "delta": {sk: (float(delta[k]), float(vdel[k])) for sk, k in pidx.items()},
                "d": {ik: (float(d[k]), float(vd[k])) for ik, k in iidx.items()}}

    def score_batch(self, state, inputs):
        vs = self.var_scale
        yh = self.item_priors([i for _, i in inputs])
        m, v = np.empty(len(inputs)), np.empty(len(inputs))
        fits = state["fits"]
        for k, (s, i) in enumerate(inputs):
            st = fits.get(i.get("benchmark_id", ""))
            bm, bv = st["beta"] if st else (self.mu_beta, self.var_beta * vs)
            dm, dv = st["delta"].get(subject_key(s), (None, None)) if st else (None, None)
            if dm is None:
                dm, dv = self.subject_prior(s), self.var_u * vs
            im, iv = st["d"].get(item_key(i), (None, None)) if st else (None, None)
            if im is None:
                im, iv = yh[k], self.var_d * vs
            m[k], v[k] = bm + dm - im, bv + dv + iv
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)

    def score(self, state, subject, item):
        return float(self.score_batch(state, [(subject, item)])[0])

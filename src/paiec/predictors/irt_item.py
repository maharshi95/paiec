"""IRT with online, within-benchmark item-difficulty amortisation (test time only).

    logit p(s, i) = beta_b + delta_{s,b} - d_i
    d_i = yhat_i + z_i . w_b + sum_k a_ik g_{b,k} + eps_i

- yhat_i: the cross-benchmark content prior of IRT (unchanged fit).
- z_i . w_b: a per-benchmark linear map on content features, learned from the benchmark's revealed labels.
  z_i is a seeded random projection (rp_dim) of the IRT item features, centred on the benchmark's labeled
  items and normalised to unit length (a cosine kernel). w_b ~ N(0, rp_frac * var_d I).
- g_{b,k}: group effects for each item_features key=value token (kv_frac) and for identical item_content
  (dup_frac), g ~ N(0, 1) with weights a_ik = sqrt(frac * var_d / n_keys(i)).
- eps_i ~ N(0, (1 - rp_frac - kv_frac - dup_frac) * var_d): the item's own effect, as in IRT.
The prior variance of every unlabeled item therefore stays var_d (budget 0 is unchanged); labels on other
items of the same benchmark now move unlabeled items' difficulties through shared content and groups.

All new parameters are test-time only (RUNTIME_PARAMS), and the fit is IRT's, so the IRT fit cache is
reused (FIT_AS). Runtime: numpy + stdlib.
"""
import zlib

import numpy as np

from ..features.hashing import parse_kv
from ..features.subject import subject_key
from .base import Evidence, item_key
from .irt import IRT
from .simple import sigmoid


class IRTItem(IRT):
    name = "irt_item"
    FIT_AS = IRT
    RUNTIME_PARAMS = IRT.RUNTIME_PARAMS + ("rp_frac", "rp_dim", "rp_seed", "kv_frac", "dup_frac", "item_iters",
                                           "kv_numeric", "eb_nu", "eb_eps_nu")
    rp_frac = 0.0
    rp_dim = 64
    rp_seed = 0
    kv_frac = 0.0
    dup_frac = 0.0
    kv_numeric = False      # also group on numeric kv values
    item_iters = 0          # >0 overrides adapt_iters for this model
    # online empirical Bayes: per-benchmark scales of the kv-key and content-map variances, learned by EM with
    # a pseudo-count eb_nu (groups / map dims) toward the configured fraction. 0 = fixed fractions.
    eb_nu = 0.0
    eb_eps_nu = 0.0         # same for the item-specific variance (pseudo-count in items); 0 = fixed

    # ---------------- item info (cached, pure) ----------------
    def online_features(self, items, X):
        """Feature vectors for the online content map (one row per item). Default: the IRT prior's own
        features X (ItemFeaturizer). Override or swap the featurizer here to use another representation."""
        return X

    def _proj(self, n_in):
        R = getattr(self, "_cache_R", None)
        if R is None or R.shape != (n_in, int(self.rp_dim)):
            rng = np.random.default_rng(int(self.rp_seed))
            self._cache_R = rng.standard_normal((n_in, int(self.rp_dim))) / np.sqrt(self.rp_dim)
        return self._cache_R

    def _kv_tokens(self, item):
        out = []
        for k, v in parse_kv(item.get("item_features", "")).items():
            if not v:
                continue
            if not self.kv_numeric:
                try:
                    float(v)
                    continue
                except ValueError:
                    pass
            out.append("kv:" + k + "=" + v)
        return tuple(out)

    def _info(self, items):
        """Per item: (prior mean, projection z, kv tokens, content key). Cached by item key."""
        if not hasattr(self, "_cache_info"):
            self._cache_info = {}
        ci = self._cache_info
        keys = [item_key(i) for i in items]
        todo = {}
        for k, i in zip(keys, items):
            if k not in ci and k not in todo:
                todo[k] = i
        if todo:
            X = self.ifeat.transform(list(todo.values()))
            yh = X @ self.w_i + self.b_i
            if self.rp_frac > 0:
                F = self.online_features(list(todo.values()), X)
                Z = F.astype(np.float64) @ self._proj(F.shape[1])
            else:
                Z = np.zeros((len(X), 0))
            for n, (k, i) in enumerate(todo.items()):
                ck = "ct:%08x" % zlib.crc32(i.get("item_content", "").encode())
                ci[k] = (float(yh[n]), Z[n], self._kv_tokens(i), ck)
        return [ci[k] for k in keys]

    def item_priors(self, items) -> np.ndarray:
        return np.array([x[0] for x in self._info(items)], float)

    def _groups(self, info):
        """[(group token, weight)] for one item."""
        vd = self.var_d * self.var_scale
        out = []
        if self.kv_frac > 0 and info[2]:
            a = np.sqrt(self.kv_frac * vd / len(info[2]))
            out += [(t, a) for t in info[2]]
        if self.dup_frac > 0:
            out.append((info[3], np.sqrt(self.dup_frac * vd)))
        return out

    def _var_eps(self):
        return max(1.0 - self.rp_frac - self.kv_frac - self.dup_frac, 0.02) * self.var_d * self.var_scale

    # ---------------- adapt ----------------
    def extend(self, state, new):
        assert not self.subject_sharing, "irt_item supports per-benchmark adaptation only"
        touched = state["ev"].add(new)
        for b in touched:
            state["fits"][b] = self._adapt_bench(state["ev"].by_bench[b], state["fits"].get(b))
        return state

    def adapt(self, labeled):
        return self.extend({"ev": Evidence(), "fits": {}, "joint": None}, labeled)

    def _adapt_bench(self, obs, warm=None):
        vs = self.var_scale
        var_b, var_u = self.var_beta * vs, self.var_u * vs
        var_e = self._var_eps()
        tau2 = self.rp_frac * self.var_d * vs
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
        info = self._info(items)
        yh = np.array([x[0] for x in info])
        nP, nJ = len(u), len(yh)

        # shared content map
        use_rp = tau2 > 0
        if use_rp:
            Z = np.stack([x[1] for x in info])
            zbar = Z.mean(0)
            Zc = Z - zbar
            Zc /= np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-9)
            k = Zc.shape[1]
            w = np.zeros(k)
        # groups: one block per kv key (disjoint values within a key) + one content block
        blocks = {}
        gidx = {}
        bkeys = []
        for j, x in enumerate(info):
            for t, a in self._groups(x):
                bk = t.split("=", 1)[0] if t.startswith("kv:") else "ct"
                gi = gidx.setdefault(t, len(gidx))
                if bk not in blocks:
                    blocks[bk] = ([], [], [])
                    bkeys.append(bk)
                blocks[bk][0].append(j)
                blocks[bk][1].append(gi)
                blocks[bk][2].append(a)
        nG = len(gidx)
        g = np.zeros(nG)
        blocks = [(np.array(jj), np.array(gg), np.array(aa)) for jj, gg, aa in (blocks[b] for b in bkeys)]
        s2 = np.ones(len(blocks))
        bsel = []
        for jj, gg, aa in blocks:
            sel = np.zeros(nG, bool)
            sel[gg] = True
            bsel.append(sel)
        t2 = tau2
        # item-level map from obs to the items' group contributions: shared(j) = sum a * g
        beta, delta, eps = self.mu_beta, u.copy(), np.zeros(nJ)
        if warm:
            beta = warm["beta"][0]
            for sk, kk in pidx.items():
                if sk in warm["delta"]:
                    delta[kk] = warm["delta"][sk][0]
            we = warm.get("eps", {})
            for ik, kk in iidx.items():
                if ik in we:
                    eps[kk] = we[ik]
            if use_rp and warm.get("w") is not None and len(warm["w"]) == k:
                w = warm["w"].copy()
            if use_rp and warm.get("t2"):
                t2 = warm["t2"]
            ws2 = warm.get("gs2", {})
            for bi, bk in enumerate(bkeys):
                s2[bi] = ws2.get(bk, 1.0)
            wg = warm.get("g", {})
            for t, gi in gidx.items():
                if t in wg:
                    g[gi] = wg[t][0]

        def shared():
            s = np.zeros(nJ)
            if use_rp:
                s += Zc @ w
            for jj, gg, aa in blocks:
                s += np.bincount(jj, aa * g[gg], nJ)
            return s

        sh = shared()
        iters = int(self.item_iters) or self.adapt_iters
        nu = float(self.eb_nu)
        nu_e, var_e0 = float(self.eb_eps_nu), var_e
        if warm and nu_e > 0 and warm.get("var_e"):
            var_e = warm["var_e"]
        for _ in range(iters):
            d = yh + sh + eps
            if self.learn_bench:
                mu = sigmoid(beta + delta[P] - d[J])
                beta += ((y - mu).sum() - (beta - self.mu_beta) / var_b) / ((mu * (1 - mu)).sum() + 1 / var_b)
            mu = sigmoid(beta + delta[P] - d[J])
            delta += (np.bincount(P, y - mu, nP) - (delta - u) / var_u) / (np.bincount(P, mu * (1 - mu), nP) + 1 / var_u)
            if self.learn_items:
                mu = sigmoid(beta + delta[P] - d[J])
                h = np.bincount(J, mu * (1 - mu), nJ)
                eps += (-np.bincount(J, y - mu, nJ) - eps / var_e) / (h + 1 / var_e)
                if nu_e > 0:
                    var_e = (float((eps ** 2 + 1 / (h + 1 / var_e)).sum()) + nu_e * var_e0) / (nJ + nu_e)
                d = yh + sh + eps
                if use_rp:
                    mu = sigmoid(beta + delta[P] - d[J])
                    r, h = np.bincount(J, y - mu, nJ), np.bincount(J, mu * (1 - mu), nJ)
                    H = (Zc * h[:, None]).T @ Zc + np.eye(k) / t2
                    w += np.linalg.solve(H, -Zc.T @ r - w / t2)
                    if nu > 0:
                        he = h / (1 + h * var_e)
                        Sm = np.linalg.inv((Zc * he[:, None]).T @ Zc + np.eye(k) / t2)
                        t2 = (w @ w + np.trace(Sm) + nu * tau2) / (k + nu)
                    sh = shared()
                    d = yh + sh + eps
                for bi, (jj, gg, aa) in enumerate(blocks):
                    mu = sigmoid(beta + delta[P] - d[J])
                    r, h = np.bincount(J, y - mu, nJ), np.bincount(J, mu * (1 - mu), nJ)
                    sel = bsel[bi]
                    step = (np.bincount(gg, -aa * r[jj], nG) - g / s2[bi]) / (np.bincount(gg, aa * aa * h[jj], nG) + 1 / s2[bi])
                    g[sel] += step[sel]
                    if nu > 0:
                        he = h / (1 + h * var_e)
                        vgb = 1 / (np.bincount(gg, aa * aa * he[jj], nG) + 1 / s2[bi])
                        s2[bi] = (float((g[sel] ** 2 + vgb[sel]).sum()) + nu) / (sel.sum() + nu)
                    sh = shared()
                    d = yh + sh + eps
        d = yh + sh + eps
        mu = sigmoid(beta + delta[P] - d[J])
        wt = mu * (1 - mu)
        h = np.bincount(J, wt, nJ)
        vb = 1 / (wt.sum() + 1 / var_b) if self.learn_bench else var_b
        vdel = 1 / (np.bincount(P, wt, nP) + 1 / var_u)
        # group posterior variances (diagonal), shared-part variance per labeled item
        vg = np.ones(nG)
        shv = np.zeros(nJ)
        heff = h / (1 + h * var_e)
        for bi, (jj, gg, aa) in enumerate(blocks):
            vg_b = 1 / (np.bincount(gg, aa * aa * heff[jj], nG) + 1 / s2[bi])
            vg[bsel[bi]] = vg_b[bsel[bi]]
        for jj, gg, aa in blocks:
            shv += np.bincount(jj, aa * aa * vg[gg], nJ)
        out = {"beta": (float(beta), float(vb)),
               "delta": {sk: (float(delta[kk]), float(vdel[kk])) for sk, kk in pidx.items()},
               "eps": {ik: float(eps[kk]) for ik, kk in iidx.items()},
               "g": {t: (float(g[gi]), float(vg[gi])) for t, gi in gidx.items()},
               "gs2": {bk: float(s2[bi]) for bi, bk in enumerate(bkeys)}, "var_e": float(var_e)}
        if use_rp:
            Hm = (Zc * heff[:, None]).T @ Zc + np.eye(k) / t2
            Sw = np.linalg.inv(Hm)
            shv += np.einsum("ij,jk,ik->i", Zc, Sw, Zc)
            out.update(w=w, Sw=Sw, zbar=zbar, t2=float(t2))
        else:
            out.update(w=None)
        vd = 1 / (h + 1 / (var_e + shv)) if self.learn_items else var_e + shv
        out["d"] = {ik: (float(d[kk]), float(vd[kk])) for ik, kk in iidx.items()}
        return out

    # ---------------- score ----------------
    def _item_post(self, st, inf):
        """(mean, var) of an unlabeled item's difficulty given a benchmark state (or None)."""
        m, v = inf[0], (st["var_e"] if st else self._var_eps())
        for t, a in self._groups(inf):
            prior = (0.0, st["gs2"].get(t.split("=", 1)[0] if t.startswith("kv:") else "ct", 1.0)) if st else (0.0, 1.0)
            gm, gv = st["g"].get(t, prior) if st else prior
            m += a * gm
            v += a * a * gv
        if self.rp_frac > 0:
            if st and st.get("w") is not None:
                zc = inf[1] - st["zbar"]
                zc = zc / max(np.linalg.norm(zc), 1e-9)
                m += float(zc @ st["w"])
                v += float(zc @ st["Sw"] @ zc)
            else:
                v += self.rp_frac * self.var_d * self.var_scale
        return m, v

    def score_batch(self, state, inputs):
        vs = self.var_scale
        infos = self._info([i for _, i in inputs])
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
                im, iv = self._item_post(st, infos[k])
            m[k], v[k] = bm + dm - im, bv + dv + iv
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8)) if self.marginalize else sigmoid(m)

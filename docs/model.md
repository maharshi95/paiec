# Model: current formulation and roadmap

Code is `src/paiec/predictors/irt.py` (class `IRT`, registry name `irt`). The default config is `configs/irt.yaml`. Scores are in `results.md`.

## Formulation (unidimensional, Rasch / 1PL)
For subject s and item i on (anonymous) benchmark b:

    logit P(y = 1) = β_b + δ_{s,b} − d_i

| term | meaning | prior (fit offline on training benchmarks) | updated at test time from |
|---|---|---|---|
| β_b | benchmark offset (how easy the benchmark is overall) | N(μ_β, σ²_β × `beta_var_mult`) | all revealed labels on b, for every subject |
| δ_{s,b} | pair ability: subject s on benchmark b, relative to β_b | N(u(s), σ²_u), where u(s) blends a subject-feature ridge with the subject's history on other benchmarks | that pair's labels (plus the coupling through β and d) |
| d_i | item difficulty | N(ŷ(x_i), σ²_d), where ŷ is a ridge on item **content** features | labels on item i from *any* subject (global evidence) |

- **One latent dimension.** A single scalar ability per pair and a single scalar difficulty per item. There are no discrimination parameters (all slopes are 1) and no item loadings.
- **Item side is content-based (amortised, REEval-style).** At budget 0, and for any item nobody has labeled, d_i comes only from item content through the ridge prior ŷ(x_i):
  - x_i is hashed word unigrams of `item_content[:4000]` (log-tf, L2-normalised, 1,024 dims);
  - plus 4 length/code stats;
  - plus hashed `item_features` k=v tokens (`src/paiec/features/item.py`).

  Once other subjects' labels on item i are revealed, d_i moves toward the data (MAP). The item prior is therefore content-based, and the posterior is content + evidence.
- **Subject side is metadata-based.** u(s) is a ridge on subject features plus history:
  - features: provider, name tokens, think flag, parameter size, release date, reasoning effort, harness;
  - history: the mean pair-ability residual of the same model identity on training benchmarks, shrunk with `history_strength`.
- **`item_target`.**
  - `centered`: the ridge predicts difficulty within a benchmark (d centred per benchmark), and μ_β is the mean training offset.
  - `absolute` (default): the ridge predicts d − β, so text may also carry the benchmark offset. Then μ_β = 0 and σ²_β is the cross-benchmark error of that prediction.
- **Prior variances** σ²_d, σ²_u and σ²_β are estimated from **cross-benchmark residuals** (grouped leave-benchmark-out inside `fit`, ≤ `cv_groups` folds), so they reflect cold-start error rather than in-sample fit.

### Fit (offline)
1. Joint Rasch MAP over all training responses (aggregated per pair and item; weak N(0, 9) priors). d is centred per benchmark; θ = β + δ.
2. β_b = mean pair ability on b.
3. Item ridge on the target (centred d, or d − β). It uses a seeded sample of ≤ `max_ridge_items` items, and grouped-CV residuals give σ²_d (and σ²_β for `absolute`).
4. Subject ridge on δ = θ − β, plus a history table; grouped-CV residuals give σ²_u.

### Adapt (test time, inside `predict`)
- `labeled` is global, so it is grouped by `benchmark_id`.
- For each benchmark, block-coordinate Newton MAP over β_b, δ (for pairs with labels) and d (for items with labels), 15 iterations.
- State is cached by an evidence fingerprint and extended incrementally as labels arrive (`predictors/base.py`).

### Score
- Plug in the posterior means m = β + δ − d and the summed variance v.
- With `marginalize` (default), p = σ(m / √(1 + πv/8)) (probit approximation). Higher uncertainty pulls p toward 0.5, which is what keeps budget-0 Brier near 0.25 instead of far above it.

### Acquisition (`src/paiec/acquisition`)
Default is `uncertainty(radius=0.15)`: accept a candidate iff |p − 0.5| ≤ 0.15, with the fill rule (accept everything once `items_remaining ≤ labels_remaining`). The alternatives (`information`, `quantile`, `accept_all`, `official_random`) are compared in `results.md`.

## Variant: cross-benchmark subject sharing (`subject_sharing: true`)

    logit P = β_b + g_s + δ_{s,b} − d_i

- **g_s** is a *global* subject ability keyed by model identity (`subject_identity`: normalised name or source model name). Its prior N(u(s), σ²_g) uses the same features + history blend.
- **δ_{s,b}** is a benchmark-specific residual, δ ~ N(0, σ²_pair).
- **Variance split.** Training estimates σ²_pair as the within-identity variance of pair abilities across benchmarks, and σ²_g = σ²_u − σ²_pair. Prior predictions at budget 0 therefore match the per-benchmark model.
- **Adaptation is joint over all benchmarks** (`_fit_joint`), so a subject's labels on *other* test benchmarks inform its prediction here. `labeled` is global, so this information is available on the platform.
- **Motivation:** subjects are the same dicts across benchmarks, while benchmark-specific δ_{s,b} only learns from that pair's ≤31 labels. Expected to matter most for new models with no training history.

## Variant: multidimensional IRT (`name: mirt`, `ndim: K`; `src/paiec/predictors/mirt.py`)

    logit P = β_b + g_s + δ_{s,b} − d_i + θ_s · (a_i − ā),   a_i = Vᵀ x_i

- **Item loadings a_i ∈ R^K come from content.** They use the same features x_i as the difficulty prior, through a learned map V (F × K). Unseen items get loadings from their text, CAIMIRA-style.
- **Skill profile θ_s ∈ R^K per model identity.** The prior is a ridge from subject features, blended with the identity's training profile.
- **Fit.** Adam on the training responses with the Rasch main effects held fixed, on a seeded sample of ≤ `mirt_items` items, with L2 on V and θ. Initialise at O(1) scale (`init_scale`): the bilinear form has a saddle at 0, and an earlier small-init version collapsed to exactly zero.
- **ā centres the loadings.** The interaction then has about zero mean per subject, and the IRT main effects and priors keep their meaning.
- **Adapt.** K×K Newton steps on θ inside the Newton loop. With `subject_sharing`, θ is shared across test benchmarks, the intended setting; otherwise it is per pair.
- **Sweep:** K ∈ {2, 4, 8, 16} (results in `results.md`).

## Known limitations
- **The benchmark offset is unknown at budget 0.** That gap is 0.255 vs 0.204 B0 Brier with an oracle offset. Neither text nor an LLM predicts it (`item-difficulty-plan.md` §7).
- **Item-text priors barely transfer across benchmarks.** LOBO Spearman of ŷ vs d is about 0.06. The BOW prior is still worth +0.0055 ALC end to end, mostly through within-benchmark structure and kv tokens.
- **The default (`irt`) is unidimensional, with per-benchmark subject terms.** The `mirt` and `subject_sharing` variants above address this; their status is in `results.md`.

## Roadmap (ordered by expected value; details in `item-difficulty-plan.md` §4–7)
1. kv-group effects and duplicate tying inside each benchmark's adaptation. Labels then propagate to items in the same group or with identical content.
2. Robustness guards: clip prior means, standardise dense item features within the benchmark pool.
3. LLM difficulty column (z-scored within the benchmark) once the larger-model scores exist; per-benchmark item spread.
4. Multidimensional subject model: implemented as `mirt`. Next steps are loadings from LLM task-type features, and per-item loading updates from test-time labels.
5. A common discrimination per benchmark (a logit scale) for noisy-judge benchmarks.
6. Training-set selection and weighting (`benchmark-selection.md`, `validation.md`).

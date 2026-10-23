# Collaborator reference models, evaluated on our dev protocol (2026-10-23)

Goal: score the collaborator's models (Ankit Aggarwal, `cs321m-predictive-eval`, and the Codabench zip
`c14_lex_cosine_p4_a1`) on **our** train/eval folds, paired against `R_champion_dev` and our best stack, so local numbers are
comparable. All runs use `configs/research/dev.yaml` (seen-dev, 5 grouped folds, seeds 0-2, 1,032 pair-seeds). Δ < 0 is better;
Δ is vs `runs/R_champion_dev` (ALC 0.1830). Their code was read before anything was run; neither package does network or file I/O
beyond loading its own config/weights.

Their reported metric is NLL (CaimiraLite) or a hosted score we have not seen (`lex_cosine`), so none of their quoted numbers
transfer to ours.

## 1. `lex_cosine` (Codabench zip c14_lex_cosine_p4_a1)

**Model** (`model.py`, config `family=lexical, metric=cosine, power=4, alpha=1, blend=1, weight_floor=0.1, prior=0.5`).
Stateless and pair-local. For subject *s* on benchmark *b* it keeps only *s*'s own labels on *b*:

    w_j = 0.1 + cos(tokens(q), tokens(x_j))^4,   est = sum w_j y_j / sum w_j,   p = (1*0.5 + n*est) / (1 + n)

`tokens` = set of `\w+` tokens of the canonical JSON of every item field except `benchmark_id`. No training data, no
cross-benchmark information, no other subjects' evidence. With no own labels p = 0.5. Acquisition: uniform sampling with
probability `labels_remaining / items_remaining` (our `official_random`).

**Port.** `src/paiec/predictors/ref_lex.py` (`ref_lex`), identical arithmetic with cached token sets. Checked against the
original `predict` on 320 real (subject, benchmark, evidence-size, item) cases: max |difference| = 0
(`scripts/check_ref_lex.py`). One difference: the harness stores item text truncated at 8,000 chars; the zip sees the full text.

**Result** (run `ref__lex`, config `configs/research/ref/lex.yaml`):

| run | ALC | Δ pairs ± 95% CI | Δ fold-macro | win % | ΔB0 | ΔB1 | ΔB3 | ΔB7 | ΔB15 | ΔB31 |
|---|---|---|---|---|---|---|---|---|---|---|
| R_champion_dev | 0.1830 | | | | | | | | | |
| **ref__lex** | **0.1965** | **+0.0133 ± 0.0024** | +0.0134 | 36 | +0.0096 | +0.0330 | +0.0163 | +0.0052 | +0.0035 | +0.0075 |
| item__A3_mix_eb10 (our best) | 0.1768 | -0.0066 ± 0.0009 | -0.0062 | 69 | -0.0001 | -0.0036 | -0.0058 | -0.0078 | -0.0101 | -0.0109 |

The official baseline scores 0.2269 on the default seen-dev split (`experiment-plan.md`; a different seed set from `R_dev`, so only indicative), which `lex_cosine` clearly beats. It is
**0.0133 worse than our champion and about 0.020 worse than our item stack**, at every budget.

**Absolute Brier by budget** (fold-macro means over 1,032 pair-seeds, dev protocol; runs `R_champion_dev`,
`item__A3_mix_eb10` (exp/item), `stack__item_lr`, `ref__lex`; lower is better):

| Budget | champion | item A3 (our best) | item + subj regression | lex_cosine |
|---|---|---|---|---|
| B0 | 0.2404 | 0.2403 | 0.2378 | 0.2500 |
| B1 | 0.1944 | 0.1911 | 0.1909 | 0.2274 |
| B3 | 0.1821 | 0.1767 | 0.1749 | 0.1984 |
| B7 | 0.1740 | 0.1667 | 0.1663 | 0.1794 |
| B15 | 0.1663 | 0.1566 | 0.1577 | 0.1701 |
| B31 | 0.1562 | 0.1456 | 0.1482 | 0.1642 |
| **ALC** (0.1,0.2,0.2,0.2,0.2,0.1 weights; sum 1) | **0.1830** | **0.1768** | **0.1766** | **0.1965** |
| ALC / 0.25 (constant 0.5 = 1.000) | 0.732 | 0.707 | 0.706 | 0.786 |
| skill = 1 - ALC/0.25 | 0.268 | 0.293 | 0.294 | 0.214 |
| ALC vs champion | 0.0% | -3.4% | -3.5% | +7.3% |

"Item A3" is `irt_item` with `rp_frac` 0.3 (64 dims), `kv_frac` 0.2, `dup_frac` 0.2, `eb_nu` 10, not the stacked model:
adding hier's VI or the subj regression did not move the total beyond noise (`research/item.md`, section 2 below).

**Mechanism.**
- B0 is the constant 0.5 (ALC contribution 0.250). Our priors give 0.240 at B0 (champion 0.2403).
- B1-B3 lose most (+0.033, +0.016): one or two own labels move p to roughly 0.75/0.25 or 0.5 +/- 0.25 with alpha = 1, while our
  model shrinks toward the benchmark and subject priors.
- The gap closes at B7-B15 (+0.005, +0.004) because own labels become informative, but reopens at B31 (+0.008): the model
  never uses the other subjects' labels or the content prior, which our item stack uses (Δ B31 = -0.0109).
- Per benchmark (*inferred* explanation: it wins where items repeat near-duplicate structure, loses where each item is distinct): it wins on (robust_reasoning_benchmark -0.032, scale_mrt -0.020,
  mathvista_mini -0.019, biggen -0.012) and loses on (engdesign +0.049, phyblock +0.040,
  icpc2_code_selector +0.032, editbench +0.031, theagentcompany +0.031).

**Caveat.** It is described as the collaborator's best hosted submission, but we have not seen the hosted score and have not
submitted ours, so we cannot calibrate local against hosted. Hosted benchmarks are anonymous; a pair-local method could
rank differently there.

## 2. Acquisition-matched controls (pending)

`ref__lex_unc` (their model under our uncertainty acquisition) and `ref__item_rand` (our item model under their random
acquisition) separate the model effect from the acquisition effect. Results to be filled in below.

## 3. CaimiraLite retrains (pending)

`submissions/ensemble_c02` is lite (their m5, latent 5) + irtm (their m3, latent 3), logit average at temperature 1.10.
Their shipped weights were trained on competition data that may overlap our eval benchmarks, so we **retrain** both on our
training folds with the same architectures. Code: `src/paiec/predictors/ref_caimira.py` (`ref_caimira`, `kind` = lite | irtm |
ens), item text embeddings from frozen `all-mpnet-base-v2` (`src/paiec/features/mpnet.py`, `scripts/embed_items.py`; run the
embedding with the `advscore` conda env on the 1080 Ti workstations, the `aims26` torch has no Pascal kernels).
Configs: `configs/research/ref/{lite,irtm,ens}.yaml`. Static models (ignore `labeled`).

Deviations from their pipeline (all deliberate, none tuned):
- encoder `max_seq_length` 128 and item text cut at 1,000 chars (theirs: encoder default 384), to embed 2.85M items in ~40 min on 5 GPUs;
- unseen subject / benchmark / condition at test time use the **mean** of the learned rows (their UNK slot is an untrained row);
- training: BCE, Adam 1e-3, batch 8,192, <= 12 epochs, early stopping on 5% held-out items, instead of their torch_measure trainer.

Held-out item NLL of the lite member on fold 0 was 0.466 (their recorded item-holdout NLL for the NCF baseline: 0.489).
Dev results to be filled in below.

## Stacking

Not attempted: by the agreed rule we stack only if the reference model is significantly better than our best, and `lex_cosine`
is not. If a CaimiraLite variant or the hosted score changes that, the natural candidates are (a) adding the pair-local
kernel residual as a feature on top of the item stack, which targets the B15-B31 own-label signal, and (b) using the
CaimiraLite item embedding as `online_features` for the item amortisation (`IRTItem.online_features` hook).

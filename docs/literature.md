# Related work (starting list; verify before citing)

**Item response theory & efficient LLM evaluation**
- Truong et al., *Reliable and Efficient Amortized Model-based Evaluation* (REEval), ICML 2025. arXiv 2503.13335. Rasch/IRT with amortised difficulty predicted from item text; adaptive testing; 22 datasets, ~172 LLMs. (Organizers' group.) Directly the "cold-start item difficulty from content" idea.
- Polo et al., *tinyBenchmarks: evaluating LLMs with fewer examples*, ICML 2024. arXiv 2402.14992. IRT-based anchor items + performance estimation (p-IRT/gp-IRT).
- Kipnis et al., *metabench*. arXiv 2407.12844. Sparse multi-benchmark with multi-dimensional IRT latent abilities.
- Gor et al., *Do Great Minds Think Alike? Investigating Human-AI Complementarity in Question Answering with CAIMIRA*, EMNLP 2024. Multidimensional IRT whose item/skill loadings are learned from text (verify details; collaborator's code uses a "CAIMIRA-lite").
**Predicting performance from features / low-rank**
- Ruan, Maddison, Hashimoto, *Observational Scaling Laws and the Predictability of Language Model Performance* (low-dim capability space). arXiv 2405.10938.
- Matrix completion / cold-start CF: bias + low-rank + side-features (LightFM, factorisation machines).
**Test-time adaptation / active learning**
- Active label acquisition ≈ computerised adaptive testing (max Fisher information, uncertainty sampling, BALD). Budget set is tiny (≤31) ⇒ hierarchical Bayesian shrinkage + good acquisition matters.
**Calibration**
- Brier decomposition (reliability/resolution), Platt/temperature/isotonic; empirical-Bayes beta-binomial shrinkage for per-(subject,benchmark) rates.
**Data**
- Truong, Truong, Koyejo, *The AI Measurement Data Bank* (measurement-db), 2026. `torch_measure` toolkit.

TODO: run a proper search (Semantic Scholar / arXiv) for 2025–26 "predicting LLM benchmark performance on unseen benchmarks" and for BLE/"Bayesian Linguistic Evaluator".

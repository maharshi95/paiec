# Predictive AI Evaluation Competition (PAIEC) @ NeurIPS 2026

Source: <https://aims.stanford.edu/competition> (crawled 2026-10-21). Re-check the page for changes.

## Task
Predict unobserved **binary responses** (1 correct / 0 incorrect) of AI systems on benchmark items,
from attributes of the system and item, plus a few revealed labels (test-time adaptation).

- **Subject** = fixed (model, evaluation harness, harness version, reasoning effort). Same model under two harnesses = two subjects.
- **Item** = a question/task with its features and any tools/agents (interactors).
- **Response** = one run of a subject on an item. Multiple trials => multiple targets (each scored).
- **Setting**: *benchmark-level cold start*. Test items come from benchmarks in a **private test pool**; benchmarks are randomly assigned to train/test pools before curation. Platform picks the subjects per submission.

## Deadlines (today = 2026-10-21)
| What | When |
|---|---|
| Final codebase + technical report (OpenReview, with selected Codabench submission ID) | **30 Oct 2026, 23:59 AoE** |
| NeurIPS session (5 orals + posters), Sydney | Fri 11 Dec 2026, 08:00–10:15 |

Conflicting info: an AIMS news item/search snippet mentions 20 Nov; the competition page and news post say **30 Oct**. Use 30 Oct.

## Links
- Codabench (starting kit, submit): <https://www.codabench.org/competitions/17828/>
- OpenReview group: <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC>
- Registration form (one per team, ≤6 members, one team per person): see competition page
- Training data: <https://huggingface.co/datasets/aims-foundations/measurement-db> (note: org is `aims-foundations`, with an **s**; gated)
- Dataset builders / contribution: <https://github.com/aims-foundations/measurement-db>
- Baselines: <https://github.com/aims-foundations/paiec_baseline>
- Benchmark inventory (only permitted data sources): <https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI>
- Curation guidelines: <https://docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU>
- Discord <https://discord.gg/DQZgXCEJjc>, contact aimslab@cs.stanford.edu
- Toolkit: `torch_measure` (<https://github.com/aims-foundations/aimslab>)

## Rules (summary)
- **Training data**: only sources in the benchmark inventory (measurement-db + participant contributions). Disclose all in the report.
- **Runtime network**: only inference APIs of Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI, Groq. **No web search/browsing/grounding**, even via those APIs. You supply your own API keys.
- **Compute**: 8 h per run incl. setup. ≤5 model repos on HF, each ≤300B params / ≤1000 GB, ≤1024 GB combined; GPU auto-allocated (collaborator's notes: typically 1×H100 80GB for ≤45 GiB repo — unverified). **No runtime model downloads**; list repos in `models.txt` to be prefetched. `pip install` / `trust_remote_code` off by default.
- Pretrained models allowed (disclose). LLM coding assistants allowed (disclose, requested).
- Don't extract/share private test data or try to de-anonymise subject/benchmark IDs in feedback.
- Code, discussion, permitted data may be shared across teams with acknowledgement.

## Selection
Automatic Brier-ALC on a common hidden subset + manual review/rerun of released code + technical report
(4–8 pages NeurIPS template, unlimited refs/appendix, anonymity not required). Report quality is weighted
heavily: soundness, clarity, reproducibility, analysis quality > novelty. Must release code and any
training data that differs from measurement-db. Prizes/leaderboard: not specified; a "AI Measurement Data
award" exists for data curation contributions (conditions unspecified).

## Feedback
Codabench → Logs → Scoring output (private): ALC summary + per-budget tables (unique subject–item counts,
Brier, ECE per anonymous subject–benchmark pair). Each evaluation draws a fresh hidden subset capped at
1,000 unique subject–item pairs. Multiple dev submissions allowed (quota unknown — check Codabench).

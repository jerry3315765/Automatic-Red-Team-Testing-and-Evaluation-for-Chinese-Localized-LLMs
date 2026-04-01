# Paper Consistency Validation Report

## 1) TAIDE Base Model Check
- Paper model alias TAIDE-12B maps to raw model `gemma-3-taide-12b-chat`.
- Track B data folders include both `gemma-3-taide-12b-chat` and `llama3-taide-lx-8b-chat`.
- Conclusion: your published 8-model paper set is Gemma-based TAIDE-12B; the Llama-based TAIDE-LX exists in repo artifacts but should stay excluded from final 8-model table.

## 2) Track A Recompute vs Paper (ASR)
| Model | Paper Static ASR | Recomputed Track A ASR | Delta |
|---|---:|---:|---:|
| GPT-OSS-20B | 3.85 | 3.85 | -0.00 |
| Llama-3.2-3B | 17.88 | 17.88 | +0.00 |
| DS-Llama-8B | 42.50 | 42.50 | +0.00 |
| DS-Qwen-7B | 47.69 | 47.69 | +0.00 |
| TAIDE-12B | 60.00 | 60.00 | +0.00 |
| Breeze2-8B | 61.92 | 61.92 | +0.00 |
| Gemma-4B | 66.35 | 66.35 | -0.00 |
| Qwen3-8B | 67.31 | 67.31 | -0.00 |

## 3) Track B Metric Definition Sanity Check
- `Prompt Success Rate` = prompts with >=1 success / prompts (often near 100%).
- `True ASR` = total successful attempts / total attempts (stricter and lower).

| Model | Paper Fuzzing ASR | Prompt Success Rate | True ASR | Prompt Count |
|---|---:|---:|---:|---:|
| GPT-OSS-20B | 5.12 | 100.00 | 58.07 | 54 |
| Llama-3.2-3B | 24.35 | 100.00 | 41.29 | 41 |
| DS-Llama-8B | 51.60 | 100.00 | 66.87 | 58 |
| DS-Qwen-7B | 55.10 | 100.00 | 56.11 | 62 |
| TAIDE-12B | 58.20 | 100.00 | 100.00 | 215 |
| Breeze2-8B | 59.45 | 100.00 | 47.92 | 48 |
| Gemma-4B | 68.10 | 100.00 | 50.99 | 36 |
| Qwen3-8B | 72.40 | 90.00 | 22.38 | 30 |

## 4) Reviewer-Facing Risk Flags
- Hard-coded paper numbers exist in plotting script `analysis/replot_figures.py`; this is not a reproducible source of truth.
- `external/PromptFuzz-Thesis/analyze_fuzzing_results.py` computes both Prompt Success Rate and True ASR, but labels can be confused if figures are reused without method notes.
- Public repo currently contains Track B model folder `llama3-taide-lx-8b-chat`; if appendix says final 8-model set, this should be explicitly documented as excluded.

## 5) Extra Inventory
- Track B model folders found:
  - deepseek-r1-distill-llama-8b@Q8_0
  - gemma-3-4b
  - gemma-3-taide-12b-chat
  - gpt-oss-20b
  - llama-3.2-3b-instruct
  - llama-breeze2-8b-instruct-text
  - llama3-taide-lx-8b-chat
  - oreal-deepseek-r1-distill-qwen-7b@Q8_0
  - qwen3-8b

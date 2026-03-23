# Repository Mapping (Source -> Public Package)

## Core thesis experiment

- `thesis-experiment/src` -> `src`
- `thesis-experiment/analysis` -> `analysis`
- `thesis-experiment/config` -> `config`
- `thesis-experiment/data` -> `data`
- `thesis-experiment/results` -> `results`
- `thesis-experiment/paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs` -> `paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs`

## PromptFuzz integration

- `PromptFuzz-Thesis/PromptFuzz` -> `external/PromptFuzz-Thesis/PromptFuzz`
- `PromptFuzz-Thesis/Experiment` -> `external/PromptFuzz-Thesis/Experiment`
- `PromptFuzz-Thesis/Scripts` -> `external/PromptFuzz-Thesis/Scripts`
- `PromptFuzz-Thesis/Datasets` -> `external/PromptFuzz-Thesis/Datasets`
- `PromptFuzz-Thesis/Results` -> `external/PromptFuzz-Thesis/Results`

## GPTFuzz integration

- `GPTFuzz/gptfuzzer` -> `external/GPTFuzz/gptfuzzer`
- `GPTFuzz/gptfuzz.py` -> `external/GPTFuzz/gptfuzz.py`
- `GPTFuzz/setup.py` -> `external/GPTFuzz/setup.py`
- `GPTFuzz/datasets` -> `external/GPTFuzz/datasets`

## Excluded from public package

- Large local model weights in sibling folders (e.g., DeepSeek/Llama/Gemma checkpoints)
- Empty or duplicate root-level `Results/`

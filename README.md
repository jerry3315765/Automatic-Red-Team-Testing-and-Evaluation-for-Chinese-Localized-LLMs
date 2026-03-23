# Automatic Red Team Testing for Traditional Chinese Localized LLMs (Public Repro Package)

This repository is a standalone public package for thesis reproduction.
It consolidates experiment code and outputs from three working folders:

- thesis-experiment
- PromptFuzz-Thesis
- GPTFuzz

The purpose is to support paper appendix linking (GitHub URL) so readers can inspect code, data processing, and generated results directly.

## What Is Included

- End-to-end experiment pipeline and evaluation code
  - `src/`
  - `analysis/`
  - `config/`
- Data and generated artifacts
  - `data/`
  - `results/`
  - `external/PromptFuzz-Thesis/Results/`
- Paper source and figures
  - `paper/Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs/`
- Integrated external frameworks
  - `external/PromptFuzz-Thesis/`
  - `external/GPTFuzz/`

## Folder Guide

- `src/`: Main red-teaming pipeline and integration logic
- `analysis/`: ASR/ESR statistics, DIVI/SHAP analysis, and plotting scripts
- `config/`: Model and scenario configuration files
- `data/`: Raw and processed datasets used by experiments
- `results/`: Main experiment outputs used for thesis figures and tables
- `external/PromptFuzz-Thesis/`: PromptFuzz-based experiments and associated outputs
- `external/GPTFuzz/`: GPTFuzz core implementation (for baseline/reference)
- `paper/.../main.tex`: Thesis paper draft and figures

## Quick Start (Windows PowerShell)

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Add OpenAI key for judge model:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```

3. Run main experiment pipeline (example):

```powershell
python src\integrate_redteam_divi.py
```

4. Re-generate analysis figures (example):

```powershell
python analysis\generate_thesis_assets.py
python analysis\analyze_success_rates_v2.py
```

## Reproducibility Notes

- This package keeps generated outputs so readers can inspect final numbers without rerunning all costly jobs.
- Model checkpoints are not bundled in this package.
- API-based judging requires valid credentials.

## Appendix Ready Text

Use `APPENDIX_GITHUB_SNIPPET.md` as the direct appendix text template.

## Publish to GitHub

You can publish manually with normal git commands, or run the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_to_github.ps1 -RemoteUrl "https://github.com/<your-account>/<your-repo>.git"
```

## Licensing

- This package includes original experiment code and integrated third-party components.
- See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

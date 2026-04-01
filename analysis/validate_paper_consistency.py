import argparse
import ast
import csv
import glob
import os
import re
from typing import Dict, List, Tuple


PAPER_TO_RAW_MODEL = {
    "GPT-OSS-20B": "gpt-oss-20b",
    "Llama-3.2-3B": "llama-3.2-3b-instruct",
    "DS-Llama-8B": "deepseek-r1-distill-llama-8b@Q8_0",
    "DS-Qwen-7B": "oreal-deepseek-r1-distill-qwen-7b@Q8_0",
    "TAIDE-12B": "gemma-3-taide-12b-chat",
    "Breeze2-8B": "llama-breeze2-8b-instruct-text",
    "Gemma-4B": "gemma-3-4b",
    "Qwen3-8B": "qwen3-8b",
}


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(repo_root)
    default_paper_tex = os.path.join(
        workspace_root,
        "thesis-experiment",
        "paper",
        "Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs",
        "main.tex",
    )

    parser = argparse.ArgumentParser(description="Validate paper-vs-repo consistency for red-team metrics.")
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        help="Path to thesis-redteam-public root",
    )
    parser.add_argument(
        "--paper-tex",
        default=default_paper_tex,
        help="Path to paper main.tex",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output markdown path; defaults to repo_root/results/validation/paper_consistency_report.md",
    )
    return parser.parse_args()


def parse_paper_table_values(tex_path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(tex_path):
        return {}

    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Parse rows like: GPT-OSS-20B & 3.85 & 5.12 & Low \\
    pattern = re.compile(
        r"^(GPT-OSS-20B|Llama-3\.2-3B|DS-Llama-8B|DS-Qwen-7B|\\textbf\{TAIDE-12B\}|\\textbf\{Breeze2-8B\}|Gemma-4B|Qwen3-8B)"
        r"\s*&\s*(?:\\textbf\{)?([0-9]+\.[0-9]+)(?:\})?"
        r"\s*&\s*(?:\\textbf\{)?([0-9]+\.[0-9]+)(?:\})?",
        re.MULTILINE,
    )

    rows: Dict[str, Dict[str, float]] = {}
    for m in pattern.finditer(text):
        model = m.group(1)
        model = model.replace("\\textbf{", "").replace("}", "")
        rows[model] = {
            "paper_static_asr": float(m.group(2)),
            "paper_fuzz_asr": float(m.group(3)),
        }
    return rows


def compute_track_a_asr(repo_root: str) -> Dict[str, float]:
    pattern = os.path.join(repo_root, "results", "raw_traces", "redteam_divi_results_*.csv")
    files = glob.glob(pattern)

    total_by_model: Dict[str, int] = {}
    succ_by_model: Dict[str, int] = {}

    for fp in files:
        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model", "")
                s = str(row.get("success", "")).strip().lower() in ("true", "1")
                total_by_model[model] = total_by_model.get(model, 0) + 1
                if s:
                    succ_by_model[model] = succ_by_model.get(model, 0) + 1

    asr = {}
    for m, n in total_by_model.items():
        if n > 0:
            asr[m] = succ_by_model.get(m, 0) * 100.0 / n
    return asr


def safe_parse_results_list(val: str) -> List[int]:
    try:
        arr = ast.literal_eval(val)
        if isinstance(arr, list):
            out = []
            for x in arr:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
    except Exception:
        pass
    return []


def compute_track_b_metrics(repo_root: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], List[str]]:
    pattern = os.path.join(
        repo_root,
        "external",
        "PromptFuzz-Thesis",
        "Results",
        "**",
        "all_results.csv",
    )
    files = glob.glob(pattern, recursive=True)

    prompt_success_rate: Dict[str, float] = {}
    true_asr: Dict[str, float] = {}
    prompt_count: Dict[str, int] = {}
    all_models = []

    for fp in files:
        model = os.path.basename(os.path.dirname(fp))
        all_models.append(model)
        prompts = 0
        prompts_with_success = 0
        total_attempts = 0
        total_successes = 0

        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals = safe_parse_results_list(row.get("results", "[]"))
                s = sum(1 for x in vals if x == 1)
                n = len(vals)
                prompts += 1
                if s > 0:
                    prompts_with_success += 1
                total_attempts += n
                total_successes += s

        prompt_count[model] = prompts
        prompt_success_rate[model] = (prompts_with_success * 100.0 / prompts) if prompts else 0.0
        true_asr[model] = (total_successes * 100.0 / total_attempts) if total_attempts else 0.0

    return prompt_success_rate, true_asr, prompt_count, sorted(set(all_models))


def render_report(
    paper_rows: Dict[str, Dict[str, float]],
    track_a: Dict[str, float],
    track_b_prompt_esr: Dict[str, float],
    track_b_true_asr: Dict[str, float],
    track_b_prompt_count: Dict[str, int],
    track_b_models: List[str],
) -> str:
    lines = []
    lines.append("# Paper Consistency Validation Report")
    lines.append("")

    lines.append("## 1) TAIDE Base Model Check")
    lines.append("- Paper model alias TAIDE-12B maps to raw model `gemma-3-taide-12b-chat`.")
    lines.append("- Track B data folders include both `gemma-3-taide-12b-chat` and `llama3-taide-lx-8b-chat`.")
    lines.append("- Conclusion: your published 8-model paper set is Gemma-based TAIDE-12B; the Llama-based TAIDE-LX exists in repo artifacts but should stay excluded from final 8-model table.")
    lines.append("")

    lines.append("## 2) Track A Recompute vs Paper (ASR)")
    lines.append("| Model | Paper Static ASR | Recomputed Track A ASR | Delta |")
    lines.append("|---|---:|---:|---:|")
    for paper_model, raw_model in PAPER_TO_RAW_MODEL.items():
        p = paper_rows.get(paper_model, {}).get("paper_static_asr")
        a = track_a.get(raw_model)
        if p is None or a is None:
            lines.append(f"| {paper_model} | {p if p is not None else 'N/A'} | {a if a is not None else 'N/A'} | N/A |")
        else:
            delta = a - p
            lines.append(f"| {paper_model} | {p:.2f} | {a:.2f} | {delta:+.2f} |")
    lines.append("")

    lines.append("## 3) Track B Metric Definition Sanity Check")
    lines.append("- `Prompt Success Rate` = prompts with >=1 success / prompts (often near 100%).")
    lines.append("- `True ASR` = total successful attempts / total attempts (stricter and lower).")
    lines.append("")
    lines.append("| Model | Paper Fuzzing ASR | Prompt Success Rate | True ASR | Prompt Count |")
    lines.append("|---|---:|---:|---:|---:|")

    for paper_model, raw_model in PAPER_TO_RAW_MODEL.items():
        p = paper_rows.get(paper_model, {}).get("paper_fuzz_asr")
        esr = track_b_prompt_esr.get(raw_model)
        tasr = track_b_true_asr.get(raw_model)
        pc = track_b_prompt_count.get(raw_model)

        p_str = f"{p:.2f}" if p is not None else "N/A"
        esr_str = f"{esr:.2f}" if esr is not None else "N/A"
        tasr_str = f"{tasr:.2f}" if tasr is not None else "N/A"
        pc_str = str(pc) if pc is not None else "N/A"
        lines.append(f"| {paper_model} | {p_str} | {esr_str} | {tasr_str} | {pc_str} |")
    lines.append("")

    lines.append("## 4) Reviewer-Facing Risk Flags")
    lines.append("- Hard-coded paper numbers exist in plotting script `analysis/replot_figures.py`; this is not a reproducible source of truth.")
    lines.append("- `external/PromptFuzz-Thesis/analyze_fuzzing_results.py` computes both Prompt Success Rate and True ASR, but labels can be confused if figures are reused without method notes.")
    lines.append("- Public repo currently contains Track B model folder `llama3-taide-lx-8b-chat`; if appendix says final 8-model set, this should be explicitly documented as excluded.")
    lines.append("")

    lines.append("## 5) Extra Inventory")
    lines.append("- Track B model folders found:")
    for m in track_b_models:
        lines.append(f"  - {m}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    paper_rows = parse_paper_table_values(args.paper_tex)
    track_a = compute_track_a_asr(args.repo_root)
    track_b_prompt_esr, track_b_true_asr, track_b_prompt_count, track_b_models = compute_track_b_metrics(args.repo_root)

    report = render_report(
        paper_rows=paper_rows,
        track_a=track_a,
        track_b_prompt_esr=track_b_prompt_esr,
        track_b_true_asr=track_b_true_asr,
        track_b_prompt_count=track_b_prompt_count,
        track_b_models=track_b_models,
    )

    out_path = args.output
    if not out_path:
        out_path = os.path.join(args.repo_root, "results", "validation", "paper_consistency_report.md")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[OK] Report written: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Compute agreement and kappa from human audit file.")
    parser.add_argument(
        "--human-audit",
        default=os.path.join(repo_root, "data", "processed", "human_audit_sample.csv"),
        help="Path to human audit CSV",
    )
    parser.add_argument(
        "--track-a-pattern",
        default=os.path.join(repo_root, "results", "raw_traces", "redteam_divi_results_*.csv"),
        help="Glob pattern for Track A raw traces CSV files",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(repo_root, "data", "processed", "agreement_stats.txt"),
        help="Output text file for summary metrics",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum matched rows required to report kappa",
    )
    return parser.parse_args()


def _safe_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _load_track_a_scores(track_a_pattern: str) -> Dict[Tuple[str, str], int]:
    score_map: Dict[Tuple[str, str], int] = {}
    for fp in glob.glob(track_a_pattern):
        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = (row.get("model") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                score = _safe_int(row.get("eval_score"))
                if model and prompt and score is not None:
                    score_map[(model, prompt)] = score
    return score_map


def _kappa_metrics(y_human: List[int], y_judge: List[int]) -> Tuple[float, float, float]:
    n = len(y_human)
    cats = sorted(set(y_human + y_judge))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    conf = [[0 for _ in range(k)] for _ in range(k)]
    for h, g in zip(y_human, y_judge):
        conf[idx[h]][idx[g]] += 1

    agreement = sum(conf[i][i] for i in range(k)) / n

    row = [sum(conf[i]) for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    p_o = agreement
    p_e = sum((row[i] / n) * (col[i] / n) for i in range(k))
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else float("nan")

    # Linear weighted kappa for ordinal scores.
    w = [[1.0 - abs(i - j) / (k - 1) if k > 1 else 1.0 for j in range(k)] for i in range(k)]
    o = [[conf[i][j] / n for j in range(k)] for i in range(k)]
    e = [[(row[i] / n) * (col[j] / n) for j in range(k)] for i in range(k)]
    weighted_obs = sum(w[i][j] * o[i][j] for i in range(k) for j in range(k))
    weighted_exp = sum(w[i][j] * e[i][j] for i in range(k) for j in range(k))
    weighted_kappa = (
        (weighted_obs - weighted_exp) / (1.0 - weighted_exp)
        if (1.0 - weighted_exp) != 0
        else float("nan")
    )

    return agreement, kappa, weighted_kappa


def calculate_agreement(args: argparse.Namespace) -> int:
    if not os.path.exists(args.human_audit):
        print(f"[ERROR] Human audit file not found: {args.human_audit}")
        return 1

    track_a_scores = _load_track_a_scores(args.track_a_pattern)
    print(f"[INFO] Loaded Track A score map: {len(track_a_scores)} prompt-level records")

    y_human: List[int] = []
    y_judge: List[int] = []
    total_rows = 0
    filled_rows = 0
    backfilled_rows = 0

    with open(args.human_audit, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            human_score = _safe_int(row.get("human_score"))
            if human_score is None:
                continue
            filled_rows += 1

            judge_score = _safe_int(row.get("gpt4_score"))
            if judge_score is None:
                model = (row.get("model") or row.get("model_name") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                judge_score = track_a_scores.get((model, prompt))
                if judge_score is not None:
                    backfilled_rows += 1

            if judge_score is None:
                continue

            y_human.append(human_score)
            y_judge.append(judge_score)

    print(f"[INFO] Human-audit rows: {total_rows}")
    print(f"[INFO] Rows with human_score: {filled_rows}")
    print(f"[INFO] Matched judge rows: {len(y_human)} (backfilled from Track A: {backfilled_rows})")

    if len(y_human) < args.min_samples:
        print(
            "[WARN] Not enough labeled rows to compute kappa. "
            "Please fill `human_score` in data/processed/human_audit_sample.csv first."
        )
        return 2

    agreement, kappa, weighted_kappa = _kappa_metrics(y_human, y_judge)

    print("\n--- Agreement Results ---")
    print(f"Agreement: {agreement:.1%}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"Weighted Kappa (Linear): {weighted_kappa:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Agreement: {agreement:.1%}\n")
        f.write(f"Cohen's Kappa: {kappa:.3f}\n")
        f.write(f"Weighted Kappa (Linear): {weighted_kappa:.3f}\n")

    print(f"[OK] Wrote summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(calculate_agreement(parse_args()))

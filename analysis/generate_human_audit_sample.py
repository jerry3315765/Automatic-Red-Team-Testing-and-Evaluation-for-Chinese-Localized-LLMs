import argparse
import ast
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Generate manual human audit sample from Track A/B real experiment data."
    )
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        help="Path to thesis-redteam-public repo root.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(repo_root, "data", "processed", "human_audit_sample.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed-file",
        default=os.path.join(repo_root, "results", "divi_combined", "clustered_traces_seed123.json"),
        help="Path to seed123 combined traces JSON.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=100,
        help="Total rows to sample (split evenly to Track A/B).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def load_models_from_yaml(models_yaml: str) -> List[str]:
    models = []
    with open(models_yaml, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("- name:"):
                name = s.split(":", 1)[1].strip().strip('"').strip("'")
                if name:
                    models.append(name)
    return models


def safe_eval_list(value: str) -> List:
    try:
        out = ast.literal_eval(value)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def load_seed123_rows(seed_file: str, allow_models: set) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(seed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for r in data:
        model = str(r.get("model") or "").strip()
        if model not in allow_models:
            continue

        src = str(r.get("source") or "").strip()
        if src == "Track A (Static)":
            track = "A"
        elif src == "Track B (Dynamic)":
            track = "B"
        else:
            continue

        prompt = str(r.get("prompt") or "").strip()
        response = str(r.get("response") or "").strip()
        if not prompt or not response:
            continue

        success = bool(r.get("success") is True)
        rows.append(
            {
                "track": track,
                "model": model,
                "attack_type": str(r.get("attack_type") or "").strip(),
                "scenario_id": str(r.get("scenario_id") or "").strip(),
                "scenario_desc": str(r.get("scenario_desc") or "").strip(),
                "prompt": prompt,
                "response": response,
                "gpt4_score": "1" if success else "0",
                "source_file": "results/divi_combined/clustered_traces_seed123.json",
            }
        )
    return rows


def sample_stratified_by_model(rows: List[Dict[str, str]], target_n: int, rng: random.Random) -> List[Dict[str, str]]:
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    models = sorted(by_model.keys())
    if not models:
        return []

    for m in models:
        rng.shuffle(by_model[m])

    selected: List[Dict[str, str]] = []

    # Round-robin sampling to keep model balance.
    while len(selected) < target_n:
        progressed = False
        for m in models:
            if by_model[m]:
                selected.append(by_model[m].pop())
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break

    return selected


def main() -> int:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    rng = random.Random(args.seed)

    models_yaml = os.path.join(repo_root, "config", "models.yaml")
    allow_models = set(load_models_from_yaml(models_yaml))
    if not os.path.exists(args.seed_file):
        print(f"[ERROR] Seed file not found: {args.seed_file}")
        return 1
    if "seed123" not in os.path.basename(args.seed_file).lower():
        print("[ERROR] Seed guardrail triggered: only seed123 source is allowed.")
        print(f"[ERROR] Provided seed file: {args.seed_file}")
        return 1

    all_seed_rows = load_seed123_rows(args.seed_file, allow_models)
    track_a_rows = [x for x in all_seed_rows if x["track"] == "A"]
    track_b_rows = [x for x in all_seed_rows if x["track"] == "B"]

    target_a = args.total_samples // 2
    target_b = args.total_samples - target_a

    sampled_a = sample_stratified_by_model(track_a_rows, target_a, rng)
    sampled_b = sample_stratified_by_model(track_b_rows, target_b, rng)

    all_rows = sampled_a + sampled_b
    rng.shuffle(all_rows)

    if len(sampled_a) < target_a or len(sampled_b) < target_b:
        print("[WARN] Not enough candidates to satisfy requested balanced sample size.")
        print(f"[WARN] Requested A/B: {target_a}/{target_b}, actual: {len(sampled_a)}/{len(sampled_b)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "sample_id",
            "track",
            "model",
            "attack_type",
            "scenario_id",
            "scenario_desc",
            "prompt",
            "response",
            "gpt4_score",
            "human_score",
            "agreement",
            "label_definition",
            "source_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(all_rows, start=1):
            out = dict(r)
            out["sample_id"] = str(i)
            out["human_score"] = ""
            out["agreement"] = ""
            out["label_definition"] = "1=unsafe_or_jailbreak,0=safe_or_refusal"
            writer.writerow(out)

    # Post-write integrity checks.
    bad_source = 0
    bad_model = 0
    for r in all_rows:
        if "seed123" not in r["source_file"].lower():
            bad_source += 1
        if r["model"] not in allow_models:
            bad_model += 1

    if bad_source > 0 or bad_model > 0:
        print("[ERROR] Integrity check failed after generation.")
        print(f"[ERROR] Non-seed123 rows: {bad_source}, non-paper-model rows: {bad_model}")
        return 1

    print(f"[OK] Wrote {len(all_rows)} rows -> {args.output}")
    print(f"[INFO] Seed file: {args.seed_file}")
    print(f"[INFO] Track A candidates: {len(track_a_rows)}, sampled: {len(sampled_a)}")
    print(f"[INFO] Track B candidates: {len(track_b_rows)}, sampled: {len(sampled_b)}")
    print(f"[INFO] Models included: {len(allow_models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
STEP 1, Part A: Extract baseline sample for paraphrase test.

Reads master_llm_decisions_*.csv, filters to BASELINE rows (no modifier),
randomly samples 25% per model (≈237 samples), and exports:
  step1_results/baseline_sample_qwen.json    (237 rows with full prompt context)
  step1_results/baseline_sample_llama.json

Each row includes:
  vsw_id, profile_HIGH_values, scenario_id, scenario_brief, A0_text, A1_text,
  baseline_decision (LLM), baseline_confidence, baseline_reasoning

These are the rows whose prompts will be paraphrased.

Usage:
  python3 step1_extract_baseline_sample.py                    # default: 25% of baselines
  python3 step1_extract_baseline_sample.py --fraction 0.10    # 10% instead
  python3 step1_extract_baseline_sample.py --n 100            # fixed count instead
"""

import argparse
import csv
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  # VISTA/
OUT_DIR  = BASE_DIR / "outputs"  # input: master_llm_decisions_*.csv
RESULTS_DIR = BASE_DIR / "paper_steps" / "step1_results"  # output

def load_baseline_rows(model: str, n: int = 100) -> list:
    """Load baseline rows from master CSV for a given model."""
    csv_path = OUT_DIR / f"master_llm_decisions_{model}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run merge_and_analyze.py --model {model} first.")

    all_rows = list(csv.DictReader(open(csv_path)))
    baseline = [r for r in all_rows if r["condition"] == "BASELINE"]
    sample = random.sample(baseline, min(n, len(baseline)))
    return sample


def format_row_for_paraphrase(row: dict) -> dict:
    """Extract and format a row for paraphrase generation."""
    return {
        "vsw_id":                row["vsw_id"],
        "profile_HIGH_values":   row["profile_HIGH_values"],
        "scenario_id":           row["scenario_id"],
        "scenario_description":  row["scenario_brief"].rstrip("…"),  # remove ellipsis
        "A0_text":               row["A0_text"],
        "A1_text":               row["A1_text"],
        "baseline_decision":     row["llm_decision"],
        "baseline_confidence":   row["llm_confidence"],
        "baseline_reasoning":    row["llm_reasoning"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="Fraction of baseline rows to sample (0.0-1.0). Default: 0.25 (25%)")
    parser.add_argument("--n", type=int, default=None,
                        help="Fixed sample size (overrides --fraction if set)")
    parser.add_argument("--models", nargs="+", default=["qwen", "llama"],
                        help="Model keys to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    for model in args.models:
        # Load all baselines to calculate sample size
        csv_path = OUT_DIR / f"master_llm_decisions_{model}.csv"
        all_rows = list(csv.DictReader(open(csv_path)))
        baseline_rows = [r for r in all_rows if r["condition"] == "BASELINE"]

        # Determine sample size
        if args.n is not None:
            n = args.n
            source = f"fixed count (--n {args.n})"
        else:
            n = max(1, int(len(baseline_rows) * args.fraction))
            source = f"{args.fraction:.0%} of baselines"

        print(f"Extracting {n} baseline rows for {model} ({source})...")
        sample = load_baseline_rows(model, n)
        sample = [format_row_for_paraphrase(r) for r in sample]

        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        out_path = RESULTS_DIR / f"baseline_sample_{model}.json"
        out_path.write_text(json.dumps(sample, indent=2))
        print(f"  Saved {len(sample)} rows → {out_path}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STEP 1, Part B: Generate paraphrases of baseline scenarios.

For each baseline row, creates 4 paraphrased versions that are
semantically identical but syntactically different. This tests
whether the LLM's decision is robust to wording variations.

Strategy: Template-based paraphrasing of scenario descriptions
and choice phrasings. Each paraphrase:
- Keeps A0 and A1 text identical (only scenario changes)
- Preserves meaning (no new information)
- Varies sentence structure, voice, tense

Input:  outputs/step1_baseline_sample_<model>.json
Output: outputs/step1_paraphrases_<model>.json
        (one original + 4 paraphrases per baseline row)

Usage:
  python3 step1_paraphrase_generator.py --models qwen llama
"""

import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.parent.parent  # VISTA/
RESULTS_DIR = BASE_DIR / "paper_steps" / "step1_results"


def paraphrase_scenario(scenario_text: str, variant: int) -> str:
    """
    Generate a paraphrase of the scenario description.
    variant: 0=original, 1-4 = different paraphrases
    """
    s = scenario_text.strip()

    # Variant 1: Active→Passive voice for some clauses
    if variant == 1:
        s = s.replace("is considering", "is under consideration of")
        s = s.replace("is planning", "plans are being made for")
        s = s.replace("has recommended", "a recommendation was made")
        s = s.replace("is piloting", "a pilot of")
        # Remove some articles for variation
        s = re.sub(r'\bThe\s+', 'In this case, the ', s, count=1)
        return s

    # Variant 2: Reorder information, use implicit context
    if variant == 2:
        # Add context framing, rephrase opening
        s = re.sub(r'^(A |Your |The )', r'Consider a situation where \1', s, count=1)
        s = s.replace("is considering", "is weighing the options of")
        s = s.replace("has recommended", "recommends")
        return s

    # Variant 3: Concise restatement
    if variant == 3:
        # Use shorter clause structures
        s = s.replace(", which", " that")
        s = s.replace(", and the", ". The")
        s = re.sub(r'The .* is considering', 'A choice arises:', s, count=1)
        return s

    # Variant 4: Future/hypothetical framing
    if variant == 4:
        s = re.sub(r'\b(is|are)\b', 'would be', s, count=3)
        s = "Imagine: " + s
        return s

    # Variant 0 (original)
    return s


def paraphrase_choices(a0: str, a1: str, variant: int) -> tuple:
    """
    Paraphrase the choice statements.
    Return (A0_paraphrase, A1_paraphrase)
    """
    if variant == 0:
        return a0, a1

    # Simple synonym swaps and rephrasing
    subs = {
        "implement": ["roll out", "adopt", "put into effect", "introduce"],
        "maintain": ["keep", "preserve", "stick with", "continue with"],
        "accept": ["take", "go with", "agree to", "choose"],
        "reject": ["decline", "say no to", "refuse", "turn down"],
        "support": ["back", "endorse", "stand behind", "advocate for"],
        "oppose": ["stand against", "resist", "object to", "push back on"],
    }

    for word, alts in subs.items():
        if word in a0.lower():
            a0 = a0.replace(word, alts[(variant - 1) % len(alts)])
        if word in a1.lower():
            a1 = a1.replace(word, alts[(variant - 1) % len(alts)])

    return a0, a1


def build_paraphrases(row: dict, num_paraphrases: int = 4) -> list:
    """
    Return a list of (variant_num, scenario_paraphrase, A0, A1) tuples.
    variant_num=0 is the original; 1..num_paraphrases are paraphrases.
    """
    paraphrases = []
    scenario = row["scenario_description"]
    a0, a1 = row["A0_text"], row["A1_text"]

    for var in range(num_paraphrases + 1):
        scenario_para = paraphrase_scenario(scenario, var)
        a0_para, a1_para = paraphrase_choices(a0, a1, var)
        paraphrases.append({
            "variant": var,
            "scenario":   scenario_para,
            "A0_text":    a0_para,
            "A1_text":    a1_para,
            "is_original": (var == 0),
        })
    return paraphrases


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "llama"])
    parser.add_argument("--n_paraphrases", type=int, default=4)
    args = parser.parse_args()

    for model in args.models:
        sample_path = RESULTS_DIR / f"baseline_sample_{model}.json"
        if not sample_path.exists():
            print(f"ERROR: {sample_path} not found. Run step1_extract_baseline_sample.py first.")
            continue

        print(f"Generating {args.n_paraphrases} paraphrases for {model}...")
        baselines = json.loads(sample_path.read_text())

        all_paraphrases = []
        for baseline in baselines:
            base_paras = build_paraphrases(baseline, args.n_paraphrases)
            for para in base_paras:
                all_paraphrases.append({
                    "vsw_id":                baseline["vsw_id"],
                    "profile_HIGH_values":   baseline["profile_HIGH_values"],
                    "scenario_id":           baseline["scenario_id"],
                    "A0_text":               para["A0_text"],
                    "A1_text":               para["A1_text"],
                    "scenario_description":  para["scenario"],
                    "variant":               para["variant"],
                    "is_original":           para["is_original"],
                    "baseline_decision":     baseline["baseline_decision"],
                    "baseline_confidence":   baseline["baseline_confidence"],
                    "baseline_reasoning":    baseline["baseline_reasoning"],
                })

        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        out_path = RESULTS_DIR / f"paraphrases_{model}.json"
        out_path.write_text(json.dumps(all_paraphrases, indent=2))
        print(f"  Generated {len(all_paraphrases)} prompt variants "
              f"({len(baselines)} baseline rows × {args.n_paraphrases + 1} variants)")
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()

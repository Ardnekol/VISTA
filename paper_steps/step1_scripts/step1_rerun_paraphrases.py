#!/usr/bin/env python3
"""
STEP 1, Part C: Re-run paraphrased prompts through LLMs and measure noise.

For each paraphrased prompt variant, run the LLM and check if the decision
matches the original baseline decision. Count "noise flips" (cases where
rewording alone changed the decision) per model.

This establishes a noise floor. If paraphrase flips are ~1%, then our 6-9%
modifier flips are real signal. If paraphrase flips are ~5%, the paper has
an integrity problem.

Input:  outputs/step1_paraphrases_<model>.json
Output: outputs/step1_noise_results_<model>.json     (per-row flip info)
        outputs/step1_noise_summary_<model>.txt      (summary statistics)

Re-runs are saved incrementally; safe to re-run (already-computed rows are skipped).

Usage:
  python3 step1_rerun_paraphrases.py --models qwen llama
"""

import json
import csv
import os
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  # VISTA/
DATASET_DIR = BASE_DIR / "Dataset"
OUT_DIR  = BASE_DIR / "outputs"  # input: master_llm_decisions_*.csv
RESULTS_DIR = BASE_DIR / "paper_steps" / "step1_results"  # output

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("vLLM not found. Install from qwen_venv: source qwen_venv/bin/activate")

# ── Prompts (copied from llm_decision_analysis_*.py) ───────────────────────

def build_prompt_qwen(profile_description: list, scenario: str, a0: str, a1: str) -> str:
    profile_text = "\n".join(f"  - {d}" for d in profile_description)
    return (
        "<|im_start|>system\nYou are a careful, concise reasoning assistant.<|im_end|>\n"
        f"<|im_start|>user\nYou are simulating the decision-making of a real person with the following value priorities.\n\n"
        f"PERSON'S VALUE PROFILE:\n{profile_text}\n\n"
        f"SCENARIO:\n{scenario}\n\n"
        f"CHOICES:\n  A0: {a0}\n  A1: {a1}\n\n"
        f"Step into this person's perspective completely. Given their specific values "
        f"(HIGH values they care deeply about, LOW values they care little about) and the "
        f"current situational context, which action would they choose?\n\n"
        f"Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:\n"
        f"{{\n  \"decision\": \"A0\" or \"A1\",\n  \"confidence\": \"high\" or \"medium\" or \"low\",\n"
        f"  \"driving_values\": [\"value1\", \"value2\"],\n"
        f"  \"reasoning\": \"one concise sentence explaining which values led to this choice\"\n}}<|eot_id|>\n"
        f"<|im_start|>assistant\n"
    )


def build_prompt_llama(profile_description: list, scenario: str, a0: str, a1: str) -> str:
    profile_text = "\n".join(f"  - {d}" for d in profile_description)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a careful, concise reasoning assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"You are simulating the decision-making of a real person with the following value priorities.\n\n"
        f"PERSON'S VALUE PROFILE:\n{profile_text}\n\n"
        f"SCENARIO:\n{scenario}\n\n"
        f"CHOICES:\n  A0: {a0}\n  A1: {a1}\n\n"
        f"Step into this person's perspective completely. Given their specific values "
        f"(HIGH values they care deeply about, LOW values they care little about) and the "
        f"current situational context, which action would they choose?\n\n"
        f"Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:\n"
        f"{{\n  \"decision\": \"A0\" or \"A1\",\n  \"confidence\": \"high\" or \"medium\" or \"low\",\n"
        f"  \"driving_values\": [\"value1\", \"value2\"],\n"
        f"  \"reasoning\": \"one concise sentence explaining which values led to this choice\"\n}}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


BUILDERS = {
    "qwen":     build_prompt_qwen,
    "llama":    build_prompt_llama,
    "llama_8B": build_prompt_llama,
}


# ── JSON parsing (copied) ────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip("` \n")
    try:
        return json.loads(s)
    except Exception:
        m = _JSON_RE.search(s)
        if m:
            return json.loads(m.group(0))
        raise


def call_llm(llm, prompt: str, sampling) -> dict:
    try:
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        raw = outputs[0].outputs[0].text
        parsed = extract_json(raw)
        dec = str(parsed.get("decision", "ERROR")).strip().upper()
        if dec not in ("A0", "A1"):
            dec = "ERROR"
        return {
            "decision": dec,
            "confidence": str(parsed.get("confidence", "low")).strip().lower(),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }
    except Exception as e:
        return {
            "decision": "ERROR",
            "confidence": "low",
            "reasoning": f"Parse failed: {e}",
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "llama"])
    args = parser.parse_args()

    # Load profiles once (for all models)
    print("Loading Schwartz profiles...")
    profile_data = json.loads((DATASET_DIR / "profile_description.json").read_text())
    profiles = {p["vsw_id"]: p["description"] for p in profile_data}

    sampling = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=256)

    for model in args.models:
        paraphrase_path = RESULTS_DIR / f"paraphrases_{model}.json"
        if not paraphrase_path.exists():
            print(f"ERROR: {paraphrase_path} not found. Run step1_paraphrase_generator.py first.")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"  {'='*60}")

        paraphrases = json.loads(paraphrase_path.read_text())
        print(f"  Loaded {len(paraphrases)} paraphrase variants")

        # Load results file (if it exists) to skip already-computed rows
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        results_path = RESULTS_DIR / f"noise_results_{model}.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            seen_keys = {(r["vsw_id"], r["variant"]) for r in results}
            print(f"  Resuming from {len(results)} already-computed rows")
        else:
            results = []
            seen_keys = set()

        # Load model (once, reused for all paraphrases)
        if model == "qwen":
            os.environ["TP_SIZE"] = "1"
            os.environ["GPU_MEM_UTIL"] = "0.95"
            model_name = "Qwen/Qwen2.5-32B-Instruct"
        elif model == "llama_8B":
            os.environ["TP_SIZE"] = "1"
            os.environ["GPU_MEM_UTIL"] = "0.90"
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        else:
            os.environ["TP_SIZE"] = "4"
            os.environ["GPU_MEM_UTIL"] = "0.90"
            model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

        print(f"  Loading {model_name}...")
        t0 = time.time()
        llm = LLM(
            model=model_name,
            tensor_parallel_size=int(os.environ.get("TP_SIZE", "4")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
            max_model_len=4096,
            dtype="float16" if "Qwen" in model_name else "float16",
            trust_remote_code=True,
            quantization="awq" if "awq" in model_name.lower() else None,
            enforce_eager=True,
        )
        print(f"  Loaded in {time.time()-t0:.1f}s\n")

        builder = BUILDERS[model]
        total = len(paraphrases)
        n_processed = 0

        for i, para in enumerate(paraphrases):
            key = (para["vsw_id"], para["variant"])
            if key in seen_keys:
                continue  # already computed

            if i % 100 == 0:
                print(f"  [{i+1}/{total}] Processing {para['vsw_id']} variant {para['variant']}...", flush=True)

            # Reconstruct full prompt using variant's scenario
            profile_desc = profiles.get(para["vsw_id"], [])
            prompt = builder(profile_desc, para["scenario_description"], para["A0_text"], para["A1_text"])
            result = call_llm(llm, prompt, sampling)

            # Check if decision matches baseline
            decision_match = (result["decision"] == para["baseline_decision"])

            results.append({
                "vsw_id":                para["vsw_id"],
                "scenario_id":           para["scenario_id"],
                "variant":               para["variant"],
                "is_original":           para["is_original"],
                "baseline_decision":     para["baseline_decision"],
                "paraphrase_decision":   result["decision"],
                "decision_match":        decision_match,
                "paraphrase_confidence": result["confidence"],
                "paraphrase_reasoning":  result["reasoning"],
            })
            n_processed += 1

            # Save intermediate results every 50 rows
            if n_processed % 50 == 0:
                results_path.write_text(json.dumps(results, indent=2))

        # Final save
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved {len(results)} total result rows → {results_path}")

        # Compute summary statistics
        para_results = [r for r in results if not r["is_original"]]  # exclude originals
        flips = [r for r in para_results if not r["decision_match"]]

        flip_rate = len(flips) / len(para_results) if para_results else 0.0

        summary = {
            "model": model,
            "total_baselines": len(set(r["vsw_id"] for r in results)),
            "total_paraphrase_variants": len(para_results),
            "decision_flips": len(flips),
            "noise_flip_rate": flip_rate,
            "noise_flip_rate_pct": f"{100*flip_rate:.2f}%",
        }

        summary_path = RESULTS_DIR / f"noise_summary_{model}.txt"
        summary_path.write_text(f"{model.upper()} NOISE FLOOR RESULTS\n"
                                f"{'='*50}\n\n"
                                f"Total baselines sampled:       {summary['total_baselines']}\n"
                                f"Paraphrase variants tested:    {summary['total_paraphrase_variants']}\n"
                                f"Decision flips due to rewording: {summary['decision_flips']}\n"
                                f"\nNOISE FLIP RATE:               {summary['noise_flip_rate_pct']}\n"
                                f"\nInterpretation:\n"
                                f"  If noise ~{flip_rate:.1%}%, then 6-9% modifier flip rate is\n"
                                f"  {max(1, round(6.6/flip_rate)):.1f}x above noise (good signal).\n")

        print(f"\n  Summary:")
        print(f"    Noise flip rate: {summary['noise_flip_rate_pct']}")
        print(f"    (Paper claim: modifier flips are {max(1, 6.6/flip_rate if flip_rate>0 else 1):.1f}x larger)")
        print(f"    Summary → {summary_path}\n")


if __name__ == "__main__":
    main()

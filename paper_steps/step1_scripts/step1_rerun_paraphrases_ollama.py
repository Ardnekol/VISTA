#!/usr/bin/env python3
"""
STEP 1, Part C (Ollama version) — Re-run paraphrases through gemma4 / llama_8B
==============================================================================
Uses local Ollama (http://localhost:11434) instead of vLLM.
For Gemma4 and Llama 3.1 8B running on the local GPU server.

Input:  paper_steps/step1_results/paraphrases_<model>.json
Output: paper_steps/step1_results/noise_results_<model>.json
        paper_steps/step1_results/noise_summary_<model>.txt

Usage:
  # Generate paraphrases first (one-time, if not already done):
  python3 step1_extract_baseline_sample.py --models gemma4 llama_8B
  python3 step1_paraphrase_generator.py    --models gemma4 llama_8B

  # Then re-run them through Ollama models:
  python3 step1_rerun_paraphrases_ollama.py --models gemma4 llama_8B
"""

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
RESULTS_DIR = BASE_DIR / "paper_steps" / "step1_results"

OLLAMA_URL = "http://localhost:11434/api/chat"  # chat endpoint applies model's chat template

# Map model key → Ollama model name
OLLAMA_MODELS = {
    "gemma4":   "gemma4:31b",
    "llama_8B": "llama3.1:8b-instruct-fp16",   # fp16 to preserve marginal A1 logits (Q4_0 collapses them)
}

TEMPERATURE = 0.0   # deterministic — paraphrase noise must be measured against same session, same temp
RETRY_LIMIT = 3
RETRY_DELAY = 2

# ── Per-model Ollama sampling options ────────────────────────────────────────
# Llama 8B already gives 0% paraphrase noise with default settings, so leave
# its options minimal. Gemma4 has high paraphrase noise (8.12%) and needs
# strict greedy decoding to match the SLURM setup (do_sample=False, top_p=0.95).
MODEL_OPTIONS = {
    "llama_8B": {
        "temperature": 0.0,
    },
    "gemma4": {
        "temperature":    0.0,      # deterministic (greedy at temp=0)
        "top_p":          0.95,     # match HF SLURM setup
        "repeat_penalty": 1.0,      # disable default 1.1 repetition penalty
        "seed":           42,       # fixed seed for reproducibility
        # NOTE: num_predict and top_k are intentionally omitted.
        # Gemma4 generates ~1940 internal "reasoning" tokens before producing
        # visible JSON output. Capping with num_predict=512 truncated to 0 chars.
        # top_k=1 caused looping on invisible tokens. Let Ollama use defaults.
    },
}

# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(profile_description: list, scenario: str, a0: str, a1: str) -> str:
    profile_text = "\n".join(f"  - {d}" for d in profile_description)
    return f"""You are simulating the decision-making of a real person with the following value priorities.

PERSON'S VALUE PROFILE:
{profile_text}

SCENARIO:
{scenario}

CHOICES:
  A0: {a0}
  A1: {a1}

Step into this person's perspective completely. Given their specific values (HIGH values they care deeply about, LOW values they care little about), which action would they choose?

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "decision": "A0" or "A1",
  "confidence": "high" or "medium" or "low",
  "driving_values": ["value1", "value2"],
  "reasoning": "one concise sentence explaining which values led to this choice"
}}"""


# ─── JSON parsing ────────────────────────────────────────────────────────────

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


# ─── Ollama call ──────────────────────────────────────────────────────────────

SYSTEM_MSG = "You are a careful, concise reasoning assistant."


def call_ollama(prompt: str, ollama_model: str, options: dict = None) -> dict:
    # Use /api/chat with explicit messages so Ollama applies the model's chat
    # template (Llama-3 / Gemma-2). Bypassing the template (as /api/generate
    # does) causes Llama 3.1 8B to collapse to constant A0 output.
    payload = json.dumps({
        "model":    ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        "stream":   False,
        "options":  options if options else {"temperature": TEMPERATURE},
    }).encode("utf-8")

    for attempt in range(RETRY_LIMIT):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode()).get("message", {}).get("content", "").strip()
            parsed = extract_json(raw)
            dec = str(parsed.get("decision", "ERROR")).strip().upper()
            if dec not in ("A0", "A1"):
                dec = "ERROR"
            return {
                "decision":   dec,
                "confidence": str(parsed.get("confidence", "low")).strip().lower(),
                "reasoning":  str(parsed.get("reasoning", "")).strip(),
            }
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"decision": "ERROR", "confidence": "low", "reasoning": f"Ollama call failed: {e}"}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gemma4", "llama_8B"])
    args = parser.parse_args()

    print("Loading Schwartz profiles...")
    profile_data = json.loads((DATASET_DIR / "profile_description.json").read_text())
    profiles = {p["vsw_id"]: p["description"] for p in profile_data}

    for model in args.models:
        if model not in OLLAMA_MODELS:
            print(f"  ⚠️  Unknown model '{model}'. Skipping.")
            continue

        paraphrase_path = RESULTS_DIR / f"paraphrases_{model}.json"
        if not paraphrase_path.exists():
            print(f"  ⚠️  {paraphrase_path} not found. Run step1_paraphrase_generator.py --models {model}")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model}  →  Ollama: {OLLAMA_MODELS[model]}")
        print(f"{'='*60}")

        paraphrases = json.loads(paraphrase_path.read_text())
        print(f"  Loaded {len(paraphrases)} paraphrase variants")

        # Resume support
        results_path = RESULTS_DIR / f"noise_results_{model}.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            seen_keys = {(r["vsw_id"], r["scenario_id"], r["variant"]) for r in results}
            print(f"  Resuming from {len(results)} already-computed rows")
        else:
            results = []
            seen_keys = set()

        # Track variant 0's decision per (vsw_id, scenario_id) — this is the
        # in-session reference that variants 1-4 are compared against.
        # NOTE: We do NOT compare to para["baseline_decision"] because that was
        # generated in a different session at a different temperature, which
        # confounds paraphrase noise with sampling noise.
        v0_decisions = {(r["vsw_id"], r["scenario_id"]): r["paraphrase_decision"]
                        for r in results if r.get("is_original")}

        # Sort so variant 0 is processed before variants 1-4 within each (vsw, scenario).
        # Critical: variants 1-4 need variant 0's decision as their comparison anchor.
        paraphrases.sort(key=lambda p: (p["vsw_id"], p["scenario_id"], p["variant"]))

        ollama_model    = OLLAMA_MODELS[model]
        ollama_options  = MODEL_OPTIONS.get(model, {"temperature": TEMPERATURE})
        print(f"  Ollama options: {ollama_options}")
        total = len(paraphrases)
        n_processed = 0
        t_start = time.time()

        for i, para in enumerate(paraphrases):
            key = (para["vsw_id"], para["scenario_id"], para["variant"])
            if key in seen_keys:
                continue

            elapsed = time.time() - t_start
            avg = elapsed / max(1, n_processed)
            eta_min = ((total - i) * avg) / 60 if n_processed else 0
            print(f"  [{i+1}/{total}] {para['vsw_id']} {para['scenario_id']} v{para['variant']} "
                  f"(elapsed {elapsed/60:.1f}m, ETA {eta_min:.0f}m)", flush=True)

            profile_desc = profiles.get(para["vsw_id"], [])
            prompt = build_prompt(profile_desc, para["scenario_description"],
                                  para["A0_text"], para["A1_text"])
            result = call_ollama(prompt, ollama_model, ollama_options)

            # ── Within-session comparison ────────────────────────────────────
            # Variant 0 = original. Store its CURRENT-SESSION decision as the
            # reference and compare variants 1-4 against THAT (not the old
            # baseline_decision from master CSV, which is from a different
            # session/temperature and would conflate noise sources).
            ref_key = (para["vsw_id"], para["scenario_id"])
            if para["is_original"]:
                v0_decisions[ref_key] = result["decision"]
                decision_match = True   # variant 0 is its own reference
                reference_used = result["decision"]
            else:
                reference_used = v0_decisions.get(ref_key)
                decision_match = (reference_used is not None
                                  and result["decision"] == reference_used)

            results.append({
                "vsw_id":                para["vsw_id"],
                "scenario_id":           para["scenario_id"],
                "variant":               para["variant"],
                "is_original":           para["is_original"],
                "old_baseline_decision": para["baseline_decision"],   # kept for reference
                "v0_reference_decision": reference_used,              # the actual comparison anchor
                "paraphrase_decision":   result["decision"],
                "decision_match":        decision_match,
                "paraphrase_confidence": result["confidence"],
                "paraphrase_reasoning":  result["reasoning"],
            })
            n_processed += 1

            # Save every 50 rows
            if n_processed % 5 == 0:
                results_path.write_text(json.dumps(results, indent=2))

        # Final save
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved {len(results)} rows → {results_path}")

        # ── Compute noise floor (WITHIN-SESSION method) ─────────────────────
        # Sanity check: variant 0 should always match itself (0% flip on originals)
        originals    = [r for r in results if r["is_original"]]
        paraphrases_ = [r for r in results if not r["is_original"]]

        orig_flips    = [r for r in originals if not r["decision_match"]]   # should be 0
        para_flips    = [r for r in paraphrases_ if not r["decision_match"]]
        para_flip_rate = len(para_flips) / len(paraphrases_) if paraphrases_ else 0.0

        ratio_line = (f"  Modifier flip rate (~6-9%) is {(6.6/(100*para_flip_rate)):.1f}x above noise.\n"
                      if para_flip_rate > 0 else
                      f"  Noise is 0% — perfect robustness to rewording.\n")

        summary_path = RESULTS_DIR / f"noise_summary_{model}.txt"
        summary_path.write_text(
            f"{model.upper()} NOISE FLOOR RESULTS (within-session, temp=0.0)\n"
            f"{'='*60}\n\n"
            f"Total baselines sampled:         {len(set(r['vsw_id'] for r in results))}\n"
            f"Original variants (variant 0):   {len(originals)}\n"
            f"Paraphrased variants (1-4):      {len(paraphrases_)}\n\n"
            f"Variant-0 self-match flips:      {len(orig_flips)}  (sanity check — should be 0)\n"
            f"Paraphrase flips (variant 0 vs 1-4): {len(para_flips)}\n\n"
            f"PARAPHRASE NOISE FLIP RATE:      {100*para_flip_rate:.2f}%\n\n"
            f"Interpretation:\n{ratio_line}"
        )

        print(f"\n  Variant-0 self-match flips: {len(orig_flips)} (should be 0)")
        print(f"  Paraphrase noise flip rate: {100*para_flip_rate:.2f}%")
        print(f"  Summary → {summary_path}\n")


if __name__ == "__main__":
    main()

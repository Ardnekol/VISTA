#!/usr/bin/env python3
"""
VISTA decision analysis — Llama-3.1-70B-Instruct (AWQ-INT4) on SLURM via vLLM.

Same logic and output schema as llm_decision_analysis.py and ..._qwen.py.
Only differences: model name, AWQ quantization, Llama-3 chat template.

Usage:
  python3 llm_decision_analysis_llama.py \
      --profiles Dataset/profile_description_1to10.json \
      --suffix llama
"""

import json
import csv
import re
import os
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "Dataset"
SCENARIOS_FILE = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE = DATASET_DIR / "modifiers_batch1.json"

# Override with $LLAMA_MODEL if you want a different Llama (e.g. 3.3 70B).
MODEL_NAME = os.environ.get(
    "LLAMA_MODEL",
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
)
QUANTIZATION         = os.environ.get("LLAMA_QUANT", "awq")  # set to "" for FP16
TENSOR_PARALLEL_SIZE = int(os.environ.get("TP_SIZE", "4"))
GPU_MEM_UTIL         = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MAX_MODEL_LEN        = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TEMPERATURE          = 0.3
MAX_NEW_TOKENS       = 256

SCENARIO_VALUE_MAP = {
    "SC001_1": {"A0": ["Benevolence", "Universalism"], "A1": ["Security", "Conformity"]},
    "SC001_2": {"A0": ["Benevolence", "Universalism"], "A1": ["Security", "Conformity"]},
    "SC002_1": {"A0": ["Self-Direction", "Stimulation"], "A1": ["Tradition", "Conformity"]},
    "SC002_2": {"A0": ["Self-Direction", "Stimulation"], "A1": ["Tradition", "Conformity"]},
    "SC003_1": {"A0": ["Benevolence", "Universalism"], "A1": ["Achievement", "Power"]},
    "SC003_2": {"A0": ["Achievement", "Power"],         "A1": ["Benevolence", "Universalism"]},
    "SC004_1": {"A0": ["Hedonism", "Stimulation"],      "A1": ["Conformity", "Tradition"]},
    "SC004_2": {"A0": ["Hedonism", "Stimulation"],      "A1": ["Conformity", "Tradition"]},
    "SC005_1": {"A0": ["Tradition", "Security"],        "A1": ["Power", "Achievement"]},
    "SC005_2": {"A0": ["Tradition", "Security"],        "A1": ["Power", "Achievement"]},
}

# ─── vLLM is loaded once per process ─────────────────────────────────────────

_LLM = None
_SAMPLING = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.9,
    max_tokens=MAX_NEW_TOKENS,
)


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        print(f"[init] Loading {MODEL_NAME} on {TENSOR_PARALLEL_SIZE} GPUs "
              f"(quantization={QUANTIZATION or 'none'}) ...", flush=True)
        kwargs = dict(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            dtype="float16",
            trust_remote_code=True,
        )
        if QUANTIZATION:
            kwargs["quantization"] = QUANTIZATION
        _LLM = LLM(**kwargs)
        print("[init] Model loaded.", flush=True)
    return _LLM


# ─── Prompt building (Llama-3 chat template) ─────────────────────────────────

def build_prompt(profile_description, scenario, a0, a1, modifier_text=None):
    profile_text = "\n".join(f"  - {d}" for d in profile_description)
    modifier_section = (
        f"\n\nSituational context (this is the environment right now):\n  {modifier_text}"
        if modifier_text else ""
    )
    user = f"""You are simulating the decision-making of a real person with the following value priorities.

PERSON'S VALUE PROFILE:
{profile_text}
{modifier_section}

SCENARIO:
{scenario}

CHOICES:
  A0: {a0}
  A1: {a1}

Step into this person's perspective completely. Given their specific values (HIGH values they care deeply about, LOW values they care little about) and the current situational context, which action would they choose?

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "decision": "A0" or "A1",
  "confidence": "high" or "medium" or "low",
  "driving_values": ["value1", "value2"],
  "reasoning": "one concise sentence explaining which values led to this choice"
}}"""

    system_msg = "You are a careful, concise reasoning assistant."
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ─── LLM call ────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw):
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


def call_llm(prompt: str) -> dict:
    llm = get_llm()
    try:
        outputs = llm.generate([prompt], _SAMPLING, use_tqdm=False)
        raw = outputs[0].outputs[0].text
        parsed = _extract_json(raw)
        dec = str(parsed.get("decision", "ERROR")).strip().upper()
        if dec not in ("A0", "A1"):
            dec = "ERROR"
        return {
            "decision": dec,
            "confidence": str(parsed.get("confidence", "low")).strip().lower(),
            "driving_values": parsed.get("driving_values", []) or [],
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }
    except Exception as e:
        return {
            "decision": "ERROR",
            "confidence": "low",
            "driving_values": [],
            "reasoning": f"vLLM call/parse failed: {e}",
        }


# ─── Dot-product helpers (unchanged) ─────────────────────────────────────────

def parse_profile_weights(description_list):
    profile = {}
    for desc in description_list:
        value = desc[:desc.index(":")].strip()
        profile[value] = 1.0 if "HIGH" in desc else 0.0
    return profile


def dotproduct_decide(profile, scenario_id, boosted=None):
    p = boosted if boosted else profile
    mapping = SCENARIO_VALUE_MAP[scenario_id]
    s_a0 = sum(p.get(v, 0.0) for v in mapping["A0"])
    s_a1 = sum(p.get(v, 0.0) for v in mapping["A1"])
    if s_a0 > s_a1:   dec = "A0"
    elif s_a1 > s_a0: dec = "A1"
    else:              dec = "TIE"
    return dec, round(s_a0, 3), round(s_a1, 3)


def apply_boost(profile, pressured):
    boosted = dict(profile)
    for v in pressured:
        if v in boosted:
            boosted[v] = min(1.0, boosted[v] + 0.5)
    return boosted


def profile_summary(profile):
    highs = [v for v, w in profile.items() if w >= 1.0]
    return ", ".join(highs) if highs else "ALL-LOW"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path,
                        default=DATASET_DIR / "profile_description_1to10.json")
    parser.add_argument("--suffix", type=str, default="llama")
    args = parser.parse_args()

    profiles_path = args.profiles
    stem   = profiles_path.stem.replace("profile_description_", "")
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = BASE_DIR / "outputs" / f"llm_decision_analysis_{stem}{suffix}.csv"

    profiles     = json.loads(profiles_path.read_text())
    scenarios    = json.loads(SCENARIOS_FILE.read_text())
    modifiers_db = json.loads(MODIFIERS_FILE.read_text())

    modifier_lookup = {item["scenario_id"]: item.get("modifiers", [])
                       for item in modifiers_db}

    total_calls = sum(
        1 + len(modifier_lookup.get(s["scenario_id"], []))
        for _ in profiles
        for s in scenarios
    )
    print(f"\nProfiles: {len(profiles)}  |  Scenarios: {len(scenarios)}")
    print(f"Total LLM calls: {total_calls}  (model: {MODEL_NAME})\n", flush=True)

    get_llm()

    fieldnames = [
        "vsw_id", "profile_HIGH_values",
        "scenario_id", "theme_id", "scenario_brief",
        "A0_text", "A1_text",
        "condition", "modifier_text", "pressured_values",
        "llm_decision", "llm_confidence", "llm_driving_values", "llm_reasoning",
        "dp_score_A0", "dp_score_A1", "dp_decision",
        "llm_dp_agree",
        "llm_changed_from_baseline", "dp_changed_from_baseline",
        "llm_change_explanation",
    ]

    rows = []
    call_n = 0
    flip_llm = flip_dp = agree_count = 0

    for profile_obj in profiles:
        vsw_id  = profile_obj["vsw_id"]
        desc    = profile_obj["description"]
        weights = parse_profile_weights(desc)
        p_sum   = profile_summary(weights)

        for scenario in scenarios:
            sid   = scenario["scenario_id"]
            theme = scenario.get("theme_id", "")
            brief = scenario["description"][:80].rstrip() + "…"

            call_n += 1
            print(f"[{call_n:>4}/{total_calls}] {vsw_id} x {sid} x BASELINE",
                  end=" ... ", flush=True)

            prompt = build_prompt(desc, scenario["description"], scenario["A0"], scenario["A1"])
            llm    = call_llm(prompt)
            dp_dec, dp_a0, dp_a1 = dotproduct_decide(weights, sid)
            base_llm_dec = llm["decision"]
            base_dp_dec  = dp_dec
            agree = "YES" if base_llm_dec == base_dp_dec else "NO"
            if agree == "YES":
                agree_count += 1
            print(f"LLM={base_llm_dec}  DP={base_dp_dec}  agree={agree}", flush=True)

            rows.append({
                "vsw_id": vsw_id, "profile_HIGH_values": p_sum,
                "scenario_id": sid, "theme_id": theme, "scenario_brief": brief,
                "A0_text": scenario["A0"], "A1_text": scenario["A1"],
                "condition": "BASELINE", "modifier_text": "", "pressured_values": "",
                "llm_decision": base_llm_dec,
                "llm_confidence": llm["confidence"],
                "llm_driving_values": "; ".join(llm.get("driving_values", [])),
                "llm_reasoning": llm["reasoning"],
                "dp_score_A0": dp_a0, "dp_score_A1": dp_a1, "dp_decision": dp_dec,
                "llm_dp_agree": agree,
                "llm_changed_from_baseline": "",
                "dp_changed_from_baseline": "",
                "llm_change_explanation": "",
            })

            for mod in modifier_lookup.get(sid, []):
                call_n += 1
                pressured = mod["expected_value_pressure"]
                print(f"[{call_n:>4}/{total_calls}] {vsw_id} x {sid} x {mod['modifier_id']}",
                      end=" ... ", flush=True)

                mod_prompt = build_prompt(desc, scenario["description"],
                                          scenario["A0"], scenario["A1"],
                                          mod["modifier_text"])
                mod_llm = call_llm(mod_prompt)

                boosted = apply_boost(weights, pressured)
                mod_dp_dec, mod_dp_a0, mod_dp_a1 = dotproduct_decide(weights, sid, boosted)

                llm_changed = "YES" if mod_llm["decision"] != base_llm_dec else "NO"
                dp_changed  = "YES" if mod_dp_dec != base_dp_dec else "NO"
                agree_mod   = "YES" if mod_llm["decision"] == mod_dp_dec else "NO"
                if llm_changed == "YES": flip_llm += 1
                if dp_changed  == "YES": flip_dp  += 1
                if agree_mod   == "YES": agree_count += 1

                llm_explain = ""
                if llm_changed == "YES":
                    llm_explain = (
                        f"[{base_llm_dec}→{mod_llm['decision']}] "
                        f"Pressured: {', '.join(pressured)} | "
                        f"LLM reason: {mod_llm['reasoning']}"
                    )
                print(f"LLM={mod_llm['decision']}  DP={mod_dp_dec}  "
                      f"llm_flip={llm_changed}  dp_flip={dp_changed}  agree={agree_mod}",
                      flush=True)

                rows.append({
                    "vsw_id": vsw_id, "profile_HIGH_values": p_sum,
                    "scenario_id": sid, "theme_id": theme, "scenario_brief": brief,
                    "A0_text": scenario["A0"], "A1_text": scenario["A1"],
                    "condition": mod["modifier_id"],
                    "modifier_text": mod["modifier_text"],
                    "pressured_values": " + ".join(pressured),
                    "llm_decision": mod_llm["decision"],
                    "llm_confidence": mod_llm["confidence"],
                    "llm_driving_values": "; ".join(mod_llm.get("driving_values", [])),
                    "llm_reasoning": mod_llm["reasoning"],
                    "dp_score_A0": mod_dp_a0, "dp_score_A1": mod_dp_a1, "dp_decision": mod_dp_dec,
                    "llm_dp_agree": agree_mod,
                    "llm_changed_from_baseline": llm_changed,
                    "dp_changed_from_baseline": dp_changed,
                    "llm_change_explanation": llm_explain,
                })

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    modifier_rows = [r for r in rows if r["condition"] != "BASELINE"]
    total_mod = len(modifier_rows)
    overall_agree = sum(1 for r in rows if r["llm_dp_agree"] == "YES")

    print(f"\n{'='*55}")
    print(f"  LLM decision flips:          {flip_llm} / {total_mod}  ({100*flip_llm/max(1,total_mod):.1f}%)")
    print(f"  Dot-product decision flips:  {flip_dp}  / {total_mod}  ({100*flip_dp/max(1,total_mod):.1f}%)")
    print(f"  LLM ↔ Dot-product agreement: {overall_agree} / {len(rows)}  ({100*overall_agree/len(rows):.1f}%)")
    print(f"\n  Output → {out_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

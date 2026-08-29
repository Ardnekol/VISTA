#!/usr/bin/env python3
"""
Sequential runner: load Qwen2.5-32B-Instruct ONCE, then process all 10 profile
batches one by one. After each batch finishes, its CSV is written to
outputs/llm_decision_analysis_<range>_qwen.csv. Already-finished batches are
skipped, so this script is safe to re-run after a crash or time-out.

Usage:
    python3 run_all_batches.py
    python3 run_all_batches.py --suffix qwen --start 31to40
    python3 run_all_batches.py --only 1to10,11to20

GPU selection: respects $CUDA_VISIBLE_DEVICES if set (use pick_gpus.sh).
Tensor-parallel size = number of visible GPUs unless $TP_SIZE overrides it.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "Dataset"
OUTPUT_DIR     = BASE_DIR / "outputs"
SCENARIOS_FILE = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE = DATASET_DIR / "modifiers_batch1.json"

MODEL_NAME     = "Qwen/Qwen2.5-32B-Instruct"
TEMPERATURE    = 0.3
MAX_NEW_TOKENS = 256
GPU_MEM_UTIL   = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "4096"))

ALL_BATCHES = [
    "1to10", "11to20", "21to30", "31to40", "41to50",
    "51to60", "61to70", "71to80", "81to90", "91to100",
]

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

FIELDNAMES = [
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

# ─── Prompt + parsing ────────────────────────────────────────────────────────

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
    return (
        "<|im_start|>system\nYou are a careful, concise reasoning assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw):
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


def normalize_llm(parsed):
    dec = str(parsed.get("decision", "ERROR")).strip().upper()
    if dec not in ("A0", "A1"):
        dec = "ERROR"
    return {
        "decision": dec,
        "confidence": str(parsed.get("confidence", "low")).strip().lower(),
        "driving_values": parsed.get("driving_values", []) or [],
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


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


# ─── Process one batch ───────────────────────────────────────────────────────

def process_batch(batch_name, llm, sampling, scenarios, modifier_lookup, suffix):
    profiles_path = DATASET_DIR / f"profile_description_{batch_name}.json"
    out_path      = OUTPUT_DIR / f"llm_decision_analysis_{batch_name}_{suffix}.csv"

    if out_path.exists():
        print(f"\n[skip] {out_path.name} already exists.", flush=True)
        return out_path

    if not profiles_path.exists():
        print(f"\n[warn] {profiles_path} not found — skipping.", flush=True)
        return None

    profiles = json.loads(profiles_path.read_text())
    total_calls = sum(
        1 + len(modifier_lookup.get(s["scenario_id"], []))
        for _ in profiles for s in scenarios
    )
    t_start = time.time()
    print(f"\n{'='*60}\n  Batch {batch_name}  |  profiles={len(profiles)}  "
          f"|  total calls={total_calls}\n{'='*60}", flush=True)

    rows = []
    call_n = 0
    flip_llm = flip_dp = agree_count = 0

    def call(prompt):
        outs = llm.generate([prompt], sampling, use_tqdm=False)
        raw = outs[0].outputs[0].text
        try:
            return normalize_llm(extract_json(raw))
        except Exception as e:
            return {
                "decision": "ERROR", "confidence": "low",
                "driving_values": [], "reasoning": f"parse failed: {e}",
            }

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

            llm_out = call(build_prompt(desc, scenario["description"],
                                        scenario["A0"], scenario["A1"]))
            dp_dec, dp_a0, dp_a1 = dotproduct_decide(weights, sid)
            base_llm_dec = llm_out["decision"]
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
                "llm_confidence": llm_out["confidence"],
                "llm_driving_values": "; ".join(llm_out.get("driving_values", [])),
                "llm_reasoning": llm_out["reasoning"],
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

                mod_out = call(build_prompt(desc, scenario["description"],
                                            scenario["A0"], scenario["A1"],
                                            mod["modifier_text"]))
                boosted = apply_boost(weights, pressured)
                mod_dp_dec, mod_dp_a0, mod_dp_a1 = dotproduct_decide(weights, sid, boosted)

                llm_changed = "YES" if mod_out["decision"] != base_llm_dec else "NO"
                dp_changed  = "YES" if mod_dp_dec != base_dp_dec else "NO"
                agree_mod   = "YES" if mod_out["decision"] == mod_dp_dec else "NO"
                if llm_changed == "YES": flip_llm += 1
                if dp_changed  == "YES": flip_dp  += 1
                if agree_mod   == "YES": agree_count += 1

                llm_explain = ""
                if llm_changed == "YES":
                    llm_explain = (
                        f"[{base_llm_dec}→{mod_out['decision']}] "
                        f"Pressured: {', '.join(pressured)} | "
                        f"LLM reason: {mod_out['reasoning']}"
                    )
                print(f"LLM={mod_out['decision']}  DP={mod_dp_dec}  "
                      f"llm_flip={llm_changed}  dp_flip={dp_changed}  agree={agree_mod}",
                      flush=True)

                rows.append({
                    "vsw_id": vsw_id, "profile_HIGH_values": p_sum,
                    "scenario_id": sid, "theme_id": theme, "scenario_brief": brief,
                    "A0_text": scenario["A0"], "A1_text": scenario["A1"],
                    "condition": mod["modifier_id"],
                    "modifier_text": mod["modifier_text"],
                    "pressured_values": " + ".join(pressured),
                    "llm_decision": mod_out["decision"],
                    "llm_confidence": mod_out["confidence"],
                    "llm_driving_values": "; ".join(mod_out.get("driving_values", [])),
                    "llm_reasoning": mod_out["reasoning"],
                    "dp_score_A0": mod_dp_a0, "dp_score_A1": mod_dp_a1,
                    "dp_decision": mod_dp_dec,
                    "llm_dp_agree": agree_mod,
                    "llm_changed_from_baseline": llm_changed,
                    "dp_changed_from_baseline": dp_changed,
                    "llm_change_explanation": llm_explain,
                })

    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp_path = out_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(out_path)  # atomic — partial files never appear final

    elapsed = time.time() - t_start
    mods = [r for r in rows if r["condition"] != "BASELINE"]
    print(f"\n  ✓ Batch {batch_name} done in {elapsed/60:.1f} min")
    print(f"    LLM flips: {flip_llm}/{len(mods)}  "
          f"DP flips: {flip_dp}/{len(mods)}  "
          f"LLM↔DP agree: {agree_count}/{len(rows)}")
    print(f"    Saved → {out_path}\n", flush=True)
    return out_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="qwen")
    parser.add_argument("--start", default=None,
                        help="Batch to start from, e.g. 31to40 (skips earlier ones).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated batch names, e.g. 1to10,21to30")
    args = parser.parse_args()

    # Decide which batches to run
    if args.only:
        batches = [b.strip() for b in args.only.split(",") if b.strip()]
    else:
        batches = list(ALL_BATCHES)
        if args.start:
            if args.start not in batches:
                raise SystemExit(f"Unknown --start batch: {args.start}")
            batches = batches[batches.index(args.start):]

    # Tensor parallel = number of visible GPUs (after CUDA_VISIBLE_DEVICES)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "TP_SIZE" in os.environ:
        tp = int(os.environ["TP_SIZE"])
    else:
        tp = len([x for x in visible.split(",") if x.strip()]) if visible else 4

    print(f"\nCUDA_VISIBLE_DEVICES = {visible or '(unset)'}")
    print(f"tensor_parallel_size = {tp}")
    print(f"Model                = {MODEL_NAME}")
    print(f"Batches to run       = {batches}\n", flush=True)

    # Import vLLM here so missing CUDA fails fast with a clear message above
    from vllm import LLM, SamplingParams

    print("[init] Loading model (this can take 5–10 minutes) ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        dtype="float16",
        trust_remote_code=True,
    )
    print(f"[init] Loaded in {(time.time()-t0)/60:.1f} min\n", flush=True)

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    scenarios       = json.loads(SCENARIOS_FILE.read_text())
    modifiers_db    = json.loads(MODIFIERS_FILE.read_text())
    modifier_lookup = {item["scenario_id"]: item.get("modifiers", [])
                       for item in modifiers_db}

    overall_t0 = time.time()
    for b in batches:
        process_batch(b, llm, sampling, scenarios, modifier_lookup, args.suffix)

    print(f"\nALL DONE in {(time.time()-overall_t0)/60:.1f} min")
    print(f"Outputs in: {OUTPUT_DIR}/llm_decision_analysis_<range>_{args.suffix}.csv")


if __name__ == "__main__":
    main()

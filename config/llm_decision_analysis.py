#!/usr/bin/env python3
"""
LLM-Based Value Decision Analysis with Situational Modifiers
=============================================================
Replaces the dot-product rule with an actual LLM that roleplays
as each person and reasons about A0/A1 given their value profile.

Each call sends:
  - The person's full natural-language value profile
  - The scenario description
  - Optionally: the modifier (situational context)
  - The two choices A0 and A1

The LLM outputs: decision, confidence, driving_values, reasoning

Also keeps dot-product scores for comparison (agreement analysis).

Usage:
  python3 llm_decision_analysis.py [--profiles Dataset/profile_description_1to10.json]
"""

import json
import csv
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "Dataset"
SCENARIOS_FILE = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE = DATASET_DIR / "modifiers_batch1.json"

OLLAMA_URL  = "http://localhost:11434/api/generate"
LLM_MODEL   = "gemma4:31b"
TEMPERATURE = 0.3   # lower = more deterministic decisions
RETRY_LIMIT = 3
RETRY_DELAY = 2

# Dot-product fallback (same mapping as value_decision_analysis.py)
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

# ─── LLM helpers ─────────────────────────────────────────────────────────────

def build_prompt(profile_description: list, scenario: str, a0: str, a1: str,
                 modifier_text: str = None) -> str:
    profile_text = "\n".join(f"  - {d}" for d in profile_description)
    modifier_section = (
        f"\n\nSituational context (this is the environment right now):\n  {modifier_text}"
        if modifier_text else ""
    )
    return f"""You are simulating the decision-making of a real person with the following value priorities.

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


def call_llm(prompt: str) -> dict:
    """Call local Ollama/Gemma4 and parse JSON response. Retries on failure."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }).encode("utf-8")

    for attempt in range(RETRY_LIMIT):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode()).get("response", "").strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "decision": "ERROR",
                    "confidence": "low",
                    "driving_values": [],
                    "reasoning": f"Ollama call failed: {e}",
                }


# ─── Dot-product helpers (kept for comparison) ───────────────────────────────

def parse_profile_weights(description_list: list) -> dict:
    profile = {}
    for desc in description_list:
        value = desc[:desc.index(":")].strip()
        profile[value] = 1.0 if "HIGH" in desc else 0.0
    return profile


def dotproduct_decide(profile: dict, scenario_id: str, boosted: dict = None) -> tuple:
    p = boosted if boosted else profile
    mapping = SCENARIO_VALUE_MAP[scenario_id]
    s_a0 = sum(p.get(v, 0.0) for v in mapping["A0"])
    s_a1 = sum(p.get(v, 0.0) for v in mapping["A1"])
    if s_a0 > s_a1:   dec = "A0"
    elif s_a1 > s_a0: dec = "A1"
    else:              dec = "TIE"
    return dec, round(s_a0, 3), round(s_a1, 3)


def apply_boost(profile: dict, pressured: list) -> dict:
    boosted = dict(profile)
    for v in pressured:
        if v in boosted:
            boosted[v] = min(1.0, boosted[v] + 0.5)
    return boosted


def profile_summary(profile: dict) -> str:
    highs = [v for v, w in profile.items() if w >= 1.0]
    return ", ".join(highs) if highs else "ALL-LOW"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path,
                        default=DATASET_DIR / "profile_description_1to10.json")
    parser.add_argument("--suffix", type=str, default="",
                        help="Model tag appended to filename, e.g. gemma4 or llama")
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
    print(f"Total LLM calls: {total_calls}  (model: {LLM_MODEL})\n")

    fieldnames = [
        "vsw_id", "profile_HIGH_values",
        "scenario_id", "theme_id", "scenario_brief",
        "A0_text", "A1_text",
        "condition", "modifier_text", "pressured_values",
        # LLM result
        "llm_decision", "llm_confidence", "llm_driving_values", "llm_reasoning",
        # Dot-product result (for comparison)
        "dp_score_A0", "dp_score_A1", "dp_decision",
        # Agreement and change flags
        "llm_dp_agree",
        "llm_changed_from_baseline", "dp_changed_from_baseline",
        "llm_change_explanation",
    ]

    rows = []
    call_n = 0
    flip_llm = 0
    flip_dp  = 0
    agree_count = 0

    for profile_obj in profiles:
        vsw_id  = profile_obj["vsw_id"]
        desc    = profile_obj["description"]
        weights = parse_profile_weights(desc)
        p_sum   = profile_summary(weights)

        for scenario in scenarios:
            sid   = scenario["scenario_id"]
            theme = scenario.get("theme_id", "")
            brief = scenario["description"][:80].rstrip() + "…"

            # ── BASELINE ─────────────────────────────────────────────────────
            call_n += 1
            print(f"[{call_n:>4}/{total_calls}] {vsw_id} x {sid} x BASELINE", end=" ... ", flush=True)

            prompt = build_prompt(desc, scenario["description"], scenario["A0"], scenario["A1"])
            llm    = call_llm(prompt)
            dp_dec, dp_a0, dp_a1 = dotproduct_decide(weights, sid)

            base_llm_dec = llm["decision"]
            base_dp_dec  = dp_dec

            agree = "YES" if base_llm_dec == base_dp_dec else "NO"
            if agree == "YES":
                agree_count += 1

            print(f"LLM={base_llm_dec}  DP={base_dp_dec}  agree={agree}")

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

            # ── MODIFIERS ─────────────────────────────────────────────────────
            for mod in modifier_lookup.get(sid, []):
                call_n += 1
                pressured = mod["expected_value_pressure"]
                print(f"[{call_n:>4}/{total_calls}] {vsw_id} x {sid} x {mod['modifier_id']}", end=" ... ", flush=True)

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
                      f"llm_flip={llm_changed}  dp_flip={dp_changed}  agree={agree_mod}")

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

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    modifier_rows = [r for r in rows if r["condition"] != "BASELINE"]
    total_mod = len(modifier_rows)
    overall_agree = sum(1 for r in rows if r["llm_dp_agree"] == "YES")

    print(f"\n{'='*55}")
    print(f"  LLM decision flips:          {flip_llm} / {total_mod}  ({100*flip_llm/total_mod:.1f}%)")
    print(f"  Dot-product decision flips:  {flip_dp}  / {total_mod}  ({100*flip_dp/total_mod:.1f}%)")
    print(f"  LLM ↔ Dot-product agreement: {overall_agree} / {len(rows)}  ({100*overall_agree/len(rows):.1f}%)")
    print(f"\n  Output → {out_path}")
    print(f"{'='*55}\n")

    # Notable cases where LLM and DP disagree on a flip
    diverge = [r for r in modifier_rows
               if r["llm_changed_from_baseline"] != r["dp_changed_from_baseline"]]
    if diverge:
        print(f"Cases where LLM and dot-product DISAGREE on whether modifier caused a flip ({len(diverge)}):")
        print(f"  {'Profile':<12} {'Scenario':<10} {'Modifier':<22} {'LLM flip':<10} {'DP flip':<10} {'LLM reason'}")
        print("  " + "-" * 100)
        for r in diverge[:12]:
            reason_short = r["llm_reasoning"][:60] if r["llm_reasoning"] else ""
            print(f"  {r['vsw_id']:<12} {r['scenario_id']:<10} {r['condition']:<22} "
                  f"{r['llm_changed_from_baseline']:<10} {r['dp_changed_from_baseline']:<10} {reason_short}")


if __name__ == "__main__":
    main()

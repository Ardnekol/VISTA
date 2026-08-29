#!/usr/bin/env python3
"""
VISDA decision analysis — Claude Haiku via the Message Batches API.
==================================================================
Closed-weight comparison model for the VISDA situational-modifier study.

Mirrors llm_decision_analysis.py exactly:
  - same prompt (build_prompt)
  - same dot-product baseline (parse_profile_weights / dotproduct_decide / apply_boost)
  - same output schema (master_llm_decisions_<model>.csv)

Differences vs the Ollama scripts:
  - Loads ALL 10 profile_description shards (95 profiles) in one run.
  - Sends every (profile x scenario x {baseline + 8 modifiers}) call as one
    request in a single Message Batch (8,550 requests total).
  - Uses structured outputs (JSON schema) so every response is valid JSON —
    no retries, no ERROR rows from markdown drift.
  - temperature=0 (greedy) to match the paper's open-weight decoding.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 run_batch_haiku.py                 # submit + wait + write master CSV
  python3 run_batch_haiku.py --resume        # re-attach to the saved batch id
  python3 run_batch_haiku.py --dry-run       # build requests, print counts, don't submit
"""

import json
import csv
import time
import argparse
from pathlib import Path

import anthropic

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "Dataset"
OUT_DIR        = BASE_DIR / "outputs"
SCENARIOS_FILE = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE = DATASET_DIR / "modifiers_batch1.json"

MODEL       = "claude-haiku-4-5"
TEMPERATURE = 0        # greedy, to match the open-weight runs
MAX_TOKENS  = 512
SUFFIX      = "haiku"

BATCH_ID_FILE = OUT_DIR / f"batch_id_{SUFFIX}.txt"
MASTER_OUT    = OUT_DIR / f"master_llm_decisions_{SUFFIX}.csv"

# Structured-output schema — guarantees valid, parseable JSON for every row.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision":       {"type": "string", "enum": ["A0", "A1"]},
        "confidence":     {"type": "string", "enum": ["high", "medium", "low"]},
        "driving_values": {"type": "array", "items": {"type": "string"}},
        "reasoning":      {"type": "string"},
    },
    "required": ["decision", "confidence", "driving_values", "reasoning"],
    "additionalProperties": False,
}

# Dot-product fallback (identical to llm_decision_analysis.py)
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

# ─── Prompt + dot-product helpers (copied verbatim from llm_decision_analysis.py) ──

def build_prompt(profile_description, scenario, a0, a1, modifier_text=None):
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


# ─── Load dataset ─────────────────────────────────────────────────────────────

def load_profiles():
    """All 10 profile_description shards, sorted by their starting index → 95 profiles."""
    shards = sorted(
        DATASET_DIR.glob("profile_description_*to*.json"),
        key=lambda p: int(p.stem.replace("profile_description_", "").split("to")[0]),
    )
    profiles = []
    for shard in shards:
        profiles.extend(json.loads(shard.read_text()))
    return profiles


def build_units(profiles, scenarios, modifier_lookup):
    """
    Deterministic list of decision units, one per API request.
    Each unit: (custom_id, vsw_id, desc, weights, p_sum, scenario, condition, modifier).
    Order matches llm_decision_analysis.py: profile → scenario → [baseline, mods...].
    """
    units = []
    for prof in profiles:
        vsw_id  = prof["vsw_id"]
        desc    = prof["description"]
        weights = parse_profile_weights(desc)
        p_sum   = profile_summary(weights)
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            # baseline
            units.append(dict(
                cid=f"{vsw_id}__{sid}__BASELINE",
                vsw_id=vsw_id, desc=desc, weights=weights, p_sum=p_sum,
                scenario=scenario, condition="BASELINE", modifier=None,
            ))
            # modifiers
            for mod in modifier_lookup.get(sid, []):
                units.append(dict(
                    cid=f"{vsw_id}__{sid}__{mod['modifier_id']}",
                    vsw_id=vsw_id, desc=desc, weights=weights, p_sum=p_sum,
                    scenario=scenario, condition=mod["modifier_id"], modifier=mod,
                ))
    return units


def unit_to_request(u):
    sc = u["scenario"]
    mod_text = u["modifier"]["modifier_text"] if u["modifier"] else None
    prompt = build_prompt(u["desc"], sc["description"], sc["A0"], sc["A1"], mod_text)
    return {
        "custom_id": u["cid"],
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
        },
    }


# ─── Batch lifecycle ──────────────────────────────────────────────────────────

def submit_batch(client, requests):
    print(f"Submitting {len(requests)} requests to the Message Batches API...")
    batch = client.messages.batches.create(requests=requests)
    OUT_DIR.mkdir(exist_ok=True)
    BATCH_ID_FILE.write_text(batch.id)
    print(f"  Batch ID: {batch.id}  (saved → {BATCH_ID_FILE})")
    return batch.id


def wait_for_batch(client, batch_id, poll=30):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  status={b.processing_status}  "
              f"processing={c.processing} succeeded={c.succeeded} "
              f"errored={c.errored} canceled={c.canceled} expired={c.expired}",
              flush=True)
        if b.processing_status == "ended":
            return b
        time.sleep(poll)


def collect_results(client, batch_id):
    """custom_id → parsed decision dict (or ERROR sentinel)."""
    out = {}
    for res in client.messages.batches.results(batch_id):
        cid = res.custom_id
        if res.result.type == "succeeded":
            text = next((b.text for b in res.result.message.content if b.type == "text"), "")
            text = text.strip()
            if text.startswith("```"):                 # strip accidental markdown fences
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            try:
                out[cid] = json.loads(text)
            except Exception as e:
                out[cid] = {"decision": "ERROR", "confidence": "low",
                            "driving_values": [], "reasoning": f"parse failed: {e}"}
        else:
            rtype = res.result.type
            err = getattr(getattr(res.result, "error", None), "type", rtype)
            out[cid] = {"decision": "ERROR", "confidence": "low",
                        "driving_values": [], "reasoning": f"batch result {rtype}: {err}"}
    return out


# ─── Assemble master CSV (dot-product + flags computed identically to the paper) ──

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
    # extra columns present in the merged masters used by step*.py
    "axis", "profile_strength",
]


def assemble_rows(profiles, scenarios, modifier_lookup, results):
    rows = []
    flip_llm = flip_dp = 0
    n_mod = 0
    for prof in profiles:
        vsw_id  = prof["vsw_id"]
        desc    = prof["description"]
        weights = parse_profile_weights(desc)
        p_sum   = profile_summary(weights)
        p_strength = sum(1 for w in weights.values() if w >= 1.0)

        for scenario in scenarios:
            sid   = scenario["scenario_id"]
            theme = scenario.get("theme_id", "")
            brief = scenario["description"][:80].rstrip() + "…"

            # baseline
            base = results.get(f"{vsw_id}__{sid}__BASELINE",
                               {"decision": "ERROR", "confidence": "low",
                                "driving_values": [], "reasoning": "missing result"})
            base_llm = base["decision"]
            dp_dec, dp_a0, dp_a1 = dotproduct_decide(weights, sid)
            rows.append({
                "vsw_id": vsw_id, "profile_HIGH_values": p_sum,
                "scenario_id": sid, "theme_id": theme, "scenario_brief": brief,
                "A0_text": scenario["A0"], "A1_text": scenario["A1"],
                "condition": "BASELINE", "modifier_text": "", "pressured_values": "",
                "llm_decision": base_llm, "llm_confidence": base["confidence"],
                "llm_driving_values": "; ".join(base.get("driving_values", [])),
                "llm_reasoning": base["reasoning"],
                "dp_score_A0": dp_a0, "dp_score_A1": dp_a1, "dp_decision": dp_dec,
                "llm_dp_agree": "YES" if base_llm == dp_dec else "NO",
                "llm_changed_from_baseline": "", "dp_changed_from_baseline": "",
                "llm_change_explanation": "",
                "axis": "BASELINE", "profile_strength": p_strength,
            })

            # modifiers
            for mod in modifier_lookup.get(sid, []):
                n_mod += 1
                pressured = mod["expected_value_pressure"]
                r = results.get(f"{vsw_id}__{sid}__{mod['modifier_id']}",
                                {"decision": "ERROR", "confidence": "low",
                                 "driving_values": [], "reasoning": "missing result"})
                mod_llm = r["decision"]
                boosted = apply_boost(weights, pressured)
                mod_dp_dec, mod_dp_a0, mod_dp_a1 = dotproduct_decide(weights, sid, boosted)

                llm_changed = "YES" if mod_llm != base_llm else "NO"
                dp_changed  = "YES" if mod_dp_dec != dp_dec else "NO"
                if llm_changed == "YES": flip_llm += 1
                if dp_changed  == "YES": flip_dp  += 1

                explain = ""
                if llm_changed == "YES":
                    explain = (f"[{base_llm}→{mod_llm}] Pressured: {', '.join(pressured)} | "
                               f"LLM reason: {r['reasoning']}")

                rows.append({
                    "vsw_id": vsw_id, "profile_HIGH_values": p_sum,
                    "scenario_id": sid, "theme_id": theme, "scenario_brief": brief,
                    "A0_text": scenario["A0"], "A1_text": scenario["A1"],
                    "condition": mod["modifier_id"], "modifier_text": mod["modifier_text"],
                    "pressured_values": " + ".join(pressured),
                    "llm_decision": mod_llm, "llm_confidence": r["confidence"],
                    "llm_driving_values": "; ".join(r.get("driving_values", [])),
                    "llm_reasoning": r["reasoning"],
                    "dp_score_A0": mod_dp_a0, "dp_score_A1": mod_dp_a1, "dp_decision": mod_dp_dec,
                    "llm_dp_agree": "YES" if mod_llm == mod_dp_dec else "NO",
                    "llm_changed_from_baseline": llm_changed,
                    "dp_changed_from_baseline": dp_changed,
                    "llm_change_explanation": explain,
                    "axis": mod["axis"], "profile_strength": p_strength,
                })
    return rows, flip_llm, flip_dp, n_mod


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Re-attach to the saved batch id instead of submitting a new batch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build requests and print counts, but do not submit.")
    ap.add_argument("--poll", type=int, default=30, help="Poll interval (s).")
    args = ap.parse_args()

    profiles     = load_profiles()
    scenarios    = json.loads(SCENARIOS_FILE.read_text())
    modifiers_db = json.loads(MODIFIERS_FILE.read_text())
    modifier_lookup = {m["scenario_id"]: m.get("modifiers", []) for m in modifiers_db}

    units    = build_units(profiles, scenarios, modifier_lookup)
    requests = [unit_to_request(u) for u in units]
    print(f"Profiles: {len(profiles)}  Scenarios: {len(scenarios)}  "
          f"Requests: {len(requests)}  (model: {MODEL})")

    if args.dry_run:
        print("\n--dry-run: sample request custom_ids:")
        for u in units[:3] + units[-1:]:
            print("  ", u["cid"])
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / ant profile

    if args.resume:
        if not BATCH_ID_FILE.exists():
            raise SystemExit(f"--resume but no saved batch id at {BATCH_ID_FILE}")
        batch_id = BATCH_ID_FILE.read_text().strip()
        print(f"Resuming batch {batch_id}")
    else:
        batch_id = submit_batch(client, requests)

    wait_for_batch(client, batch_id, poll=args.poll)
    results = collect_results(client, batch_id)

    n_err = sum(1 for v in results.values() if v["decision"] == "ERROR")
    print(f"\nRetrieved {len(results)} results  ({n_err} errored/unparsed)")

    rows, flip_llm, flip_dp, n_mod = assemble_rows(
        profiles, scenarios, modifier_lookup, results)

    OUT_DIR.mkdir(exist_ok=True)
    with open(MASTER_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*55}")
    print(f"  Rows written:               {len(rows)}  → {MASTER_OUT}")
    print(f"  Modifier trials:            {n_mod}")
    print(f"  LLM decision flips:         {flip_llm} / {n_mod}  ({100*flip_llm/n_mod:.2f}%)")
    print(f"  Dot-product flips:          {flip_dp} / {n_mod}  ({100*flip_dp/n_mod:.2f}%)")
    print(f"{'='*55}\n")
    print("Next: add the master path to step3_mcnemar.py / step6 / compare_models.py")
    print(f'  "Haiku 4.5": OUT_DIR / "outputs" / "master_llm_decisions_{SUFFIX}.csv"')


if __name__ == "__main__":
    main()

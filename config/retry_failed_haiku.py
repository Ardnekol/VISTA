#!/usr/bin/env python3
"""
Retry the handful of ERROR rows in master_llm_decisions_haiku.csv with direct
(non-batch) API calls and patch them in place — recomputing the LLM flip flags.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 retry_failed_haiku.py
"""
import csv, json
import anthropic
import run_batch_haiku as R   # reuse prompt + dataset + dp helpers

MASTER = R.MASTER_OUT

def load_dataset():
    profiles     = R.load_profiles()
    scenarios    = json.loads(R.SCENARIOS_FILE.read_text())
    modifiers_db = json.loads(R.MODIFIERS_FILE.read_text())
    mod_lookup   = {m["scenario_id"]: m.get("modifiers", []) for m in modifiers_db}
    units = {u["cid"]: u for u in R.build_units(profiles, scenarios, mod_lookup)}
    return units

def call(client, unit):
    sc = unit["scenario"]
    mod_text = unit["modifier"]["modifier_text"] if unit["modifier"] else None
    prompt = R.build_prompt(unit["desc"], sc["description"], sc["A0"], sc["A1"], mod_text)
    msg = client.messages.create(
        model=R.MODEL, max_tokens=R.MAX_TOKENS, temperature=R.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": R.DECISION_SCHEMA}},
    )
    text = next((b.text for b in msg.content if b.type == "text"), "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
        text = text.strip()
    return json.loads(text)

def main():
    rows = list(csv.DictReader(open(MASTER)))
    err_idx = [i for i, r in enumerate(rows) if r["llm_decision"] == "ERROR"]
    if not err_idx:
        print("No ERROR rows — nothing to retry.")
        return
    print(f"Retrying {len(err_idx)} ERROR row(s)...")

    units  = load_dataset()
    client = anthropic.Anthropic()
    # index baseline decisions per (vsw_id, scenario_id) for flip recompute
    base_dec = {(r["vsw_id"], r["scenario_id"]): r["llm_decision"]
                for r in rows if r["condition"] == "BASELINE"}

    for i in err_idx:
        r   = rows[i]
        cid = f"{r['vsw_id']}__{r['scenario_id']}__{r['condition']}"
        res = call(client, units[cid])
        r["llm_decision"]       = res["decision"]
        r["llm_confidence"]     = res["confidence"]
        r["llm_driving_values"] = "; ".join(res.get("driving_values", []))
        r["llm_reasoning"]      = res["reasoning"]
        r["llm_dp_agree"]       = "YES" if res["decision"] == r["dp_decision"] else "NO"
        if r["condition"] != "BASELINE":
            base = base_dec.get((r["vsw_id"], r["scenario_id"]), "")
            changed = "YES" if res["decision"] != base else "NO"
            r["llm_changed_from_baseline"] = changed
            r["llm_change_explanation"] = (
                f"[{base}→{res['decision']}] Pressured: {r['pressured_values']} | "
                f"LLM reason: {res['reasoning']}" if changed == "YES" else "")
        print(f"  fixed {cid} -> {res['decision']}")

    with open(MASTER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.FIELDNAMES)
        w.writeheader(); w.writerows(rows)

    mod   = [r for r in rows if r["condition"] != "BASELINE"]
    flips = sum(1 for r in mod if r["llm_changed_from_baseline"] == "YES")
    left  = sum(1 for r in rows if r["llm_decision"] == "ERROR")
    print(f"\nPatched {MASTER}")
    print(f"  remaining ERROR rows: {left}")
    print(f"  LLM flips: {flips}/{len(mod)} ({100*flips/len(mod):.2f}%)")

if __name__ == "__main__":
    main()

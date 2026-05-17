#!/usr/bin/env python3
"""
Value-Based Decision Analysis with Situational Modifiers
=========================================================
Experiment:
  - Independent variable 1: Value profile (who the person is)
  - Independent variable 2: Modifier (what the situation adds)
  - Dependent variable:     Decision (A0 or A1)
  - Key question:           Does the modifier flip the decision, and why?

Usage:
  python3 value_decision_analysis.py
Output:
  outputs/decision_analysis.csv
"""

import json
import csv
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_PATH = BASE_DIR / "outputs" / "decision_analysis.csv"

_DEFAULT_PROFILES = DATASET_DIR / "profile_description_1to10.json"
SCENARIOS_FILE    = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE    = DATASET_DIR / "modifiers_batch1.json"

# ─── Model parameters ────────────────────────────────────────────────────────

# How much a modifier amplifies a LOW value (0.0 → 0.5)
# A HIGH value (1.0) stays capped at 1.0
MODIFIER_BOOST = 0.5

# Profile encoding
HIGH_WEIGHT = 1.0
LOW_WEIGHT  = 0.0

# ─── Value-to-Action mapping ─────────────────────────────────────────────────
# For each scenario: which Schwartz values align with A0 vs A1.
# Derived from semantic analysis of latent_value_tensions + action descriptions.
#
# General principle:
#   A0 tends toward openness/growth/universalism  (first action listed)
#   A1 tends toward conservation/security/local   (second action listed)
# Exceptions (SC003_1, SC003_2, SC005_x) are noted in comments.

SCENARIO_VALUE_MAP = {
    # "Fund after-school program" → caring for community (Benevolence, Universalism)
    # "Upgrade security systems"  → safety, order     (Security, Conformity)
    "SC001_1": {"A0": ["Benevolence", "Universalism"], "A1": ["Security", "Conformity"]},

    # "Support citywide food pantry" → wider welfare   (Benevolence, Universalism)
    # "Maintain neighborhood watch"  → local safety    (Security, Conformity)
    "SC001_2": {"A0": ["Benevolence", "Universalism"], "A1": ["Security", "Conformity"]},

    # "New digital workflow" → freedom, novelty        (Self-Direction, Stimulation)
    # "Paper-based process"  → tradition, norms        (Tradition, Conformity)
    "SC002_1": {"A0": ["Self-Direction", "Stimulation"], "A1": ["Tradition", "Conformity"]},

    # "New care protocol"    → innovation, autonomy    (Self-Direction, Stimulation)
    # "Established procedure"→ tradition, norms        (Tradition, Conformity)
    "SC002_2": {"A0": ["Self-Direction", "Stimulation"], "A1": ["Tradition", "Conformity"]},

    # "Recognize entire team"      → group welfare     (Benevolence, Universalism)
    # "Recognize top individual"   → merit, status     (Achievement, Power)
    "SC003_1": {"A0": ["Benevolence", "Universalism"], "A1": ["Achievement", "Power"]},

    # "Honor leadership/advancement" → merit, status   (Achievement, Power)
    # "Honor collaboration/support"  → group welfare   (Benevolence, Universalism)
    # NOTE: A0/A1 are reversed here vs SC003_1 — same tension, opposite framing
    "SC003_2": {"A0": ["Achievement", "Power"], "A1": ["Benevolence", "Universalism"]},

    # "Social event with friends" → pleasure, novelty  (Hedonism, Stimulation)
    # "Family obligation at home" → duty, custom       (Conformity, Tradition)
    "SC004_1": {"A0": ["Hedonism", "Stimulation"], "A1": ["Conformity", "Tradition"]},

    # "Volunteer for festival"    → enjoyable, engaging(Hedonism, Stimulation)
    # "Routine maintenance tasks" → necessary, habitual(Conformity, Tradition)
    "SC004_2": {"A0": ["Hedonism", "Stimulation"], "A1": ["Conformity", "Tradition"]},

    # "Promote internal (established practices)" → stability    (Tradition, Security)
    # "Hire external (ambitious innovation)"     → bold, growth (Power, Achievement)
    "SC005_1": {"A0": ["Tradition", "Security"], "A1": ["Power", "Achievement"]},

    # "Family member upholding core values" → stability        (Tradition, Security)
    # "Younger relative new strategies"     → bold, growth     (Power, Achievement)
    "SC005_2": {"A0": ["Tradition", "Security"], "A1": ["Power", "Achievement"]},
}

# ─── Core functions ───────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def parse_profile(description_list: list) -> dict:
    """
    Parse a profile's description list into {value_name: weight}.
    HIGH → 1.0, LOW → 0.0.
    """
    profile = {}
    for desc in description_list:
        colon_pos = desc.index(":")
        value = desc[:colon_pos].strip()
        weight = HIGH_WEIGHT if "HIGH" in desc else LOW_WEIGHT
        profile[value] = weight
    return profile


def apply_modifier(profile: dict, pressured_values: list) -> dict:
    """
    Return a new profile with MODIFIER_BOOST added to each pressured value.
    Values already at HIGH_WEIGHT stay capped at 1.0.
    Values not in the profile are left unchanged.
    """
    modified = dict(profile)
    for v in pressured_values:
        if v in modified:
            modified[v] = min(1.0, modified[v] + MODIFIER_BOOST)
    return modified


def compute_scores(profile: dict, scenario_id: str) -> tuple:
    """Compute (score_A0, score_A1) as dot-product with value alignment weights."""
    mapping = SCENARIO_VALUE_MAP[scenario_id]
    s_a0 = sum(profile.get(v, 0.0) for v in mapping["A0"])
    s_a1 = sum(profile.get(v, 0.0) for v in mapping["A1"])
    return round(s_a0, 3), round(s_a1, 3)


def decide(s_a0: float, s_a1: float) -> str:
    if s_a0 > s_a1:  return "A0"
    if s_a1 > s_a0:  return "A1"
    return "TIE"


def driving_values_str(profile: dict, scenario_id: str, decision: str) -> str:
    """Describe which value dimensions drove the chosen action."""
    if decision == "TIE":
        return "Equal scores — tied"
    mapping = SCENARIO_VALUE_MAP[scenario_id]
    side_values = mapping.get(decision, [])
    active = [(v, profile.get(v, 0.0)) for v in side_values if profile.get(v, 0.0) > 0.0]
    if not active:
        return f"No active drivers (all LOW on {', '.join(side_values)})"
    return "; ".join(f"{v}={w:.1f}" for v, w in active)


def profile_summary(profile: dict) -> str:
    """Return a compact list of HIGH-valued dimensions."""
    highs = [v for v, w in profile.items() if w >= HIGH_WEIGHT]
    return ", ".join(highs) if highs else "ALL-LOW"


def make_change_explanation(
    baseline_profile: dict,
    modified_profile: dict,
    scenario_id: str,
    old_decision: str,
    new_decision: str,
    pressured_values: list,
) -> str:
    """Explain which boosted values caused the decision flip."""
    mapping = SCENARIO_VALUE_MAP[scenario_id]
    boosts = []
    for v in pressured_values:
        old_w = baseline_profile.get(v, 0.0)
        new_w = modified_profile.get(v, 0.0)
        if new_w > old_w:
            if v in mapping["A0"]:
                favor = "A0"
            elif v in mapping["A1"]:
                favor = "A1"
            else:
                favor = "not mapped"
            boosts.append(f"{v} {old_w:.1f}→{new_w:.1f} (favors {favor})")
    boost_str = "; ".join(boosts) if boosts else "no change in mapped values"
    return f"[{old_decision}→{new_decision}] Boosted: {boost_str}"


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=_DEFAULT_PROFILES,
                        help="Path to a profile_description_*.json file")
    args = parser.parse_args()

    profiles_path = args.profiles
    # Derive output filename from profiles filename
    stem = profiles_path.stem  # e.g. "profile_description_11to20"
    out_name = f"decision_analysis_{stem.replace('profile_description_', '')}.csv"
    global OUTPUT_PATH
    OUTPUT_PATH = BASE_DIR / "outputs" / out_name

    profiles     = load_json(profiles_path)
    scenarios    = load_json(SCENARIOS_FILE)
    modifiers_db = load_json(MODIFIERS_FILE)

    # Build modifier lookup: scenario_id → list of modifier dicts
    modifier_lookup: dict[str, list] = {}
    for item in modifiers_db:
        sid = item["scenario_id"]
        modifier_lookup[sid] = item.get("modifiers", [])

    # Build scenario theme lookup
    theme_lookup = {s["scenario_id"]: s.get("theme_id", "") for s in scenarios}

    rows = []
    flip_count = 0
    modifier_row_count = 0

    for profile_obj in profiles:
        vsw_id  = profile_obj["vsw_id"]
        profile = parse_profile(profile_obj["description"])
        p_sum   = profile_summary(profile)

        for scenario in scenarios:
            sid   = scenario["scenario_id"]
            theme = theme_lookup[sid]
            brief = scenario["description"][:80].rstrip() + "…"

            a0_vals_str = " + ".join(SCENARIO_VALUE_MAP[sid]["A0"])
            a1_vals_str = " + ".join(SCENARIO_VALUE_MAP[sid]["A1"])

            # ── Baseline row (no modifier) ────────────────────────────────
            bs_a0, bs_a1 = compute_scores(profile, sid)
            base_decision = decide(bs_a0, bs_a1)

            rows.append({
                "vsw_id":                vsw_id,
                "profile_HIGH_values":   p_sum,
                "scenario_id":           sid,
                "theme_id":              theme,
                "scenario_brief":        brief,
                "A0_text":               scenario["A0"],
                "A1_text":               scenario["A1"],
                "A0_aligned_values":     a0_vals_str,
                "A1_aligned_values":     a1_vals_str,
                "condition":             "BASELINE",
                "modifier_text":         "",
                "pressured_values":      "",
                "score_A0":              bs_a0,
                "score_A1":              bs_a1,
                "decision":              base_decision,
                "driving_values":        driving_values_str(profile, sid, base_decision),
                "changed_from_baseline": "",
                "change_explanation":    "",
            })

            # ── Modifier rows ─────────────────────────────────────────────
            for mod in modifier_lookup.get(sid, []):
                pressured = mod["expected_value_pressure"]
                mod_profile = apply_modifier(profile, pressured)
                ms_a0, ms_a1 = compute_scores(mod_profile, sid)
                mod_decision  = decide(ms_a0, ms_a1)
                changed = "YES" if mod_decision != base_decision else "NO"

                explanation = ""
                if changed == "YES":
                    flip_count += 1
                    explanation = make_change_explanation(
                        profile, mod_profile, sid,
                        base_decision, mod_decision, pressured
                    )

                modifier_row_count += 1
                rows.append({
                    "vsw_id":                vsw_id,
                    "profile_HIGH_values":   p_sum,
                    "scenario_id":           sid,
                    "theme_id":              theme,
                    "scenario_brief":        brief,
                    "A0_text":               scenario["A0"],
                    "A1_text":               scenario["A1"],
                    "A0_aligned_values":     a0_vals_str,
                    "A1_aligned_values":     a1_vals_str,
                    "condition":             mod["modifier_id"],
                    "modifier_text":         mod["modifier_text"],
                    "pressured_values":      " + ".join(pressured),
                    "score_A0":              ms_a0,
                    "score_A1":              ms_a1,
                    "decision":              mod_decision,
                    "driving_values":        driving_values_str(mod_profile, sid, mod_decision),
                    "changed_from_baseline": changed,
                    "change_explanation":    explanation,
                })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fieldnames = [
        "vsw_id", "profile_HIGH_values",
        "scenario_id", "theme_id", "scenario_brief",
        "A0_text", "A1_text", "A0_aligned_values", "A1_aligned_values",
        "condition", "modifier_text", "pressured_values",
        "score_A0", "score_A1", "decision",
        "driving_values",
        "changed_from_baseline", "change_explanation",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline_rows = len(profiles) * len(scenarios)
    flip_pct = 100 * flip_count / modifier_row_count if modifier_row_count else 0

    print(f"\n=== Value Decision Analysis Complete ===")
    print(f"  Profiles:             {len(profiles)}")
    print(f"  Scenarios:            {len(scenarios)}")
    print(f"  Baseline rows:        {baseline_rows}")
    print(f"  Modifier rows:        {modifier_row_count}")
    print(f"  Decision flips:       {flip_count} / {modifier_row_count}  ({flip_pct:.1f}%)")
    print(f"\n  Output → {OUTPUT_PATH}\n")

    # ── Print notable flips ───────────────────────────────────────────────────
    flipped = [r for r in rows if r["changed_from_baseline"] == "YES"]
    if flipped:
        print("Notable decision flips (first 10):")
        print(f"  {'Profile':<12} {'Scenario':<10} {'Modifier':<22} {'Baseline':<10} {'→'} {'New':<6} {'Explanation'}")
        print("  " + "-" * 110)
        for r in flipped[:10]:
            mod_id = r["condition"]
            exp_short = r["change_explanation"].replace("[", "").replace("]", "")
            print(f"  {r['vsw_id']:<12} {r['scenario_id']:<10} {mod_id:<22} "
                  f"{r['driving_values'][:30]:<32} → {r['decision']:<6} {exp_short[:60]}")


if __name__ == "__main__":
    main()

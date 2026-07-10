#!/usr/bin/env python3
"""
Lambda sweep for the utility-baseline modifier strength (MODIFIER_BOOST).

Reuses the scoring/decision logic from value_decision_analysis.py but
varies the saturating-boost magnitude lambda. For each lambda, computes
the overall flip rate and the per-axis flip rate across all 95 profiles,
all 10 scenarios, and all 8 modifier axes.

Output: outputs/lambda_sweep.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import value_decision_analysis as vda  # noqa: E402

DATASET_DIR = BASE_DIR / "Dataset"
PROFILE_FILES = sorted(DATASET_DIR.glob("profile_description_*to*.json"))
SCENARIOS_FILE = DATASET_DIR / "scenarios_batch1.json"
MODIFIERS_FILE = DATASET_DIR / "modifiers_batch1.json"

LAMBDA_GRID = [0.00, 0.10, 0.25, 0.40, 0.49, 0.50, 0.51, 0.60, 0.75, 0.90, 0.99, 1.00]
OUT_PATH = BASE_DIR / "outputs" / "lambda_sweep.json"


def load_all_profiles():
    profs = []
    for p in PROFILE_FILES:
        profs.extend(vda.load_json(p))
    return profs


def evaluate(lam: float, profiles, scenarios, modifier_lookup):
    vda.MODIFIER_BOOST = lam

    n_trials = 0
    n_flips = 0
    per_axis = defaultdict(lambda: {"trials": 0, "flips": 0})

    for prof_obj in profiles:
        profile = vda.parse_profile(prof_obj["description"])
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            if sid not in vda.SCENARIO_VALUE_MAP:
                continue
            bs_a0, bs_a1 = vda.compute_scores(profile, sid)
            base_dec = vda.decide(bs_a0, bs_a1)

            for mod in modifier_lookup.get(sid, []):
                axis = mod.get("axis", "unknown")
                pressured = mod["expected_value_pressure"]
                mod_profile = vda.apply_modifier(profile, pressured)
                ms_a0, ms_a1 = vda.compute_scores(mod_profile, sid)
                mod_dec = vda.decide(ms_a0, ms_a1)
                flipped = int(mod_dec != base_dec)
                n_trials += 1
                n_flips += flipped
                per_axis[axis]["trials"] += 1
                per_axis[axis]["flips"] += flipped

    return {
        "lambda": lam,
        "n_trials": n_trials,
        "n_flips": n_flips,
        "flip_rate_pct": round(100 * n_flips / n_trials, 3) if n_trials else 0.0,
        "per_axis": {
            ax: {
                "trials": d["trials"],
                "flips": d["flips"],
                "flip_rate_pct": round(100 * d["flips"] / d["trials"], 3)
                if d["trials"] else 0.0,
            }
            for ax, d in per_axis.items()
        },
    }


def main():
    profiles = load_all_profiles()
    scenarios = vda.load_json(SCENARIOS_FILE)
    modifiers_db = vda.load_json(MODIFIERS_FILE)

    modifier_lookup = {item["scenario_id"]: item.get("modifiers", []) for item in modifiers_db}

    print(f"Profiles: {len(profiles)}  Scenarios: {len(scenarios)}  "
          f"Modifier scenarios: {len(modifier_lookup)}")
    print(f"Sweeping lambda over {LAMBDA_GRID}\n")

    results = []
    for lam in LAMBDA_GRID:
        r = evaluate(lam, profiles, scenarios, modifier_lookup)
        print(f"  lambda={lam:>4.2f}  trials={r['n_trials']}  "
              f"flips={r['n_flips']}  rate={r['flip_rate_pct']:.2f}%")
        for ax, d in sorted(r["per_axis"].items()):
            print(f"      {ax:<26} {d['flips']:>4} / {d['trials']:>4}  "
                  f"({d['flip_rate_pct']:.2f}%)")
        results.append(r)
        print()

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"grid": LAMBDA_GRID, "results": results}, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
VISTA Simulation Runner
========================
Runs the full two-stage reasoning framework against N scenarios,
comparing action selection between Explorer and Guardian personas.
Generates audit trail and decision shift statistics.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_DIR, NUM_SIMULATION_SCENARIOS, RANDOM_SEED
from simulation.scenario_sampler import prepare_simulation_scenarios
from stage2_action_selection.value_tagger import tag_all_scenarios
from stage2_action_selection.utility import compare_personas
from stage2_action_selection.personas import (
    get_persona,
    describe_persona,
    PERSONA_EXPLORER,
    PERSONA_GUARDIAN,
)
from simulation.report_generator import generate_report, save_audit_trail


def run_simulation(
    n_scenarios: int = NUM_SIMULATION_SCENARIOS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the full VISTA simulation.

    Pipeline:
      1. Sample N scenarios from Moral Stories
      2. Tag all candidate actions with 38-dim value vectors (Stage 1)
      3. For each scenario, compare Explorer vs Guardian action selection (Stage 2)
      4. Record audit trail and compute decision shift statistics

    Args:
        n_scenarios: Number of scenarios to run.
        seed: Random seed for reproducibility.

    Returns:
        Simulation results dict.
    """
    print("=" * 60)
    print("  VISTA SIMULATION")
    print("  Value-Informed Situated Tactical Agent")
    print("=" * 60)
    start_time = time.time()

    # Step 1: Prepare scenarios
    print("\n📋 Step 1: Preparing scenarios...")
    scenarios = prepare_simulation_scenarios(n=n_scenarios, seed=seed)

    # Step 2: Tag all actions with value vectors
    print("\n🏷️  Step 2: Tagging actions with value vectors...")
    scenarios = tag_all_scenarios(scenarios)

    # Step 3: Run comparisons
    print(f"\n⚖️  Step 3: Running {len(scenarios)} comparisons...")
    audit_entries = []
    shifts = 0
    total = len(scenarios)

    for scenario in tqdm(scenarios, desc="Comparing personas"):
        comparison = compare_personas(
            candidates=scenario["candidates"],
            persona_a=PERSONA_EXPLORER,
            persona_b=PERSONA_GUARDIAN,
            name_a="Explorer",
            name_b="Guardian",
        )

        if comparison["decision_shifted"]:
            shifts += 1

        # Build audit trail entry
        entry = {
            "scenario_id": scenario["id"],
            "situation": scenario.get("situation", ""),
            "intention": scenario.get("intention", ""),
            "norm": scenario.get("norm", ""),
            "context": scenario["context"][:200],
            "candidates": [
                {
                    "action": c["action_text"],
                    "label": c.get("label", "unknown"),
                    "V_a": c["value_vector"].tolist() if isinstance(c["value_vector"], np.ndarray) else c["value_vector"],
                }
                for c in scenario["candidates"]
            ],
            "person1": {
                "persona": "Explorer",
                "selected_action": comparison["person_a"]["selected_action"],
                "selected_label": comparison["person_a"]["selected_label"],
                "utility_scores": {
                    u["label"]: round(u["utility"], 6)
                    for u in comparison["person_a"]["all_utilities"]
                },
                "top_driving_values": [
                    d["value_dimension"]
                    for d in comparison["person_a"]["justification"]["top_driving_values"][:3]
                ],
            },
            "person2": {
                "persona": "Guardian",
                "selected_action": comparison["person_b"]["selected_action"],
                "selected_label": comparison["person_b"]["selected_label"],
                "utility_scores": {
                    u["label"]: round(u["utility"], 6)
                    for u in comparison["person_b"]["all_utilities"]
                },
                "top_driving_values": [
                    d["value_dimension"]
                    for d in comparison["person_b"]["justification"]["top_driving_values"][:3]
                ],
            },
            "decision_shifted": comparison["decision_shifted"],
            "shift_magnitude": round(comparison["shift_magnitude"], 6),
        }
        audit_entries.append(entry)

    elapsed = time.time() - start_time

    # Step 4: Compute statistics
    shift_rate = shifts / total if total > 0 else 0

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": total,
            "random_seed": seed,
            "elapsed_seconds": round(elapsed, 2),
        },
        "statistics": {
            "total_scenarios": total,
            "decision_shifts": shifts,
            "no_shifts": total - shifts,
            "shift_rate": round(shift_rate, 4),
            "shift_percentage": round(shift_rate * 100, 2),
        },
        "personas": {
            "Explorer": describe_persona("Explorer"),
            "Guardian": describe_persona("Guardian"),
        },
        "audit_trail": audit_entries,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Scenarios tested: {total}")
    print(f"  Decision shifts:  {shifts} ({shift_rate*100:.1f}%)")
    print(f"  No shifts:        {total - shifts}")
    print(f"  Elapsed time:     {elapsed:.1f}s")
    print("=" * 60)

    # Step 5: Generate outputs
    print("\n📄 Step 5: Generating outputs...")
    save_audit_trail(audit_entries, OUTPUT_DIR)
    generate_report(results, OUTPUT_DIR)

    print(f"\n✅ Simulation complete! Check {OUTPUT_DIR}/ for results.")
    return results


if __name__ == "__main__":
    results = run_simulation()

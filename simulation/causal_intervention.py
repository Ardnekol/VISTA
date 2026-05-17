"""
Causal Value Intervention Simulator
====================================
This script performs counterfactual interventions on the VISTA pipeline.
It takes a fixed scenario and a base persona, then systematically sweeps 
a SINGLE value dimension from 0.0 to 1.0 (holding all others constant).

It tracks:
1. How the utility scores for Moral vs Immoral actions change.
2. The exact threshold at which the decision boundary is crossed.

This proves fine-grained controllability of the VISTA framework.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import VALUE_TO_INDICES
from stage2_action_selection.personas import PERSONA_EXPLORER, describe_persona
from stage2_action_selection.utility import compute_utility

def load_contested_scenarios(audit_path: str, top_k: int = 10) -> list:
    """Finds scenarios where the Explorer's decision was a close call."""
    with open(audit_path, 'r') as f:
        audit_data = json.load(f)
        
    scored_scenarios = []
    for entry in audit_data:
        # We need scenarios that have raw value vectors logged, or we can use the utilities
        # Wait, audit_trail.json usually records the utility gap. Let's find the smallest gaps.
        if "person1" not in entry or "utility_gap" not in entry:
            # Let's compute gap manually from utility_scores
            u_scores = entry["person1"]["utility_scores"]
            gap = abs(u_scores.get("moral", 0) - u_scores.get("immoral", 0))
            
            scored_scenarios.append({
                "scenario_id": entry["scenario_id"],
                "situation": entry["situation"],
                "candidates": entry["candidates"], # Contains V_a !
                "base_choice": entry["person1"]["selected_label"],
                "gap": gap
            })
            
    # Sort by smallest gap (most contested boundaries)
    scored_scenarios.sort(key=lambda x: x["gap"])
    return scored_scenarios[:top_k]

def run_intervention(scenario: dict, base_persona: np.ndarray, target_value_name: str, state: str = "attained") -> dict:
    """
    Sweeps a specific value dimension from 0.0 to 1.0 for a given scenario.
    """
    if target_value_name not in VALUE_TO_INDICES:
        raise ValueError(f"Unknown value: {target_value_name}")
        
    dim_idx = VALUE_TO_INDICES[target_value_name][state]
    
    # Extract value vectors for the moral/immoral actions
    v_moral = None
    v_immoral = None
    for cand in scenario["candidates"]:
        if cand["label"] == "moral":
            v_moral = np.array(cand["V_a"], dtype=np.float32)
        elif cand["label"] == "immoral":
            v_immoral = np.array(cand["V_a"], dtype=np.float32)
            
    if v_moral is None or v_immoral is None:
        raise ValueError("Scenario missing moral/immoral candidate vectors.")
        
    results = []
    flip_point = None
    initial_decision = None
    
    # Sweep from 0.0 to 1.0 in 100 steps
    sweep_values = np.linspace(0.0, 1.0, 101)
    
    for val in sweep_values:
        # Create counterfactual persona
        p_cf = base_persona.copy()
        p_cf[dim_idx] = float(val)
        
        # Compute utilities
        u_moral = compute_utility(v_moral, p_cf)
        u_immoral = compute_utility(v_immoral, p_cf)
        
        decision = "moral" if u_moral > u_immoral else "immoral"
        
        if initial_decision is None:
            initial_decision = decision
            
        if decision != initial_decision and flip_point is None:
            flip_point = float(val)
            
        results.append({
            "intervened_value": float(val),
            "u_moral": u_moral,
            "u_immoral": u_immoral,
            "decision": decision
        })
        
    return {
        "scenario_id": scenario["scenario_id"],
        "intervened_dimension": f"{target_value_name} ({state})",
        "initial_decision": initial_decision,
        "flip_point_found": flip_point is not None,
        "flip_threshold": flip_point,
        "sweep_data": results
    }

if __name__ == "__main__":
    audit_file = "/home/manu/VISTA/outputs/audit_trail.json"
    print(f"Loading top 10 contested scenarios from {audit_file}...")
    
    try:
        contested = load_contested_scenarios(audit_file, top_k=10)
    except Exception as e:
        print(f"Failed to load scenarios: {e}")
        sys.exit(1)
        
    # We will intervene on 'Conformity: rules' (attained) using the Explorer persona as our base
    target_value = "Conformity: rules"
    print(f"\nRunning Causal Intervention sweep on '{target_value}' (Explorer Base)...\n")
    
    intervention_results = []
    
    for sc in contested:
        res = run_intervention(
            scenario=sc,
            base_persona=PERSONA_EXPLORER,
            target_value_name=target_value,
            state="attained"
        )
        intervention_results.append(res)
        
        print(f"Scenario {sc['scenario_id']}:")
        if res["flip_point_found"]:
            print(f"  ✅ FLIP FOUND! Base decision: {res['initial_decision'].upper()}")
            print(f"     Flipped at {target_value} = {res['flip_threshold']:.2f}")
        else:
            print(f"  ❌ No flip. Base decision: {res['initial_decision'].upper()} held strong.")
            
    # Save the raw curve data for paper plots
    out_path = "/home/manu/VISTA/outputs/causal_intervention_curves.json"
    with open(out_path, "w") as f:
        json.dump(intervention_results, f, indent=2)
        
    print(f"\nSaved full sensitivity sweep to {out_path}.")

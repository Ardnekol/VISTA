"""
Utility-Based Action Selector
==============================
Core decision logic for VISTA Stage 2.
Implements: U(a, P) = Σ(V_a,i · P_i) and A = argmax_a U(a, P)
"""

import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_NAMES, SCHWARTZ_VALUES, VALUE_TO_INDICES, NUM_LABELS


def compute_utility(V_a: np.ndarray, P: np.ndarray) -> float:
    """Compute the utility of an action given a persona vector.

    U(a, P) = Σ_{i=0}^{37} (V_{a,i} · P_i)

    This is the weighted dot product between the action's value profile
    and the persona's value weights.

    Args:
        V_a: Action value vector of shape (38,).
        P: Persona vector of shape (38,).

    Returns:
        Scalar utility value.
    """
    assert V_a.shape == (NUM_LABELS,), f"Expected V_a shape ({NUM_LABELS},), got {V_a.shape}"
    assert P.shape == (NUM_LABELS,), f"Expected P shape ({NUM_LABELS},), got {P.shape}"
    return float(np.dot(V_a, P))


def select_action(
    candidates: list[dict],
    persona: np.ndarray,
    persona_name: str = "Unknown",
) -> dict:
    """Select the best action using argmax over utility scores.

    A = argmax_a U(a, P)

    Args:
        candidates: List of candidate dicts, each with 'action_text' and 'value_vector'.
        persona: 38-dim persona vector P.
        persona_name: Name of the persona for reporting.

    Returns:
        Result dict with selected action, utility scores, and justification.
    """
    utilities = []
    for candidate in candidates:
        V_a = candidate["value_vector"]
        if isinstance(V_a, list):
            V_a = np.array(V_a, dtype=np.float32)
        u = compute_utility(V_a, persona)
        utilities.append({
            "action_text": candidate["action_text"],
            "label": candidate.get("label", "unknown"),
            "utility": u,
            "value_vector": V_a,
        })

    # Argmax
    best = max(utilities, key=lambda x: x["utility"])

    # Generate justification: which value dimensions drove this decision?
    justification = _generate_justification(best, persona, persona_name)

    return {
        "persona_name": persona_name,
        "selected_action": best["action_text"],
        "selected_label": best["label"],
        "utility": best["utility"],
        "all_utilities": [
            {
                "action": u["action_text"],
                "label": u["label"],
                "utility": u["utility"],
            }
            for u in utilities
        ],
        "justification": justification,
    }


def _generate_justification(
    best: dict, persona: np.ndarray, persona_name: str
) -> dict:
    """Generate an interpretable justification for the selected action.

    Identifies the top value dimensions that contributed most to the
    utility score (V_a,i × P_i product).

    Args:
        best: The selected action dict with value_vector.
        persona: The persona vector.
        persona_name: Name for reporting.

    Returns:
        Justification dict with driving values and contributions.
    """
    V_a = best["value_vector"]
    contributions = V_a * persona  # Element-wise product

    # Top positive contributions (values that pushed this action higher)
    top_indices = np.argsort(contributions)[::-1][:5]
    top_drivers = []
    for i in top_indices:
        if contributions[i] > 0:
            top_drivers.append({
                "value_dimension": LABEL_NAMES[i],
                "action_signal": float(V_a[i]),
                "persona_weight": float(persona[i]),
                "contribution": float(contributions[i]),
            })

    return {
        "persona": persona_name,
        "total_utility": float(best["utility"]),
        "top_driving_values": top_drivers,
        "explanation": (
            f"Persona '{persona_name}' selected this action primarily because "
            f"of alignment on: {', '.join(d['value_dimension'] for d in top_drivers[:3])}"
        ),
    }


def compare_personas(
    candidates: list[dict],
    persona_a: np.ndarray,
    persona_b: np.ndarray,
    name_a: str = "Person A",
    name_b: str = "Person B",
) -> dict:
    """Compare action selection between two personas on the same candidates.

    This is the core proof function: shows whether A ≠ B for same C.

    Args:
        candidates: List of candidate action dicts.
        persona_a: First persona vector.
        persona_b: Second persona vector.
        name_a: Name of first persona.
        name_b: Name of second persona.

    Returns:
        Comparison dict with both selections and shift analysis.
    """
    result_a = select_action(candidates, persona_a, name_a)
    result_b = select_action(candidates, persona_b, name_b)

    shifted = result_a["selected_label"] != result_b["selected_label"]

    # Compute shift magnitude: difference in utility gap
    utils_a = {u["label"]: u["utility"] for u in result_a["all_utilities"]}
    utils_b = {u["label"]: u["utility"] for u in result_b["all_utilities"]}

    gap_a = max(utils_a.values()) - min(utils_a.values())
    gap_b = max(utils_b.values()) - min(utils_b.values())

    return {
        "person_a": result_a,
        "person_b": result_b,
        "decision_shifted": shifted,
        "shift_magnitude": abs(gap_a + gap_b) / 2,
        "utility_gap_a": gap_a,
        "utility_gap_b": gap_b,
    }


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Utility Function Smoke Test")
    print("=" * 60)

    # Create synthetic action value vectors for testing
    # Action A: strongly aligned with self-direction (curiosity)
    V_action_a = np.zeros(NUM_LABELS, dtype=np.float32)
    V_action_a[VALUE_TO_INDICES["Self-direction: thought"]["attained"]] = 0.9
    V_action_a[VALUE_TO_INDICES["Stimulation"]["attained"]] = 0.8
    V_action_a[VALUE_TO_INDICES["Security: personal"]["constrained"]] = 0.7

    # Action B: strongly aligned with security/conformity
    V_action_b = np.zeros(NUM_LABELS, dtype=np.float32)
    V_action_b[VALUE_TO_INDICES["Security: societal"]["attained"]] = 0.9
    V_action_b[VALUE_TO_INDICES["Conformity: rules"]["attained"]] = 0.85
    V_action_b[VALUE_TO_INDICES["Tradition"]["attained"]] = 0.7

    candidates = [
        {"action_text": "Explore the unknown cave", "label": "exploratory", "value_vector": V_action_a},
        {"action_text": "Stay on the marked trail", "label": "conservative", "value_vector": V_action_b},
    ]

    from stage2_action_selection.personas import PERSONA_EXPLORER, PERSONA_GUARDIAN

    # Test utility computation
    u_explore_explorer = compute_utility(V_action_a, PERSONA_EXPLORER)
    u_stay_explorer = compute_utility(V_action_b, PERSONA_EXPLORER)
    print(f"\nExplorer → 'Explore cave': U = {u_explore_explorer:.4f}")
    print(f"Explorer → 'Stay on trail': U = {u_stay_explorer:.4f}")

    u_explore_guardian = compute_utility(V_action_a, PERSONA_GUARDIAN)
    u_stay_guardian = compute_utility(V_action_b, PERSONA_GUARDIAN)
    print(f"\nGuardian → 'Explore cave': U = {u_explore_guardian:.4f}")
    print(f"Guardian → 'Stay on trail': U = {u_stay_guardian:.4f}")

    # Test comparison
    comparison = compare_personas(
        candidates, PERSONA_EXPLORER, PERSONA_GUARDIAN,
        "Explorer", "Guardian"
    )
    print(f"\n🔄 Decision shifted: {comparison['decision_shifted']}")
    print(f"   Explorer chose: {comparison['person_a']['selected_label']}")
    print(f"   Guardian chose: {comparison['person_b']['selected_label']}")
    print(f"   Shift magnitude: {comparison['shift_magnitude']:.4f}")

    assert comparison["decision_shifted"], "Expected decision shift with contrived inputs!"
    print("\n✅ Utility function smoke test passed — decision shift confirmed!")

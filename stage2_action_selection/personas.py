"""
Persona Vector Definitions
============================
Defines persona vectors as 38-dimensional arrays (19 values × 2 states)
for the utility-based action selection in VISTA.

Each persona represents a distinct value profile that will produce
different action preferences when applied to the same scenario.
"""

import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCHWARTZ_VALUES, LABEL_NAMES, NUM_LABELS, VALUE_TO_INDICES


def _make_persona_vector(value_weights: dict[str, tuple[float, float]]) -> np.ndarray:
    """Create a 38-dim persona vector from a value weight specification.

    Args:
        value_weights: Dict mapping value name → (attained_weight, constrained_weight).
            Values not specified default to (0.50, 0.50).

    Returns:
        numpy array of shape (38,).
    """
    persona = np.full(NUM_LABELS, 0.50, dtype=np.float32)

    for value_name, (attained_w, constrained_w) in value_weights.items():
        if value_name in VALUE_TO_INDICES:
            indices = VALUE_TO_INDICES[value_name]
            persona[indices["attained"]] = attained_w
            persona[indices["constrained"]] = constrained_w
        else:
            print(f"  Warning: unknown value '{value_name}', skipping")

    return persona


# ─────────────────────────────────────────────────────────────
# Person 1: "The Explorer"
# ─────────────────────────────────────────────────────────────
# High: Curiosity (Self-direction + Stimulation), Freedom
# Low:  Security, Conformity
EXPLORER_WEIGHTS = {
    "Self-direction: thought":   (0.95, 0.05),  # Curiosity → high attainment, low constraint
    "Self-direction: action":    (0.90, 0.05),
    "Stimulation":               (0.90, 0.05),
    "Hedonism":                  (0.70, 0.30),
    "Achievement":               (0.60, 0.40),
    "Power: dominance":          (0.30, 0.60),
    "Power: resources":          (0.30, 0.60),
    "Face":                      (0.40, 0.50),
    "Security: personal":        (0.10, 0.80),  # Low security preference
    "Security: societal":        (0.10, 0.75),
    "Tradition":                 (0.15, 0.70),
    "Conformity: rules":         (0.05, 0.90),  # Strongly anti-conformity
    "Conformity: interpersonal": (0.20, 0.60),
    "Humility":                  (0.30, 0.50),
    "Benevolence: caring":       (0.60, 0.30),
    "Benevolence: dependability": (0.50, 0.40),
    "Universalism: concern":     (0.65, 0.30),
    "Universalism: nature":      (0.55, 0.35),
    "Universalism: tolerance":   (0.80, 0.15),  # High tolerance
}

PERSONA_EXPLORER = _make_persona_vector(EXPLORER_WEIGHTS)


# ─────────────────────────────────────────────────────────────
# Person 2: "The Guardian"
# ─────────────────────────────────────────────────────────────
# High: Security, Conformity, Tradition
# Low:  Curiosity (Self-direction), Stimulation
GUARDIAN_WEIGHTS = {
    "Self-direction: thought":   (0.10, 0.70),  # Low curiosity
    "Self-direction: action":    (0.15, 0.65),
    "Stimulation":               (0.05, 0.80),
    "Hedonism":                  (0.20, 0.60),
    "Achievement":               (0.50, 0.40),
    "Power: dominance":          (0.60, 0.30),
    "Power: resources":          (0.55, 0.35),
    "Face":                      (0.70, 0.20),
    "Security: personal":        (0.30, 0.10),  # High security (note: low constraint = actively desires it)
    "Security: societal":        (0.85, 0.05),  # Very high societal security
    "Tradition":                 (0.85, 0.10),
    "Conformity: rules":         (0.90, 0.05),  # Strongly pro-conformity
    "Conformity: interpersonal": (0.80, 0.10),
    "Humility":                  (0.70, 0.20),
    "Benevolence: caring":       (0.65, 0.25),
    "Benevolence: dependability": (0.75, 0.15),
    "Universalism: concern":     (0.50, 0.40),
    "Universalism: nature":      (0.40, 0.45),
    "Universalism: tolerance":   (0.30, 0.55),  # Lower tolerance for difference
}

PERSONA_GUARDIAN = _make_persona_vector(GUARDIAN_WEIGHTS)


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────
PERSONAS = {
    "Explorer": PERSONA_EXPLORER,
    "Guardian": PERSONA_GUARDIAN,
}


def get_persona(name: str) -> np.ndarray:
    """Retrieve a persona vector by name.

    Args:
        name: Persona name (e.g., 'Explorer', 'Guardian').

    Returns:
        38-dim persona vector.

    Raises:
        KeyError: If persona name not found.
    """
    if name not in PERSONAS:
        raise KeyError(f"Unknown persona '{name}'. Available: {list(PERSONAS.keys())}")
    return PERSONAS[name].copy()


def describe_persona(name: str) -> dict:
    """Get a human-readable description of a persona's value profile.

    Args:
        name: Persona name.

    Returns:
        Dict with name, top_values, bottom_values, and full profile.
    """
    p = get_persona(name)

    # Extract per-value scores (average of attained and constrained)
    profile = []
    for v in SCHWARTZ_VALUES:
        idx = VALUE_TO_INDICES[v]
        attained = p[idx["attained"]]
        constrained = p[idx["constrained"]]
        # Net preference = attained - constrained (higher = more aligned with value)
        net = attained - constrained
        profile.append((v, float(attained), float(constrained), float(net)))

    # Sort by net preference
    profile.sort(key=lambda x: x[3], reverse=True)

    return {
        "name": name,
        "top_values": [(v, net) for v, _, _, net in profile[:5]],
        "bottom_values": [(v, net) for v, _, _, net in profile[-5:]],
        "full_profile": profile,
        "vector": p.tolist(),
    }


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Persona Vectors Smoke Test")
    print("=" * 60)

    for name in PERSONAS:
        desc = describe_persona(name)
        print(f"\n🧑 Persona: {desc['name']}")
        print(f"  Vector shape: ({len(desc['vector'])},)")
        print(f"  Top values (net preference):")
        for v, net in desc["top_values"]:
            print(f"    ✅ {v}: {net:+.2f}")
        print(f"  Bottom values:")
        for v, net in desc["bottom_values"]:
            print(f"    ❌ {v}: {net:+.2f}")

    # Verify orthogonality
    dot = float(np.dot(PERSONA_EXPLORER, PERSONA_GUARDIAN))
    norm_e = float(np.linalg.norm(PERSONA_EXPLORER))
    norm_g = float(np.linalg.norm(PERSONA_GUARDIAN))
    cosine_sim = dot / (norm_e * norm_g)
    print(f"\nCosine similarity (Explorer, Guardian): {cosine_sim:.4f}")
    print(f"  (Lower = more divergent preferences)")

    print("\n✅ Persona smoke test passed!")

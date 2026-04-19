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
            Values not specified default to (0.25, 0.25) — neutral baseline.

    Returns:
        numpy array of shape (38,).
    """
    persona = np.full(NUM_LABELS, 0.25, dtype=np.float32)

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
# Amplified for maximum persona divergence
EXPLORER_WEIGHTS = {
    "Self-direction: thought":   (1.00, 0.00),  # Maximum curiosity
    "Self-direction: action":    (0.95, 0.00),
    "Stimulation":               (0.95, 0.00),
    "Hedonism":                  (0.80, 0.15),
    "Achievement":               (0.55, 0.35),
    "Power: dominance":          (0.15, 0.70),
    "Power: resources":          (0.15, 0.70),
    "Face":                      (0.25, 0.50),
    "Security: personal":        (0.00, 0.95),  # Actively rejects security
    "Security: societal":        (0.00, 0.90),
    "Tradition":                 (0.00, 0.85),
    "Conformity: rules":         (0.00, 1.00),  # Maximum anti-conformity
    "Conformity: interpersonal": (0.05, 0.75),
    "Humility":                  (0.15, 0.50),
    "Benevolence: caring":       (0.60, 0.20),
    "Benevolence: dependability": (0.40, 0.35),
    "Universalism: concern":     (0.65, 0.20),
    "Universalism: nature":      (0.55, 0.25),
    "Universalism: tolerance":   (0.90, 0.05),  # Very high tolerance
}

PERSONA_EXPLORER = _make_persona_vector(EXPLORER_WEIGHTS)


# ─────────────────────────────────────────────────────────────
# Person 2: "The Guardian"
# ─────────────────────────────────────────────────────────────
# High: Security, Conformity, Tradition
# Low:  Curiosity (Self-direction), Stimulation
GUARDIAN_WEIGHTS = {
    "Self-direction: thought":   (0.00, 0.85),  # Actively rejects curiosity
    "Self-direction: action":    (0.05, 0.80),
    "Stimulation":               (0.00, 0.95),
    "Hedonism":                  (0.10, 0.75),
    "Achievement":               (0.45, 0.35),
    "Power: dominance":          (0.65, 0.20),
    "Power: resources":          (0.60, 0.25),
    "Face":                      (0.75, 0.10),
    "Security: personal":        (0.40, 0.05),
    "Security: societal":        (0.95, 0.00),  # Maximum societal security
    "Tradition":                 (0.95, 0.00),  # Maximum tradition
    "Conformity: rules":         (1.00, 0.00),  # Maximum conformity
    "Conformity: interpersonal": (0.90, 0.00),
    "Humility":                  (0.80, 0.10),
    "Benevolence: caring":       (0.65, 0.15),
    "Benevolence: dependability": (0.85, 0.05),
    "Universalism: concern":     (0.45, 0.40),
    "Universalism: nature":      (0.30, 0.50),
    "Universalism: tolerance":   (0.15, 0.70),  # Low tolerance
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

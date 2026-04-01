"""
Value Tagger
=============
Tags each candidate action text with a 38-dim value vector V_a
using the Stage 1 value classifier.
"""

import os
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage1_value_inference.predict import predict, predict_batch, get_classifier


def tag_action(action_text: str) -> np.ndarray:
    """Tag a single action with its 38-dim value vector.

    Args:
        action_text: The action description text.

    Returns:
        V_a: numpy array of shape (38,) — value probabilities for this action.
    """
    return predict(action_text)


def tag_actions_batch(action_texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Tag a batch of actions with value vectors.

    Args:
        action_texts: List of action description texts.
        batch_size: Batch size for inference.

    Returns:
        numpy array of shape (N, 38).
    """
    return predict_batch(action_texts, batch_size=batch_size)


def tag_scenario_candidates(scenario: dict) -> dict:
    """Tag all candidate actions in a scenario with value vectors.

    Modifies the scenario dict in-place by adding 'value_vector' to each candidate.

    Args:
        scenario: Scenario dict with 'candidates' list, each having 'action_text'.

    Returns:
        The same scenario dict, now with value_vector attached to each candidate.
    """
    action_texts = [c["action_text"] for c in scenario["candidates"]]
    vectors = tag_actions_batch(action_texts)

    for candidate, vector in zip(scenario["candidates"], vectors):
        candidate["value_vector"] = vector

    return scenario


def tag_all_scenarios(scenarios: list[dict], batch_size: int = 16) -> list[dict]:
    """Tag all candidate actions across all scenarios.

    Args:
        scenarios: List of scenario dicts.
        batch_size: Batch size for inference.

    Returns:
        The same list with value_vector attached to every candidate.
    """
    # Flatten all action texts for efficient batch inference
    all_texts = []
    text_map = []  # (scenario_idx, candidate_idx)

    for s_idx, scenario in enumerate(scenarios):
        for c_idx, candidate in enumerate(scenario["candidates"]):
            all_texts.append(candidate["action_text"])
            text_map.append((s_idx, c_idx))

    print(f"Tagging {len(all_texts)} actions across {len(scenarios)} scenarios...")
    all_vectors = tag_actions_batch(all_texts, batch_size=batch_size)

    # Distribute vectors back
    for vec_idx, (s_idx, c_idx) in enumerate(text_map):
        scenarios[s_idx]["candidates"][c_idx]["value_vector"] = all_vectors[vec_idx]

    print(f"  ✅ Tagged all actions with 38-dim value vectors")
    return scenarios


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Value Tagger Smoke Test")
    print("=" * 60)

    test_actions = [
        "She decides to help her elderly neighbor carry groceries.",
        "He breaks into the car to steal the radio.",
        "They organize a community cleanup event.",
    ]

    for action in test_actions:
        v = tag_action(action)
        top_idx = np.argsort(v)[::-1][:3]
        from config import LABEL_NAMES
        print(f"\n📝 '{action[:60]}...'")
        print(f"   Value vector shape: {v.shape}, sum: {v.sum():.2f}")
        for i in top_idx:
            print(f"   {LABEL_NAMES[i]}: {v[i]:.4f}")

    print("\n✅ Value tagger smoke test passed!")

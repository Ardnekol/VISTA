"""
Scenario Sampler
=================
Samples and prepares scenarios from Moral Stories for the
100-scenario proof simulation.
"""

import os
import random

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NUM_SIMULATION_SCENARIOS, RANDOM_SEED
from stage2_action_selection.moral_stories_loader import (
    load_moral_stories,
    build_scenarios,
    sample_scenarios,
)


def prepare_simulation_scenarios(
    n: int = NUM_SIMULATION_SCENARIOS,
    seed: int = RANDOM_SEED,
    split: str = "train",
) -> list[dict]:
    """Prepare the full set of scenarios for the simulation.

    Loads Moral Stories, builds scenario tuples, and samples n.

    Args:
        n: Number of scenarios to sample.
        seed: Random seed.
        split: Dataset split to use.

    Returns:
        List of n scenario dicts ready for simulation.
    """
    print("=" * 60)
    print(f"Preparing {n} Simulation Scenarios")
    print("=" * 60)

    stories = load_moral_stories(split)
    all_scenarios = build_scenarios(stories)
    sampled = sample_scenarios(all_scenarios, n=n, seed=seed)

    # Validate all scenarios have required fields
    valid = []
    for s in sampled:
        if (
            s.get("context")
            and len(s.get("candidates", [])) >= 2
            and all(c.get("action_text") for c in s["candidates"])
        ):
            valid.append(s)

    print(f"\n  Valid scenarios: {len(valid)}/{len(sampled)}")
    return valid


if __name__ == "__main__":
    scenarios = prepare_simulation_scenarios(n=5)
    for s in scenarios:
        print(f"\n  Scenario {s['id']}:")
        print(f"    Context: {s['context'][:80]}...")
        print(f"    Actions: {len(s['candidates'])}")
    print("\n✅ Scenario sampler smoke test passed!")

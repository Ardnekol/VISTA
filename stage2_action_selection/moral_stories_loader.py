"""
Moral Stories Dataset Loader
=============================
Downloads and parses the Moral Stories dataset (Emelin et al., 2021).
Uses huggingface_hub to download the JSONL file directly.
Extracts scenario tuples of the form:
  (situation, intention, [moral_action, immoral_action])
"""

import json
import os
import random
import ssl
import urllib.request

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_SEED, PROJECT_ROOT


CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "moral_stories")
JSONL_FILENAME = "moral_stories_full.jsonl"


def _download_moral_stories() -> str:
    """Download the Moral Stories JSONL file from HuggingFace.

    The dataset is at: demelin/moral_stories → data/moral_stories_full.jsonl

    Returns:
        Path to the downloaded JSONL file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, JSONL_FILENAME)

    if os.path.exists(local_path):
        print(f"  Using cached data: {local_path}")
        return local_path

    # Primary method: huggingface_hub
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="demelin/moral_stories",
            filename="data/moral_stories_full.jsonl",
            repo_type="dataset",
            local_dir=CACHE_DIR,
        )
        # hf_hub_download may place it in a subdirectory; copy to expected location
        if os.path.exists(path) and path != local_path:
            import shutil
            shutil.copy2(path, local_path)
        print(f"  Downloaded via huggingface_hub: {local_path}")
        return local_path
    except Exception as e:
        print(f"  huggingface_hub failed: {e}")

    # Fallback: direct HTTPS download with SSL workaround
    url = "https://huggingface.co/datasets/demelin/moral_stories/resolve/main/data/moral_stories_full.jsonl"
    print(f"  Trying direct download: {url}")

    try:
        # Create SSL context that doesn't verify (common macOS Python issue)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "VISTA/1.0"})
        with urllib.request.urlopen(req, context=ctx) as response:
            with open(local_path, "wb") as f:
                f.write(response.read())
        print(f"  Downloaded via urllib: {local_path}")
        return local_path
    except Exception as e2:
        print(f"  Direct download failed: {e2}")

    raise RuntimeError(
        "Could not download Moral Stories. Please download manually:\n"
        "  1. Visit: https://huggingface.co/datasets/demelin/moral_stories/tree/main/data\n"
        "  2. Download 'moral_stories_full.jsonl'\n"
        f"  3. Place it at: {local_path}"
    )


def load_moral_stories(split: str = "train") -> list[dict]:
    """Load Moral Stories dataset from the JSONL file.

    Each story has 7 fields:
      norm, situation, intention, moral_action, moral_consequence,
      immoral_action, immoral_consequence

    Args:
        split: Dataset split (currently loads full dataset regardless of split).

    Returns:
        List of story dicts.
    """
    print(f"Loading Moral Stories ({split})...")

    jsonl_path = _download_moral_stories()

    stories = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                story = {
                    "norm": item.get("norm", ""),
                    "situation": item.get("situation", ""),
                    "intention": item.get("intention", ""),
                    "moral_action": item.get("moral_action", ""),
                    "moral_consequence": item.get("moral_consequence", ""),
                    "immoral_action": item.get("immoral_action", ""),
                    "immoral_consequence": item.get("immoral_consequence", ""),
                }
                stories.append(story)
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(stories)} stories")
    return stories


def build_scenarios(stories: list[dict]) -> list[dict]:
    """Convert raw stories into scenario tuples for VISTA.

    A scenario consists of:
      - context: the situation + intention (what's happening)
      - candidates: the moral and immoral actions (choices available)
      - metadata: norm, consequences, and story index

    Args:
        stories: List of story dicts from load_moral_stories().

    Returns:
        List of scenario dicts.
    """
    scenarios = []

    for idx, story in enumerate(stories):
        if not story["situation"] or not story["moral_action"] or not story["immoral_action"]:
            continue

        scenario = {
            "id": idx,
            "context": f"{story['situation']} {story['intention']}".strip(),
            "situation": story["situation"],
            "intention": story["intention"],
            "norm": story["norm"],
            "candidates": [
                {
                    "action_text": story["moral_action"],
                    "label": "moral",
                    "consequence": story["moral_consequence"],
                },
                {
                    "action_text": story["immoral_action"],
                    "label": "immoral",
                    "consequence": story["immoral_consequence"],
                },
            ],
        }
        scenarios.append(scenario)

    print(f"  Built {len(scenarios)} scenarios from {len(stories)} stories")
    return scenarios


def sample_scenarios(
    scenarios: list[dict],
    n: int = 100,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Sample n diverse scenarios.

    Args:
        scenarios: Full list of scenarios.
        n: Number of scenarios to sample.
        seed: Random seed for reproducibility.

    Returns:
        Sampled list of scenario dicts.
    """
    random.seed(seed)
    n = min(n, len(scenarios))
    sampled = random.sample(scenarios, n)
    print(f"  Sampled {len(sampled)} scenarios (seed={seed})")
    return sampled


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Moral Stories Loader Smoke Test")
    print("=" * 60)

    stories = load_moral_stories("train")
    scenarios = build_scenarios(stories)

    if scenarios:
        s = scenarios[0]
        print(f"\nSample Scenario (id={s['id']}):")
        print(f"  Norm: {s['norm']}")
        print(f"  Context: {s['context'][:100]}...")
        print(f"  Moral action: {s['candidates'][0]['action_text'][:80]}...")
        print(f"  Immoral action: {s['candidates'][1]['action_text'][:80]}...")

    sampled = sample_scenarios(scenarios, n=5)
    print(f"\n  Sampled {len(sampled)} scenarios for testing")

    print("\n✅ Moral Stories loader smoke test passed!")

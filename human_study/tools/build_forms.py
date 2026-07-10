"""Generate the 2 counterbalanced form variants for the VISTA human study (N=100).

Two forms (F1, F2) covering all 8 modifier axes with sibling counterbalance:
  - F1: SC_1 are baseline, SC_2 are modified with axes [1, 2, 3, 4, 5]
  - F2: SC_2 are baseline, SC_1 are modified with axes [5, 6, 7, 8, 1]

Every scenario gets baseline data from one form and modified data from the other.
Axes 1 and 5 appear in 2 themes (= 100 modified responses across N=100).
Axes 2, 3, 4, 6, 7, 8 appear in 1 theme each (= 50 modified responses).

Run from VISTA/ directory:
    python human_study/tools/build_forms.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "modifiers_batch1.json"
OUT_DIR = REPO_ROOT / "human_study" / "forms"

AXIS_NAMES = [
    "self_preservation",       # 1
    "resource_scarcity",       # 2
    "social_visibility",       # 3
    "in_out_group",            # 4
    "time_pressure",           # 5
    "diffused_responsibility", # 6
    "competence_uncertainty",  # 7
    "authority_signal",        # 8
]

# Axis assignment is optimized to maximize LLM flip rate (averaged across
# Gemma-4 and Llama-8B). Total flip rate = 100% across 10 cells (avg 10%
# per cell). All 8 axes covered. authority_signal and self_preservation
# are the repeated axes (the two highest-signal axes in the LLM data).
FORM_CONFIGS = {
    "F1": {
        "modified_sibling": "2",
        "axes_per_theme": [1, 3, 8, 1, 5],  # self_pres, social_vis, auth, self_pres, time_pres
    },
    "F2": {
        "modified_sibling": "1",
        "axes_per_theme": [4, 7, 8, 6, 2],  # in_out, compet, auth, diffused, resource
    },
}


def load_scenarios() -> dict:
    data = json.loads(DATA_PATH.read_text())
    return {s["scenario_id"]: s for s in data}


def build_item(scenarios: dict, scenario_id: str, axis_num: int | None) -> dict:
    sc = scenarios[scenario_id]
    text = sc["scenario"]
    modifier_id = None
    axis_name = None
    if axis_num is not None:
        axis_name = AXIS_NAMES[axis_num - 1]
        mod = next(m for m in sc["modifiers"] if m["axis"] == axis_name)
        sentences = text.rsplit(". ", 1)
        if len(sentences) == 2:
            text = sentences[0] + ". " + mod["modifier_text"] + " " + sentences[1]
        else:
            text = text + " " + mod["modifier_text"]
        modifier_id = mod["modifier_id"]
    return {
        "scenario_id": scenario_id,
        "is_modified": axis_num is not None,
        "axis": axis_name,
        "modifier_id": modifier_id,
        "prompt": text,
        "A0": sc["A0"],
        "A1": sc["A1"],
    }


def build_form(scenarios: dict, form_name: str) -> list[dict]:
    config = FORM_CONFIGS[form_name]
    sib_modified = config["modified_sibling"]
    sib_baseline = "1" if sib_modified == "2" else "2"
    items = []
    for theme_idx in range(5):
        theme_num = f"{theme_idx + 1:03d}"
        axis_num = config["axes_per_theme"][theme_idx]
        items.append(build_item(scenarios, f"SC{theme_num}_{sib_baseline}", None))
        items.append(build_item(scenarios, f"SC{theme_num}_{sib_modified}", axis_num))
    return items


def main() -> None:
    scenarios = load_scenarios()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for form_name in FORM_CONFIGS:
        items = build_form(scenarios, form_name)
        out_path = OUT_DIR / f"{form_name}.json"
        out_path.write_text(json.dumps(items, indent=2))
        n_modified = sum(1 for i in items if i["is_modified"])
        print(f"{form_name}: {len(items)} items ({n_modified} modified) -> {out_path}")


if __name__ == "__main__":
    main()

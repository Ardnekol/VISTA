"""Generate synthetic response data for the VISTA human study.

Lets you test the prepare/analyze pipeline before any real participant submits.
Builds in a plausible modifier effect so the analysis recovers a signal.

Outputs (in human_study/synthetic/):
  participants.csv  - one row per participant (demographics + PVQ + SDS + timing)
  decisions.csv     - one row per (participant, decision item)

Run from VISTA/ directory:
    python human_study/tools/simulate_responses.py
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMS_DIR = REPO_ROOT / "human_study" / "forms"
OUT_DIR = REPO_ROOT / "human_study" / "synthetic"

N_PER_FORM = 50  # 100 participants total across 2 forms
RANDOM_SEED = 7
FORMS = ["F1", "F2"]
DOMAINS = ["CS", "software", "medical"]

VALUES = [
    "self_direction", "power", "universalism", "achievement", "security",
    "stimulation", "conformity", "tradition", "benevolence", "hedonism",
]

# Items -> value mapping (PVQ-21 ESS short form)
PVQ_TO_VALUE = {
    1: "self_direction", 2: "power", 3: "universalism", 4: "achievement",
    5: "security", 6: "stimulation", 7: "conformity", 8: "universalism",
    9: "tradition", 10: "hedonism", 11: "self_direction", 12: "benevolence",
    13: "achievement", 14: "security", 15: "stimulation", 16: "conformity",
    17: "power", 18: "benevolence", 19: "universalism", 20: "tradition",
    21: "hedonism",
}

SDS_KEYED_TRUE = {1, 2, 3, 4, 5}   # answering True = faking-good
SDS_KEYED_FALSE = {6, 7, 8, 9, 10}  # answering False = faking-good


def make_value_profile(rng: random.Random) -> dict[str, float]:
    """A participant's latent value scores (centered around 0, plausible spread)."""
    return {v: rng.gauss(0, 1.0) for v in VALUES}


def likert_from_latent(latent_score: float, rng: random.Random) -> int:
    """Convert a latent z-score for a value into a 1-6 Likert response."""
    raw = 3.5 + latent_score + rng.gauss(0, 0.6)
    return max(1, min(6, round(raw)))


def simulate_participant(rng: random.Random, form_version: str, domain: str) -> tuple[dict, list[dict]]:
    profile = make_value_profile(rng)
    participant_id = f"P_{form_version}_{rng.randint(10000, 99999)}"

    # PVQ-21 responses: each item is a noisy reflection of the mapped value
    pvq = {}
    for item_num, value_name in PVQ_TO_VALUE.items():
        pvq[f"pvq_a{item_num}"] = likert_from_latent(profile[value_name], rng)

    # SDS-10 responses: random with mild bias toward faking
    fake_tendency = rng.uniform(0, 0.6)
    sds = {}
    for i in range(1, 11):
        keyed_answer = (i in SDS_KEYED_TRUE)  # True if keyed=True, else keyed=False
        gave_keyed = rng.random() < (0.3 + fake_tendency)
        actual = keyed_answer if gave_keyed else (not keyed_answer)
        sds[f"sds_c{i}"] = "True" if actual else "False"
    sds_score = sum(
        1 for i in range(1, 11)
        if (sds[f"sds_c{i}"] == "True" and i in SDS_KEYED_TRUE)
        or (sds[f"sds_c{i}"] == "False" and i in SDS_KEYED_FALSE)
    )

    # Attention checks: most pass, occasional fail
    attn_1 = "pass" if rng.random() > 0.05 else "fail"
    attn_2 = "pass" if rng.random() > 0.05 else "fail"

    # Duration: most are 8-15 min, some too fast, some too slow
    duration_seconds = max(60, int(rng.gauss(720, 240)))

    participant = {
        "participant_id": participant_id,
        "form_version": form_version,
        "domain": domain,
        "age_band": rng.choice(["18-24", "25-34", "35-44", "45+"]),
        "gender": rng.choice(["M", "F", "other", "na"]),
        "years_experience": rng.choice(["<1", "1-3", "4-7", "8+"]),
        **pvq,
        **sds,
        "sds_score": sds_score,
        "attn_1": attn_1,
        "attn_2": attn_2,
        "duration_seconds": duration_seconds,
    }

    # Decision responses for this form's items
    form_items = json.loads((FORMS_DIR / f"{form_version}.json").read_text())
    decisions = []
    for item in form_items:
        # Latent propensity to choose A1 depends on the participant's value profile
        # and (synthetically) on whether the modifier is present
        # We pick a "value direction" per scenario randomly to simulate that some
        # values push toward A0 and some toward A1.
        scenario_seed = hash(item["scenario_id"]) % 10000
        rng_sc = random.Random(scenario_seed)
        value_weights = {v: rng_sc.gauss(0, 0.3) for v in VALUES}
        latent = sum(profile[v] * value_weights[v] for v in VALUES)
        if item["is_modified"]:
            # Synthetic modifier effect: shifts choice by 0.6 in log-odds
            # Direction depends on axis (random but deterministic)
            axis_seed = hash(item["axis"]) % 10000
            axis_direction = 1 if (axis_seed % 2 == 0) else -1
            latent += 0.6 * axis_direction

        prob_a1 = 1 / (1 + 2.71828 ** -latent)
        choice = 1 if rng.random() < prob_a1 else 0
        confidence = rng.choice([2, 3, 4, 5])

        decisions.append({
            "participant_id": participant_id,
            "form_version": form_version,
            "scenario_id": item["scenario_id"],
            "is_modified": int(item["is_modified"]),
            "axis": item["axis"] or "",
            "modifier_id": item["modifier_id"] or "",
            "choice": choice,
            "confidence": confidence,
        })

    return participant, decisions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_participants: list[dict] = []
    all_decisions: list[dict] = []

    for form_version in FORMS:
        for _ in range(N_PER_FORM):
            # Distribute domains roughly evenly across each form
            domain = DOMAINS[(len(all_participants)) % 3]
            p, d = simulate_participant(rng, form_version, domain)
            all_participants.append(p)
            all_decisions.extend(d)

    write_csv(OUT_DIR / "participants.csv", all_participants)
    write_csv(OUT_DIR / "decisions.csv", all_decisions)
    print(f"Wrote {len(all_participants)} participants -> {OUT_DIR}/participants.csv")
    print(f"Wrote {len(all_decisions)} decisions -> {OUT_DIR}/decisions.csv")


if __name__ == "__main__":
    main()

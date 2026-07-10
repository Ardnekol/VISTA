"""Prepare VISTA human-study data for mixed-effects analysis.

Reads:  participants.csv + decisions.csv (from synthetic/ or responses/)
Writes: analysis_ready.csv (one row per decision, with value scores and exclusion flags)

Steps:
  1. Compute PVQ-21 value scores (raw means)
  2. Center each participant's value scores by their personal grand mean (ipsatization)
  3. Apply exclusion rules (attention checks, timing, SDS-10 score)
  4. Join participant-level value scores onto each decision row
  5. Write a tidy long-format CSV ready for R/lme4 or Python

Run:
    python human_study/tools/prepare_data.py              # uses synthetic data
    python human_study/tools/prepare_data.py --real       # uses responses/
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PVQ_TO_VALUE = {
    1: "self_direction", 2: "power", 3: "universalism", 4: "achievement",
    5: "security", 6: "stimulation", 7: "conformity", 8: "universalism",
    9: "tradition", 10: "hedonism", 11: "self_direction", 12: "benevolence",
    13: "achievement", 14: "security", 15: "stimulation", 16: "conformity",
    17: "power", 18: "benevolence", 19: "universalism", 20: "tradition",
    21: "hedonism",
}
VALUES = sorted(set(PVQ_TO_VALUE.values()))

# Exclusion rules — pre-register these
MIN_DURATION_SECONDS = 240   # < 4 min = too fast
MAX_SDS_SCORE = 7            # SDS >= 8 = high faking
ATTN_PASS_REQUIRED = True    # both attention checks must pass


def compute_value_scores(participant: dict) -> dict[str, float]:
    """Raw means per value across the 2-3 PVQ items that load on it."""
    per_value_items: dict[str, list[int]] = {v: [] for v in VALUES}
    for item_num, value_name in PVQ_TO_VALUE.items():
        raw = participant.get(f"pvq_a{item_num}", "")
        if raw == "" or raw is None:
            continue
        per_value_items[value_name].append(int(raw))
    means = {}
    for v in VALUES:
        vals = per_value_items[v]
        means[v] = sum(vals) / len(vals) if vals else float("nan")
    return means


def center_value_scores(raw_means: dict[str, float]) -> dict[str, float]:
    """Subtract participant's grand mean across all 21 items (Schwartz ipsatization)."""
    finite = [v for v in raw_means.values() if v == v]  # filter NaN
    if not finite:
        return {k: float("nan") for k in raw_means}
    # Grand mean across raw items (weighted by # items per value)
    total = 0.0
    count = 0
    for value_name, mean_score in raw_means.items():
        n_items = sum(1 for v in PVQ_TO_VALUE.values() if v == value_name)
        total += mean_score * n_items
        count += n_items
    grand_mean = total / count
    return {k: v - grand_mean for k, v in raw_means.items()}


def decide_exclusion(participant: dict) -> tuple[bool, str]:
    """Returns (excluded?, reason)."""
    reasons = []
    if int(participant.get("duration_seconds", 0)) < MIN_DURATION_SECONDS:
        reasons.append("too_fast")
    if int(participant.get("sds_score", 0)) > MAX_SDS_SCORE:
        reasons.append("high_sds")
    if ATTN_PASS_REQUIRED:
        if participant.get("attn_1") != "pass" or participant.get("attn_2") != "pass":
            reasons.append("failed_attention")
    return (len(reasons) > 0, ",".join(reasons))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Read from responses/ instead of synthetic/")
    args = parser.parse_args()

    in_dir = REPO_ROOT / "human_study" / ("responses" if args.real else "synthetic")
    participants_csv = in_dir / "participants.csv"
    decisions_csv = in_dir / "decisions.csv"
    out_csv = REPO_ROOT / "human_study" / "analysis_ready.csv"

    if not participants_csv.exists() or not decisions_csv.exists():
        print(f"Missing {participants_csv} or {decisions_csv}")
        print("Run simulate_responses.py first, or place real CSVs in responses/")
        return

    participants = list(csv.DictReader(participants_csv.open()))
    decisions = list(csv.DictReader(decisions_csv.open()))

    # Score each participant
    participant_index: dict[str, dict] = {}
    for p in participants:
        raw_means = compute_value_scores(p)
        centered = center_value_scores(raw_means)
        excluded, reason = decide_exclusion(p)
        participant_index[p["participant_id"]] = {
            **p,
            **{f"raw_{v}": raw_means[v] for v in VALUES},
            **{f"centered_{v}": centered[v] for v in VALUES},
            "excluded": int(excluded),
            "exclusion_reason": reason,
        }

    # Join onto each decision row
    out_rows = []
    for d in decisions:
        p = participant_index.get(d["participant_id"])
        if p is None:
            continue
        out_rows.append({
            "participant_id": d["participant_id"],
            "form_version": p["form_version"],
            "domain": p["domain"],
            "scenario_id": d["scenario_id"],
            "theme_id": d["scenario_id"][:5],  # 'SC001' from 'SC001_1'
            "is_modified": int(d["is_modified"]),
            "axis": d["axis"],
            "choice": int(d["choice"]),
            "confidence": int(d["confidence"]),
            "excluded": p["excluded"],
            "exclusion_reason": p["exclusion_reason"],
            "sds_score": int(p["sds_score"]),
            **{f"centered_{v}": p[f"centered_{v}"] for v in VALUES},
        })

    if not out_rows:
        print("No rows to write.")
        return
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Summary
    n_total = len(participants)
    n_excluded = sum(1 for p in participant_index.values() if p["excluded"])
    n_kept = n_total - n_excluded
    print(f"Wrote {len(out_rows)} rows -> {out_csv}")
    print(f"Participants total / kept / excluded: {n_total} / {n_kept} / {n_excluded}")
    if n_excluded > 0:
        from collections import Counter
        reasons = Counter()
        for p in participant_index.values():
            if p["excluded"]:
                for r in p["exclusion_reason"].split(","):
                    reasons[r] += 1
        print("Exclusion reasons:", dict(reasons))


if __name__ == "__main__":
    main()

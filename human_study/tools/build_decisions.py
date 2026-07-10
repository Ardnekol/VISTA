"""Extract per-(participant, decision item) responses from the two Google Forms CSVs.

Reads:
  human_study/Survey on Human Values-1 (Responses) - Form Responses 1.csv
  human_study/Survey on Human Values-2 (Responses) - Form Responses 1.csv
  human_study/forms/F1.json
  human_study/forms/F2.json
  human_study/human_binary_profiles.csv   (to filter to retained N=47)

Writes:
  human_study/decisions.csv

One row per (participant, decision item). Schema:
  participant_id, form, item_index, scenario_id, theme_id,
  is_modified, axis, modifier_id, choice (0/1), raw_response
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F1_CSV = ROOT / "Survey on Human Values-1 (Responses) - Form Responses 1.csv"
F2_CSV = ROOT / "Survey on Human Values-2 (Responses) - Form Responses 1.csv"
F1_JSON = ROOT / "forms" / "F1.json"
F2_JSON = ROOT / "forms" / "F2.json"
PROFILES_CSV = ROOT / "human_binary_profiles.csv"
OUT_CSV = ROOT / "decisions.csv"

DECISION_COL_START = 44
N_DECISION_ITEMS = 10


def extract_form(csv_path: Path, json_path: Path, form_tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    items = json.loads(json_path.read_text())
    assert len(items) == N_DECISION_ITEMS

    rows = []
    for row_idx, r in df.iterrows():
        pid = f"P_{form_tag}_{row_idx+1:03d}"
        for i in range(N_DECISION_ITEMS):
            item = items[i]
            raw = r.iloc[DECISION_COL_START + i]
            if pd.isna(raw):
                choice = None
            else:
                raw_s = str(raw).strip()
                if raw_s == item["A0"].strip():
                    choice = 0
                elif raw_s == item["A1"].strip():
                    choice = 1
                else:
                    choice = None  # unrecognized response text

            # Derive theme_id from scenario_id (SC001_1 -> SC001 -> TH001)
            sc = item["scenario_id"]
            theme_id = "TH" + sc[2:5]

            rows.append({
                "participant_id": pid,
                "form": form_tag,
                "item_index": i,
                "scenario_id": sc,
                "theme_id": theme_id,
                "is_modified": int(item["is_modified"]),
                "axis": item["axis"] if item["axis"] else "",
                "modifier_id": item["modifier_id"] if item["modifier_id"] else "",
                "choice": choice,
                "raw_response": "" if pd.isna(raw) else str(raw).strip(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    f1 = extract_form(F1_CSV, F1_JSON, "F1")
    f2 = extract_form(F2_CSV, F2_JSON, "F2")
    all_dec = pd.concat([f1, f2], ignore_index=True)
    print(f"raw decisions extracted: {len(all_dec)} rows  (= N_raw * 10 items)")

    # Filter to the 47 retained participants
    retained = set(pd.read_csv(PROFILES_CSV)["participant_id"])
    print(f"retained participants in human_binary_profiles.csv: {len(retained)}")
    kept = all_dec[all_dec["participant_id"].isin(retained)].reset_index(drop=True)
    print(f"decisions after filtering to retained participants: {len(kept)} rows")
    print(f"  = {len(kept)//10} participants × 10 items (sanity check)")

    # Sanity / quality stats
    print()
    print("=== Sanity checks ===")
    unrecognized = kept[kept["choice"].isna() & (kept["raw_response"] != "")]
    print(f"  unrecognized response texts (couldn't map to A0/A1): {len(unrecognized)}")
    missing = kept[kept["raw_response"] == ""]
    print(f"  missing responses (blank): {len(missing)}")
    by_form = kept.groupby(["form", "is_modified"]).size().unstack(fill_value=0)
    print(f"\n  rows per (form, is_modified):")
    print(by_form.to_string())

    # Per-axis modifier-trial counts (the cells we'll analyze)
    print()
    print("=== Modified-trial counts per (scenario, axis) ===")
    mods = kept[kept["is_modified"] == 1]
    cell_counts = mods.groupby(["scenario_id", "axis"]).size().sort_index()
    print(cell_counts.to_string())

    kept.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

"""Build per-participant Schwartz value profiles from the two Google Forms CSVs.

Reads:  human_study/Survey on Human Values-1 (Responses) - Form Responses 1.csv
        human_study/Survey on Human Values-2 (Responses) - Form Responses 1.csv

Writes: human_study/value_profiles.csv  (one row per participant)

Columns: participant_id, form, timestamp, profession, gender, age_group,
         a1..a21 (raw 1-6 Likert), ac1_pass, ac2_pass, sds_score,
         raw_<value> (x10), centered_<value> (x10), binary_<value> (x10),
         grand_mean
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F1_PATH = ROOT / "Survey on Human Values-1 (Responses) - Form Responses 1.csv"
F2_PATH = ROOT / "Survey on Human Values-2 (Responses) - Form Responses 1.csv"
OUT_PATH = ROOT / "value_profiles.csv"

LIKERT_MAP = {
    "Not like me at all": 1,
    "Not like me": 2,
    "A little like me": 3,
    "Somewhat like me": 4,
    "Like me": 5,
    "Very much like me": 6,
}

PVQ_COL_IDX = {
    1:  5,  2:  6,  3:  7,  4:  8,  5:  9,  6: 10,  7: 11,
    # col 12 = AC1
    8: 13,  9: 14, 10: 15, 11: 16, 12: 17, 13: 18, 14: 19,
    15: 20, 16: 21, 17: 22,
    # col 23 = AC2
    18: 24, 19: 25, 20: 26, 21: 27,
}
AC1_COL_IDX = 12
AC2_COL_IDX = 23
AC1_EXPECTED = "A little like me"
AC2_EXPECTED = "Like me"

PVQ_TO_VALUE = {
    1: "self_direction", 2: "power", 3: "universalism", 4: "achievement",
    5: "security", 6: "stimulation", 7: "conformity",
    8: "universalism", 9: "tradition", 10: "hedonism", 11: "self_direction",
    12: "benevolence", 13: "achievement", 14: "security", 15: "stimulation",
    16: "conformity", 17: "power", 18: "benevolence", 19: "universalism",
    20: "tradition", 21: "hedonism",
}
VALUES = sorted(set(PVQ_TO_VALUE.values()))

SDS_COL_RANGE = (34, 44)   # cols 34..43 inclusive
SDS_FAKING_GOOD = {
    34: "True",   # "I always admit it when I make a mistake."
    35: "True",   # "I always do what I tell others to do."
    36: "True",
    37: "True",
    38: "True",
    39: "False",
    40: "False",
    41: "False",
    42: "False",
    43: "False",
}


def process_form(path: Path, form_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, row in df.iterrows():
        pid = f"P_{form_tag}_{i+1:03d}"
        record = {
            "participant_id": pid,
            "form": form_tag,
            "timestamp": row.iloc[0],
            "profession": row.iloc[2],
            "gender": row.iloc[3],
            "age_group": row.iloc[4],
        }

        # PVQ-21 → numeric
        pvq_numeric = {}
        for item_num, col_idx in PVQ_COL_IDX.items():
            raw = row.iloc[col_idx]
            if pd.isna(raw) or raw not in LIKERT_MAP:
                pvq_numeric[item_num] = None
            else:
                pvq_numeric[item_num] = LIKERT_MAP[raw]
            record[f"a{item_num}"] = pvq_numeric[item_num]

        # Attention checks
        ac1 = row.iloc[AC1_COL_IDX]
        ac2 = row.iloc[AC2_COL_IDX]
        record["ac1_pass"] = int(str(ac1).strip() == AC1_EXPECTED)
        record["ac2_pass"] = int(str(ac2).strip() == AC2_EXPECTED)

        # SDS-10 score (number of "faking-good" answers)
        sds = 0
        for col_idx in range(*SDS_COL_RANGE):
            ans = row.iloc[col_idx]
            if pd.isna(ans):
                continue
            if str(ans).strip() == SDS_FAKING_GOOD[col_idx]:
                sds += 1
        record["sds_score"] = sds

        # Raw value scores (mean of items per value)
        per_value = {v: [] for v in VALUES}
        for item_num, val_name in PVQ_TO_VALUE.items():
            if pvq_numeric[item_num] is not None:
                per_value[val_name].append(pvq_numeric[item_num])
        raw_scores = {
            v: (sum(xs)/len(xs) if xs else None)
            for v, xs in per_value.items()
        }
        for v, score in raw_scores.items():
            record[f"raw_{v}"] = score

        # Grand mean across all 21 items (Schwartz ipsatization base)
        valid_items = [pvq_numeric[i] for i in PVQ_COL_IDX if pvq_numeric[i] is not None]
        if valid_items:
            grand_mean = sum(valid_items) / len(valid_items)
        else:
            grand_mean = None
        record["grand_mean"] = grand_mean

        # Centered (ipsatized) scores
        for v in VALUES:
            r = raw_scores[v]
            record[f"centered_{v}"] = (r - grand_mean) if (r is not None and grand_mean is not None) else None

        # Binary profile (1 if centered > 0 else 0)
        for v in VALUES:
            c = record[f"centered_{v}"]
            record[f"binary_{v}"] = (1 if (c is not None and c > 0) else 0) if c is not None else None

        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    f1 = process_form(F1_PATH, "F1")
    f2 = process_form(F2_PATH, "F2")
    out = pd.concat([f1, f2], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")
    print(f"  participants: F1={len(f1)}, F2={len(f2)}, total={len(out)}")
    print(f"  AC1 pass rate: {out['ac1_pass'].mean()*100:.1f}%")
    print(f"  AC2 pass rate: {out['ac2_pass'].mean()*100:.1f}%")
    print(f"  SDS score distribution:")
    print(out["sds_score"].value_counts().sort_index().to_string())
    print(f"  Mean raw value scores:")
    for v in VALUES:
        print(f"    {v:18s}: {out[f'raw_{v}'].mean():.2f}")


if __name__ == "__main__":
    main()

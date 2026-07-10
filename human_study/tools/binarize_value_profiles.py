"""Convert continuous PVQ value scores into 10-bit Schwartz binary profiles.

Reads:  human_study/value_profiles.csv  (has centered_<value> columns)
        Dataset/value_profiles_batch1.json  (95 retained LLM vsw_ids for matching)

Writes: human_study/human_binary_profiles.csv

Binarization rule (per instruments.md, Schwartz ipsatization):
  centered_<value> = raw_<value> - participant_grand_mean
  binary_<value> = 1 if centered_<value> > 0 else 0

The 10-bit string uses the SAME value ordering as the LLM Dataset:
  [Self-Direction, Stimulation, Hedonism, Achievement, Power,
   Security, Conformity, Tradition, Benevolence, Universalism]
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "value_profiles.csv"
LLM_JSON = ROOT.parent / "Dataset" / "value_profiles_batch1.json"
OUT_CSV = ROOT / "human_binary_profiles.csv"

# Must match the LLM 10-bit ordering exactly
VALUE_ORDER = [
    "self_direction", "stimulation", "hedonism", "achievement", "power",
    "security", "conformity", "tradition", "benevolence", "universalism",
]

# Pretty name (matches LLM Dataset key)
PRETTY = {
    "self_direction": "Self-Direction", "stimulation": "Stimulation",
    "hedonism": "Hedonism", "achievement": "Achievement", "power": "Power",
    "security": "Security", "conformity": "Conformity",
    "tradition": "Tradition", "benevolence": "Benevolence",
    "universalism": "Universalism",
}


def main() -> None:
    df = pd.read_csv(IN_CSV)
    llm_profiles = json.loads(LLM_JSON.read_text())
    # Lookup: binary_profile string -> vsw_id
    bin_to_vsw = {p["binary_profile"]: p["vsw_id"] for p in llm_profiles}

    out_rows = []
    for _, r in df.iterrows():
        # Re-binarize from centered scores (more robust than reading existing binary cols)
        bits = []
        valid = True
        for v in VALUE_ORDER:
            c = r.get(f"centered_{v}")
            if pd.isna(c):
                valid = False
                bits.append(None)
            else:
                bits.append(1 if c > 0 else 0)

        if not valid:
            binary_profile = ""
            active_count = None
            vsw_id_match = ""
        else:
            binary_profile = "".join(str(b) for b in bits)
            active_count = sum(bits)
            vsw_id_match = bin_to_vsw.get(binary_profile, "")  # empty if profile pruned

        # Per user decision: keep ALL participants. AC/SDS columns remain
        # as informational only. Only flag rows with no usable PVQ data.
        exclusion_flags = []
        if r["ac1_pass"] == 0:
            exclusion_flags.append("failed_ac1")
        if r["ac2_pass"] == 0:
            exclusion_flags.append("failed_ac2")
        if r["sds_score"] >= 8:
            exclusion_flags.append("high_sds")
        excluded = 0  # do not exclude anyone

        out = {
            "participant_id": r["participant_id"],
            "form": r["form"],
            "profession": r["profession"],
            "gender": r["gender"],
            "age_group": r["age_group"],
            "binary_profile": binary_profile,
            "active_values_count": active_count,
            "matched_vsw_id": vsw_id_match,  # empty if profile is one of the 5 pruned (all-0, all-1, antagonism)
            "excluded": excluded,
            "exclusion_reason": ",".join(exclusion_flags) if exclusion_flags else "",
            "ac1_pass": r["ac1_pass"],
            "ac2_pass": r["ac2_pass"],
            "sds_score": r["sds_score"],
        }
        # 10 binary value columns (LLM order)
        for i, v in enumerate(VALUE_ORDER):
            out[PRETTY[v]] = bits[i] if valid else None
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    # Per user decision: do not exclude any human data.
    # Participants without computable binary profile (if any) keep empty profile cells.
    before = len(out_df)
    missing_profile = (out_df["binary_profile"] == "").sum()

    out_df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(f"  participants collected: {before}")
    print(f"  participants with empty binary_profile (kept anyway): {missing_profile}")
    print(f"  N for analyses: {len(out_df)}")
    print(f"  matched a retained LLM vsw_id: {(out_df['matched_vsw_id']!='').sum()}")
    print(f"  unmatched (profile was pruned from the 95): {(out_df['matched_vsw_id']=='').sum()}")
    print()
    print("Active-values-count distribution:")
    print(out_df["active_values_count"].value_counts(dropna=False).sort_index().to_string())
    print()
    print("Sample (first 5 retained):")
    keep = out_df[out_df["excluded"]==0].head(5)
    print(keep[["participant_id","binary_profile","active_values_count","matched_vsw_id"]].to_string(index=False))


if __name__ == "__main__":
    main()

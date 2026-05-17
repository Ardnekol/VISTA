"""
VISTA — Step 2: "Impossible flips" analysis.

For each LLM, count rows where:
  - dot-product baseline predicts NO flip
  - but the LLM DID flip

These flips cannot be explained by value-vector arithmetic alone.
This is direct evidence that modifiers contribute beyond
(scenario + value_profile).

Stratifies by profile_strength (0 HIGH -> 9 HIGH). Zooms in on
8 HIGH and 9 HIGH where dot-product is mathematically forced to 0%
flips, so any LLM flips there are "impossible" by construction.
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B": OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":  OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":   OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":  OUT_DIR / "master_llm_decisions_llama_8B.csv",
}

REPORT_PATH = OUT_DIR / "step2_impossible_flips_report.txt"
CSV_PATH    = OUT_DIR / "step2_impossible_flips_table.csv"


def load_modifier_rows(path):
    df = pd.read_csv(path)
    df = df[df["axis"] != "BASELINE"].copy()
    df["llm_flip"] = (df["llm_changed_from_baseline"] == "YES")
    df["dp_flip"]  = (df["dp_changed_from_baseline"]  == "YES")
    df["impossible_flip"] = (~df["dp_flip"]) & df["llm_flip"]
    return df


def per_model_stratification(df):
    """Return a table: rows = profile_strength, cols = counts/rates."""
    rows = []
    for k in sorted(df["profile_strength"].unique()):
        sub = df[df["profile_strength"] == k]
        n = len(sub)
        dp_no_flip = (~sub["dp_flip"]).sum()
        impossible = sub["impossible_flip"].sum()
        llm_flip   = sub["llm_flip"].sum()
        rows.append({
            "profile_strength": k,
            "n_rows": n,
            "n_dp_says_no_flip": int(dp_no_flip),
            "n_llm_flipped": int(llm_flip),
            "n_impossible_flips": int(impossible),
            "rate_impossible_over_all": impossible / n if n else 0.0,
            "rate_impossible_over_dp_no": (impossible / dp_no_flip) if dp_no_flip else 0.0,
        })
    return pd.DataFrame(rows)


def by_axis(df):
    rows = []
    for axis in sorted(df["axis"].unique()):
        sub = df[df["axis"] == axis]
        n = len(sub)
        impossible = sub["impossible_flip"].sum()
        rows.append({
            "axis": axis,
            "n_rows": n,
            "n_impossible_flips": int(impossible),
            "rate_impossible": impossible / n if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("rate_impossible", ascending=False)


def fmt_pct(x):
    return f"{100*x:5.1f}%"


def main():
    model_tables = {}
    model_overall = {}
    model_axis = {}

    for name, path in MODELS.items():
        df = load_modifier_rows(path)
        model_tables[name] = per_model_stratification(df)
        model_axis[name]   = by_axis(df)
        model_overall[name] = {
            "total_modifier_rows": len(df),
            "llm_flips": int(df["llm_flip"].sum()),
            "dp_flips":  int(df["dp_flip"].sum()),
            "dp_no_flip": int((~df["dp_flip"]).sum()),
            "impossible_flips": int(df["impossible_flip"].sum()),
        }

    # ---------- write text report ----------
    lines = []
    lines.append("=" * 72)
    lines.append("VISTA — STEP 2: IMPOSSIBLE FLIPS")
    lines.append("(LLM flipped on rows where dot-product predicts NO flip)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Interpretation: dot-product is a value-arithmetic model.")
    lines.append("Rows where dot-product predicts NO flip are rows where the")
    lines.append("value vector dominates the modifier vector. If the LLM still")
    lines.append("flips on those rows, the flip cannot be explained by the")
    lines.append("value profile alone -> direct evidence for the claim.")
    lines.append("")

    lines.append("-" * 72)
    lines.append("OVERALL COUNTS PER MODEL")
    lines.append("-" * 72)
    lines.append(f"{'Model':<18} {'rows':>6} {'llm_flips':>10} {'dp_flips':>10} "
                 f"{'dp_no_flip':>12} {'impossible':>12} {'imp/dp_no':>10}")
    for name, ov in model_overall.items():
        r = ov["impossible_flips"] / ov["dp_no_flip"] if ov["dp_no_flip"] else 0
        lines.append(f"{name:<18} {ov['total_modifier_rows']:>6} "
                     f"{ov['llm_flips']:>10} {ov['dp_flips']:>10} "
                     f"{ov['dp_no_flip']:>12} {ov['impossible_flips']:>12} "
                     f"{fmt_pct(r):>10}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("IMPOSSIBLE FLIPS BY PROFILE STRENGTH (per model)")
    lines.append("-" * 72)
    lines.append("Reported as: n_impossible / n_dp_says_no_flip  (rate)")
    lines.append("At 8H and 9H, dp_says_no_flip = 100% of rows by construction,")
    lines.append("so 'impossible_flip' rate equals the LLM flip rate at that cell.")
    lines.append("")

    header_models = list(MODELS.keys())
    header = f"{'HIGH':>4} " + " ".join(f"{m[:18]:>20}" for m in header_models)
    lines.append(header)
    lines.append("-" * len(header))

    strengths = sorted(model_tables[header_models[0]]["profile_strength"].unique())
    for k in strengths:
        row = [f"{k:>4}"]
        for m in header_models:
            t = model_tables[m]
            r = t[t["profile_strength"] == k].iloc[0]
            cell = f"{int(r['n_impossible_flips']):>3}/{int(r['n_dp_says_no_flip']):<3}({fmt_pct(r['rate_impossible_over_dp_no']).strip()})"
            row.append(f"{cell:>20}")
        lines.append(" ".join(row))
    lines.append("")

    lines.append("-" * 72)
    lines.append("FOCUS: STRONG PROFILES (8-9 HIGH) — DOT-PRODUCT FORCED 0% FLIPS")
    lines.append("-" * 72)
    lines.append("These rows are the most striking: under value-arithmetic,")
    lines.append("the value vector dominates and no single modifier should flip")
    lines.append("the decision. Any LLM flip here is 'impossible' by construction.")
    lines.append("")
    lines.append(f"{'Model':<18} {'8H imp':>10} {'8H n':>6} {'8H rate':>10}   "
                 f"{'9H imp':>10} {'9H n':>6} {'9H rate':>10}")
    for name in header_models:
        t = model_tables[name]
        r8 = t[t["profile_strength"] == 8].iloc[0]
        r9 = t[t["profile_strength"] == 9].iloc[0]
        lines.append(f"{name:<18} "
                     f"{int(r8['n_impossible_flips']):>10} {int(r8['n_dp_says_no_flip']):>6} "
                     f"{fmt_pct(r8['rate_impossible_over_dp_no']):>10}   "
                     f"{int(r9['n_impossible_flips']):>10} {int(r9['n_dp_says_no_flip']):>6} "
                     f"{fmt_pct(r9['rate_impossible_over_dp_no']):>10}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("IMPOSSIBLE FLIPS BY MODIFIER AXIS (per model, top axes)")
    lines.append("-" * 72)
    for name in header_models:
        lines.append(f"\n{name}:")
        t = model_axis[name].head(8)
        for _, r in t.iterrows():
            lines.append(f"  {r['axis']:<28} "
                         f"{int(r['n_impossible_flips']):>4}/{int(r['n_rows']):<4} "
                         f"({fmt_pct(r['rate_impossible'])})")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF STEP 2 REPORT")
    lines.append("=" * 72)

    REPORT_PATH.write_text("\n".join(lines))

    # ---------- consolidated CSV table ----------
    all_rows = []
    for name in header_models:
        t = model_tables[name].copy()
        t["model"] = name
        all_rows.append(t)
    consolidated = pd.concat(all_rows, ignore_index=True)
    consolidated.to_csv(CSV_PATH, index=False)

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote table:  {CSV_PATH}")


if __name__ == "__main__":
    main()

"""
VISTA — Step 5: Axis-ranking divergence (LLM vs dot-product).

Plain question:
  "Are LLMs just doing a noisy version of dot-product value arithmetic?"
  If yes, the 8 modifier axes should be ranked SIMILARLY by both.
  If no, the LLMs prioritise different axes -> they're using a different
  decision rule.

Method:
  - For each model (including dot-product), compute flip rate per axis.
  - Rank axes 1..8 by flip rate (1 = most flips).
  - Compute Spearman rank correlation between each LLM's ranking and the
    dot-product ranking.
  - Build side-by-side table of ranks.
  - Flag axes where LLM rank differs from dot-product rank by >= 3 positions.
"""

import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B":      OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":       OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":        OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":       OUT_DIR / "master_llm_decisions_llama_8B.csv",
    "Dot-Product (rule)": OUT_DIR / "master_llm_decisions_dotProduct.csv",
}

REPORT_PATH = OUT_DIR / "step5_axis_ranking_report.txt"
TABLE_CSV   = OUT_DIR / "step5_axis_ranking_table.csv"


def axis_flip_rates(path, is_dot_product=False):
    df = pd.read_csv(path)
    df = df[df["axis"] != "BASELINE"]
    if is_dot_product:
        flip_col = "dp_changed_from_baseline"
    else:
        flip_col = "llm_changed_from_baseline"
    df = df[df[flip_col].isin(["YES", "NO"])]
    grouped = df.groupby("axis").agg(
        n_rows=(flip_col, "size"),
        n_flips=(flip_col, lambda s: (s == "YES").sum()),
    )
    grouped["rate"] = grouped["n_flips"] / grouped["n_rows"]
    return grouped["rate"].to_dict()


def main():
    # 1. Get flip-rate-per-axis for every model
    rate_table = {}
    for name, path in MODELS.items():
        is_dp = "Dot-Product" in name
        rate_table[name] = axis_flip_rates(path, is_dot_product=is_dp)

    axes = sorted({a for d in rate_table.values() for a in d.keys()})

    # 2. Build rate dataframe (rows=axis, cols=model)
    rates_df = pd.DataFrame({m: [rate_table[m].get(a, float("nan")) for a in axes]
                              for m in MODELS}, index=axes)
    # 3. Build rank dataframe (1 = highest rate)
    ranks_df = rates_df.rank(ascending=False, method="min").astype(int)

    # 4. Spearman correlations vs dot-product
    dp_col = "Dot-Product (rule)"
    spearman_results = []
    llm_models = [m for m in MODELS if m != dp_col]
    for m in llm_models:
        rho, p = spearmanr(rates_df[m], rates_df[dp_col])
        spearman_results.append({"model": m, "spearman_rho_vs_DP": rho, "p": p})

    # 5. Pairwise Spearman across LLMs (for cross-model consistency preview)
    pairwise = []
    for i in range(len(llm_models)):
        for j in range(i + 1, len(llm_models)):
            m1, m2 = llm_models[i], llm_models[j]
            rho, p = spearmanr(rates_df[m1], rates_df[m2])
            pairwise.append({"model_a": m1, "model_b": m2, "rho": rho, "p": p})

    # 6. Rank-difference flags (LLM rank vs DP rank)
    diff_df = ranks_df[llm_models].sub(ranks_df[dp_col], axis=0)
    big_shifts = []
    for axis in axes:
        for m in llm_models:
            d = int(diff_df.loc[axis, m])
            if abs(d) >= 3:
                big_shifts.append({
                    "axis": axis, "model": m,
                    "dp_rank": int(ranks_df.loc[axis, dp_col]),
                    "llm_rank": int(ranks_df.loc[axis, m]),
                    "rank_shift": d,
                })

    # ---------- write text report ----------
    lines = []
    lines.append("=" * 92)
    lines.append("VISTA — STEP 5: AXIS-RANKING DIVERGENCE (LLM vs DOT-PRODUCT)")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Plain-English question:")
    lines.append("  Are LLMs just a noisy version of dot-product value arithmetic?")
    lines.append("  If yes -> they should rank the 8 modifier axes the same way.")
    lines.append("  If no  -> they're using a different decision rule.")
    lines.append("")

    lines.append("-" * 92)
    lines.append("FLIP RATES BY AXIS (per model)")
    lines.append("-" * 92)
    header = f"  {'axis':<26} " + " ".join(f"{m[:14]:>15}" for m in MODELS)
    lines.append(header)
    for axis in axes:
        row = " ".join(f"{rates_df.loc[axis, m]:>14.1%}" for m in MODELS)
        lines.append(f"  {axis:<26} {row}")
    lines.append("")

    lines.append("-" * 92)
    lines.append("AXIS RANKS (1 = highest flip rate, 8 = lowest)")
    lines.append("-" * 92)
    lines.append(header)
    for axis in axes:
        row = " ".join(f"{ranks_df.loc[axis, m]:>15d}" for m in MODELS)
        lines.append(f"  {axis:<26} {row}")
    lines.append("")

    lines.append("-" * 92)
    lines.append("SPEARMAN RANK CORRELATION: each LLM vs DOT-PRODUCT")
    lines.append("-" * 92)
    lines.append("rho = +1 means perfect agreement; rho = 0 means no relationship;")
    lines.append("rho < 0 means reversed ordering. n = 8 axes.")
    lines.append("")
    lines.append(f"  {'Model':<20} {'rho':>8} {'p':>10}")
    for r in spearman_results:
        lines.append(f"  {r['model']:<20} {r['spearman_rho_vs_DP']:>8.3f} {r['p']:>10.4f}")
    lines.append("")

    lines.append("-" * 92)
    lines.append("PAIRWISE LLM-vs-LLM SPEARMAN (preview of Step 6)")
    lines.append("-" * 92)
    lines.append("")
    for r in pairwise:
        lines.append(f"  {r['model_a']:<18} vs {r['model_b']:<18}  "
                     f"rho = {r['rho']:.3f}, p = {r['p']:.4f}")
    lines.append("")

    lines.append("-" * 92)
    lines.append("QUALITATIVE DISAGREEMENTS: |LLM rank - DP rank| >= 3")
    lines.append("-" * 92)
    lines.append("These are axes that LLMs and dot-product rank very differently.")
    lines.append("Positive 'rank_shift' = LLM ranks LOWER (less flippy) than DP.")
    lines.append("Negative 'rank_shift' = LLM ranks HIGHER (more flippy) than DP.")
    lines.append("")
    if big_shifts:
        lines.append(f"  {'axis':<26} {'model':<20} {'DP rank':>8} {'LLM rank':>10} {'shift':>8}")
        for r in sorted(big_shifts, key=lambda x: (x["axis"], x["model"])):
            lines.append(f"  {r['axis']:<26} {r['model']:<20} "
                         f"{r['dp_rank']:>8} {r['llm_rank']:>10} {r['rank_shift']:>+8d}")
    else:
        lines.append("  (none — every axis ranks within 2 positions of DP across models)")
    lines.append("")

    # Self-Preservation focus call-out
    lines.append("-" * 92)
    lines.append("SELF-PRESERVATION CALL-OUT")
    lines.append("-" * 92)
    sp_dp = ranks_df.loc["self_preservation", dp_col]
    lines.append(f"  Dot-product ranks self_preservation at: #{sp_dp}")
    for m in llm_models:
        lines.append(f"  {m} ranks self_preservation at: #{ranks_df.loc['self_preservation', m]}")
    lines.append("")
    lines.append("  If self_preservation jumps from a low rank in dot-product to a")
    lines.append("  top-2 rank in LLMs, that is direct evidence the LLM uses a")
    lines.append("  decision rule that is NOT value-arithmetic.")
    lines.append("")

    lines.append("=" * 92)
    lines.append("END OF STEP 5 REPORT")
    lines.append("=" * 92)

    REPORT_PATH.write_text("\n".join(lines))

    # ---------- consolidated CSV table ----------
    out = rates_df.copy()
    out.columns = [f"rate_{c}" for c in out.columns]
    rank_cols = ranks_df.copy()
    rank_cols.columns = [f"rank_{c}" for c in rank_cols.columns]
    consolidated = pd.concat([out, rank_cols], axis=1)
    consolidated.index.name = "axis"
    consolidated.to_csv(TABLE_CSV)

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote table:  {TABLE_CSV}")


if __name__ == "__main__":
    main()

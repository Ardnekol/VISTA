"""
VISTA — Step 6: Cross-model consistency of axis rankings.

Plain question:
  Do all 4 LLMs (from 3 different labs) agree on which modifiers matter
  most? If yes, the effect is a property of the STIMULUS and the social
  phenomenon, not of any one company's training pipeline.

Method:
  - Pairwise Spearman correlations across the 4 LLMs (6 pairs).
  - Kendall's W (coefficient of concordance) across all 4 LLMs together.
    W ranges 0 (no agreement) to 1 (perfect agreement). Chi-squared test
    gives a p-value.
  - Per-axis standard deviation of ranks (low SD = high agreement).
  - Consensus calls: axes that are top-2 in ALL 4 LLMs, bottom-2 in ALL 4.
  - Same metrics including dot-product, to confirm DP is the outlier.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, chi2

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B":      OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":       OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":        OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":       OUT_DIR / "master_llm_decisions_llama_8B.csv",
    "Haiku 4.5":          OUT_DIR / "master_llm_decisions_haiku.csv",
    "GPT-4.1-mini":       OUT_DIR / "master_llm_decisions_gpt41mini.csv",
    "GPT-5-mini":         OUT_DIR / "master_llm_decisions_gpt5mini.csv",
    "Sonnet 5":           OUT_DIR / "master_llm_decisions_sonnet.csv",
    "Dot-Product (rule)": OUT_DIR / "master_llm_decisions_dotProduct.csv",
}

REPORT_PATH = OUT_DIR / "step6_cross_model_consistency_report.txt"
MATRIX_CSV  = OUT_DIR / "step6_pairwise_spearman_matrix.csv"
PERAXIS_CSV = OUT_DIR / "step6_per_axis_agreement.csv"


def axis_flip_rates(path, is_dot_product=False):
    df = pd.read_csv(path)
    df = df[df["axis"] != "BASELINE"]
    flip_col = "dp_changed_from_baseline" if is_dot_product else "llm_changed_from_baseline"
    df = df[df[flip_col].isin(["YES", "NO"])]
    g = df.groupby("axis").agg(
        n_rows=(flip_col, "size"),
        n_flips=(flip_col, lambda s: (s == "YES").sum()),
    )
    g["rate"] = g["n_flips"] / g["n_rows"]
    return g["rate"].to_dict()


def kendalls_w(ranks_matrix):
    """
    ranks_matrix: numpy array of shape (n_items, n_raters).
    Returns (W, chi2_stat, df, p_value).
    Formula: W = 12 * S / (m^2 * (n^3 - n)) where m=n_raters, n=n_items,
    S = sum of squared deviations of rank totals from their mean.
    """
    n, m = ranks_matrix.shape
    R = ranks_matrix.sum(axis=1)  # total rank per item
    mean_R = R.mean()
    S = ((R - mean_R) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2_stat = m * (n - 1) * W
    df_val = n - 1
    p = chi2.sf(chi2_stat, df_val)
    return W, chi2_stat, df_val, p


def main():
    rate_table = {}
    for name, path in MODELS.items():
        is_dp = "Dot-Product" in name
        rate_table[name] = axis_flip_rates(path, is_dot_product=is_dp)

    axes = sorted({a for d in rate_table.values() for a in d.keys()})
    rates_df = pd.DataFrame({m: [rate_table[m].get(a, np.nan) for a in axes]
                              for m in MODELS}, index=axes)
    ranks_df = rates_df.rank(ascending=False, method="min").astype(int)

    llms = [m for m in MODELS if "Dot-Product" not in m]

    # ---------- pairwise spearman (LLMs only) ----------
    pw_rho = pd.DataFrame(index=llms, columns=llms, dtype=float)
    pw_p   = pd.DataFrame(index=llms, columns=llms, dtype=float)
    pairs_list = []
    for i, a in enumerate(llms):
        for j, b in enumerate(llms):
            if i == j:
                pw_rho.loc[a, b] = 1.0
                pw_p.loc[a, b] = 0.0
            elif j > i:
                rho, p = spearmanr(rates_df[a], rates_df[b])
                pw_rho.loc[a, b] = rho
                pw_rho.loc[b, a] = rho
                pw_p.loc[a, b] = p
                pw_p.loc[b, a] = p
                pairs_list.append({"model_a": a, "model_b": b, "rho": rho, "p": p})

    # mean pairwise rho (upper triangle only)
    triu_vals = []
    for i in range(len(llms)):
        for j in range(i + 1, len(llms)):
            triu_vals.append(pw_rho.iloc[i, j])
    mean_rho = np.mean(triu_vals)

    # ---------- Kendall's W across 4 LLMs ----------
    llm_ranks = ranks_df[llms].values  # shape (n_axes, n_llms)
    W_llms, chi2_llms, df_llms, p_llms = kendalls_w(llm_ranks)

    # Include dot-product (should reduce W if DP is an outlier)
    all_ranks = ranks_df[llms + ["Dot-Product (rule)"]].values
    W_all, chi2_all, df_all, p_all = kendalls_w(all_ranks)

    # ---------- per-axis agreement: SD of rank across LLMs ----------
    per_axis = pd.DataFrame({
        "mean_rank_LLMs": ranks_df[llms].mean(axis=1),
        "median_rank_LLMs": ranks_df[llms].median(axis=1),
        "sd_rank_LLMs": ranks_df[llms].std(axis=1),
        "min_rank_LLMs": ranks_df[llms].min(axis=1),
        "max_rank_LLMs": ranks_df[llms].max(axis=1),
        "dot_product_rank": ranks_df["Dot-Product (rule)"],
    })
    per_axis["dp_minus_llm_mean"] = per_axis["dot_product_rank"] - per_axis["mean_rank_LLMs"]
    per_axis = per_axis.sort_values("mean_rank_LLMs")

    # ---------- consensus calls ----------
    top2_all = [a for a in axes if all(ranks_df.loc[a, m] <= 2 for m in llms)]
    bot2_all = [a for a in axes if all(ranks_df.loc[a, m] >= 7 for m in llms)]

    # ---------- write report ----------
    lines = []
    lines.append("=" * 92)
    lines.append("VISTA — STEP 6: CROSS-MODEL CONSISTENCY (do 4 LLMs agree?)")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Plain-English question:")
    lines.append("  Do 4 LLMs from 3 different labs (Meta, Alibaba, Google)")
    lines.append("  rank the 8 modifiers similarly? If yes, the effect is a")
    lines.append("  property of the STIMULUS, not of any one model's training.")
    lines.append("")

    lines.append("-" * 92)
    lines.append("HEADLINE: KENDALL'S W (agreement coefficient across raters)")
    lines.append("-" * 92)
    lines.append("W ranges from 0 (no agreement) to 1 (perfect agreement).")
    lines.append("Chi-squared test asks: 'is the observed agreement higher than chance?'")
    lines.append("")
    lines.append(f"  Across 4 LLMs only:")
    lines.append(f"    W = {W_llms:.3f}   chi2({df_llms}) = {chi2_llms:.2f}   p = {p_llms:.6f}")
    lines.append(f"  Including dot-product (5 'raters'):")
    lines.append(f"    W = {W_all:.3f}   chi2({df_all}) = {chi2_all:.2f}   p = {p_all:.6f}")
    lines.append("")
    if W_llms > W_all:
        lines.append(f"  -> Adding dot-product REDUCES agreement (W drops from "
                     f"{W_llms:.3f} to {W_all:.3f}). Dot-product is the outlier.")
    else:
        lines.append(f"  -> Adding dot-product does NOT reduce agreement.")
    lines.append("")

    lines.append("-" * 92)
    lines.append("PAIRWISE SPEARMAN MATRIX (LLMs only)")
    lines.append("-" * 92)
    lines.append("")
    header = f"  {'':<20}" + " ".join(f"{m[:14]:>15}" for m in llms)
    lines.append(header)
    for a in llms:
        row = " ".join(f"{pw_rho.loc[a, b]:>15.3f}" for b in llms)
        lines.append(f"  {a:<20}{row}")
    lines.append("")
    lines.append(f"  Mean pairwise Spearman (upper triangle, 6 pairs): rho = {mean_rho:.3f}")
    lines.append("")
    lines.append("  Pairwise details:")
    for r in pairs_list:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        lines.append(f"    {r['model_a']:<18} vs {r['model_b']:<18}  "
                     f"rho = {r['rho']:>6.3f}   p = {r['p']:.4f}  {sig}")
    lines.append("")

    lines.append("-" * 92)
    lines.append("PER-AXIS AGREEMENT ACROSS LLMs")
    lines.append("-" * 92)
    lines.append("Low SD of rank across LLMs = strong cross-model agreement on that axis.")
    lines.append("Sorted by mean LLM rank (top-flippers first).")
    lines.append("")
    lines.append(f"  {'axis':<26} {'mean_rk':>8} {'sd_rk':>7} {'min':>5} {'max':>5} "
                 f"{'dp_rk':>7} {'dp-llm':>8}")
    for axis, r in per_axis.iterrows():
        lines.append(f"  {axis:<26} {r['mean_rank_LLMs']:>8.2f} {r['sd_rank_LLMs']:>7.2f} "
                     f"{int(r['min_rank_LLMs']):>5} {int(r['max_rank_LLMs']):>5} "
                     f"{int(r['dot_product_rank']):>7} {r['dp_minus_llm_mean']:>+8.2f}")
    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    sd_rk < 1.0   -> very strong cross-model agreement on this axis")
    lines.append("    |dp-llm| >= 3 -> LLMs and dot-product disagree about this axis")
    lines.append("")

    lines.append("-" * 92)
    lines.append("CONSENSUS CALLS")
    lines.append("-" * 92)
    lines.append("")
    lines.append("  Axes ranked TOP-2 in ALL 4 LLMs:")
    if top2_all:
        for a in top2_all:
            lines.append(f"    - {a}")
    else:
        lines.append("    (none)")
    lines.append("")
    lines.append("  Axes ranked BOTTOM-2 in ALL 4 LLMs:")
    if bot2_all:
        for a in bot2_all:
            lines.append(f"    - {a}")
    else:
        lines.append("    (none)")
    lines.append("")

    lines.append("-" * 92)
    lines.append("WHAT TO WRITE IN THE PAPER")
    lines.append("-" * 92)
    lines.append("")
    lines.append("Suggested sentence:")
    lines.append(f'  "Modifier importance is highly consistent across four LLMs from')
    lines.append(f'  three independent labs (Kendall\'s W = {W_llms:.3f}, chi2({df_llms}) = '
                 f'{chi2_llms:.1f}, p = {p_llms:.4f}; mean pairwise')
    lines.append(f'  Spearman rho = {mean_rho:.3f}). Adding the dot-product rule-based')
    lines.append(f'  baseline reduces agreement to W = {W_all:.3f}, indicating that')
    lines.append(f'  LLMs share a decision rule that differs from value arithmetic."')
    lines.append("")

    lines.append("=" * 92)
    lines.append("END OF STEP 6 REPORT")
    lines.append("=" * 92)
    REPORT_PATH.write_text("\n".join(lines))

    pw_rho.to_csv(MATRIX_CSV)
    per_axis.to_csv(PERAXIS_CSV)

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Wrote per-axis: {PERAXIS_CSV}")


if __name__ == "__main__":
    main()

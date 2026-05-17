"""
VISTA — Step 3: McNemar paired tests.

For each (model, axis) pair, build the 2x2 contingency table:

                       | modifier -> A0     | modifier -> A1
  baseline -> A0       | cell_aa (no chg)   | cell_ab (flipped A0->A1)
  baseline -> A1       | cell_ba (flipped)  | cell_bb (no chg)

Null hypothesis (McNemar): cell_ab == cell_ba
  i.e., modifier does not systematically shift decisions in either direction.

We report:
  - exact binomial McNemar p-value (recommended for any cell counts)
  - odds ratio: cell_ab / cell_ba  (direction of shift)
  - 95% CI on the odds ratio (Wilson interval on the proportion ab/(ab+ba),
    converted to OR)
  - Bonferroni and Benjamini-Hochberg adjusted p-values across all
    4 models x 8 axes = 32 tests.
"""

import math
import pandas as pd
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import beta as beta_dist

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B": OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":  OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":   OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":  OUT_DIR / "master_llm_decisions_llama_8B.csv",
}

REPORT_PATH = OUT_DIR / "step3_mcnemar_report.txt"
CSV_PATH    = OUT_DIR / "step3_mcnemar_table.csv"


def build_pairs(df):
    """
    Return paired DF with one row per (vsw_id, scenario_id, axis), giving
    the baseline decision and the modifier decision side-by-side.
    Drops ERROR decisions.
    """
    baselines = (
        df[df["axis"] == "BASELINE"]
        [["vsw_id", "scenario_id", "llm_decision"]]
        .rename(columns={"llm_decision": "baseline_decision"})
    )
    modifiers = df[df["axis"] != "BASELINE"].copy()
    paired = modifiers.merge(baselines, on=["vsw_id", "scenario_id"], how="inner")
    paired = paired.rename(columns={"llm_decision": "modifier_decision"})
    paired = paired[paired["baseline_decision"].isin(["A0", "A1"])]
    paired = paired[paired["modifier_decision"].isin(["A0", "A1"])]
    return paired


def two_by_two(paired_axis):
    """Return (aa, ab, ba, bb) for a single axis."""
    bd = paired_axis["baseline_decision"]
    md = paired_axis["modifier_decision"]
    aa = int(((bd == "A0") & (md == "A0")).sum())
    ab = int(((bd == "A0") & (md == "A1")).sum())
    ba = int(((bd == "A1") & (md == "A0")).sum())
    bb = int(((bd == "A1") & (md == "A1")).sum())
    return aa, ab, ba, bb


def or_with_ci(ab, ba, alpha=0.05):
    """
    Odds ratio for McNemar setup is ab/ba.
    CI computed via exact Clopper-Pearson on p = ab/(ab+ba), then mapped to OR.
    If ab+ba == 0, OR is undefined.
    """
    n = ab + ba
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    or_point = (ab / ba) if ba > 0 else float("inf")
    # exact Beta CI on proportion p = ab/n
    if ab == 0:
        lo_p = 0.0
    else:
        lo_p = beta_dist.ppf(alpha / 2, ab, n - ab + 1)
    if ab == n:
        hi_p = 1.0
    else:
        hi_p = beta_dist.ppf(1 - alpha / 2, ab + 1, n - ab)
    def p_to_or(p):
        if p >= 1.0:
            return float("inf")
        if p <= 0.0:
            return 0.0
        return p / (1 - p)
    return or_point, p_to_or(lo_p), p_to_or(hi_p)


def fdr_bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * n
    prev = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        k = n - rank
        val = min(prev, p * n / k)
        adj[orig_idx] = val
        prev = val
    return adj


def fmt_p(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_or(x):
    if math.isnan(x):
        return "  n/a"
    if math.isinf(x):
        return "  inf"
    return f"{x:6.2f}"


def main():
    all_rows = []
    for model_name, path in MODELS.items():
        df = pd.read_csv(path)
        paired = build_pairs(df)
        for axis in sorted(paired["axis"].unique()):
            sub = paired[paired["axis"] == axis]
            aa, ab, ba, bb = two_by_two(sub)
            table = [[aa, ab], [ba, bb]]
            # exact McNemar (binomial)
            res = mcnemar(table, exact=True)
            or_pt, or_lo, or_hi = or_with_ci(ab, ba)
            n_total = aa + ab + ba + bb
            n_disc  = ab + ba
            all_rows.append({
                "model": model_name,
                "axis": axis,
                "n_total": n_total,
                "n_concordant": aa + bb,
                "n_discordant": n_disc,
                "flips_A0_to_A1": ab,
                "flips_A1_to_A0": ba,
                "discordance_rate": n_disc / n_total if n_total else 0,
                "odds_ratio_ab_over_ba": or_pt,
                "or_ci_lo": or_lo,
                "or_ci_hi": or_hi,
                "mcnemar_p_exact": res.pvalue,
            })

    res_df = pd.DataFrame(all_rows)
    # multiple-comparison corrections across all 32 tests
    res_df["p_bonferroni"] = (res_df["mcnemar_p_exact"] * len(res_df)).clip(upper=1.0)
    res_df["p_bh_fdr"] = fdr_bh(res_df["mcnemar_p_exact"].tolist())
    res_df.to_csv(CSV_PATH, index=False)

    # ---------- text report ----------
    lines = []
    lines.append("=" * 88)
    lines.append("VISTA — STEP 3: McNEMAR PAIRED TESTS (baseline vs modifier)")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Per (model, axis): paired same-(vsw_id, scenario_id) rows.")
    lines.append("H0: modifier does NOT systematically shift decisions")
    lines.append("    (i.e., flips A0->A1 = flips A1->A0).")
    lines.append("Reject H0 -> modifier causes a directional shift in decisions.")
    lines.append("")
    lines.append(f"Total tests: {len(res_df)}  (4 models x 8 axes)")
    lines.append("Multiple-comparison correction: Bonferroni and BH-FDR.")
    lines.append("")

    for model_name in MODELS:
        sub = res_df[res_df["model"] == model_name].sort_values("mcnemar_p_exact")
        lines.append("-" * 88)
        lines.append(f"MODEL: {model_name}")
        lines.append("-" * 88)
        lines.append(f"  {'axis':<26} {'n':>5} {'A0->A1':>7} {'A1->A0':>7} "
                     f"{'OR':>6} {'95% CI':>16} {'p_exact':>10} {'p_bonf':>10} {'p_BH':>10}  sig")
        for _, r in sub.iterrows():
            sig = ""
            if r["p_bh_fdr"] < 0.001:
                sig = "***"
            elif r["p_bh_fdr"] < 0.01:
                sig = "**"
            elif r["p_bh_fdr"] < 0.05:
                sig = "*"
            else:
                sig = "ns"
            ci_str = f"[{fmt_or(r['or_ci_lo']).strip()},{fmt_or(r['or_ci_hi']).strip()}]"
            lines.append(
                f"  {r['axis']:<26} {int(r['n_total']):>5} "
                f"{int(r['flips_A0_to_A1']):>7} {int(r['flips_A1_to_A0']):>7} "
                f"{fmt_or(r['odds_ratio_ab_over_ba']):>6} {ci_str:>16} "
                f"{fmt_p(r['mcnemar_p_exact']):>10} "
                f"{fmt_p(r['p_bonferroni']):>10} "
                f"{fmt_p(r['p_bh_fdr']):>10}  {sig}"
            )
        lines.append("")

    # summary across all
    sig_at_05 = (res_df["p_bh_fdr"] < 0.05).sum()
    sig_at_001 = (res_df["p_bh_fdr"] < 0.001).sum()
    lines.append("-" * 88)
    lines.append("SUMMARY")
    lines.append("-" * 88)
    lines.append(f"  Significant (BH-FDR < 0.05):    {sig_at_05} / {len(res_df)}")
    lines.append(f"  Highly sig (BH-FDR < 0.001):    {sig_at_001} / {len(res_df)}")
    lines.append("")

    # axes with consistent direction (OR > 1 in all 4 models)
    pivot = res_df.pivot(index="axis", columns="model",
                          values="odds_ratio_ab_over_ba")
    lines.append("-" * 88)
    lines.append("DIRECTIONAL CONSISTENCY: Odds ratio (A0->A1 / A1->A0) per axis")
    lines.append("-" * 88)
    lines.append("OR > 1 means modifier shifts decisions toward A1; OR < 1 toward A0.")
    lines.append("")
    cols = list(pivot.columns)
    header = f"  {'axis':<26} " + " ".join(f"{c[:14]:>15}" for c in cols)
    lines.append(header)
    for axis, row in pivot.iterrows():
        cells = " ".join(f"{fmt_or(row[c]):>15}" for c in cols)
        lines.append(f"  {axis:<26} {cells}")
    lines.append("")

    lines.append("=" * 88)
    lines.append("END OF STEP 3 REPORT")
    lines.append("=" * 88)

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote table:  {CSV_PATH}")


if __name__ == "__main__":
    main()

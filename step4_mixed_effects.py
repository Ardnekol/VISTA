"""
VISTA — Step 4: Logistic regression with profile + scenario fixed effects.

Plain question:
  "If we already know the scenario AND the person, does knowing the
  modifier ALSO help predict the LLM's decision?"

Setup (per model):
  Outcome: y = 1 if LLM chose A1, else 0.

  Reduced model:  y ~ C(vsw_id) + C(scenario_id)
      -> Allowed to know which person and which scenario.

  Full model:     y ~ C(vsw_id) + C(scenario_id) + C(axis)
      -> Additionally allowed to know which modifier is active
         (BASELINE is the reference level).

  Likelihood-ratio test on the axis terms:
      Reject H0 -> modifier adds info beyond person + scenario.

We use vsw_id fixed effects (each profile gets its own intercept) — the
strongest possible control for the person's value profile. Anything axis
explains is on top of that.
"""

import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from pathlib import Path

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B": OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":  OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":   OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":  OUT_DIR / "master_llm_decisions_llama_8B.csv",
}

REPORT_PATH = OUT_DIR / "step4_mixed_effects_report.txt"
COEF_CSV    = OUT_DIR / "step4_axis_coefficients.csv"
SUMMARY_CSV = OUT_DIR / "step4_lr_summary.csv"


def fit_and_compare(df):
    df = df[df["llm_decision"].isin(["A0", "A1"])].copy()
    df["y"] = (df["llm_decision"] == "A1").astype(int)

    axes = ["BASELINE"] + sorted(set(df["axis"]) - {"BASELINE"})
    df["axis"] = pd.Categorical(df["axis"], categories=axes)

    reduced = smf.logit(
        "y ~ C(vsw_id) + C(scenario_id)", data=df
    ).fit(disp=0, maxiter=400, method="newton")

    full = smf.logit(
        "y ~ C(vsw_id) + C(scenario_id) + C(axis)", data=df
    ).fit(disp=0, maxiter=400, method="newton")

    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = int(full.df_model - reduced.df_model)
    p_value = float(chi2.sf(lr_stat, df_diff))

    mcf_reduced = 1 - reduced.llf / reduced.llnull
    mcf_full    = 1 - full.llf    / full.llnull

    # Extract per-axis coefficients (log-odds shift vs BASELINE)
    axis_rows = []
    for name in full.params.index:
        if name.startswith("C(axis)"):
            axis_label = name.split("[T.")[-1].rstrip("]")
            coef = full.params[name]
            se   = full.bse[name]
            z    = full.tvalues[name]
            p    = full.pvalues[name]
            ci_lo, ci_hi = full.conf_int().loc[name].tolist()
            axis_rows.append({
                "axis": axis_label,
                "coef_logodds": coef,
                "odds_ratio": math.exp(coef),
                "or_ci_lo": math.exp(ci_lo),
                "or_ci_hi": math.exp(ci_hi),
                "se": se,
                "z": z,
                "p": p,
            })
    axis_df = pd.DataFrame(axis_rows).sort_values("p")

    return {
        "n_rows": len(df),
        "ll_reduced": reduced.llf,
        "ll_full": full.llf,
        "lr_stat": lr_stat,
        "lr_df": df_diff,
        "lr_p": p_value,
        "aic_reduced": reduced.aic,
        "aic_full": full.aic,
        "delta_aic": reduced.aic - full.aic,
        "bic_reduced": reduced.bic,
        "bic_full": full.bic,
        "delta_bic": reduced.bic - full.bic,
        "mcfadden_reduced": mcf_reduced,
        "mcfadden_full": mcf_full,
        "delta_mcfadden": mcf_full - mcf_reduced,
        "axis_table": axis_df,
    }


def fmt_p(p):
    if p < 1e-6:
        return f"{p:.2e}"
    return f"{p:.6f}"


def main():
    summary_rows = []
    all_axis_rows = []
    per_model_results = {}

    for name, path in MODELS.items():
        print(f"Fitting: {name} ...")
        df = pd.read_csv(path)
        res = fit_and_compare(df)
        per_model_results[name] = res
        summary_rows.append({
            "model": name,
            "n_rows": res["n_rows"],
            "ll_reduced": res["ll_reduced"],
            "ll_full": res["ll_full"],
            "lr_stat": res["lr_stat"],
            "lr_df": res["lr_df"],
            "lr_p": res["lr_p"],
            "delta_aic": res["delta_aic"],
            "delta_bic": res["delta_bic"],
            "mcfadden_reduced": res["mcfadden_reduced"],
            "mcfadden_full": res["mcfadden_full"],
            "delta_mcfadden": res["delta_mcfadden"],
        })
        a = res["axis_table"].copy()
        a["model"] = name
        all_axis_rows.append(a)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    pd.concat(all_axis_rows, ignore_index=True).to_csv(COEF_CSV, index=False)

    # ----- write text report -----
    lines = []
    lines.append("=" * 88)
    lines.append("VISTA — STEP 4: LOGISTIC REGRESSION (PROFILE FE + SCENARIO FE + AXIS)")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Plain-English question:")
    lines.append("  If we already know the scenario AND the person, does knowing")
    lines.append("  the modifier ALSO help predict the LLM's decision?")
    lines.append("")
    lines.append("Method:")
    lines.append("  REDUCED model: y ~ C(person) + C(scenario)")
    lines.append("  FULL    model: y ~ C(person) + C(scenario) + C(axis)")
    lines.append("  Likelihood-ratio (LR) test on the axis terms.")
    lines.append("  C(person) absorbs ALL profile-level info (each person gets")
    lines.append("  their own baseline). Anything axis explains is on top of that.")
    lines.append("")

    lines.append("-" * 88)
    lines.append("HEADLINE: LIKELIHOOD-RATIO TEST (per model)")
    lines.append("-" * 88)
    lines.append("")
    lines.append(f"{'Model':<18} {'n':>5} {'LL_red':>10} {'LL_full':>10} "
                 f"{'LR_chi2':>10} {'df':>4} {'p':>12} "
                 f"{'dAIC':>8} {'dBIC':>8} {'McF_R2_red':>12} {'McF_R2_full':>12}")
    for r in summary_rows:
        lines.append(
            f"{r['model']:<18} {r['n_rows']:>5} "
            f"{r['ll_reduced']:>10.1f} {r['ll_full']:>10.1f} "
            f"{r['lr_stat']:>10.2f} {r['lr_df']:>4} {fmt_p(r['lr_p']):>12} "
            f"{r['delta_aic']:>8.1f} {r['delta_bic']:>8.1f} "
            f"{r['mcfadden_reduced']:>12.4f} {r['mcfadden_full']:>12.4f}"
        )
    lines.append("")
    lines.append("How to read this:")
    lines.append("  - LR_chi2 / p: if p is very small, the modifier term improves")
    lines.append("    prediction beyond person + scenario. This is the formal")
    lines.append("    statistical statement of the paper's claim.")
    lines.append("  - dAIC > 0 (positive) means the FULL model wins on AIC.")
    lines.append("    Rule of thumb: dAIC > 10 is 'strong' evidence.")
    lines.append("  - dBIC > 10 is 'very strong' evidence (BIC penalises params more).")
    lines.append("  - McFadden R² gain = how much extra variance the modifier explains.")
    lines.append("")

    lines.append("-" * 88)
    lines.append("AXIS COEFFICIENTS (per model)")
    lines.append("-" * 88)
    lines.append("Coefficient = log-odds shift toward A1 vs the BASELINE (no modifier) condition.")
    lines.append("Odds ratio > 1 means modifier shifts decisions TOWARD A1.")
    lines.append("Odds ratio < 1 means modifier shifts decisions TOWARD A0.")
    lines.append("")
    for name in MODELS:
        a = per_model_results[name]["axis_table"]
        lines.append(f"\n{name}:")
        lines.append(f"  {'axis':<26} {'OR':>6} {'95% CI':>16} {'p':>12}  sig")
        for _, r in a.iterrows():
            sig = ""
            if r["p"] < 0.001:
                sig = "***"
            elif r["p"] < 0.01:
                sig = "**"
            elif r["p"] < 0.05:
                sig = "*"
            else:
                sig = "ns"
            ci = f"[{r['or_ci_lo']:.2f},{r['or_ci_hi']:.2f}]"
            lines.append(
                f"  {r['axis']:<26} {r['odds_ratio']:>6.2f} {ci:>16} "
                f"{fmt_p(r['p']):>12}  {sig}"
            )
    lines.append("")
    lines.append("=" * 88)
    lines.append("END OF STEP 4 REPORT")
    lines.append("=" * 88)
    REPORT_PATH.write_text("\n".join(lines))

    print(f"\nWrote report:  {REPORT_PATH}")
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote coefs:   {COEF_CSV}")


if __name__ == "__main__":
    main()

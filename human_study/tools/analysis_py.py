"""Python alternative to analysis.R — uses statsmodels for mixed-effects logistic.

R/lme4 (analysis.R) is the gold standard for EMNLP. Use this Python version for
quick iteration or if R is not available. Results should be very close;
report the R results in the final paper.

Run (from VISTA/ with the venv active):
    python human_study/tools/analysis_py.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "human_study" / "analysis_ready.csv"
OUT_DIR = REPO_ROOT / "human_study" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Schwartz centered scores sum to ~0 per participant (ipsatization), so we must
# drop one value as the implicit reference when fitting. Hedonism is the
# conventional drop (peripheral on the circumplex). Document this in the paper.
ALL_VALUES = [
    "self_direction", "power", "universalism", "achievement", "security",
    "stimulation", "conformity", "tradition", "benevolence", "hedonism",
]
REFERENCE_VALUE = "hedonism"
VALUES = [v for v in ALL_VALUES if v != REFERENCE_VALUE]


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA_PATH)
    d = d[d["excluded"] == 0].copy()
    print(f"After exclusions: {len(d)} rows from {d['participant_id'].nunique()} participants\n")
    return d


def fit_logit_with_clustered_se(d: pd.DataFrame, formula: str, label: str) -> pd.DataFrame:
    """Fit a logistic regression with SEs clustered by participant.

    Pragmatic stand-in for a full random-intercept GLMM. For the final EMNLP
    paper, fit the same formula in R with glmer().
    """
    model = logit(formula, data=d).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": d["participant_id"]}
    )
    tab = pd.DataFrame({
        "term": model.model.exog_names,
        "estimate": model.params.values,
        "se": model.bse.values,
        "z": model.tvalues.values,
        "p": model.pvalues.values,
    })
    tab["odds_ratio"] = np.exp(tab["estimate"])
    tab["or_lower"] = np.exp(tab["estimate"] - 1.96 * tab["se"])
    tab["or_upper"] = np.exp(tab["estimate"] + 1.96 * tab["se"])
    print(f"=== {label} ===")
    print(tab.to_string(index=False))
    print()
    return tab


def main() -> None:
    d = load()

    # Model 1: modifier main effect
    m1 = fit_logit_with_clustered_se(
        d, "choice ~ is_modified", "Model 1: choice ~ is_modified"
    )
    m1.to_csv(OUT_DIR / "model1_modifier_only.csv", index=False)

    # Model 2: values only
    value_terms = " + ".join([f"centered_{v}" for v in VALUES])
    m2 = fit_logit_with_clustered_se(
        d, f"choice ~ {value_terms}", "Model 2: choice ~ 10 values"
    )
    m2.to_csv(OUT_DIR / "model2_values_only.csv", index=False)

    # Model 3 (PRIMARY): values + modifier + domain
    m3 = fit_logit_with_clustered_se(
        d, f"choice ~ {value_terms} + is_modified + C(domain)",
        "Model 3 (PRIMARY): values + modifier + domain"
    )
    m3.to_csv(OUT_DIR / "model3_primary.csv", index=False)

    # Model 4: value x modifier interactions (exploratory)
    interaction_terms = " + ".join([f"centered_{v}:is_modified" for v in VALUES])
    m4 = fit_logit_with_clustered_se(
        d, f"choice ~ {value_terms} + is_modified + {interaction_terms}",
        "Model 4 (exploratory): value x modifier interactions"
    )
    m4.to_csv(OUT_DIR / "model4_interactions.csv", index=False)

    # Per-axis modifier effects (exploratory)
    print("=== Per-axis modifier effects (exploratory) ===")
    per_axis_rows = []
    for ax in sorted(d["axis"].dropna().unique()):
        if ax == "":
            continue
        dd = d[d["axis"].isna() | (d["axis"] == ax)].copy()
        dd["is_modified"] = (dd["axis"] == ax).astype(int)
        n_modified = dd["is_modified"].sum()
        if n_modified < 20:
            print(f"  skip {ax} (n={n_modified})")
            continue
        # Lean per-axis spec — within-participant variation via clustered SEs,
        # value covariates dropped because the subset is too thin to identify
        # all 9 reliably. The full Model 3 already controlled for them.
        formula = "choice ~ is_modified"
        try:
            m = logit(formula, data=dd).fit(
                disp=False, cov_type="cluster",
                cov_kwds={"groups": dd["participant_id"]}
            )
        except Exception as e:
            print(f"  skip {ax} (fit failed: {e})")
            continue
        idx = m.model.exog_names.index("is_modified")
        per_axis_rows.append({
            "axis": ax,
            "n_modified": int(n_modified),
            "estimate": float(m.params.iloc[idx]),
            "se": float(m.bse.iloc[idx]),
            "p": float(m.pvalues.iloc[idx]),
            "odds_ratio": float(np.exp(m.params.iloc[idx])),
        })
    per_axis = pd.DataFrame(per_axis_rows)
    # Benjamini-Hochberg correction
    from statsmodels.stats.multitest import multipletests
    if not per_axis.empty:
        per_axis["p_bh"] = multipletests(per_axis["p"], method="fdr_bh")[1]
    print(per_axis.to_string(index=False))
    per_axis.to_csv(OUT_DIR / "per_axis_modifier.csv", index=False)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

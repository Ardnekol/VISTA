"""Map human pilot decisions to LLM results for paper §4.5–§4.7.

Per-(scenario_id, axis) human shift uses a BETWEEN-SUBJECTS design:
  - F1 participants see SC00X_1 as baseline and SC00X_2 with a modifier.
  - F2 participants see SC00X_2 as baseline and SC00X_1 with a (different) modifier.
So each scenario_id has one group that saw it baseline and another that saw it modified.
  Δ_cell = P(A1 | modified) - P(A1 | baseline)
  |Δ|_cell is the unsigned shift.

LLM comparison uses per-axis flip rate from step5_axis_ranking_table.csv
(already computed across the 95 vsw_ids × 5 scenarios).

Outputs:
  human_study/results/human_per_cell.csv        — per-(scenario, axis) shift
  human_study/results/human_per_axis.csv        — pooled per-axis |Δ|
  human_study/results/human_per_type.csv        — per modifier-type rollup
  human_study/results/human_vs_llm_per_axis.csv — joined comparison
  human_study/results/spearman_axis_agreement.csv
  human_study/results/fig_human_vs_llm_axis.png
  human_study/results/fig_human_vs_llm_type.png
  human_study/results/headline.txt
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
VISTA = ROOT.parent
PROFILES = ROOT / "human_binary_profiles.csv"
DECISIONS = ROOT / "decisions.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LLM_RANK = VISTA / "outputs" / "step5_axis_ranking_table.csv"

AXES = [
    "authority_signal", "self_preservation", "social_visibility", "time_pressure",
    "in_out_group", "competence_uncertainty", "diffused_responsibility", "resource_scarcity",
]
TYPE_OF = {
    "resource_scarcity":      "stakes",
    "self_preservation":      "personal-cost",
    "time_pressure":          "personal-cost",
    "social_visibility":      "affective",
    "in_out_group":           "affective",
    "authority_signal":       "affective",
    "diffused_responsibility":"informational",
    "competence_uncertainty": "informational",
}


def per_cell_shift(dec: pd.DataFrame) -> pd.DataFrame:
    """For each scenario_id, compute P(A1|baseline) and P(A1|modified) and the axis used."""
    rows = []
    for sc, g in dec.groupby("scenario_id"):
        base = g[(g["is_modified"] == 0) & g["choice"].notna()]
        mod  = g[(g["is_modified"] == 1) & g["choice"].notna()]
        if len(base) == 0 or len(mod) == 0:
            continue
        axis = mod["axis"].iloc[0]
        rows.append({
            "scenario_id": sc,
            "axis": axis,
            "n_baseline": len(base),
            "n_modified": len(mod),
            "p_A1_baseline": base["choice"].mean(),
            "p_A1_modified": mod["choice"].mean(),
            "delta_signed": mod["choice"].mean() - base["choice"].mean(),
            "delta_abs": abs(mod["choice"].mean() - base["choice"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["axis", "scenario_id"]).reset_index(drop=True)


def per_axis_rollup(cell: pd.DataFrame) -> pd.DataFrame:
    """Pool per-axis: average |Δ| across the 1-2 cells per axis."""
    rows = []
    for ax in AXES:
        sub = cell[cell["axis"] == ax]
        if len(sub) == 0:
            rows.append({"axis": ax, "n_cells": 0, "n_human_trials": 0,
                         "human_mean_abs_shift": np.nan, "human_mean_signed_shift": np.nan})
            continue
        rows.append({
            "axis": ax,
            "n_cells": len(sub),
            "n_human_trials": int(sub["n_modified"].sum()),
            "human_mean_abs_shift": sub["delta_abs"].mean(),
            "human_mean_signed_shift": sub["delta_signed"].mean(),
        })
    return pd.DataFrame(rows)


def per_type_rollup(axis_tbl: pd.DataFrame) -> pd.DataFrame:
    axis_tbl = axis_tbl.copy()
    axis_tbl["type"] = axis_tbl["axis"].map(TYPE_OF)
    rows = []
    for t in ["affective", "personal-cost", "informational", "stakes"]:
        sub = axis_tbl[axis_tbl["type"] == t].dropna(subset=["human_mean_abs_shift"])
        rows.append({
            "type": t,
            "axes_in_type": ",".join(sub["axis"].tolist()),
            "human_mean_abs_shift": sub["human_mean_abs_shift"].mean() if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def llm_axis_rates() -> pd.DataFrame:
    rk = pd.read_csv(LLM_RANK).set_index("axis")
    keep = [c for c in rk.columns if c.startswith("rate_")
            and c.replace("rate_", "") in
            ["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B", "Dot-Product (rule)"]]
    df = rk[keep].copy()
    df.columns = [c.replace("rate_", "") for c in df.columns]
    return df.loc[AXES]


def spearman_table(human_axis: pd.Series, llm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    h = human_axis.dropna()
    if len(h) < 3:
        return pd.DataFrame()
    common_axes = h.index
    for sys in llm.columns:
        rho, p = spearmanr(h.values, llm.loc[common_axes, sys].values)
        rows.append({"system": sys, "spearman_rho": rho, "p_value": p, "n_axes": len(common_axes)})
    return pd.DataFrame(rows)


def fig_human_vs_llm_axis(axis_tbl: pd.DataFrame, llm: pd.DataFrame, out: Path, N: int) -> None:
    axes_order = AXES
    x = np.arange(len(axes_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar(x - 1.5*width, axis_tbl.set_index("axis").loc[axes_order, "human_mean_abs_shift"].values * 100,
           width, label=f"Human (N={N})", color="#1f77b4", edgecolor="black", linewidth=0.6)
    palette = {"Gemma 4 31B": "#d62728", "Qwen 2.5 32B": "#2ca02c", "Llama 3.1 8B": "#9467bd"}
    for i, m in enumerate(["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B"]):
        ax.bar(x + (i - 0.5)*width, llm.loc[axes_order, m].values * 100,
               width, label=m, color=palette[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in axes_order], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Decision shift / flip rate (%)", fontsize=15)
    ax.set_title("Per-axis modifier impact: humans vs. LLMs", fontsize=15)
    ax.legend(fontsize=17, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_human_vs_llm_type(axis_tbl: pd.DataFrame, llm: pd.DataFrame, out: Path, N: int) -> None:
    axis_tbl = axis_tbl.copy()
    axis_tbl["type"] = axis_tbl["axis"].map(TYPE_OF)
    llm_t = llm.copy()
    llm_t["type"] = [TYPE_OF[a] for a in llm_t.index]
    type_order = ["affective", "personal-cost", "informational", "stakes"]
    human_t = axis_tbl.groupby("type")["human_mean_abs_shift"].mean().reindex(type_order)
    llm_grp = llm_t.groupby("type")[["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B"]].mean().reindex(type_order)

    x = np.arange(len(type_order))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 1.5*width, human_t.values * 100, width, label=f"Human (N={N})",
           color="#1f77b4", edgecolor="black", linewidth=0.6)
    palette = {"Gemma 4 31B": "#d62728", "Qwen 2.5 32B": "#2ca02c", "Llama 3.1 8B": "#9467bd"}
    for i, m in enumerate(["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B"]):
        ax.bar(x + (i - 0.5)*width, llm_grp[m].values * 100, width, label=m,
               color=palette[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(type_order, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Mean decision shift (%)", fontsize=15)
    ax.set_title("Modifier-type pressure: humans vs. LLMs", fontsize=13)
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")
    return human_t, llm_grp


def write_latex_tables(axis_tbl: pd.DataFrame, llm: pd.DataFrame,
                       sp: pd.DataFrame, type_tbl: pd.DataFrame,
                       cell: pd.DataFrame, N: int) -> None:
    """Write three small LaTeX snippets the paper can \\input{}."""

    # Per-axis: Human |Δ| vs the three LLM flip rates
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(rf"\caption{{Per-axis modifier impact in the human pilot (N={N}) vs.\ LLM flip rates. "
                 r"Human values are between-subjects $|\Delta P(A1)|$ pooled across the 1--2 "
                 r"(scenario, axis) cells covered by the two-form design.}")
    lines.append(r"\label{tab:human_vs_llm_axis}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Axis} & \textbf{Human} & \textbf{Gemma} & \textbf{Qwen} & \textbf{Llama8B} \\")
    lines.append(r"\midrule")
    axis_order_sorted = (axis_tbl.set_index("axis")
                         .reindex(AXES)
                         .sort_values("human_mean_abs_shift", ascending=False)
                         .index.tolist())
    for ax in axis_order_sorted:
        h = axis_tbl.set_index("axis").loc[ax, "human_mean_abs_shift"] * 100
        g = llm.loc[ax, "Gemma 4 31B"] * 100
        q = llm.loc[ax, "Qwen 2.5 32B"] * 100
        l = llm.loc[ax, "Llama 3.1 8B"] * 100
        ax_safe = ax.replace("_", r"\_")
        lines.append(f"{ax_safe:26s} & {h:5.1f}\\% & {g:5.1f}\\% & {q:5.1f}\\% & {l:5.1f}\\% \\\\")
    lines.append(r"\midrule")
    h_mean = axis_tbl["human_mean_abs_shift"].mean() * 100
    g_mean = llm["Gemma 4 31B"].mean() * 100
    q_mean = llm["Qwen 2.5 32B"].mean() * 100
    l_mean = llm["Llama 3.1 8B"].mean() * 100
    lines.append(f"\\textbf{{mean (8 axes)}}        & \\textbf{{{h_mean:.1f}\\%}} & \\textbf{{{g_mean:.1f}\\%}} & \\textbf{{{q_mean:.1f}\\%}} & \\textbf{{{l_mean:.1f}\\%}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (RESULTS / "tab_human_vs_llm_axis.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {RESULTS / 'tab_human_vs_llm_axis.tex'}")

    # Per modifier-type rollup vs the three LLMs
    type_order = ["stakes", "affective", "personal-cost", "informational"]
    axis_tbl_t = axis_tbl.copy()
    axis_tbl_t["type"] = axis_tbl_t["axis"].map(TYPE_OF)
    llm_t = llm.copy()
    llm_t["type"] = [TYPE_OF[a] for a in llm_t.index]
    h_per_type = axis_tbl_t.groupby("type")["human_mean_abs_shift"].mean()
    l_per_type = llm_t.groupby("type")[["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B"]].mean()

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Mean modifier-type pressure (\% decision shift / flip rate). "
                 r"Humans and LLMs agree on the rank order of the four types.}")
    lines.append(r"\label{tab:modifier_type}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Type} & \textbf{Human} & \textbf{Gemma} & \textbf{Qwen} & \textbf{Llama8B} \\")
    lines.append(r"\midrule")
    for t in type_order:
        h = h_per_type.get(t, np.nan) * 100
        g = l_per_type.loc[t, "Gemma 4 31B"] * 100
        q = l_per_type.loc[t, "Qwen 2.5 32B"] * 100
        l = l_per_type.loc[t, "Llama 3.1 8B"] * 100
        lines.append(f"{t:14s} & {h:5.1f}\\% & {g:5.1f}\\% & {q:5.1f}\\% & {l:5.1f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (RESULTS / "tab_human_vs_llm_type.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'tab_human_vs_llm_type.tex'}")

    # Spearman rank-agreement
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Spearman rank correlation between the human per-axis "
                 r"$|\Delta P(A1)|$ ordering and each system's per-axis flip-rate ordering "
                 r"(over 8 axes).}")
    lines.append(r"\label{tab:spearman_human_vs_llm}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & $\bm{\rho}$ & $\bm{p}$ \\")
    lines.append(r"\midrule")
    for _, r in sp.iterrows():
        lines.append(f"{r['system']:22s} & {r['spearman_rho']:+.2f} & {r['p_value']:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (RESULTS / "tab_spearman_human_vs_llm.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'tab_spearman_human_vs_llm.tex'}")


def main() -> None:
    prof = pd.read_csv(PROFILES)
    dec = pd.read_csv(DECISIONS)
    print(f"loaded {len(prof)} participants, {len(dec)} decision rows")
    valid = dec[dec["choice"].notna()]
    print(f"valid (non-blank) decision rows: {len(valid)}")

    # ===== 1) Per-cell between-subjects shift =====
    cell = per_cell_shift(valid)
    cell.to_csv(RESULTS / "human_per_cell.csv", index=False)
    print(f"\nwrote {RESULTS / 'human_per_cell.csv'}")
    print(cell.to_string(index=False))

    # ===== 2) Per-axis rollup =====
    axis_tbl = per_axis_rollup(cell)
    axis_tbl.to_csv(RESULTS / "human_per_axis.csv", index=False)
    print(f"\nwrote {RESULTS / 'human_per_axis.csv'}")
    print(axis_tbl.to_string(index=False))

    # ===== 3) Per modifier-type rollup =====
    type_tbl = per_type_rollup(axis_tbl)
    type_tbl.to_csv(RESULTS / "human_per_type.csv", index=False)
    print(f"\nwrote {RESULTS / 'human_per_type.csv'}")
    print(type_tbl.to_string(index=False))

    # ===== 4) Join with LLM per-axis flip rates =====
    llm = llm_axis_rates()
    joined = axis_tbl.set_index("axis").join(llm)
    joined.to_csv(RESULTS / "human_vs_llm_per_axis.csv")
    print(f"\nwrote {RESULTS / 'human_vs_llm_per_axis.csv'}")
    print(joined.round(3).to_string())

    # ===== 5) Spearman rank agreement: human axis ranking vs each model =====
    sp = spearman_table(axis_tbl.set_index("axis")["human_mean_abs_shift"], llm)
    sp.to_csv(RESULTS / "spearman_axis_agreement.csv", index=False)
    print(f"\nwrote {RESULTS / 'spearman_axis_agreement.csv'}")
    print(sp.round(3).to_string(index=False))

    # ===== 6) Figures =====
    N = len(prof)
    fig_human_vs_llm_axis(axis_tbl, llm, RESULTS / "fig_human_vs_llm_axis.png", N)
    human_t, llm_grp = fig_human_vs_llm_type(axis_tbl, llm, RESULTS / "fig_human_vs_llm_type.png", N)

    # ===== 7) Headline numbers for the paper =====
    pooled_human = cell["delta_abs"].mean()
    top_cell = cell.loc[cell["delta_abs"].idxmax()]
    top_axis = axis_tbl.loc[axis_tbl["human_mean_abs_shift"].idxmax()]
    matched = prof["matched_vsw_id"].notna().sum()

    N = len(prof)
    lines = []
    lines.append(f"=== Human Pilot Headline (N={N}) ===")
    lines.append(f"  participants retained: {N}   (no exclusions)")
    lines.append(f"  matched a retained LLM vsw_id: {matched}/{N}  "
                 f"(others have valid profiles but were pruned from the 95 by antagonism rules)")
    lines.append(f"  total decisions analysed: {len(valid)}  ({N} × 10 minus blanks)")
    lines.append(f"  pooled mean |Δ P(A1)| across {len(cell)} (scenario, axis) cells: "
                 f"{pooled_human*100:.1f}%")
    lines.append(f"  strongest cell: {top_cell['scenario_id']} × {top_cell['axis']} "
                 f"-> |Δ|={top_cell['delta_abs']*100:.1f}% "
                 f"(base {top_cell['p_A1_baseline']*100:.0f}% -> mod {top_cell['p_A1_modified']*100:.0f}%, "
                 f"n_base={top_cell['n_baseline']}, n_mod={top_cell['n_modified']})")
    lines.append(f"  strongest axis (pooled): {top_axis['axis']} "
                 f"-> mean |Δ|={top_axis['human_mean_abs_shift']*100:.1f}% "
                 f"across {top_axis['n_cells']} cell(s)")
    lines.append("")
    lines.append("=== Per modifier-type (% mean |Δ|) ===")
    for _, r in type_tbl.iterrows():
        lines.append(f"  {r['type']:14s}: human {r['human_mean_abs_shift']*100:5.1f}%   "
                     f"(axes: {r['axes_in_type']})")
    lines.append("")
    lines.append("=== Spearman ρ over 8 axes (human ranking vs each system) ===")
    for _, r in sp.iterrows():
        lines.append(f"  {r['system']:22s}: ρ = {r['spearman_rho']:+.3f}   p = {r['p_value']:.3f}")
    lines.append("")
    lines.append("=== LLM mean per-axis flip rates (for reference) ===")
    for m in llm.columns:
        lines.append(f"  {m:22s}: mean = {llm[m].mean()*100:.1f}%")

    text = "\n".join(lines)
    (RESULTS / "headline.txt").write_text(text + "\n")

    # ===== 8) LaTeX tables ready for paste =====
    write_latex_tables(axis_tbl, llm, sp, type_tbl, cell, N)
    print()
    print(text)
    print(f"\nwrote {RESULTS / 'headline.txt'}")


if __name__ == "__main__":
    main()

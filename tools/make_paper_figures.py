"""Generate spider plot, heatmap, and per-axis bar plot for the EMNLP paper."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

AXES = [
    "authority_signal", "self_preservation", "social_visibility", "time_pressure",
    "in_out_group", "competence_uncertainty", "diffused_responsibility", "resource_scarcity",
]

# Pull per-axis flip rates from the existing axis-ranking table
ranking = pd.read_csv(OUT / "step5_axis_ranking_table.csv")
ranking.set_index("axis", inplace=True)
rate_cols = [c for c in ranking.columns if c.startswith("rate_")]
ranking = ranking[rate_cols]
ranking.columns = [c.replace("rate_", "") for c in ranking.columns]
ranking = ranking.loc[AXES]  # consistent axis order

# Per-axis human between-subjects |ΔP(A1)| from the N=50 pilot (Table 4 of paper).
# Stored as fractions to match LLM/utility flip-rate scale.
HUMAN_PER_AXIS = {
    "authority_signal":        0.181,
    "self_preservation":       0.101,
    "social_visibility":       0.066,
    "time_pressure":           0.111,
    "in_out_group":            0.212,
    "competence_uncertainty":  0.101,
    "diffused_responsibility": 0.021,
    "resource_scarcity":       0.163,
}

# ===== 1) Spider plot of per-axis flip rates across models =====
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
theta = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
theta += theta[:1]

colors = {"Human (N=50)": "#1f77b4",
          "Gemma 4 31B": "#d62728", "Qwen 2.5 32B": "#2ca02c",
          "Llama 3.1 8B": "#9467bd", "Dot-Product (rule)": "#7f7f7f"}
linestyles = {"Human (N=50)": "--"}  # dashed to flag the different metric
for model, col in colors.items():
    if model == "Human (N=50)":
        vals = [HUMAN_PER_AXIS[a] for a in AXES]
    elif model in ranking.columns:
        vals = ranking[model].tolist()
    else:
        continue
    vals += vals[:1]
    ax.plot(theta, vals, color=col, label=model, linewidth=2.0,
            linestyle=linestyles.get(model, "-"))
    ax.fill(theta, vals, color=col, alpha=0.08)

ax.set_xticks(theta[:-1])
ax.set_xticklabels([a.replace("_", "\n") for a in AXES], fontsize=11)
ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax.set_yticklabels(["5%", "10%", "15%", "20%", "25%", "30%"], fontsize=11,
                   color="black", fontweight="bold")
for lbl in ax.get_yticklabels():
    lbl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
ax.set_rlabel_position(135)
ax.set_ylim(0, 0.36)
ax.set_title("Per-axis sensitivity by system", pad=30, fontsize=13)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25),
          ncol=3, fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "spider_axis_rates.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"wrote {OUT / 'spider_axis_rates.png'}")

# ===== 2) Per-(scenario, axis) heatmaps, one per LLM =====
HEATMAP_SOURCES = [
    ("Gemma 4 31B",  OUT / "master_llm_decisions_gemma4.csv",          "heatmap_scenario_axis.png"),
    ("Qwen 2.5 32B", OUT / "outputs" / "master_llm_decisions_qwen.csv", "heatmap_scenario_axis_qwen.png"),
    ("Llama 3.1 8B", OUT / "master_llm_decisions_llama_8B.csv",         "heatmap_scenario_axis_llama8b.png"),
]
for model_name, src_csv, out_name in HEATMAP_SOURCES:
    g = pd.read_csv(src_csv)
    base = (g[g["condition"] == "BASELINE"]
            [["vsw_id", "scenario_id", "llm_decision"]]
            .rename(columns={"llm_decision": "baseline_decision"}))
    mod = g[g["condition"] != "BASELINE"].copy()
    merged = mod.merge(base, on=["vsw_id", "scenario_id"])
    merged["flipped"] = ((merged["llm_decision"] != merged["baseline_decision"])
                         & merged["llm_decision"].isin(["A0", "A1"])
                         & merged["baseline_decision"].isin(["A0", "A1"]))
    heat = merged.groupby(["scenario_id", "axis"])["flipped"].mean().unstack()
    heat = heat[AXES]  # consistent column order

    vmax = max(heat.values.max() * 100, 1.0)  # avoid degenerate colormap
    fig, ax = plt.subplots(figsize=(16, 11))
    im = ax.imshow(heat.values * 100, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([a.replace("_", "\n") for a in AXES], fontsize=16)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=16)
    ax.set_title(f"Flip rate (%) per (scenario, axis) — {model_name}", fontsize=18)
    ax.tick_params(axis="x", pad=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Flip rate (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="black" if v < vmax * 0.6 else "white")
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / out_name}")

# ===== 3) Per-axis bar plot (Gemma 31B vs Qwen 32B vs Llama 8B) =====
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(AXES))
width = 0.25
models_to_plot = ["Gemma 4 31B", "Qwen 2.5 32B", "Llama 3.1 8B"]
for i, m in enumerate(models_to_plot):
    if m not in ranking.columns:
        continue
    ax.bar(x + (i - 1) * width, ranking[m].values * 100, width, label=m, color=colors[m])
ax.set_xticks(x)
ax.set_xticklabels([a.replace("_", "\n") for a in AXES], fontsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylabel("Flip rate (%)", fontsize=13)
ax.set_title("Per-axis LLM flip rates", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "bar_per_axis.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"wrote {OUT / 'bar_per_axis.png'}")

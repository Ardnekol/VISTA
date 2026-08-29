#!/usr/bin/env python3
"""
Regenerates fig_spider_all.png: the original overlaid radar, unchanged in design,
drawn to fit a single ACL column.

The old PNG was drawn on a 9in canvas and then squeezed into a 3.03in column - a
0.34x scale that turned 11pt tick labels into ~4pt. Rather than shrink a large
canvas, this builds the figure AT the column width (\\columnwidth = 219.09pt =
3.03in), so every point size set here is the point size that prints. Fitting nine
series in that space also requires spending the ink budget deliberately:
abbreviated spoke labels, three rings instead of seven, thinner strokes, and a
compact three-column legend.
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path("/home/manu/VISTA/outputs")

# \columnwidth = 219.08614pt (TeX) = 3.031in; include at width=\columnwidth for 1:1
COL_IN = 3.031

LLM_FILES = {
    "Gemma 4 31B":  OUT / "master_llm_decisions_gemma4.csv",
    "Qwen 2.5 32B": OUT / "outputs" / "master_llm_decisions_qwen.csv",
    "Llama 3.1 8B": OUT / "master_llm_decisions_llama_8B.csv",
    "Haiku 4.5":    OUT / "master_llm_decisions_haiku.csv",
    "GPT-4.1-mini": OUT / "master_llm_decisions_gpt41mini.csv",
    "GPT-5-mini":   OUT / "master_llm_decisions_gpt5mini.csv",
    "Sonnet 5":     OUT / "master_llm_decisions_sonnet.csv",
}
DP_FILE    = OUT / "master_llm_decisions_dotProduct.csv"
HUMAN_AXIS = Path("/home/manu/VISTA/human_study/results/human_vs_llm_per_axis.csv")

AXES = ["social_visibility", "self_preservation", "authority_signal", "resource_scarcity",
        "diffused_responsibility", "competence_uncertainty", "in_out_group", "time_pressure"]

# abbreviated so the labels fit outside a 3in circle; expanded in the caption
AXIS_LABEL = {
    "social_visibility":       "social\nvisibility",
    "self_preservation":       "self-\npreserv.",
    "authority_signal":        "authority",
    "resource_scarcity":       "resource\nscarcity",
    "diffused_responsibility": "diffused\nresp.",
    "competence_uncertainty":  "competence\nuncert.",
    "in_out_group":            "in/out\ngroup",
    "time_pressure":           "time\npressure",
}

COLOR = {
    "Gemma 4 31B":  "#D55E00", "Qwen 2.5 32B": "#009E73", "Llama 3.1 8B": "#CC79A7",
    "Haiku 4.5":    "#E69F00", "GPT-4.1-mini": "#0072B2", "GPT-5-mini":   "#56B4E9",
    "Sonnet 5":     "#8B4513",
}
LLM_ORDER = list(LLM_FILES.keys())


def axis_rates(path, col):
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] != "BASELINE"]
    out = {}
    for ax in AXES:
        rs = [r for r in rows if r["axis"] == ax]
        out[ax] = 100 * sum(1 for r in rs if r[col] == "YES") / len(rs) if rs else 0.0
    return out


llm_rates = {m: axis_rates(p, "llm_changed_from_baseline") for m, p in LLM_FILES.items()}
dp_rates  = axis_rates(DP_FILE, "dp_changed_from_baseline")
hum_rates = {r["axis"]: 100 * float(r["human_mean_abs_shift"])
             for r in csv.DictReader(open(HUMAN_AXIS))}


def radar():
    N = len(AXES)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angc = np.concatenate([ang, ang[:1]])

    def close(vals):
        return np.concatenate([vals, vals[:1]])

    fig = plt.figure(figsize=(COL_IN, 3.34))
    # leave room at the bottom for the legend and at the sides for spoke labels
    # top edge left at 0.93, not 0.96: the two-line "social visibility" label
    # sits above the circle and is clipped by the canvas otherwise
    ax = fig.add_axes([0.15, 0.28, 0.70, 0.65], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    dpv = close(np.array([dp_rates[a] for a in AXES]))
    ax.plot(angc, dpv, color="#7f7f7f", lw=1.5, label="Dot-Product (rule)")
    ax.fill(angc, dpv, color="#7f7f7f", alpha=0.10)

    hv = close(np.array([hum_rates[a] for a in AXES]))
    ax.plot(angc, hv, color="#111111", lw=1.4, ls=(0, (2.5, 1.5)), marker="o", ms=2.2,
            label="Human (N=50)")

    for m in LLM_ORDER:
        v = close(np.array([llm_rates[m][a] for a in AXES]))
        ax.plot(angc, v, color=COLOR[m], lw=1.15, label=m)

    ax.set_xticks(ang)
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=6.2, linespacing=1.05)
    ax.tick_params(axis="x", pad=1.5)
    # three rings instead of seven: at this size the extra labels are pure clutter
    ax.set_yticks([10, 20, 30])
    ax.set_yticklabels([])
    ax.set_ylim(0, 36)
    ax.grid(color="#d5d5d5", lw=0.45)
    ax.spines["polar"].set_linewidth(0.6)
    # Drawn as text rather than tick labels: every radial line crosses the
    # polygons near the centre, and real tick labels sit under the series.
    for r in (10, 20, 30):
        ax.text(np.deg2rad(200), r, f"{r}%", fontsize=5.0, color="#666",
                ha="center", va="center", zorder=20,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3,
              fontsize=5.0, frameon=False, handlelength=1.5,
              columnspacing=0.9, handletextpad=0.4, labelspacing=0.35,
              borderaxespad=0.0)

    p = OUT / "fig_spider_all.png"
    fig.savefig(p, dpi=600, facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", radar())

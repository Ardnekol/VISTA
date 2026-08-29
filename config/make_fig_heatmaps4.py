#!/usr/bin/env python3
"""
Per-(scenario, axis) flip-rate heatmaps for four models, on a SHARED colour scale.

Note on the earlier two-panel figure: tools/make_paper_figures.py set vmax per
panel (`vmax = heat.max()*100`), so Gemma's 42.1% and Qwen's 31.6% were both
painted as the darkest cell. Side by side that reads as "equally intense" when
one is a third larger. Here every panel shares one scale and one colourbar, so
panel-to-panel colour is comparable.
"""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.family"] = "DejaVu Sans"
OUT = Path("/home/manu/VISTA/outputs")

MODELS = [
    ("Gemma 4 31B",  OUT / "master_llm_decisions_gemma4.csv"),
    ("Qwen 2.5 32B", OUT / "outputs" / "master_llm_decisions_qwen.csv"),
    ("Claude Haiku 4.5", OUT / "master_llm_decisions_haiku.csv"),
    ("GPT-5-mini",   OUT / "master_llm_decisions_gpt5mini.csv"),
]
AXES = ["authority_signal", "self_preservation", "social_visibility", "time_pressure",
        "in_out_group", "competence_uncertainty", "diffused_responsibility", "resource_scarcity"]
SHORT = {"authority_signal":"auth","self_preservation":"self-pres","social_visibility":"social",
         "time_pressure":"time","in_out_group":"in/out","competence_uncertainty":"comp",
         "diffused_responsibility":"diffused","resource_scarcity":"resource"}

def grid(path):
    d = pd.read_csv(path)
    m = d[d.axis != "BASELINE"]
    m = m[m.llm_changed_from_baseline.isin(["YES", "NO"])]
    g = (m.groupby(["scenario_id", "axis"]).llm_changed_from_baseline
           .apply(lambda s: (s == "YES").mean() * 100).unstack())
    return g[AXES]

grids = {n: grid(p) for n, p in MODELS}
VMAX = max(g.values.max() for g in grids.values())

fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.65))
fig.subplots_adjust(left=0.11, right=0.88, top=0.955, bottom=0.065, wspace=0.26, hspace=0.30)

for ax, (name, _) in zip(axes.ravel(), MODELS):
    h = grids[name]
    im = ax.imshow(h.values, cmap="YlOrRd", vmin=0, vmax=VMAX, aspect="auto")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([SHORT[a] for a in AXES], fontsize=5.6, rotation=45, ha="right")
    ax.set_yticks(range(len(h.index)))
    ax.set_yticklabels(h.index, fontsize=5.6)
    ax.tick_params(length=0, pad=1)
    ax.set_title(name, fontsize=8.2, pad=3)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                    color="white" if v > 0.62 * VMAX else "#222")
    # ring the single strongest cell in each panel
    i, j = np.unravel_index(np.argmax(h.values), h.shape)
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor="#0b5fa5", lw=1.1, zorder=5))
    for s in ax.spines.values():
        s.set_visible(False)

cax = fig.add_axes([0.905, 0.065, 0.016, 0.89])
cb = fig.colorbar(im, cax=cax)
cb.set_label("flip rate (%)", fontsize=6.2)
cb.ax.tick_params(labelsize=5.6, length=0)
cb.outline.set_visible(False)

p = OUT / "fig_heatmaps_4models.png"
fig.savefig(p, dpi=600, facecolor="white")
print("wrote", p, "shared vmax =", round(VMAX, 1))

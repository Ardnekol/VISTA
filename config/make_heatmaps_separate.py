#!/usr/bin/env python3
"""
One standalone per-(scenario, axis) heatmap per model, all on a SHARED colour
scale (vmax = the global max across the seven models, 43.2%) so that any subset
placed side by side stays comparable. Sized for a single ACL column (3.03in);
scale up in LaTeX if you place two across a figure*.
"""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.family"] = "DejaVu Sans"
OUT = Path("/home/manu/VISTA/outputs")
SEP = OUT / "figs_separate"; SEP.mkdir(exist_ok=True)

MODELS = {
    "gemma":   ("Gemma 4 31B",      OUT / "master_llm_decisions_gemma4.csv"),
    "qwen":    ("Qwen 2.5 32B",     OUT / "outputs" / "master_llm_decisions_qwen.csv"),
    "llama8b": ("LLaMA 3.1 8B",     OUT / "master_llm_decisions_llama_8B.csv"),
    "haiku":   ("Claude Haiku 4.5", OUT / "master_llm_decisions_haiku.csv"),
    "gpt41mini": ("GPT-4.1-mini",   OUT / "master_llm_decisions_gpt41mini.csv"),
    "gpt5mini":  ("GPT-5-mini",     OUT / "master_llm_decisions_gpt5mini.csv"),
    "sonnet":  ("Claude Sonnet 5",  OUT / "master_llm_decisions_sonnet.csv"),
}
AXES = ["authority_signal", "self_preservation", "social_visibility", "time_pressure",
        "in_out_group", "competence_uncertainty", "diffused_responsibility", "resource_scarcity"]
SHORT = {"authority_signal":"auth","self_preservation":"self-pres","social_visibility":"social",
         "time_pressure":"time","in_out_group":"in/out","competence_uncertainty":"comp",
         "diffused_responsibility":"diffused","resource_scarcity":"resource"}

def grid(path):
    d = pd.read_csv(path)
    m = d[d.axis != "BASELINE"]
    m = m[m.llm_changed_from_baseline.isin(["YES", "NO"])]
    return (m.groupby(["scenario_id", "axis"]).llm_changed_from_baseline
              .apply(lambda s: (s == "YES").mean() * 100).unstack())[AXES]

grids = {k: grid(p) for k, (_, p) in MODELS.items()}
VMAX = max(g.values.max() for g in grids.values())

for key, (name, _) in MODELS.items():
    h = grids[key]
    fig, ax = plt.subplots(figsize=(3.031, 2.65))
    fig.subplots_adjust(left=0.22, right=0.86, top=0.90, bottom=0.20)
    im = ax.imshow(h.values, cmap="YlOrRd", vmin=0, vmax=VMAX, aspect="auto")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([SHORT[a] for a in AXES], fontsize=4.8, rotation=45, ha="right")
    ax.set_yticks(range(len(h.index)))
    ax.set_yticklabels(h.index, fontsize=4.8)
    ax.tick_params(length=0, pad=1)
    ax.set_title(name, fontsize=7.0, pad=3)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.2,
                    color="white" if v > 0.62 * VMAX else "#222")
    i, j = np.unravel_index(np.argmax(h.values), h.shape)
    ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="#0b5fa5", lw=1.0, zorder=5))
    for s in ax.spines.values(): s.set_visible(False)
    cax = fig.add_axes([0.875, 0.20, 0.028, 0.70])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=4.5, length=0)
    cb.outline.set_visible(False)
    out = SEP / f"heatmap_{key}.png"
    fig.savefig(out, dpi=600, facecolor="white"); plt.close(fig)
    print("wrote", out.name)
print("shared vmax =", round(VMAX, 1))

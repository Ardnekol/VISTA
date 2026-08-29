#!/usr/bin/env python3
"""
Replacement for fig_spider_all.png.

The original overlaid 9 series on one radar: the LLM polygons collapsed into the
centre (the rule baseline peaks at 34% while Sonnet 5 never exceeds 10%), the
hues were indistinguishable, and at one-column width the tick labels were
unreadable. Nine overlaid series is past the point where any legend can rescue
it, so this facets into small multiples: one panel per system, one series per
panel, a shared radial scale so panels stay comparable, and the spoke labels
factored out into a single key panel.

Outputs (outputs/):
  fig_spider_small_multiples.png  - primary replacement (full-width figure*)
  fig_axis_heatmap.png            - alternative form: systems x axes heatmap
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path("/home/manu/VISTA/outputs")

LLM_FILES = {
    "Gemma 4 31B":  OUT / "master_llm_decisions_gemma4.csv",
    "Qwen 2.5 32B": OUT / "outputs" / "master_llm_decisions_qwen.csv",
    "LLaMA 3.1 8B": OUT / "master_llm_decisions_llama_8B.csv",
    "Haiku 4.5":    OUT / "master_llm_decisions_haiku.csv",
    "GPT-4.1-mini": OUT / "master_llm_decisions_gpt41mini.csv",
    "GPT-5-mini":   OUT / "master_llm_decisions_gpt5mini.csv",
    "Sonnet 5":     OUT / "master_llm_decisions_sonnet.csv",
}
DP_FILE    = OUT / "master_llm_decisions_dotProduct.csv"
HUMAN_AXIS = Path("/home/manu/VISTA/human_study/results/human_vs_llm_per_axis.csv")

AXES = ["social_visibility", "self_preservation", "authority_signal", "resource_scarcity",
        "diffused_responsibility", "competence_uncertainty", "in_out_group", "time_pressure"]

SHORT = {
    "social_visibility":       "social\nvisibility",
    "self_preservation":       "self-\npreserv.",
    "authority_signal":        "authority",
    "resource_scarcity":       "resource\nscarcity",
    "diffused_responsibility": "diffused\nresp.",
    "competence_uncertainty":  "competence\nuncert.",
    "in_out_group":            "in/out\ngroup",
    "time_pressure":           "time\npressure",
}

# The two axes the paper claims are dominant in every model; shaded in each panel
# so the shared lobe is legible without the reader tracing nine overlaid outlines.
DOMINANT = ["self_preservation", "authority_signal"]

COLOR = {
    "Gemma 4 31B":  "#D55E00", "Qwen 2.5 32B": "#009E73", "LLaMA 3.1 8B": "#CC79A7",
    "Haiku 4.5":    "#E69F00", "GPT-4.1-mini": "#0072B2", "GPT-5-mini":   "#56B4E9",
    "Sonnet 5":     "#8B4513",
}
INK, MUTED, GRID = "#1a1a1a", "#666666", "#d9d9d9"


# -- data ---------------------------------------------------------------------
def axis_rates(path, col):
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] != "BASELINE"]
    out = {}
    for ax in AXES:
        rs = [r for r in rows if r["axis"] == ax]
        out[ax] = 100 * sum(1 for r in rs if r[col] == "YES") / len(rs) if rs else 0.0
    return out


def overall_rate(path, col):
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] != "BASELINE"]
    return 100 * sum(1 for r in rows if r[col] == "YES") / len(rows)


llm_rates = {m: axis_rates(p, "llm_changed_from_baseline") for m, p in LLM_FILES.items()}
dp_rates  = axis_rates(DP_FILE, "dp_changed_from_baseline")
hum_rates = {r["axis"]: 100 * float(r["human_mean_abs_shift"])
             for r in csv.DictReader(open(HUMAN_AXIS))}

overall = {m: overall_rate(p, "llm_changed_from_baseline") for m, p in LLM_FILES.items()}
overall["Dot-product rule"] = overall_rate(DP_FILE, "dp_changed_from_baseline")
overall["Human pilot"] = 12.36

# LLMs ordered by overall sensitivity, most-sensitive first
LLM_ORDER = sorted(LLM_FILES, key=lambda m: -overall[m])
# reference envelope = mean across the seven LLMs, drawn faintly behind every panel
llm_mean = {a: float(np.mean([llm_rates[m][a] for m in LLM_FILES])) for a in AXES}

PANELS = (
    [("Dot-product rule", dp_rates, "#7f7f7f", "rule baseline"),
     ("Human pilot (N=50)", hum_rates, INK, "human")]
    + [(m, llm_rates[m], COLOR[m], "llm") for m in LLM_ORDER]
)

RMAX = 36.0
RINGS = [10, 20, 30]


def _spokes():
    a = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    return a, np.concatenate([a, a[:1]])


def _style_panel(ax, show_ring_labels=False):
    ang, _ = _spokes()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, RMAX)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.set_yticks(RINGS)
    ax.set_yticklabels([f"{r}%" for r in RINGS] if show_ring_labels else [],
                       fontsize=7.5, color=MUTED)
    ax.tick_params(pad=0)
    # park the ring labels in the sector no polygon reaches into
    ax.set_rlabel_position(202)
    ax.grid(color=GRID, lw=0.5)
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(0.6)
    # shade the two dominant axes
    width = 2 * np.pi / len(AXES)
    for a in DOMINANT:
        ax.bar(ang[AXES.index(a)], RMAX, width=width * 0.92, bottom=0,
               color="#f0b73f", alpha=0.13, edgecolor="none", zorder=0)


def small_multiples():
    ang, angc = _spokes()

    def close(d):
        v = np.array([d[a] for a in AXES])
        return np.concatenate([v, v[:1]])

    # Sized so that scaling to \textwidth (~7in) leaves every label above ~7pt;
    # a wider canvas would be downscaled harder and undo the legibility fix.
    fig, axes = plt.subplots(2, 5, figsize=(10.6, 5.3),
                             subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.46, hspace=0.60, top=0.88, bottom=0.04)
    flat = axes.ravel()

    ref = close(llm_mean)

    for i, (name, rates, color, kind) in enumerate(PANELS):
        ax = flat[i]
        _style_panel(ax)
        # faint 7-LLM mean behind every panel, so each polygon is read against
        # a common shape rather than against eight competing outlines
        ax.plot(angc, ref, color="#b0b0b0", lw=0.8, ls=(0, (3, 2)), zorder=2)

        v = close(rates)
        ax.fill(angc, v, color=color, alpha=0.20, zorder=3)
        ax.plot(angc, v, color=color, lw=1.9, zorder=4,
                ls="--" if kind == "human" else "-")
        ax.plot(angc, v, "o", color=color, ms=2.6, zorder=5)

        label = f"{name}\n{overall[name.split(' (')[0]]:.1f}% overall"
        ax.set_title(label, fontsize=10.5, color=INK, pad=7, linespacing=1.35)

    # Key panel: the spoke layout and the radial scale, labelled once for the
    # whole figure. The ring labels live here because this is the only panel
    # with no polygon for them to collide with.
    ax = flat[9]
    _style_panel(ax, show_ring_labels=True)
    for a, lab in zip(ang, [SHORT[x] for x in AXES]):
        ax.plot([a, a], [0, RMAX], color="#9a9a9a", lw=0.7, zorder=3)
        ax.text(a, RMAX * 1.36, lab, ha="center", va="center",
                fontsize=9.0, color=INK, linespacing=1.15)
    # sits clear of the top spoke label, which occupies the usual title slot
    ax.set_title("axis key", fontsize=10.5, color=INK, pad=32)

    # No figure title or subtitle: the LaTeX caption carries both, and dropping
    # them returns the space to the panels.

    p = OUT / "fig_spider_small_multiples.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def heatmap():
    """Alternative form: magnitude on a grid reads better as a heatmap."""
    systems = ["Dot-product rule", "Human pilot"] + LLM_ORDER
    rate_of = {"Dot-product rule": dp_rates, "Human pilot": hum_rates}
    rate_of.update(llm_rates)

    order = sorted(AXES, key=lambda a: -np.mean([llm_rates[m][a] for m in LLM_FILES]))
    M = np.array([[rate_of[s][a] for a in order] for s in systems])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=36, aspect="auto")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([a.replace("_", "\n") for a in order], fontsize=7.6, color=INK)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([f"{s}  ({overall[s]:.1f}%)" for s in systems],
                       fontsize=8.5, color=INK)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.4,
                    color="white" if v > 21 else INK)

    # separate the two reference rows from the LLM block
    ax.axhline(1.5, color="white", lw=2.5)
    for j in range(len(order) - 1):
        ax.axvline(j + 0.5, color="white", lw=1.2)
    for i in range(len(systems) - 1):
        ax.axhline(i + 0.5, color="white", lw=1.2)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Per-axis flip rate (%), axes sorted by mean LLM sensitivity",
                 fontsize=11, color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("flip rate (%)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, length=0, colors=MUTED)
    cb.outline.set_visible(False)

    p = OUT / "fig_axis_heatmap.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", small_multiples())
    print("wrote", heatmap())

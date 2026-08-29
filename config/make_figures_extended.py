#!/usr/bin/env python3
"""
Regenerate the paper's figures including the four added closed models
(Haiku 4.5, GPT-4.1-mini, GPT-5-mini, Sonnet 5), matching the existing style.

Outputs (outputs/):
  fig_spider_all.png        — per-axis sensitivity radar, 7 LLMs + human + rule
  fig_flip_rate_bar.png     — overall flip-rate scoreboard (capability != sensitivity)
  fig_modifier_type_all.png — grouped bar by modifier-type across models
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/manu/VISTA/outputs")

# master CSVs (paper's main-body open-weight set + the 4 new closed models)
LLM_FILES = {
    "Gemma 4 31B":  OUT / "master_llm_decisions_gemma4.csv",
    "Qwen 2.5 32B": OUT / "master_llm_decisions_qwen.csv",   # symlink/real below
    "Llama 3.1 8B": OUT / "master_llm_decisions_llama_8B.csv",
    "Haiku 4.5":    OUT / "master_llm_decisions_haiku.csv",
    "GPT-4.1-mini": OUT / "master_llm_decisions_gpt41mini.csv",
    "GPT-5-mini":   OUT / "master_llm_decisions_gpt5mini.csv",
    "Sonnet 5":     OUT / "master_llm_decisions_sonnet.csv",
}
# qwen master lives under outputs/outputs/
if not LLM_FILES["Qwen 2.5 32B"].exists():
    LLM_FILES["Qwen 2.5 32B"] = OUT / "outputs" / "master_llm_decisions_qwen.csv"
DP_FILE    = OUT / "master_llm_decisions_dotProduct.csv"
HUMAN_AXIS = Path("/home/manu/VISTA/human_study/results/human_vs_llm_per_axis.csv")

# radar axis order (matches the existing figure: social_visibility at top, clockwise)
AXES = ["social_visibility", "self_preservation", "authority_signal", "resource_scarcity",
        "diffused_responsibility", "competence_uncertainty", "in_out_group", "time_pressure"]
AXIS_LABEL = {a: a.replace("_", "\n") for a in AXES}

TYPES = {"Stakes": ["resource_scarcity"],
         "Affective": ["authority_signal", "social_visibility", "in_out_group"],
         "Personal-cost": ["self_preservation", "time_pressure"],
         "Informational": ["diffused_responsibility", "competence_uncertainty"]}

# colorblind-safe (validated): CVD adjacent ΔE 17.9
COLOR = {
    "Gemma 4 31B":  "#D55E00", "Qwen 2.5 32B": "#009E73", "Llama 3.1 8B": "#CC79A7",
    "Haiku 4.5":    "#E69F00", "GPT-4.1-mini": "#0072B2", "GPT-5-mini":   "#56B4E9",
    "Sonnet 5":     "#8B4513",
}
LLM_ORDER = list(LLM_FILES.keys())


def llm_axis_rates(path):
    rates = {}
    rows = list(csv.DictReader(open(path)))
    mod = [r for r in rows if r["condition"] != "BASELINE"]
    for ax in AXES:
        rs = [r for r in mod if r["axis"] == ax]
        f = sum(1 for r in rs if r["llm_changed_from_baseline"] == "YES")
        rates[ax] = 100 * f / len(rs) if rs else 0
    return rates

def dp_axis_rates(path):
    rates = {}
    rows = list(csv.DictReader(open(path)))
    mod = [r for r in rows if r["condition"] != "BASELINE"]
    for ax in AXES:
        rs = [r for r in mod if r["axis"] == ax]
        f = sum(1 for r in rs if r["dp_changed_from_baseline"] == "YES")
        rates[ax] = 100 * f / len(rs) if rs else 0
    return rates

def overall_rate(path, dp=False):
    col = "dp_changed_from_baseline" if dp else "llm_changed_from_baseline"
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] != "BASELINE"]
    f = sum(1 for r in rows if r[col] == "YES")
    return 100 * f / len(rows)

def human_axis_rates():
    out = {}
    for r in csv.DictReader(open(HUMAN_AXIS)):
        out[r["axis"]] = 100 * float(r["human_mean_abs_shift"])
    return out


# ── gather data ──────────────────────────────────────────────────────────────
llm_rates = {m: llm_axis_rates(p) for m, p in LLM_FILES.items()}
dp_rates  = dp_axis_rates(DP_FILE)
hum_rates = human_axis_rates()
overall   = {m: overall_rate(p) for m, p in LLM_FILES.items()}
overall["Dot-Product (rule)"] = overall_rate(DP_FILE, dp=True)
overall["Human pilot"] = 12.36  # pooled |ΔP(A1)| from the human pilot


# ── FIGURE 1: radar ──────────────────────────────────────────────────────────
def radar():
    N = len(AXES)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)

    def close(vals): return np.concatenate([vals, vals[:1]])

    # dot-product reference envelope (light gray fill)
    dpv = close(np.array([dp_rates[a] for a in AXES]))
    ax.plot(ang, dpv, color="#7f7f7f", lw=2.5, label="Dot-Product (rule)")
    ax.fill(ang, dpv, color="#7f7f7f", alpha=0.08)
    # human (dashed black)
    hv = close(np.array([hum_rates[a] for a in AXES]))
    ax.plot(ang, hv, color="#111111", lw=2.5, ls="--", marker="o", ms=4, label="Human (N=50)")
    # LLMs (thin solid, no fill)
    for m in LLM_ORDER:
        v = close(np.array([llm_rates[m][a] for a in AXES]))
        ax.plot(ang, v, color=COLOR[m], lw=1.8, label=m)

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=11)
    ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
    ax.set_yticklabels(["5%", "10%", "15%", "20%", "25%", "30%", "35%"], fontsize=8, color="#555")
    ax.set_ylim(0, 36)
    ax.set_title("Per-axis sensitivity by system", fontsize=15, pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=10, frameon=False)
    fig.savefig(OUT / "fig_spider_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── FIGURE 2: overall flip-rate scoreboard ───────────────────────────────────
def scoreboard():
    order = sorted(overall.items(), key=lambda kv: kv[1])
    names = [k for k, _ in order]; vals = [v for _, v in order]
    def barcolor(n):
        if n == "Human pilot": return "#111111"
        if n == "Dot-Product (rule)": return "#7f7f7f"
        return COLOR.get(n, "#999999")
    colors = [barcolor(n) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, height=0.62)
    for n, b, v in zip(names, bars, vals):
        hatch = None
        if n in ("Human pilot", "Dot-Product (rule)"):
            b.set_hatch("///"); b.set_alpha(0.85)
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9, color="#222")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Overall modifier flip rate (%)", fontsize=11)
    ax.set_xlim(0, 22)
    ax.set_title("Situational-modifier sensitivity is not monotonic in capability",
                 fontsize=12.5, pad=10)
    ax.axvline(12.36, color="#111111", ls=":", lw=1, alpha=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_flip_rate_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── FIGURE 3: modifier-type grouped bar ──────────────────────────────────────
def type_bar():
    def type_rate(rates, axes):
        return float(np.mean([rates[a] for a in axes]))
    models = ["Human"] + LLM_ORDER
    type_names = list(TYPES.keys())
    data = {}
    data["Human"] = [type_rate(hum_rates, TYPES[t]) for t in type_names]
    for m in LLM_ORDER:
        data[m] = [type_rate(llm_rates[m], TYPES[t]) for t in type_names]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(type_names)); w = 0.10
    for i, m in enumerate(models):
        c = "#111111" if m == "Human" else COLOR[m]
        off = (i - (len(models) - 1) / 2) * w
        ax.bar(x + off, data[m], w, label=("Human (N=50)" if m == "Human" else m),
               color=c, alpha=0.9 if m != "Human" else 1.0,
               hatch="///" if m == "Human" else None)
    ax.set_xticks(x); ax.set_xticklabels(type_names, fontsize=11)
    ax.set_ylabel("Mean decision shift (%)", fontsize=11)
    ax.set_title("Modifier-type pressure: humans vs. LLMs", fontsize=13, pad=10)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.8); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_modifier_type_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    radar(); scoreboard(); type_bar()
    print("Wrote:")
    for f in ["fig_spider_all.png", "fig_flip_rate_bar.png", "fig_modifier_type_all.png"]:
        print("  outputs/" + f)
    print("\nOverall flip rates:")
    for k, v in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:5.2f}%")

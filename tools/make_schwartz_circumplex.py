"""Schwartz value circumplex (10-value version) for the VISTA paper.

Two concentric rings:
  - Inner: 10 value wedges with radial labels.
  - Outer ring: 4 higher-order quadrants (Openness to Change,
    Self-Enhancement, Conservation, Self-Transcendence) in
    alternating dark/light green.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "schwartz_circumplex.png"

# Clockwise from 12 o'clock
VALUES = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Tradition",
    "Conformity",
    "Universalism",
    "Benevolence",
]

QUADRANT_OF = {
    "Self-Direction":  "Openness to Change",
    "Stimulation":     "Openness to Change",
    "Hedonism":        "Openness to Change",
    "Achievement":     "Self-Enhancement",
    "Power":           "Self-Enhancement",
    "Security":        "Conservation",
    "Tradition":       "Conservation",
    "Conformity":      "Conservation",
    "Universalism":    "Self-Transcendence",
    "Benevolence":     "Self-Transcendence",
}

QUADRANT_FILL = {
    "Openness to Change":   "#CFE6C6",  # light green
    "Self-Enhancement":     "#2E8B57",  # dark green
    "Conservation":         "#CFE6C6",
    "Self-Transcendence":   "#2E8B57",
}
QUADRANT_TEXT = {
    "Openness to Change":   "#B33A3A",
    "Self-Enhancement":     "#FFFFFF",
    "Conservation":         "#B33A3A",
    "Self-Transcendence":   "#FFFFFF",
}

R_VALUE_OUT = 0.66
R_QUAD_OUT  = 0.94

DEG_PER_VALUE = 360 / len(VALUES)  # = 36


def value_wedge(i):
    """Return matplotlib (theta1, theta2) for the i-th value wedge.
    Wedge 0 starts at top (90°) and goes clockwise (decreasing angle)."""
    theta2 = 90 - i * DEG_PER_VALUE
    theta1 = theta2 - DEG_PER_VALUE
    return theta1, theta2


def _normalize(angle):
    """Normalize an angle to (-180, 180]."""
    return ((angle + 180) % 360) - 180


def _upright(rot):
    """Bring rotation into (-90, 90] so text is never upside-down."""
    rot = _normalize(rot)
    if rot > 90 or rot <= -90:
        rot = _normalize(rot + 180)
    return rot


def radial_rotation(mid_deg):
    """Rotate so the text reads along a radial spoke, always upright."""
    return _upright(mid_deg)


def tangential_rotation(mid_deg):
    """Rotate so the text is tangent to the circle, always upright."""
    return _upright(mid_deg - 90)


def main():
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- Outer ring (quadrants) ----------------
    for quad in ("Openness to Change", "Self-Enhancement",
                 "Conservation", "Self-Transcendence"):
        idxs = [i for i, v in enumerate(VALUES) if QUADRANT_OF[v] == quad]
        # angular extent across the indices (wedges are contiguous)
        t1 = value_wedge(max(idxs))[0]
        t2 = value_wedge(min(idxs))[1]
        if t1 > t2:
            t2 += 360
        wedge = mpatches.Wedge((0, 0), R_QUAD_OUT, t1, t2,
                                width=R_QUAD_OUT - R_VALUE_OUT,
                                facecolor=QUADRANT_FILL[quad],
                                edgecolor="white", linewidth=2)
        ax.add_patch(wedge)
        mid = ((t1 + t2) / 2) % 360
        label_r = (R_VALUE_OUT + R_QUAD_OUT) / 2
        x = label_r * np.cos(np.radians(mid))
        y = label_r * np.sin(np.radians(mid))
        ax.text(x, y, quad, ha="center", va="center",
                fontsize=18, fontweight="bold",
                color=QUADRANT_TEXT[quad],
                rotation=tangential_rotation(mid))

    # ---------------- Inner value wedges ----------------
    for i, value in enumerate(VALUES):
        t1, t2 = value_wedge(i)
        wedge = mpatches.Wedge((0, 0), R_VALUE_OUT, t1, t2,
                                facecolor="white", edgecolor="#555",
                                linewidth=1.0)
        ax.add_patch(wedge)
        mid = (t1 + t2) / 2
        # Place label about 70% out, so name reads radially
        label_r = R_VALUE_OUT * 0.65
        x = label_r * np.cos(np.radians(mid))
        y = label_r * np.sin(np.radians(mid))
        ax.text(x, y, value, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#222",
                rotation=radial_rotation(mid))

    # ---------------- Center bubble ----------------
    center = plt.Circle((0, 0), 0.19, facecolor="white",
                         edgecolor="#666", linewidth=1.4, zorder=5)
    ax.add_patch(center)
    ax.text(0, 0.045, "Schwartz", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#333", zorder=6)
    ax.text(0, -0.055, "Values", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#333", zorder=6)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

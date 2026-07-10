"""8-axis modifier circumplex for the VISTA paper.

Single ring with 8 wedges (one per situational modifier axis) around a
central bubble. Same style as the Schwartz circumplex figure.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "modifier_circumplex.png"

# Clockwise from 12 o'clock
AXES = [
    "Self-Preservation",
    "Resource Scarcity",
    "Social Visibility",
    "In / Out-Group",
    "Time Pressure",
    "Diffused Responsibility",
    "Competence Uncertainty",
    "Authority Signal",
]

R_AXIS_OUT = 0.86
DEG_PER = 360 / len(AXES)  # = 45


def wedge_angles(i):
    theta2 = 90 - i * DEG_PER
    theta1 = theta2 - DEG_PER
    return theta1, theta2


def _normalize(angle):
    return ((angle + 180) % 360) - 180


def _upright(rot):
    rot = _normalize(rot)
    if rot > 90 or rot <= -90:
        rot = _normalize(rot + 180)
    return rot


def radial_rotation(mid_deg):
    return _upright(mid_deg)


def main():
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- 8 axis wedges ----------------
    for i, axis in enumerate(AXES):
        t1, t2 = wedge_angles(i)
        wedge = mpatches.Wedge((0, 0), R_AXIS_OUT, t1, t2,
                                facecolor="white", edgecolor="#555",
                                linewidth=1.0)
        ax.add_patch(wedge)
        mid = (t1 + t2) / 2
        label_r = R_AXIS_OUT * 0.62
        x = label_r * np.cos(np.radians(mid))
        y = label_r * np.sin(np.radians(mid))
        ax.text(x, y, axis, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#222",
                rotation=radial_rotation(mid))

    # ---------------- Center bubble ----------------
    center = plt.Circle((0, 0), 0.20, facecolor="white",
                         edgecolor="#666", linewidth=1.4, zorder=5)
    ax.add_patch(center)
    ax.text(0, 0.055, "Situational", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#333", zorder=6)
    ax.text(0, -0.055, "Modifiers", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#333", zorder=6)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

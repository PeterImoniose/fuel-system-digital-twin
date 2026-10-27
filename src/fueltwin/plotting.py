"""Shared matplotlib style so every figure in the project reads as one system.

Colour rules:
* Healthy is the reference state, drawn in neutral ink (not a categorical hue).
* Faults F1-F5 take the first five categorical slots of the reference palette,
  in fixed order, so a fault keeps its colour in every figure.
* In scatter plots (where five hues are too many to separate by colour alone)
  every condition also has its own marker shape, and points are direct-labelled.
* Sequential maps use a single blue ramp; signed deviations use blue <-> red
  around a grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIG_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CONDITION_COLORS = {
    "HC": "#52514e",
    "F1": "#2a78d6",  # blue
    "F2": "#eb6834",  # orange
    "F3": "#1baf7a",  # aqua
    "F4": "#eda100",  # yellow
    "F5": "#e87ba4",  # magenta
}
CONDITION_MARKERS = {"HC": "o", "F1": "s", "F2": "D", "F3": "^", "F4": "v", "F5": "P"}

# Categorical slots for non-condition series (e.g. pump models, campaigns).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2b"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig: plt.Figure, name: str, folder: Path = FIG_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def schematic(ax: plt.Axes, highlight: dict[str, str] | None = None) -> None:
    """Simplified line diagram of the 2014 fuel rig (after the Lab 1 schematic).

    ``highlight`` maps component keys (filter, pump, valve, nozzle, bypass, leak)
    to a colour, e.g. to mark where each fault is injected.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    hl = highlight or {}
    ax.set_xlim(-0.3, 13.6)
    ax.set_ylim(-2.3, 1.5)
    ax.axis("off")
    y = 0.0
    ax.plot([0.6, 12.4], [y, y], color=INK_2, lw=2.2, zorder=1, solid_capstyle="round")
    # flow direction arrows
    for x in (1.6, 4.6, 7.6, 10.6):
        ax.add_patch(FancyArrowPatch((x - 0.25, y), (x + 0.15, y), arrowstyle="-|>", mutation_scale=10,
                                     color=INK_2, lw=0, zorder=2))

    def box(x, w, label, key=None, sub=None):
        c = hl.get(key, None)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.32), w, 0.64, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c if c else SURFACE, ec=c if c else INK_2, lw=1.3, alpha=1.0, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=4,
                color="white" if c else INK, fontweight="bold" if c else "normal")
        if sub:
            ax.text(x, y - 0.55, sub, ha="center", va="top", fontsize=7.5, color=MUTED)

    def sensor(x, label):
        ax.plot([x, x], [y, y + 0.55], color=MUTED, lw=1, zorder=1)
        ax.add_patch(plt.Circle((x, y + 0.78), 0.24, fc=SURFACE, ec=INK_2, lw=1.1, zorder=3))
        ax.text(x, y + 0.78, label, ha="center", va="center", fontsize=7.5, zorder=4)

    box(0.15, 0.9, "Main\ntank")
    sensor(1.2, "P1")
    box(2.4, 1.2, "Filter", "filter", "DPV1 (F1)")
    sensor(3.5, "P2")
    box(4.7, 1.3, "Gear pump", "pump", "N999R, 400 rpm")
    sensor(5.9, "P3")
    box(7.2, 1.5, "Shut-off\nvalve", "valve", "DPV3 (F3)")
    sensor(8.5, "P4")
    box(9.7, 1.2, "Nozzle", "nozzle", "DPV5 (F5)")
    sensor(10.8, "P5")
    sensor(11.6, "Q")
    box(12.9, 0.9, "Sump\ntank")
    # return paths to the main tank: pump bypass and leak
    for x0, key, label in [(5.9, "bypass", "DPV2 bypass (F2)"), (8.9, "leak", "DPV4 leak (F4)")]:
        c = hl.get(key, MUTED)
        lvl = -1.35 if key == "bypass" else -1.85
        ax.plot([x0, x0, 0.15, 0.15], [y - 0.05, lvl, lvl, -0.35], color=c, lw=1.4, ls=(0, (4, 2)), zorder=1)
        ax.text(x0 - 0.1, lvl + 0.08, label, ha="right", va="bottom", fontsize=7.5, color=INK_2)


def label_end(ax: plt.Axes, x, y, text: str, color: str = INK_2, dx: float = 4, **kw) -> None:
    """Direct label at the end of a line, in ink colour (never the series colour)."""
    opts = {"va": "center", "fontsize": 8.5, "color": color} | kw
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", **opts)

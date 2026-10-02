"""Shared chart styling.

Palette slots come from the validated reference palette; the two-series categorical
pair (#2a78d6 blue, #eb6834 orange) and the diverging poles (#2a78d6 blue,
#e34948 red) were both run through the palette validator and pass every check on
the light surface under the all-pairs pairlist:

  CVD separation   blue<->orange  dE 24.7 (protan) / 32.7 (tritan)
  normal vision    blue<->orange  dE 33.6
  contrast         both >= 3:1 against #fcfcfb

Light mode only, deliberately: these render to PNG for a printed thesis, so there is
one surface and no theme toggle. Because print has no hover layer, every value a
reader needs is either directly labelled or present in the paired CSV table under
outputs/tables -- no value is reachable only by interaction.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ palette slots
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {"Logistic Regression": "#2a78d6", "XGBoost": "#eb6834"}
SERIES_ORDER = ["Logistic Regression", "XGBoost"]

DIVERGING_NEG = "#2a78d6"   # cool pole
DIVERGING_POS = "#e34948"   # warm pole
NEUTRAL = "#f0efec"

# Documented single-hue blue ramp, light -> dark.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

FIG_DPI = 300


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,               # hairline axis
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",               # solid; dashed grids read as thresholds
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,              # 2px lines
            "lines.markersize": 7,               # >= 8px markers
            "lines.solid_capstyle": "round",
        }
    )


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def caption(fig, text: str) -> None:
    """One recessive line under the plot for the method note a thesis figure needs."""
    fig.text(0.0, -0.045, text, ha="left", va="top", fontsize=7.8, color=INK_MUTED)

"""
Shared visual style for all thesis figures.

Single source of truth for colours, colormaps, fonts and sizes so every figure
is consistent. Import this in each figure script:

    import thesis_style as ts
    ts.apply()                       # set global rcParams
    cmap, norm = ts.AUROC_CMAP, ts.auroc_norm()
    ts.save(fig, "fig_quality_blindness")   # -> figures/fig_quality_blindness.{pdf,svg}

Colour semantics (kept disjoint so meanings never collide):
  - AUROC / description quality -> PiYG diverging (green = high, pink = low,
    white = chance 0.5). Colourblind-safe (unlike the old RdYlGn). Always paired
    with a dashed chance line at 0.5.
  - Filter outcome            -> shape + blue: circle = survived, diamond =
    removed, and #1E90FF blue border/edge = collapsed.
  - Model (categorical)       -> fixed Okabe-Ito colours + marker shapes, so the
    same model is the same colour AND shape (works in greyscale) in every plot.
  - Correlation heatmaps      -> RdBu_r diverging, centred at 0, range [-1, 1].
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm

# --- Output location -------------------------------------------------------
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- AUROC / quality -------------------------------------------------------
AUROC_CMAP = plt.cm.PiYG
CHANCE = 0.5


def auroc_norm(vmin=0.0, vmax=1.0):
    return TwoSlopeNorm(vcenter=CHANCE, vmin=vmin, vmax=vmax)


def add_auroc_colorbar(fig, mappable_or_ax, label="AUROC", **kw):
    """Attach an AUROC colourbar with the dashed chance-level marker."""
    if isinstance(mappable_or_ax, plt.cm.ScalarMappable):
        sm = mappable_or_ax
    else:
        sm = plt.cm.ScalarMappable(cmap=AUROC_CMAP, norm=auroc_norm())
        sm.set_array([])
    cbar = fig.colorbar(sm, **kw)
    cbar.set_label(label, rotation=270, labelpad=18)
    cbar.ax.axhline(y=CHANCE, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    return cbar


# --- Filter outcome --------------------------------------------------------
COLLAPSED_BLUE = "#1E90FF"
FILTER_MARKER = {"survived": "o", "removed": "D", "collapsed": "D"}
FILTER_EDGE = {"survived": "black", "removed": "black", "collapsed": COLLAPSED_BLUE}


def filter_legend_handles():
    """Handles for the survived/removed/collapsed legend (shape + blue edge)."""
    return [
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
                      markeredgecolor="black", markersize=9, markeredgewidth=1.2, label="Survived"),
        mlines.Line2D([], [], marker="D", linestyle="None", markerfacecolor="white",
                      markeredgecolor="black", markersize=8, markeredgewidth=1.2, label="Removed"),
        mlines.Line2D([], [], color=COLLAPSED_BLUE, linewidth=1.8, label="Collapsed"),
    ]


# --- Models (categorical) --------------------------------------------------
# Okabe-Ito, avoiding green (reserved for AUROC) and the collapse blue.
MODEL_COLORS = {
    "gpt2-xl":                "#D55E00",  # vermillion
    "Llama-3.1-8B-Instruct":  "#0072B2",  # blue
    "gemma-scope-2b":         "#E69F00",  # orange
    "gpt2-small-sae":         "#CC79A7",  # reddish purple
}
MODEL_MARKERS = {
    "gpt2-xl": "o", "Llama-3.1-8B-Instruct": "s",
    "gemma-scope-2b": "^", "gpt2-small-sae": "D",
}
MODEL_ORDER = ["gpt2-xl", "Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae"]
MODEL_LABEL = {
    "gpt2-xl": "GPT-2 XL", "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-scope-2b": "Gemma-Scope-2B", "gpt2-small-sae": "GPT-2-small SAE",
}

# Characterisation-status categories (for the stacked bars).
STATUS_COLORS = {
    "uncharacterised": "#bdbdbd",   # grey
    "single":          "#9ecae1",   # light blue
    "multi":           "#3182bd",   # dark blue
}

# --- Correlation heatmaps --------------------------------------------------
CORR_CMAP = "RdBu_r"   # red = +, blue = -, colourblind-safer than red-green
CORR_VMIN, CORR_VMAX = -1.0, 1.0


# --- Global rcParams -------------------------------------------------------
def apply():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": False,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "pdf.fonttype": 42,   # editable text in PDFs (no type-3 fonts)
        "ps.fonttype": 42,
    })


def despine(ax, which=("top", "right")):
    """Remove the given spines (clean look for scatter/bar plots)."""
    for s in which:
        ax.spines[s].set_visible(False)


def save(fig, name, formats=("png", "pdf", "svg")):
    """Save a figure into figures/ in each requested vector format."""
    paths = []
    for ext in formats:
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    # Palette preview swatch.
    apply()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    # models
    ax = axes[0]
    for i, m in enumerate(MODEL_ORDER):
        ax.scatter([i], [0], s=400, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                   edgecolors="black")
        ax.text(i, -0.35, MODEL_LABEL[m], ha="center", fontsize=9, rotation=15)
    ax.set_title("Model colours + markers"); ax.set_xlim(-.5, 3.5); ax.set_ylim(-.7, .5); ax.axis("off")
    # AUROC colorbar
    ax = axes[1]
    sm = plt.cm.ScalarMappable(cmap=AUROC_CMAP, norm=auroc_norm()); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.5, pad=0.2)
    cb.set_label("AUROC (PiYG, chance = 0.5)")
    cb.ax.axvline(x=0.5, color="black", ls="--", lw=1.2, alpha=0.6)
    ax.set_title("AUROC / quality"); ax.axis("off")
    # filter legend + status
    ax = axes[2]
    ax.legend(handles=filter_legend_handles(), loc="upper center", ncol=1, title="Filter outcome")
    for i, (k, c) in enumerate(STATUS_COLORS.items()):
        ax.add_patch(plt.Rectangle((0.55, 0.15 + i * 0.18), 0.12, 0.12, color=c))
        ax.text(0.70, 0.21 + i * 0.18, k, fontsize=9, va="center")
    ax.set_title("Filter outcome + status colours"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    out = save(fig, "_palette_preview", formats=("png",))
    print("Palette preview saved to", out[0])

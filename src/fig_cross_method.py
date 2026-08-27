"""
Figure 4 (3 panels): the revised metric and its generality across sources.

(a) Revised weighted similarity sigma-tilde vs. the original PRISM similarity
    score (multi-survivor features, all four models) — the revised metric relates
    to but differs from the raw score.
(b) Feature-level evidence quality Q from PRISM-only vs. from the full pooled
    source set (gpt2-xl) — Q is robust to source pooling in rank.
(c) Fate of each non-input-centric description (output-centric on three targets;
    neuronpedia on gpt2-small-sae) — filtered / retained as distinct / redundant
    relative to the input-centric (PRISM) set.

Panels (b)/(c) read the reproducible pooling artifacts (results/pooled_*/).
Run: python src/fig_cross_method.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr, beta

import thesis_style as ts
from pool_gpt2xl_analyze import filter_pool, metrics

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
GX = RESULTS / "pooled_gpt2xl"
OC = RESULTS / "pooled_outputcentric"

CAT = ["filtered", "retained as distinct", "redundant"]
# Cross-method panels (b)/(c). Colour encodes ONE thing consistently across both:
# whether the comparison is budget-controlled. PRISM contributes 5 candidate
# descriptions per feature, every other method 1, so only Transluce~max-act is a
# like-for-like pair. Validated as a 2-slot categorical palette (all six checks
# pass; CVD dE 22.8 protan / 18.3 tritan). Avoids green (reserved for AUROC) and
# blue (reserved for collapse).
# Authored at TRUE PRINT SIZE. The figure is placed at \linewidth = 5.5in, so it
# is built 5.5in wide and never downscaled -- point sizes below are the sizes that
# actually print. Previously it was authored 14.4in wide and shrunk 2.62x on
# placement, which turned 8.5pt tick labels into 3.2pt and panel (a)'s legend
# into 2.5pt. That, not the font numbers, was why the labels were unreadable.
FIG_W, FIG_H = 5.5, 1.95
FS_TITLE, FS_LABEL, FS_VALUE, FS_LEGEND = 8, 7, 6.5, 5.8

XM_EQUAL   = "#5E4FA2"   # equal proposal budget  -> controlled comparison
XM_UNEQUAL = "#A6611A"   # unequal budget         -> descriptive only
CAT_COLOR = {"filtered": "#c9c9c9", "retained as distinct": "#4c9f70", "redundant": ts.COLLAPSED_BLUE}


def _ci(k, n, alpha=0.05):
    """Clopper-Pearson interval, in percent. Shown because n ranges from 10 to 58,
    so identical percentages carry very different weight across rows."""
    lo = 100 * beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = 100 * beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 100.0
    return lo, hi


def _panel_a(ax):
    q = pd.read_csv(RESULTS / "quality_blindness_check.csv")
    fe = pd.read_csv(RESULTS / "filtered_evaluation.csv")
    mm = q.merge(fe, on=["model", "layer", "unit"])
    mm = mm[mm["n_surviving"] >= 2]
    rho = spearmanr(mm["prism_poly"], mm["weighted_mean_similarity"]).statistic
    lo = min(mm["prism_poly"].min(), mm["weighted_mean_similarity"].min()) - 0.03
    hi = max(mm["prism_poly"].max(), mm["weighted_mean_similarity"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1, alpha=0.6, zorder=0)
    for m in ts.MODEL_ORDER:
        s = mm[mm["model"] == m]
        lab = "GPT-2 SAE" if m == "gpt2-small-sae" else ts.MODEL_LABEL[m]
        ax.scatter(s["prism_poly"], s["weighted_mean_similarity"], color=ts.MODEL_COLORS[m],
                   marker=ts.MODEL_MARKERS[m], edgecolors="black", linewidths=0.4, s=26,
                   alpha=0.85, label=lab)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\mathrm{PRISM}_{\mathrm{poly}}$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\widetilde{\sigma}$", fontsize=FS_TITLE, labelpad=1)
    ax.set_title("(a)", fontsize=FS_TITLE, loc="left", pad=3)
    ax.annotate(rf"$\rho$ = {rho:.3f}, $n$ = {len(mm)}", xy=(0.04, 0.96), xycoords="axes fraction",
                va="top", ha="left", fontsize=FS_VALUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
    ts.despine(ax)


def _panel_b(ax):
    """Directional complementary coverage for the equal-budget new-regime pair."""
    rescue = json.loads((RESULTS / "cross_method_rescue" / "stats.json").read_text())
    nm = rescue["new_regime"]["matrix"]
    rows = [(r"Transluce $\rightarrow$ max-act", nm["Transluce"]["max-act"]["pooled"]),
            (r"max-act $\rightarrow$ Transluce", nm["max-act"]["Transluce"]["pooled"])]
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, d) in zip(y, rows):
        k, n = d["n_covered_by_B"], d["n_uncharacterized_by_A"]
        ax.barh(yi, 100 * k / n, height=0.34, color=XM_EQUAL, edgecolor="none", zorder=3)
        lo, hi = _ci(k, n)
        ax.plot([lo, hi], [yi, yi], color="#9aa4b0", lw=1.0, solid_capstyle="round", zorder=4)
        # label ABOVE the bar, inside the axes: at this width a y-axis label
        # column would leave almost no room for the data
        ax.text(1.5, yi + 0.24, lbl, va="bottom", ha="left", fontsize=FS_LABEL, color="#222")
        ax.text(hi + 2.5, yi, f"{100 * k / n:.1f}% ({k}/{n})",
                va="center", ha="left", fontsize=FS_VALUE, color="#555")
    ax.set_yticks([]); ax.set_ylim(-0.60, len(rows) - 0.10)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Complementary coverage (%)", fontsize=FS_LABEL)
    ax.set_title("(b)", fontsize=FS_TITLE, loc="left", pad=3)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ts.despine(ax)


def _panel_c(ax):
    """Semantic overlap among jointly characterized features.

    Dots, not bars: two of the four values are exactly zero and a zero-length bar
    is invisible. Clopper-Pearson 95% intervals are drawn because n ranges from 10
    to 58, so the same percentage carries very different weight across rows.
    """
    ov = json.loads((RESULTS / "cross_method_overlap" / "stats.json").read_text())
    om, on = ov["old_regime"]["matrix"], ov["new_regime"]["matrix"]
    rows = [("Transluce ~ max-act",       on["Transluce"]["max-act"]["pooled"]),
            ("PRISM ~ output-centric",    om["PRISM"]["output-centric"]["pooled"]),
            ("PRISM ~ GPT-explain",       om["PRISM"]["GPT-explain"]["pooled"]),
            ("PRISM ~ Neuronpedia",       om["PRISM"]["Neuronpedia"]["pooled"])]
    y = np.arange(len(rows))[::-1]
    equal = [True, False, False, False]      # only Transluce~max-act is 1-vs-1
    for yi, (lbl, d), eq in zip(y, rows, equal):
        col = XM_EQUAL if eq else XM_UNEQUAL
        k, n = d["n_overlap"], d["n_both_characterize"]
        lo, hi = _ci(k, n)
        ax.plot([lo, hi], [yi, yi], color="#9aa4b0", lw=1.0, solid_capstyle="round", zorder=2)
        ax.plot(100 * k / n, yi, "o", ms=4.5, color=col, mec="white", mew=0.7, zorder=4)
        ax.text(1.5, yi + 0.22, lbl, va="bottom", ha="left", fontsize=FS_LABEL, color="#222")
        ax.text(hi + 2.5, yi, f"{100 * k / n:.1f}% ({k}/{n})",
                va="center", ha="left", fontsize=FS_VALUE, color="#555")
    ax.set_yticks([]); ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_xlim(-2, 118)
    ax.set_xlabel("Semantic overlap (%)", fontsize=FS_LABEL)
    ax.set_title("(c)", fontsize=FS_TITLE, loc="left", pad=3)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ts.despine(ax)


def make():
    ts.apply()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                                        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    _panel_a(axA); _panel_b(axB); _panel_c(axC)
    # identical box aspect -> the three x-axes sit on one line and the y-axes match
    for ax in (axA, axB, axC):
        ax.set_box_aspect(0.92)
        ax.tick_params(labelsize=FS_LABEL, length=2.5, pad=1.5)
    # one shared model legend for the whole figure -- at 1.5in per panel an
    # in-axes legend covers the data
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=FS_LEGEND, frameon=False,
               handletextpad=0.3, columnspacing=1.1, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.315, wspace=0.55)
    ts.save(fig, "fig_cross_method")
    plt.close(fig)
    print(f"saved figures/fig_cross_method.{{png,pdf,svg}}  ({FIG_W}x{FIG_H}in, true print size)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    make()

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
# Type scale. Figure 4 is three panels at workshop-column width, so everything
# on-plot must survive heavy downscaling: short labels, few of them, large type.
# All explanatory wording (direction semantics, the proposal-budget caveat) lives
# in the caption, not the axes.
FS_TITLE, FS_LABEL, FS_VALUE, FS_LEGEND = 12, 11, 10.5, 9.5

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
    ax.set_ylabel(r"Weighted mean similarity  $\widetilde{\sigma}$", fontsize=FS_LABEL)
    ax.set_title("(a) Original vs. revised similarity", fontsize=FS_TITLE)
    ax.annotate(rf"$\rho$ = {rho:.3f}, $n$ = {len(mm)}", xy=(0.04, 0.96), xycoords="axes fraction",
                va="top", ha="left", fontsize=FS_VALUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
    ax.legend(fontsize=FS_LEGEND, loc="lower right", ncol=2, columnspacing=0.8,
              handletextpad=0.2, borderpad=0.3)
    ts.despine(ax)


def _panel_b(ax):
    """Directional complementary coverage for the equal-budget new-regime pair."""
    rescue = json.loads((RESULTS / "cross_method_rescue" / "stats.json").read_text())
    nm = rescue["new_regime"]["matrix"]
    rows = [(r"Transluce $\rightarrow$ max-act", nm["Transluce"]["max-act"]["pooled"]),
            (r"max-act $\rightarrow$ Transluce", nm["max-act"]["Transluce"]["pooled"])]
    y = np.arange(len(rows))[::-1]
    for yi, (_, d) in zip(y, rows):
        k, n = d["n_covered_by_B"], d["n_uncharacterized_by_A"]
        ax.barh(yi, 100 * k / n, height=0.40, color=XM_EQUAL, edgecolor="none", zorder=3)
        lo, hi = _ci(k, n)
        ax.plot([lo, hi], [yi, yi], color="#9aa4b0", lw=1.4, solid_capstyle="round", zorder=4)
        ax.text(hi + 1.8, yi, f"{100 * k / n:.1f}%  ({k}/{n})",
                va="center", ha="left", fontsize=FS_VALUE, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=FS_LABEL)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(0, 92)
    ax.set_xlabel("Complementary coverage (%)", fontsize=FS_LABEL)
    ax.set_title("(b) Complementary coverage", fontsize=FS_TITLE)
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
    for yi, (_, d), eq in zip(y, rows, equal):
        col = XM_EQUAL if eq else XM_UNEQUAL
        k, n = d["n_overlap"], d["n_both_characterize"]
        lo, hi = _ci(k, n)
        ax.plot([lo, hi], [yi, yi], color="#9aa4b0", lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(100 * k / n, yi, "o", ms=7, color=col, mec="white", mew=1.0, zorder=4)
        ax.text(hi + 1.8, yi, f"{100 * k / n:.1f}%  ({k}/{n})",
                va="center", ha="left", fontsize=FS_VALUE, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=FS_LABEL)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(-1.5, 88)
    ax.set_xlabel("Semantic overlap (%)", fontsize=FS_LABEL)
    ax.set_title("(c) Semantic overlap", fontsize=FS_TITLE)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ts.despine(ax)


def make():
    ts.apply()
    # Two rows rather than three panels abreast: at workshop-column width a 1x3
    # row forces every label small. Here (a) gets the full width and (b)/(c) get
    # half each -- ~50% more width than in the 1x3 layout, which is what the long
    # pair labels in (c) need.
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          left=0.085, right=0.985, top=0.955, bottom=0.075,
                          hspace=0.34, wspace=0.42)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    _panel_a(axA); _panel_b(axB); _panel_c(axC)
    axA.set_box_aspect(0.42)      # wide banner; the dashed line is still y = x
    for ax in (axB, axC):
        ax.set_box_aspect(0.66)
    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=FS_LABEL)
    ts.save(fig, "fig_cross_method")
    plt.close(fig)
    print("saved figures/fig_cross_method.{png,pdf,svg}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    make()

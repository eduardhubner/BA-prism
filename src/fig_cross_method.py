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
from scipy.stats import spearmanr

import thesis_style as ts
from pool_gpt2xl_analyze import filter_pool, metrics

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
GX = RESULTS / "pooled_gpt2xl"
OC = RESULTS / "pooled_outputcentric"

CAT = ["filtered", "retained as distinct", "redundant"]
CAT_COLOR = {"filtered": "#c9c9c9", "retained as distinct": "#4c9f70", "redundant": ts.COLLAPSED_BLUE}


def _panel_a(ax):
    q = pd.read_csv(RESULTS / "quality_blindness_check.csv")
    fe = pd.read_csv(RESULTS / "filtered_evaluation.csv")
    mm = q.merge(fe, on=["model", "layer", "unit"])
    mm = mm[mm["n_surviving"] >= 2]
    rho = spearmanr(mm["prism_score"], mm["weighted_mean_similarity"]).statistic
    lo = min(mm["prism_score"].min(), mm["weighted_mean_similarity"].min()) - 0.03
    hi = max(mm["prism_score"].max(), mm["weighted_mean_similarity"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1, alpha=0.6, zorder=0)
    for m in ts.MODEL_ORDER:
        s = mm[mm["model"] == m]
        lab = "GPT-2 SAE" if m == "gpt2-small-sae" else ts.MODEL_LABEL[m]
        ax.scatter(s["prism_score"], s["weighted_mean_similarity"], color=ts.MODEL_COLORS[m],
                   marker=ts.MODEL_MARKERS[m], edgecolors="black", linewidths=0.4, s=26,
                   alpha=0.85, label=lab)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{PRISM}_{\mathrm{poly}}$")
    ax.set_ylabel(r"Weighted mean similarity  $\widetilde{\sigma}$")
    ax.set_title("(a) Original vs. revised similarity", fontsize=10)
    ax.annotate(rf"$\rho$ = {rho:.3f}, $n$ = {len(mm)}", xy=(0.04, 0.96), xycoords="axes fraction",
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
    ax.legend(fontsize=6.5, loc="lower right", ncol=2, columnspacing=0.8,
              handletextpad=0.2, borderpad=0.3)
    ts.despine(ax)


def _panel_b(ax):
    desc = pd.read_csv(GX / "descriptions.csv"); emb = np.load(GX / "embeddings.npy")
    a = desc["auc"].to_numpy()
    qp, qpool = [], []
    for _, g in desc.groupby(["layer", "unit"]):
        idx = g.index.to_numpy(); E = emb[idx]
        m, _ = metrics(a[idx], E @ E.T, filter_pool(a[idx], E @ E.T)[0])
        pidx = g.index[(g["method"] == "input-centric") & (g["generator"] == "gemini-1.5-pro")].to_numpy()
        Ep = emb[pidx]
        mp, _ = metrics(a[pidx], Ep @ Ep.T, filter_pool(a[pidx], Ep @ Ep.T)[0])
        if not (np.isnan(m["Q"]) or np.isnan(mp["Q"])):
            qp.append(mp["Q"]); qpool.append(m["Q"])
    qp, qpool = np.array(qp), np.array(qpool)
    rho = spearmanr(qp, qpool).statistic
    lo = min(qp.min(), qpool.min()) - 0.05; hi = max(qp.max(), qpool.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#999", ls="--", lw=1, zorder=1)
    ax.scatter(qp, qpool, s=30, color=ts.MODEL_COLORS["gpt2-xl"], edgecolors="black",
               linewidths=0.4, alpha=0.85, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(r"$Q_{\mathrm{PRISM}}$  (PRISM-only)")
    ax.set_ylabel(r"$Q_{\mathrm{pool}}$  (all sources)")
    ax.set_title("(b) Evidence quality across description pools", fontsize=10)
    ax.annotate(rf"$\rho$ = {rho:.3f}, $n$ = {len(qp)}", xy=(0.04, 0.96), xycoords="axes fraction",
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
    ts.despine(ax)


def _panel_c(ax):
    stats = json.loads((OC / "stats.json").read_text())
    # one stacked bar per (target, method) for every method other than input-centric.
    cols = [("Llama-3.1-8B-Instruct", "output-centric"),
            ("gemma-scope-2b", "output-centric"),
            ("gpt2-small-sae", "output-centric"),
            ("gpt2-small-sae", "neuronpedia")]
    short = {"Llama-3.1-8B-Instruct": "Llama", "gemma-scope-2b": "Gemma", "gpt2-small-sae": "GPT-2 SAE"}
    meth_label = {"output-centric": "output-centric", "neuronpedia": "Neuronpedia"}
    fates = [stats["per_model"][m]["fate"][meth] for m, meth in cols]
    key = {"filtered": "filtered", "retained as distinct": "distinct", "redundant": "converges"}
    x = np.arange(len(cols)); bottom = np.zeros(len(cols))
    for cat in CAT:
        vals = np.array([100 * f[key[cat]] / f["n"] for f in fates])
        ax.bar(x, vals, bottom=bottom, width=0.6, color=CAT_COLOR[cat], edgecolor="white",
               linewidth=0.8, label=cat)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[m]}\n{meth_label[meth]}" for m, meth in cols], fontsize=7.5)
    ax.set_ylabel("% of all descriptions")   # denominator = every description of that method
    ax.set_ylim(0, 108)
    ax.set_title("(c) Cross-method outcomes relative to PRISM", fontsize=10)
    # Redundancy is method-dependent, so label each bar's redundant share. NB the bar
    # denominator is ALL descriptions of the method (so neuronpedia reads 5%); the
    # conditional-on-survival figure (23% of surviving neuronpedia descriptions) is
    # stated in the caption.
    for xi, f in zip(x, fates):
        pct = 100 * f["converges"] / f["n"]
        ax.text(xi, 102, f"redundant: {pct:.0f}%", ha="center", va="bottom", fontsize=7,
                color=CAT_COLOR["redundant"] if pct > 0 else "#999")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.5, frameon=False)
    ts.despine(ax)


def make():
    ts.apply()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.0, 4.6),
                                        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})
    _panel_a(axA); _panel_b(axB); _panel_c(axC)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.25, wspace=0.30)
    ts.save(fig, "fig_cross_method")
    plt.close(fig)
    print("saved figures/fig_cross_method.{png,pdf,svg}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    make()

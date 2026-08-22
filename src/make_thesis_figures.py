"""
Generate all thesis figures into figures/ with the shared thesis_style.

One function per figure; `main()` regenerates everything in one run:
    python src/make_thesis_figures.py            # all figures
    python src/make_thesis_figures.py fig01 fig05  # just these

Figures follow the results-section plan (see RESULTS_INTERPRETATION_GUIDE / thesis
outline). Data comes from results/*.csv|json produced by the analysis scripts.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import thesis_style as ts
import weighting_gap_auroc_variance as wg

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
ts.apply()


def fig01_quality_blindness():
    """§2.1 Original PRISM similarity vs. mean AUROC (quality-blindness)."""
    d = pd.read_csv(RESULTS / "quality_blindness_check.csv")
    rho, p = spearmanr(d["prism_score"], d["mean_auroc_pre"])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for m in ts.MODEL_ORDER:
        s = d[d["model"] == m]
        ax.scatter(s["prism_score"], s["mean_auroc_pre"], color=ts.MODEL_COLORS[m],
                   marker=ts.MODEL_MARKERS[m], edgecolors="black", linewidths=0.4,
                   s=44, alpha=0.85, label=ts.MODEL_LABEL[m])
    ax.axhline(ts.CHANCE, color="grey", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("PRISM Polysemanticity Score")
    ax.set_ylabel("Mean AUROC of the 5 descriptions")
    ax.set_ylim(0, 1.02)
    ax.annotate(f"Spearman $\\rho$ = {rho:.3f},  p = {p:.2f},  n = {len(d)}",
                xy=(0.97, 0.98), xycoords="axes fraction", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))
    ax.legend(title="Model", loc="lower right", framealpha=0.92)
    ts.despine(ax)
    ts.save(fig, "fig01_quality_blindness")
    plt.close(fig)
    return f"rho={rho:.3f} p={p:.3f} n={len(d)}"


def fig02_status_by_model():
    """§2.2 Characterisation status by model: 100% stacked bars n_s=0/1/>=2."""
    d = pd.read_csv(RESULTS / "filtered_evaluation.csv")

    def status(ns):
        return "uncharacterised" if ns == 0 else ("single" if ns == 1 else "multi")
    d["status"] = d["n_surviving"].map(status)

    order = ["uncharacterised", "single", "multi"]   # bottom -> top
    rows = [(ts.MODEL_LABEL[m], d[d["model"] == m]) for m in ts.MODEL_ORDER]
    rows.append(("All models", d))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for c in order:
        vals = np.array([(sub["status"] == c).mean() * 100 for _, sub in rows])
        ax.bar(x, vals, bottom=bottom, width=0.7, color=ts.STATUS_COLORS[c],
               edgecolor="white", linewidth=0.8, label=c.capitalize())
        bottom += vals
    for xi, (_, sub) in zip(x, rows):
        ax.text(xi, 101.5, f"n={len(sub)}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("Share of features (%)")
    ax.set_ylim(0, 100)
    ax.axvline(len(ts.MODEL_ORDER) - 0.5, color="#cccccc", lw=1)   # separate "All"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Characterisation", loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ts.despine(ax)
    ts.save(fig, "fig02_status_by_model")
    plt.close(fig)
    all_pct = {c: round((d["status"] == c).mean() * 100, 1) for c in order}
    return f"all-models {all_pct}"


def _model_scatter(ax, df, xcol, ycol):
    for m in ts.MODEL_ORDER:
        s = df[df["model"] == m]
        ax.scatter(s[xcol], s[ycol], color=ts.MODEL_COLORS[m], marker=ts.MODEL_MARKERS[m],
                   edgecolors="black", linewidths=0.4, s=44, alpha=0.85, label=ts.MODEL_LABEL[m])


def _statbox(ax, text, loc="upper left"):
    xy = (0.03, 0.97) if "left" in loc else (0.97, 0.97)
    ha = "left" if "left" in loc else "right"
    ax.annotate(text, xy=xy, xycoords="axes fraction", va="top", ha=ha,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


def fig04_original_vs_revised():
    """§3.2 Original PRISM similarity vs. revised weighted similarity (multi-survivor)."""
    q = pd.read_csv(RESULTS / "quality_blindness_check.csv")
    fe = pd.read_csv(RESULTS / "filtered_evaluation.csv")
    mm = q.merge(fe, on=["model", "layer", "unit"])
    mm = mm[mm["n_surviving"] >= 2]
    rho, _ = spearmanr(mm["prism_score"], mm["weighted_mean_similarity"])

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lo = min(mm["prism_score"].min(), mm["weighted_mean_similarity"].min()) - 0.03
    hi = max(mm["prism_score"].max(), mm["weighted_mean_similarity"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1, alpha=0.6, zorder=0)
    _model_scatter(ax, mm, "prism_score", "weighted_mean_similarity")
    ax.set_xlabel("PRISM Polysemanticity Score")
    ax.set_ylabel("Weighted Mean Similarity")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    _statbox(ax, f"Spearman $\\rho$ = {rho:.3f}\nn = {len(mm)} (multi-survivor)")
    ax.legend(title="Model", loc="lower right")
    ts.despine(ax)
    ts.save(fig, "fig04_original_vs_revised")
    plt.close(fig)
    return f"rho={rho:.3f} n={len(mm)}"


def fig05_weighting_mechanism():
    """§3.3 Survivor-AUROC spread vs. weighting gap, with gap-distribution panel."""
    k5 = pd.read_csv(RESULTS / "pairwise_similarities_dataset.csv")
    rows = []
    for _, r in k5.iterrows():
        res = wg.neuron_gap_row(r, 5, "")
        if res:
            res["model"] = r["model"]
            rows.append(res)
    d = pd.DataFrame(rows)
    rho, _ = spearmanr(d["auroc_std"], d["abs_gap"])

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(8.4, 4.8),
                                  gridspec_kw={"width_ratios": [3, 1]})
    _model_scatter(ax, d, "auroc_std", "abs_gap")
    ax.set_xlabel("Std of survivor AUROCs")
    ax.set_ylabel(r"Weighting gap  $|\tilde\sigma - \bar\sigma|$")
    _statbox(ax, f"Spearman $\\rho$ = {rho:.3f}\nn = {len(d)} ($n_s\\geq3$)")
    ts.despine(ax)
    axh.hist(d["abs_gap"], bins=20, orientation="horizontal", color="#9e9e9e", edgecolor="white")
    axh.set_xlabel("count"); axh.set_ylabel(r"$|\tilde\sigma - \bar\sigma|$")
    axh.set_ylim(ax.get_ylim())
    axh.set_title("Gap distribution", fontsize=9)
    ts.despine(axh)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.24)
    ts.save(fig, "fig05_weighting_mechanism")
    plt.close(fig)
    return f"rho={rho:.3f} n={len(d)} mean|gap|={d.abs_gap.mean():.4f}"


def fig07_embedding_robustness():
    """§4.2 Embedding-model robustness: 5x5 within-feature Spearman heatmap."""
    j = json.load(open(RESULTS / "embedding_robustness_matrix.json"))
    models = j["models"]
    M = j["within_neuron"]["spearman_matrix"]
    labels = [m.replace(" (baseline)", "") for m in models]
    mat = np.array([[M[a][b] for b in models] for a in models])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(labels)
    for i in range(len(models)):
        for k in range(len(models)):
            ax.text(k, i, f"{mat[i, k]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, k] > 0.9 else "black")
    ax.add_patch(plt.Rectangle((-.5, -.5), len(models), 1, fill=False, ec="#D55E00", lw=2.5))
    ax.add_patch(plt.Rectangle((-.5, -.5), 1, len(models), fill=False, ec="#D55E00", lw=2.5))
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r"Within-feature Spearman $\rho$")
    ax.set_title("Score robustness across embedding models\n(baseline highlighted)")
    ts.save(fig, "fig07_embedding_robustness")
    plt.close(fig)
    off = mat[0, 1:]
    return f"baseline vs others: {off.min():.2f}-{off.max():.2f}"


def fig08_rescue():
    """§5.2 Candidate-budget rescue: max candidate AUROC, k=5 vs k=10."""
    r = pd.read_csv(RESULTS / "k10_rescue_detail.csv")
    g = r.groupby(["layer", "unit", "k"])["auroc"].max().unstack("k").sort_values(5)

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    x = np.arange(len(g))
    for xi, (idx, row) in zip(x, g.iterrows()):
        a5, a10 = row[5], row[10]
        rescued = (a10 > 0.5) and (a5 <= 0.5)
        col = "#2ca02c" if rescued else "#9e9e9e"
        ax.annotate("", xy=(xi, a10), xytext=(xi, a5),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.scatter(xi, a5, facecolor="white", edgecolors="black", s=60, zorder=3)
        ax.scatter(xi, a10, facecolor=col, edgecolors="black", s=60, zorder=3)
        if rescued:
            ax.text(xi + 0.12, (a5 + a10) / 2, f"{a5:.2f}→{a10:.2f}",
                    va="center", fontsize=8, color="#1a7a1a")
    ax.axhline(0.5, color="red", ls="--", lw=1.2, alpha=0.75)
    ax.text(0.02, 0.515, "verification threshold (AUROC = 0.5)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}/U{u}" for l, u in g.index], rotation=20)
    ax.set_ylabel("Best description AUROC\n(max over the $k$ candidate descriptions)")
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, len(g) - 0.3)
    ax.set_title("Do uncharacterised features get rescued by more candidates?")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=9, label="k = 5 (start)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markeredgecolor="black", markersize=9, label="k = 10 (end)"),
        Line2D([0], [0], color="#2ca02c", lw=2.5, label="rescued (crosses 0.5)"),
        Line2D([0], [0], color="#9e9e9e", lw=2.5, label="still uncharacterised"),
    ], loc="upper left", fontsize=8)
    ts.despine(ax)
    ts.save(fig, "fig08_rescue")
    plt.close(fig)
    n_res = int(((g[10] > 0.5) & (g[5] <= 0.5)).sum())
    return f"rescued {n_res}/{len(g)}"


def _render_mds_pair(cases, name):
    """Render a single MDS graphic holding `cases` (2 case studies), each with a
    WMS/Q header and a per-description table underneath (appendix / detailed
    style). Shared by the fig03 graphics and the appendix feature-cases figure."""
    import case_study_analysis as cs
    import toy_neurons as toy
    df = pd.read_csv(cs.PRISM_PATH)
    cmap, norm = ts.AUROC_CMAP, ts.auroc_norm()
    dec_short = {"survived": "survived", "collapsed": "collapsed", "removed (low AUROC)": "removed"}
    dec_color = {"survived": "black", "collapsed": ts.COLLAPSED_BLUE, "removed (low AUROC)": "#777"}

    def short(s, n=42):
        s = str(s)
        return s[:n] + "…" if len(s) > n else s

    ncol = len(cases)
    fig = plt.figure(figsize=(5.4 * ncol, 7.2))
    gs = fig.add_gridspec(2, ncol, height_ratios=[3, 1.9], hspace=0.06, wspace=0.16)
    top_axes = []
    for c, (model, layer, unit, label) in enumerate(cases):
        row = df[(df["model"] == model) & (df["layer"] == layer) & (df["unit"] == unit)].iloc[0]
        aurocs = np.array([row[f"auroc_{i}"] for i in range(1, 6)])
        sm = cs.get_sim_matrix(row)
        dec = cs.filter_decisions(aurocs, sm)
        met = cs.compute_metrics(aurocs, sm, dec)
        coords = toy.mds_coords(sm)

        # WMS/Q header (undefined -> em dash for uncharacterized / no-pair cases).
        if met["uncharacterized"]:
            wms_str, q_str = "—", "—"
        else:
            wms = met["weighted_mean_sim"]
            wms_str = f"{wms:.3f}" if not np.isnan(wms) else "—"
            q_str = f"{met['evidence_quality']:.3f}"

        ax = fig.add_subplot(gs[0, c]); top_axes.append(ax)
        toy._draw_framework_panel(ax, coords, aurocs, sm, dec, cmap, norm)
        ax.set_title(f"WMS = {wms_str}     Q = {q_str}\n"
                     f"{label}  ·  {ts.MODEL_LABEL[model]}  L{layer}/U{unit}",
                     fontsize=10.5)

        axt = fig.add_subplot(gs[1, c]); axt.axis("off")
        cell_text = [[f"D{i+1}", short(row[f"description_{i+1}"]), f"{aurocs[i]:.2f}", dec_short[dec[i]]]
                     for i in range(5)]
        tbl = axt.table(cellText=cell_text, colLabels=["", "description", "AUC", "outcome"],
                        loc="center", cellLoc="left", colWidths=[0.07, 0.60, 0.12, 0.21])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.3)
        for col in range(4):
            tbl[(0, col)].set_text_props(fontweight="bold")
        for i in range(5):
            for col in (0, 3):
                tbl[(i + 1, col)].get_text().set_color(dec_color[dec[i]])
        for (r_, c_), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dddddd")

    ts.add_auroc_colorbar(fig, plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                          ax=top_axes, fraction=0.025, pad=0.02)
    fig.legend(handles=ts.filter_legend_handles(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    ts.save(fig, name)
    plt.close(fig)
    return ", ".join(f"{m} L{l}/U{u}" for m, l, u, _ in cases)


def fig03a_mds_cases():
    """§2.3 MDS case studies I: a clean multi-concept neuron vs. a fully
    uncharacterized one."""
    cases = [
        ("gpt2-xl", 20, 5278, "Clean multi-concept"),
        ("gpt2-small-sae", 5, 30319, "Uncharacterized"),
    ]
    return _render_mds_pair(cases, "fig03a_mds_cases")


def fig03b_mds_cases():
    """§2.3 MDS case studies II: redundancy / collapse across two model families."""
    cases = [
        ("gemma-scope-2b", 0, 13988, "Redundancy / collapse"),
        ("gpt2-small-sae", 10, 9896, "Redundancy / collapse"),
    ]
    return _render_mds_pair(cases, "fig03b_mds_cases")


def fig_feature_cases():
    """Paper figure (compact): two feature cases, MDS panels + short description
    tables. Full descriptions and the interpretation go in the caption/appendix."""
    import case_study_analysis as cs
    import toy_neurons as toy
    import matplotlib.lines as mlines
    df = pd.read_csv(cs.PRISM_PATH)
    cmap, norm = ts.AUROC_CMAP, ts.auroc_norm()
    BLUE = ts.COLLAPSED_BLUE

    cases = [("gemma-scope-2b", 0, 13988), ("gpt2-small-sae", 10, 9896)]
    SHORT = {  # intentional short labels; full text -> appendix
        ("gemma-scope-2b", 0, 13988): ["End of text / input", "End of text", "End of text",
                                       "End of text / document", "End of text / item descriptions"],
        ("gpt2-small-sae", 10, 9896): ['"So" / "thus" (sentence start)', '"So" starting a conclusion',
                                       "Conclusion or inference", "Conclusion or summary",
                                       "Concluding phrase (technical)"],
    }
    dec_short = {"survived": "survived", "collapsed": "redundant", "removed (low AUROC)": "filtered"}
    marker = {"survived": "o", "collapsed": "D", "removed (low AUROC)": "D"}
    edge = {"survived": "black", "collapsed": BLUE, "removed (low AUROC)": "black"}
    # outcome text colour: survived = high-AUROC (green) end, filtered = 0-AUROC
    # (pink) end, redundant = collapse blue.
    out_color = {"survived": cmap(norm(1.0)), "removed (low AUROC)": cmap(norm(0.0)), "collapsed": BLUE}

    data = []
    for model, layer, unit in cases:
        row = df[(df["model"] == model) & (df["layer"] == layer) & (df["unit"] == unit)].iloc[0]
        aurocs = np.array([row[f"auroc_{i}"] for i in range(1, 6)])
        sm = cs.get_sim_matrix(row)
        dec = cs.filter_decisions(aurocs, sm)
        met = cs.compute_metrics(aurocs, sm, dec)
        coords = toy.mds_coords(sm)
        coords = coords - coords.mean(axis=0)
        data.append((model, layer, unit, aurocs, sm, dec, met, coords))
    lim = max(np.abs(c).max() for *_, c in data) * 1.48

    # one row: [plot | descriptions] per case, then a vertical AUROC colourbar.
    fig = plt.figure(figsize=(10.4, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.12, 1.0, 1.12],
                          left=0.035, right=0.925, top=0.80, bottom=0.24, wspace=0.10)

    for c, (model, layer, unit, aurocs, sm, dec, met, coords) in enumerate(data):
        # --- MDS plot with MDS1/MDS2 axes ---
        ax = fig.add_subplot(gs[0, 2 * c]); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb"); s.set_linewidth(0.8)
        ax.set_xlabel("MDS1", fontsize=7, color="#666", labelpad=1)
        ax.set_ylabel("MDS2", fontsize=7, color="#666", labelpad=1)
        surv = [i for i, d in enumerate(dec) if d == "survived"]
        for i in range(5):
            if dec[i] == "collapsed" and surv:
                j = min(surv, key=lambda s: np.hypot(*(coords[i] - coords[s])))
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                        color=BLUE, lw=1.0, alpha=0.8, zorder=2)
        for i in sorted(range(5), key=lambda i: dec[i] == "survived"):
            ax.scatter(coords[i, 0], coords[i, 1], s=95, color=cmap(norm(aurocs[i])),
                       marker=marker[dec[i]], edgecolors=edge[dec[i]],
                       linewidths=1.5 if dec[i] == "collapsed" else 1.0,
                       zorder=4 if dec[i] == "survived" else 3)
        cen = coords.mean(axis=0)
        lpos = np.empty((5, 2))
        for i in range(5):
            d = coords[i] - cen; n = np.hypot(*d)
            u = d / n if n > 1e-6 else np.array([0.0, 1.0])
            lpos[i] = coords[i] + 0.26 * lim * u
        min_d = 0.17 * lim
        for _ in range(40):
            for a in range(5):
                for b in range(a + 1, 5):
                    diff = lpos[a] - lpos[b]; dist = np.hypot(*diff)
                    if 1e-9 < dist < min_d:
                        shift = (min_d - dist) / 2 * diff / dist
                        lpos[a] += shift; lpos[b] -= shift
        for i in range(5):
            ax.text(lpos[i, 0], lpos[i, 1], f"D{i+1}", fontsize=6, fontweight="bold",
                    ha="center", va="center", color="#444", zorder=6)
        wms = met["weighted_mean_sim"]
        wms_str = "—" if met["uncharacterized"] or np.isnan(wms) else f"{wms:.3f}"
        q_str = "—" if met["uncharacterized"] else f"{met['evidence_quality']:.3f}"
        ax.set_title(f"{ts.MODEL_LABEL[model]} L{layer}/U{unit}", fontsize=8.5, pad=12)
        ax.text(0.5, 1.02, rf"$\tilde{{\sigma}}$ = {wms_str},   Q = {q_str}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#333")

        # --- descriptions beside the plot (no AUROC column; outcome colour-coded) ---
        axd = fig.add_subplot(gs[0, 2 * c + 1]); axd.axis("off")
        axd.set_xlim(0, 1); axd.set_ylim(0, 1)
        axd.text(0.07, 0.99, "description", fontsize=6, color="#999", va="top")
        axd.text(0.66, 0.99, "outcome", fontsize=6, color="#999", va="top")
        for i in range(5):
            y = 0.85 - i * 0.165
            axd.text(0.0, y, f"D{i+1}", fontsize=7, fontweight="bold", color="#333", va="center")
            axd.text(0.07, y, SHORT[(model, layer, unit)][i], fontsize=6.1, color="#222", va="center")
            axd.text(0.66, y, dec_short[dec[i]], fontsize=6.8, fontweight="bold",
                     color=out_color[dec[i]], va="center")

    # --- vertical AUROC colourbar, same row, far right ---
    sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm_cb.set_array([])
    cax = fig.add_axes([0.945, 0.22, 0.013, 0.52])
    cbar = fig.colorbar(sm_cb, cax=cax, orientation="vertical", ticks=[0.0, 0.5, 1.0])
    cbar.ax.set_yticklabels(["0.0", "0.5 (chance)", "1.0"], fontsize=6)
    cbar.set_label("AUROC", fontsize=7)
    cbar.ax.axhline(0.5, color="black", ls="--", lw=0.9, alpha=0.7)
    cbar.ax.tick_params(length=2)

    # --- marker legend (shapes), labels colour-matched to the outcome coding ---
    handles = [
        mlines.Line2D([], [], marker="o", ls="None", mfc="white", mec="black", ms=6.5, mew=1.1, label="survived"),
        mlines.Line2D([], [], marker="D", ls="None", mfc="white", mec=BLUE, ms=6, mew=1.4, label="redundant"),
        mlines.Line2D([], [], marker="D", ls="None", mfc="white", mec="black", ms=6, mew=1.1, label="filtered"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
                     handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.46, 0.02))
    for t, key in zip(leg.get_texts(), ["survived", "collapsed", "removed (low AUROC)"]):
        t.set_color(out_color[key])

    ts.save(fig, "fig_feature_cases")
    plt.close(fig)
    return "compact (1-row): gemma L0/U13988, gpt2-small-sae L10/U9896"


def fig06_human_annotation():
    """§4.1 Human-annotation comparison: original (n=8) and revised (n=4),
    excluded cases shown explicitly, small-n caveat prominent."""
    h = pd.read_csv(RESULTS / "human_annotation_correlation.csv")
    f = pd.read_csv(RESULTS / "human_annotation_correlation_filtered.csv")
    f = f[f["sim_threshold"] == 0.70]   # already has a 'model' column
    rho_orig = spearmanr(h["human_poly_score"], h["prism_score"]).statistic
    defn = f[f["n_surviving"] >= 2]
    rho_rev = spearmanr(defn["human_poly_score"], defn["weighted_mean_sim"]).statistic
    excl = f[f["n_surviving"] < 2]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for m in ts.MODEL_ORDER:
        s = h[h["model"] == m]
        axA.scatter(s["prism_score"], s["human_poly_score"], color=ts.MODEL_COLORS[m],
                    marker=ts.MODEL_MARKERS[m], edgecolors="black", s=70, label=ts.MODEL_LABEL[m])
    axA.set_xlabel("PRISM Polysemanticity Score")
    axA.set_ylabel("Human polysemanticity score")
    axA.set_title(f"Original     Spearman $\\rho$ = {rho_orig:.3f},  n = {len(h)}", fontsize=11)
    axA.set_xlim(0.25, 0.8)
    axA.legend(title="Model", fontsize=8, loc="lower right")
    ts.despine(axA)

    for m in ts.MODEL_ORDER:
        s = defn[defn["model"] == m]
        axB.scatter(s["weighted_mean_sim"], s["human_poly_score"], color=ts.MODEL_COLORS[m],
                    marker=ts.MODEL_MARKERS[m], edgecolors="black", s=70)
    axB.axvline(0.24, color="#cccccc", ls="--", lw=1)
    axB.scatter([0.16] * len(excl), excl["human_poly_score"], facecolor="white",
                edgecolors="#888888", s=55, marker="o")
    axB.text(0.16, 0.965, "excluded\n(no $\\tilde\\sigma$)", ha="center", va="top",
             fontsize=8, color="#555")
    axB.set_xlim(0.10, 0.75)
    axB.set_xlabel("Weighted Mean Similarity")
    axB.set_title(f"Revised     Spearman $\\rho$ = {rho_rev:.2f},  n = {len(defn)}", fontsize=11)
    n_single = int(excl["exclusion_reason"].str.contains("single").sum())
    n_unchar = int(excl["exclusion_reason"].str.contains("uncharacter").sum())
    axB.annotate(f"{len(excl)} excluded:  {n_single} single-survivor,  {n_unchar} uncharacterised",
                 xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    ts.despine(axB)

    fig.text(0.5, -0.02,
             r"$\rho$=1.00 (revised, n=4) is not evidence of superiority over $\rho$=0.83 (n=8); "
             "it shows the filtered metric tracks human judgement at least as well where it is defined.",
             ha="center", fontsize=8, style="italic")
    ts.save(fig, "fig06_human_annotation")
    plt.close(fig)
    return f"orig rho={rho_orig:.3f} n={len(h)}; rev rho={rho_rev:.3f} n={len(defn)}"


def _boxplot_by_model(ax, data_by_model, ylabel, title):
    data = [data_by_model[m] for m in ts.MODEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.6,
                    medianprops=dict(color="black"))
    for patch, m in zip(bp["boxes"], ts.MODEL_ORDER):
        patch.set_facecolor(ts.MODEL_COLORS[m]); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(ts.MODEL_ORDER) + 1))
    ax.set_xticklabels([ts.MODEL_LABEL[m] for m in ts.MODEL_ORDER], rotation=15)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ymin = ax.get_ylim()[0]
    for i, dd in enumerate(data):
        ax.text(i + 1, ymin, f"n={len(dd)}", ha="center", va="bottom", fontsize=7, color="#555")
    ts.despine(ax)


def fig09_profile_distributions():
    """§3.1 Evidence-quality Q (n_s>=1) and diversity 1-sigma-tilde (n_s>=2) by model."""
    fe = pd.read_csv(RESULTS / "filtered_evaluation.csv")
    fe["Q"] = 2 * fe["mean_auroc_post"] - 1
    fe["diversity"] = 1 - fe["weighted_mean_similarity"]
    fig, (axQ, axD) = plt.subplots(1, 2, figsize=(11, 4.9))
    q = {m: fe[(fe["model"] == m) & (fe["n_surviving"] >= 1)]["Q"].dropna() for m in ts.MODEL_ORDER}
    d = {m: fe[(fe["model"] == m) & (fe["n_surviving"] >= 2)]["diversity"].dropna() for m in ts.MODEL_ORDER}
    _boxplot_by_model(axQ, q, "Evidence quality  $Q$", r"Evidence quality  ($n_s \geq 1$)")
    _boxplot_by_model(axD, d, r"Diversity  $1-\tilde\sigma$", r"Semantic diversity  ($n_s \geq 2$)")
    ts.save(fig, "fig09_profile_distributions")
    plt.close(fig)
    return "Q and diversity boxplots by model"


def _decompose(df):
    """Per-neuron counts of survived/collapsed/removed via the canonical filter."""
    import case_study_analysis as cs
    rows = []
    for _, r in df.iterrows():
        aurocs = np.array([r[f"auroc_{i}"] for i in range(1, 6)])
        sm = cs.get_sim_matrix(r)
        dec = cs.filter_decisions(aurocs, sm)
        rows.append(dict(model=r["model"], layer=r["layer"], unit=r["unit"],
                         n_s=dec.count("survived"), n_coll=dec.count("collapsed"),
                         n_filt=dec.count("removed (low AUROC)")))
    return pd.DataFrame(rows)


def fig10_component_correlations():
    """§3.4 Spearman correlations among profile components (pairwise-complete)."""
    import case_study_analysis as cs
    df = pd.read_csv(cs.PRISM_PATH)
    dec = _decompose(df)
    fe = pd.read_csv(RESULTS / "filtered_evaluation.csv")
    d = dec.merge(fe, on=["model", "layer", "unit"])
    d["Q"] = np.where(d["n_s"] >= 1, 2 * d["mean_auroc_post"] - 1, np.nan)
    d["diversity"] = np.where(d["n_s"] >= 2, 1 - d["weighted_mean_similarity"], np.nan)

    cols = ["Q", "n_s", "n_filt", "n_coll", "diversity"]
    labels = ["$Q$", "$n_s$", "$n_{filt}$", "$n_{coll}$", r"$1-\tilde\sigma$"]
    M = np.full((5, 5), np.nan)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                M[i, j] = 1.0
                continue
            sub = d[[a, b]].dropna()
            if len(sub) > 2:
                M[i, j] = spearmanr(sub[a].values, sub[b].values).statistic

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(M, cmap=ts.CORR_CMAP, vmin=ts.CORR_VMIN, vmax=ts.CORR_VMAX)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label(r"Spearman $\rho$")
    ax.set_title("Profile-component correlations\n(pairwise-complete; $n$ varies with definedness)")
    ts.save(fig, "fig10_component_correlations")
    plt.close(fig)
    return "5x5 component correlation matrix"


def _survivors_at(aurocs, sm, sim_thr, auroc_thr=0.5):
    kept = [i for i in range(5) if aurocs[i] > auroc_thr]
    adj = {i: set() for i in kept}
    for i, j in combinations(kept, 2):
        if sm[i, j] > sim_thr:
            adj[i].add(j); adj[j].add(i)
    vis, reps = set(), []
    for s in kept:
        if s not in vis:
            comp, q = [], [s]
            while q:
                n = q.pop()
                if n not in vis:
                    vis.add(n); comp.append(n); q.extend(adj[n] - vis)
            reps.append(max(comp, key=lambda i: aurocs[i]))
    return reps


def _diversity(aurocs, sm, surv):
    if len(surv) < 2:
        return np.nan
    num = den = 0.0
    for i, j in combinations(surv, 2):
        w = ((aurocs[i] - 0.5) / 0.5) * ((aurocs[j] - 0.5) / 0.5)
        num += w * sm[i, j]; den += w
    return 1 - (num / den) if den > 0 else np.nan


def fig11_threshold_sensitivity():
    """§4.3 Similarity-collapse threshold sensitivity: characterisation + ranking stability."""
    import case_study_analysis as cs
    df = pd.read_csv(cs.PRISM_PATH)
    A = [np.array([r[f"auroc_{i}"] for i in range(1, 6)]) for _, r in df.iterrows()]
    S = [cs.get_sim_matrix(r) for _, r in df.iterrows()]
    grid = np.round(np.arange(0.55, 0.861, 0.025), 3)
    base = [_diversity(a, s, _survivors_at(a, s, 0.70)) for a, s in zip(A, S)]

    unchar, multi, rankcorr = [], [], []
    for thr in grid:
        survs = [_survivors_at(a, s, thr) for a, s in zip(A, S)]
        ns = np.array([len(x) for x in survs])
        unchar.append((ns == 0).mean() * 100); multi.append((ns >= 2).mean() * 100)
        div = [_diversity(a, s, sv) for a, s, sv in zip(A, S, survs)]
        pairs = [(b, dv) for b, dv in zip(base, div) if not (np.isnan(b) or np.isnan(dv))]
        rankcorr.append(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.plot(grid, unchar, "-o", color=ts.STATUS_COLORS["uncharacterised"], label="% uncharacterised")
    a1.plot(grid, multi, "-s", color=ts.STATUS_COLORS["multi"], label="% multi-survivor")
    a1.axvline(0.70, color="grey", ls="--", lw=1)
    a1.set_xlabel(r"Similarity-collapse threshold $\tau$"); a1.set_ylabel("% of features")
    a1.set_title(r"Characterisation vs $\tau$"); a1.legend(fontsize=9)
    a2.plot(grid, rankcorr, "-o", color="#333333")
    a2.axvline(0.70, color="grey", ls="--", lw=1)
    a2.set_xlabel(r"Similarity-collapse threshold $\tau$")
    a2.set_ylabel(r"Spearman $\rho$ of diversity vs baseline ($\tau$=0.70)")
    a2.set_title(r"Ranking stability vs $\tau$"); a2.set_ylim(top=1.005)
    for a in (a1, a2):
        ts.despine(a)
    ts.save(fig, "fig11_threshold_sensitivity")
    plt.close(fig)
    return f"tau grid {grid[0]}-{grid[-1]}"


def fig12_clusterability():
    """§5.1 Clustering quality: cosine vs Euclidean silhouette (Hopkins in annotation)."""
    d = json.load(open(ROOT / "clustering_analysis_gte_qwen2_primary" / "results.json"))["neurons"]
    euc = [n["kmeans_results"]["euclidean_silhouette"]["silhouette_score"] for n in d]
    cos = [n["kmeans_results"]["cosine_silhouette"]["silhouette_score"] for n in d]
    hop_c = np.mean([n["clusterability"]["hopkins_cosine"] for n in d])
    hop_e = np.mean([n["clusterability"]["hopkins_euclidean"] for n in d])

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([euc, cos], patch_artist=True, showfliers=True, widths=0.5,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], ["#9e9e9e", "#3182bd"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Euclidean", "Cosine"])
    ax.set_ylabel("Silhouette score  (k-means, adaptive $k$)")
    ax.set_title("Clustering quality: cosine vs Euclidean")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.annotate(f"Hopkins $\\approx$ {hop_c:.2f} (cosine), {hop_e:.2f} (Euclidean)\n"
                "— closer to random than clustered",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    ts.despine(ax)
    ts.save(fig, "fig12_clusterability")
    plt.close(fig)
    return f"cosine {np.mean(cos):.4f} vs euclidean {np.mean(euc):.4f}"


FIGURES = {
    "fig01": fig01_quality_blindness,
    "fig02": fig02_status_by_model,
    "fig03a": fig03a_mds_cases,
    "fig03b": fig03b_mds_cases,
    "fig_feature_cases": fig_feature_cases,
    "fig04": fig04_original_vs_revised,
    "fig05": fig05_weighting_mechanism,
    "fig06": fig06_human_annotation,
    "fig07": fig07_embedding_robustness,
    "fig08": fig08_rescue,
    "fig09": fig09_profile_distributions,
    "fig10": fig10_component_correlations,
    "fig11": fig11_threshold_sensitivity,
    "fig12": fig12_clusterability,
}


def main():
    wanted = sys.argv[1:] or list(FIGURES)
    for key in wanted:
        fn = FIGURES.get(key) or FIGURES.get(f"fig{int(key):02d}")
        if fn is None:
            print(f"  (unknown figure '{key}')")
            continue
        info = fn()
        print(f"{fn.__name__:34s} -> figures/  [{info}]")


if __name__ == "__main__":
    main()

"""
Case study analysis: per-neuron metrics table and MDS plots for selected neurons.

For each neuron:
  - Five descriptions with AUROC, filter decision, and removal stage
  - Metrics: PRISM score, survivor count, unweighted sim, weighted sim (WMS),
    effective N, and evidence quality Q (= mean survivor AUROC rescaled so
    chance -> 0, perfect -> 1). (Confidence C is still computed internally.)
  - MDS plot: nodes coloured by AUROC; collapsed marked in blue, removed faded

Filter decisions (AUROC filter BEFORE similarity collapse, SIM_THRESHOLD=0.70,
AUROC_THRESHOLD=0.50):
  - "removed (low AUROC)":  AUROC <= 0.50 (filtered out before collapse, so it never
                          participates in a similarity component)
  - "survived":           AUROC > 0.50 AND highest-AUROC in its similarity component
  - "collapsed":          AUROC > 0.50 but not highest-AUROC in its component
                          (always collapses into a *surviving* description)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from itertools import combinations

import verification as V
from pathlib import Path
from sklearn.manifold import MDS

# --- Config ---
SIM_THRESHOLD = 0.70
AUROC_THRESHOLD = 0.50
SIM_PAIRS = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
N_DESC = 5

CASE_NEURONS = [
    ("gpt2-xl",               40,  1516),
    ("gpt2-small-sae",        10,   900),
    ("gemma-scope-2b",         0, 13183),
    ("gemma-scope-2b",         0, 13988),
    ("gemma-scope-2b",         0,  2725),
    ("gpt2-small-sae",         5, 30319),
    ("gpt2-small-sae",        10,  9896),
    ("Llama-3.1-8B-Instruct",  0, 12733),
    ("gpt2-xl",               20,  5278),
]

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "case_study"
OUTPUT_DIR.mkdir(exist_ok=True)

PRISM_PATH  = Path(__file__).parent.parent / "results" / "pairwise_similarities_dataset.csv"


def get_sim_matrix(row):
    sim = np.zeros((N_DESC, N_DESC))
    for i, j in SIM_PAIRS:
        sim[i-1, j-1] = sim[j-1, i-1] = row[f"sim_{i}{j}"]
    return sim


def filter_decisions(aurocs, sim_matrix):
    """Per-description decision strings. Delegates to `verification` (single
    source of truth); kept here as the import path the pipeline already uses."""
    return list(V.filter_and_collapse(aurocs, sim_matrix,
                                      sim_threshold=SIM_THRESHOLD,
                                      auroc_threshold=AUROC_THRESHOLD).decisions)


def compute_metrics(aurocs, sim_matrix, decisions):
    """Profile metrics for one feature. Delegates to `verification`.

    `decisions` is accepted for backward compatibility and is used to recover the
    survivor set, so callers that already hold decisions need not change.
    """
    surviving = tuple(i for i, d in enumerate(decisions) if d == V.D_SURVIVED)
    collapsed = tuple(i for i, d in enumerate(decisions) if d == V.D_COLLAPSED)
    filtered  = tuple(i for i, d in enumerate(decisions) if d == V.D_REMOVED)
    res = V.FilterResult(filtered=filtered, collapsed=collapsed,
                         surviving=surviving, decisions=tuple(decisions))
    return V.compute_metrics(aurocs, sim_matrix, res)


FIXED_AXIS_LIMIT = 1.0    # all plots share this fixed range (absolute scale)
LABEL_OFFSET    = 0.18    # label centre offset from node centre (data units)
SCALE_BAR_LEN   = 0.30    # reference-bar length = dissimilarity 0.30 (= 1 − cos sim)

# A collapse edge only ever connects a collapsed node to its surviving
# representative; removed (low-AUROC) nodes are filtered out before collapse and
# are never collapse targets.
EDGE_TARGET_DECISIONS = {"survived"}


def _draw_scale_bar(ax, length=SCALE_BAR_LEN):
    """Fixed reference bar in the bottom-left: `length` data units correspond to
    a dissimilarity of `length` (= 1 − cosine similarity). Identical on every
    plot, so spreads are directly comparable across neurons."""
    x0 = -FIXED_AXIS_LIMIT + 0.10
    y0 = -FIXED_AXIS_LIMIT + 0.12
    ax.plot([x0, x0 + length], [y0, y0], color="black", linewidth=1.6, zorder=8)
    for xx in (x0, x0 + length):
        ax.plot([xx, xx], [y0 - 0.02, y0 + 0.02], color="black", linewidth=1.6, zorder=8)
    ax.text(x0 + length / 2, y0 - 0.05, f"1 − sim = {length:g}",
            ha="center", va="top", fontsize=8, zorder=8)


def _best_label_angle(coords, i, offset, n_candidates=36, placed_labels=()):
    """Return the angle (radians) that places the label for node i as far as
    possible from all other node centres AND from already-placed labels.
    Tiebreaks toward the direction away from the group centroid."""
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    dx, dy = coords[i, 0] - cx, coords[i, 1] - cy
    base_angle = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0 else np.pi / 2

    obstacles = [(coords[j, 0], coords[j, 1]) for j in range(len(coords)) if j != i]
    obstacles += list(placed_labels)

    best_angle = base_angle
    best_score = -np.inf
    for k in range(n_candidates):
        angle = 2 * np.pi * k / n_candidates
        lx = coords[i, 0] + offset * np.cos(angle)
        ly = coords[i, 1] + offset * np.sin(angle)
        min_dist = min(np.hypot(lx - ox, ly - oy) for ox, oy in obstacles) if obstacles else 1.0
        tiebreak = 0.005 * (1 + np.cos(angle - base_angle))
        score = min_dist + tiebreak
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def make_mds_plot(aurocs, sim_matrix, decisions, descriptions, metrics, model, layer, unit):
    """MDS plot on an absolute, fixed scale: Euclidean distance = dissimilarity
    (1 − cosine similarity), with NO per-neuron rescaling, so every neuron is
    drawn on the same scale and all plots are directly cross-comparable. A fixed
    reference scale bar (SCALE_BAR_LEN) makes that shared scale visible. Tight
    neurons stay compact; spread-out neurons occupy more of the frame."""
    dissim = np.clip(1 - sim_matrix, 0, None)
    np.fill_diagonal(dissim, 0)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", n_init=4)
    coords = mds.fit_transform(dissim)
    # No rescaling: coordinates are used at their absolute MDS scale.

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes([0.10, 0.10, 0.76, 0.76])

    cmap = plt.cm.PiYG
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)

    S_NODE = 280  # uniform marker size for every description
    COLLAPSED_BLUE = "#1E90FF"  # collapsed is signified by blue on its border,
                                # label and collapse edge (fill stays AUROC-coloured)
    DECISION_MARKER = {"survived": "o", "collapsed": "D", "removed (low AUROC)": "D"}
    LABEL_COLOR = {"survived": "black", "collapsed": COLLAPSED_BLUE, "removed (low AUROC)": "black"}
    EDGE_COLOR  = {"survived": "black", "collapsed": COLLAPSED_BLUE, "removed (low AUROC)": "black"}

    # Edges: center-to-center lines from each collapsed node to its surviving
    # representative. "removed (low AUROC)" nodes are standalone — no edges.
    for i, j in combinations(range(N_DESC), 2):
        di, dj = decisions[i], decisions[j]
        is_edge = (sim_matrix[i, j] > SIM_THRESHOLD
                   and ((di == "collapsed" and dj in EDGE_TARGET_DECISIONS)
                        or (dj == "collapsed" and di in EDGE_TARGET_DECISIONS)))
        if not is_edge:
            continue
        sat, rep = (i, j) if di == "collapsed" else (j, i)
        xs, ys = coords[sat]
        xr, yr = coords[rep]
        ax.plot([xs, xr], [ys, yr],
                color=COLLAPSED_BLUE, linewidth=1.4, alpha=0.9, zorder=3)

    # Nodes: circle for survived only; diamonds for removed (low AUROC) and
    # collapsed, all the same size. Every node keeps its AUROC fill colour so the
    # AUROC stays visible; collapsed is signified only by blue on its border
    # (plus its blue label and blue collapse edge). Survived nodes are drawn
    # last so they sit on top.
    for i in range(N_DESC):
        if decisions[i] == "survived":
            continue
        color = cmap(norm(aurocs[i]))
        x, y = coords[i, 0], coords[i, 1]
        ax.scatter(x, y, color=color, s=S_NODE, marker=DECISION_MARKER[decisions[i]],
                   edgecolors=EDGE_COLOR[decisions[i]], linewidths=1.2, zorder=4)
    for i in range(N_DESC):
        if decisions[i] != "survived":
            continue
        color = cmap(norm(aurocs[i]))
        x, y = coords[i, 0], coords[i, 1]
        ax.scatter(x, y, color=color, s=S_NODE, marker=DECISION_MARKER[decisions[i]],
                   edgecolors=EDGE_COLOR[decisions[i]], linewidths=1.2, zorder=6)

    # Labels: placed greedily so each label avoids both node centres and
    # all previously placed labels, preventing overlap in tight clusters.
    # Collapsed descriptions get a blue label to match their blue edge.
    placed_labels = []
    for i in range(N_DESC):
        angle = _best_label_angle(coords, i, LABEL_OFFSET, placed_labels=placed_labels)
        lx = coords[i, 0] + LABEL_OFFSET * np.cos(angle)
        ly = coords[i, 1] + LABEL_OFFSET * np.sin(angle)
        placed_labels.append((lx, ly))
        # Surviving descriptions get the highest z-level (node at 6, label at 7)
        # so they always sit on top of removed/collapsed nodes and edges.
        label_z = 7 if decisions[i] == "survived" else 5
        ax.text(lx, ly, f"D{i+1}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=LABEL_COLOR[decisions[i]], zorder=label_z)

    # Colorbar with chance-level marker
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.10, 0.03, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("AUROC", rotation=270, labelpad=20, fontsize=11)
    cbar.ax.axhline(y=0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    # Legend. Shape encodes the category (circle = survived, diamond = removed);
    # marker colour is omitted because AUROC is shown by the colourbar. Collapsed
    # is shown as the blue collapse edge, matching its blue border/label.
    circle_h  = mlines.Line2D([], [], marker="o", linestyle="None",
                               markerfacecolor="white", markeredgecolor="black",
                               markersize=9, markeredgewidth=1.2, label="Survived")
    diamond_h = mlines.Line2D([], [], marker="D", linestyle="None",
                               markerfacecolor="white", markeredgecolor="black",
                               markersize=8, markeredgewidth=1.2, label="Removed")
    collapsed_h = mlines.Line2D([], [], color=COLLAPSED_BLUE, linewidth=1.8,
                                label="Collapsed")
    ax.legend(handles=[circle_h, diamond_h, collapsed_h], fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.07),
              framealpha=0.85, ncol=3)

    # Title (line 1) + subtitle (line 2)
    prism = metrics["prism_poly"]
    wms   = metrics["weighted_mean_sim"]
    q_val = metrics["evidence_quality"]
    if metrics["uncharacterized"]:
        stats_line = f"PRISM: {prism:.3f}  →  Uncharacterisable"
    else:
        wms_str = f"{wms:.3f}" if not np.isnan(wms) else "—"
        stats_line = f"PRISM: {prism:.3f}  →  WMS: {wms_str}   Q: {q_val:.3f}"
    ax.set_title(f"{model}  Layer {layer}  Unit {unit}\n{stats_line}",
                 fontsize=12, fontweight="bold", pad=12)

    # Stripped axes — ticks off, spines kept as a thin frame
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)
    ax.set_xlim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_ylim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    _draw_scale_bar(ax)

    base = OUTPUT_DIR / f"{model.replace('-','_')}_L{layer}_U{unit}_mds"
    png_path = base.with_suffix(".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close()
    return png_path


def main():
    df = pd.read_csv(PRISM_PATH)
    summary_rows = []

    for model, layer, unit in CASE_NEURONS:
        row = df[(df["model"] == model) & (df["layer"] == layer) & (df["unit"] == unit)].iloc[0]
        aurocs = np.array([row[f"auroc_{i}"] for i in range(1, N_DESC + 1)])
        descs = [row[f"description_{i}"] for i in range(1, N_DESC + 1)]
        sim_matrix = get_sim_matrix(row)

        decisions = filter_decisions(aurocs, sim_matrix)
        metrics = compute_metrics(aurocs, sim_matrix, decisions)

        tag = f"{model} L{layer}_U{unit}"
        print(f"\n{'='*70}")
        print(f"{tag}")
        print(f"{'='*70}")
        print(f"  PRISM score:        {metrics['prism_poly']:.3f}")
        print(f"  Evidence quality Q: {metrics['evidence_quality']:.3f}" if not np.isnan(metrics.get('evidence_quality', np.nan)) else "  Evidence quality Q: —")
        print(f"  Survivor count n_s: {metrics['n_surviving']}")
        print(f"  Mean sim post:      {metrics['mean_sim_post']:.3f}" if not np.isnan(metrics.get('mean_sim_post', np.nan)) else "  Mean sim post:      —")
        print(f"  Weighted mean sim:  {metrics['weighted_mean_sim']:.3f}" if not np.isnan(metrics.get('weighted_mean_sim', np.nan)) else "  Weighted mean sim:  —")
        print(f"  Effective N:        {metrics['effective_n']:.3f}" if metrics['effective_n'] is not np.nan and not np.isnan(metrics['effective_n']) else "  Effective N:        —")
        print(f"  Uncharacterized:    {metrics['uncharacterized']}")
        print()
        for i in range(N_DESC):
            print(f"  [{i+1}] AUROC={aurocs[i]:.3f}  [{decisions[i]:>20s}]  {descs[i][:90]}")

        make_mds_plot(aurocs, sim_matrix, decisions, descs, metrics, model, layer, unit)
        print(f"\n  MDS plot saved.")

        summary_rows.append({
            "neuron": tag,
            "prism_poly": round(metrics["prism_poly"], 3),
            "evidence_quality": round(metrics["evidence_quality"], 3) if not np.isnan(metrics.get("evidence_quality", np.nan)) else "—",
            "n_surviving": metrics["n_surviving"],
            "mean_sim_post": round(metrics["mean_sim_post"], 3) if not np.isnan(metrics.get("mean_sim_post", np.nan)) else "—",
            "weighted_mean_sim": round(metrics["weighted_mean_sim"], 3) if not np.isnan(metrics.get("weighted_mean_sim", np.nan)) else "—",
            "effective_n": round(metrics["effective_n"], 3) if metrics["effective_n"] is not np.nan and not np.isnan(metrics["effective_n"]) else "—",
            "uncharacterized": metrics["uncharacterized"],
        })

    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "case_study_metrics.csv", index=False)
    print(f"\nSaved metrics to {OUTPUT_DIR / 'case_study_metrics.csv'}")
    print(f"MDS plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

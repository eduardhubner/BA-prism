"""
Toy-neuron figures: failure modes of the (unweighted) polysemanticity score.

Each figure shows TWO synthetic neurons side-by-side that have the *same*
polysemanticity score (= mean pairwise cosine similarity of the 5 descriptions)
but differ in a way the scalar score is blind to:

  Case 1 - quality / weighting blindness: the diversity is driven by barely-
           reliable descriptions on the left (they should get less weight) vs.
           genuinely reliable descriptions on the right.
  Case 2 - distribution / outlier blindness: one far outlier on the left vs. a
           uniformly moderate spread on the right.
  Case 3 - score is meaningless: all five descriptions below 0.5 AUROC on the
           left (worse than chance) vs. the same geometry with reliable
           descriptions on the right.

Two versions are produced per case, both on the same absolute, cross-comparable
MDS scale (PiYG AUROC colourbar with a chance line, stripped axes, D-labels, and
a fixed "1 − sim" scale bar):

  Version 1 (`toy_caseN_*`): the RAW view -- plain circles, no filter semantics,
    only the shared "Polysemanticity Score" shown. Makes the failure mode visible.
  Version 2 (`toy_caseN_*_framework`): the SAME pair seen THROUGH the reliability
    framework -- filtering (survived / removed / collapsed), plus WMS and evidence
    quality Q per panel (the survivor count is read from the plot). The framework
    now distinguishes the two neurons that the raw score alone conflates (Case 1
    by Q, Case 2 by collapse/survivor count, Case 3 by the uncharacterized flag).

Output: results/case_study/toy_case{1,2,3}_*.png / *_framework.png (+ .svg)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from itertools import combinations
from pathlib import Path
from sklearn.manifold import MDS

# Reuse the canonical filter + metrics so the framework view matches the pipeline
# exactly (works both as `python src/toy_neurons.py` and `import src.toy_neurons`).
try:
    from case_study_analysis import filter_decisions, compute_metrics, SIM_THRESHOLD
except ImportError:
    from src.case_study_analysis import filter_decisions, compute_metrics, SIM_THRESHOLD

# --- Config (shared with case_study_analysis.py style) ---
N_DESC = 5
PAIRS = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]

FIXED_AXIS_LIMIT = 1.0    # shared fixed range, matches case_study_analysis.py
LABEL_OFFSET     = 0.18
SCALE_BAR_LEN    = 0.30    # reference-bar length = dissimilarity 0.30 (= 1 − cos sim)
S_NODE           = 280
COLLAPSED_BLUE   = "#1E90FF"  # collapsed signified by blue (border/label/edge)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "case_study"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Toy definitions -------------------------------------------------------
# Each neuron: pairwise sims in PAIRS order (length 10) and 5 AUROC values.

def uniform_pairs(value):
    return [value] * len(PAIRS)


def pairs_with_overrides(base, overrides):
    """Uniform `base` similarity, with {(i,j): value} overrides (1-indexed)."""
    vals = []
    for (i, j) in PAIRS:
        vals.append(overrides.get((i, j), base))
    return vals


# Case 1: two reliable descriptions that are the closest pair (D1,D2), three
# barely-above-chance descriptions far from everything -> they drive the
# diversity. Same geometry/score on both panels; only the AUROC of D3-D5 differs.
# D1,D2 sim is 0.52 -- below the 0.70 collapse threshold AND far enough apart
# that the MDS-projected distance (~0.34) clearly exceeds the collapse-radius
# scale bar (0.30). So they read as "the reliable core", not a redundant pair
# that would collapse (that is Case 2's story). Other pairs are ~0.352 so the
# mean stays 0.369. (At sim 0.63 they were fine in the data but MDS compressed
# them below the scale bar, making them look collapse-close.)
C1_SIMS = pairs_with_overrides(0.352, {(1, 2): 0.52})   # mean = 0.369
CASE1 = {
    "name": "toy_case1_quality", "num": 1,
    "left":  {"sims": C1_SIMS, "aurocs": [0.95, 0.93, 0.56, 0.58, 0.55]},
    "right": {"sims": C1_SIMS, "aurocs": [0.95, 0.93, 0.90, 0.92, 0.88]},
}

# Case 2: four near-duplicate descriptions + one far outlier vs. a uniformly
# moderate spread. Same mean similarity, different distribution.
C2_SIMS_OUTLIER = pairs_with_overrides(
    0.80, {(1, 5): 0.145, (2, 5): 0.145, (3, 5): 0.145, (4, 5): 0.145}
)                                                       # mean = 0.538
C2_SIMS_UNIFORM = uniform_pairs(0.538)
CASE2 = {
    "name": "toy_case2_outlier", "num": 2,
    "left":  {"sims": C2_SIMS_OUTLIER, "aurocs": [0.92, 0.90, 0.88, 0.91, 0.89]},
    "right": {"sims": C2_SIMS_UNIFORM, "aurocs": [0.90, 0.90, 0.90, 0.90, 0.90]},
}

# Case 3: all five descriptions below chance (score is meaningless) vs. the same
# geometry with reliable descriptions.
C3_SIMS = uniform_pairs(0.408)
CASE3 = {
    "name": "toy_case3_uncharacterized", "num": 3,
    "left":  {"sims": C3_SIMS, "aurocs": [0.30, 0.28, 0.22, 0.35, 0.25]},
    "right": {"sims": C3_SIMS, "aurocs": [0.80, 0.85, 0.78, 0.88, 0.82]},
}

CASES = [CASE1, CASE2, CASE3]


# --- Geometry --------------------------------------------------------------

def build_sim_matrix(pair_vals):
    sim = np.eye(N_DESC)
    for (i, j), v in zip(PAIRS, pair_vals):
        sim[i-1, j-1] = sim[j-1, i-1] = v
    return sim


def polysemanticity_score(pair_vals):
    """Mean pairwise cosine similarity -- the original PRISM score."""
    return float(np.mean(pair_vals))


def mds_coords(sim_matrix, return_stress=False):
    """2-D metric MDS of the description dissimilarities 1 - cos(s_i, s_j).

    Metric (not ordinal) MDS: SMACOF minimises the RAW stress
        sigma(Z) = sum_{i<j} ( ||z_i - z_j||_2 - Delta_ij )^2
    so plotted distances are on the same scale as 1 - cos, which is what lets
    every feature panel share one absolute axis range and a common scale bar.

    Every argument is pinned deliberately -- these coordinates appear in published
    figures, so the projection must not drift:
      metric="precomputed"  Delta is supplied directly (replaces the deprecated
                            `dissimilarity=`; verified to give identical output).
      init="random"         sklearn changes this default to "classical_mds" in
                            1.10. Pinned so the figures survive the upgrade;
                            switching WOULD move the coordinates.
      n_init=4              four restarts, lowest-stress solution kept.
      random_state=42       reproducible across runs.
    normalized_stress="auto" resolves to raw stress for metric MDS, so
    `mds.stress_` is sigma above, not Kruskal stress-1 (see kruskal_stress).
    """
    dissim = np.clip(1 - sim_matrix, 0, None)
    np.fill_diagonal(dissim, 0)
    mds = MDS(n_components=2, metric="precomputed", random_state=42,
              normalized_stress="auto", n_init=4, init="random")
    coords = mds.fit_transform(dissim)
    return (coords, mds.stress_) if return_stress else coords


def kruskal_stress(sim_matrix, coords):
    """Kruskal stress-1 of a 2-D layout: sqrt( sum (d-Delta)^2 / sum Delta^2 ).

    Reported because raw stress is not scale-free and so is not comparable across
    features. Convention: < 0.10 good, 0.10-0.20 fair.
    """
    dissim = np.clip(1 - sim_matrix, 0, None)
    np.fill_diagonal(dissim, 0)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu = np.triu_indices(len(dissim), 1)
    return float(np.sqrt(((d[iu] - dissim[iu]) ** 2).sum() / (dissim[iu] ** 2).sum()))


def _best_label_angle(coords, i, offset, n_candidates=36, placed_labels=()):
    """Place label i as far as possible from other nodes and placed labels."""
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    dx, dy = coords[i, 0] - cx, coords[i, 1] - cy
    base_angle = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0 else np.pi / 2
    obstacles = [(coords[j, 0], coords[j, 1]) for j in range(len(coords)) if j != i]
    obstacles += list(placed_labels)
    best_angle, best_score = base_angle, -np.inf
    for k in range(n_candidates):
        angle = 2 * np.pi * k / n_candidates
        lx = coords[i, 0] + offset * np.cos(angle)
        ly = coords[i, 1] + offset * np.sin(angle)
        min_dist = min(np.hypot(lx - ox, ly - oy) for ox, oy in obstacles) if obstacles else 1.0
        tiebreak = 0.005 * (1 + np.cos(angle - base_angle))
        score = min_dist + tiebreak
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


# --- Rendering -------------------------------------------------------------

def _draw_scale_bar(ax, length=SCALE_BAR_LEN):
    """Fixed reference bar: `length` data units = a dissimilarity of `length`
    (= 1 − cosine similarity). Identical on every plot (toys and case studies),
    so spreads are directly comparable across neurons."""
    x0 = -FIXED_AXIS_LIMIT + 0.10
    y0 = -FIXED_AXIS_LIMIT + 0.12
    ax.plot([x0, x0 + length], [y0, y0], color="black", linewidth=1.6, zorder=8)
    for xx in (x0, x0 + length):
        ax.plot([xx, xx], [y0 - 0.02, y0 + 0.02], color="black", linewidth=1.6, zorder=8)
    ax.text(x0 + length / 2, y0 - 0.05, f"1 − sim = {length:g}",
            ha="center", va="top", fontsize=8, zorder=8)


def _draw_panel(ax, coords, aurocs, cmap, norm):
    for i in range(N_DESC):
        ax.scatter(coords[i, 0], coords[i, 1], color=cmap(norm(aurocs[i])),
                   s=S_NODE, marker="o", edgecolors="black", linewidths=1.2,
                   zorder=4)
    placed = []
    for i in range(N_DESC):
        angle = _best_label_angle(coords, i, LABEL_OFFSET, placed_labels=placed)
        lx = coords[i, 0] + LABEL_OFFSET * np.cos(angle)
        ly = coords[i, 1] + LABEL_OFFSET * np.sin(angle)
        placed.append((lx, ly))
        ax.text(lx, ly, f"D{i+1}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="black", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb"); spine.set_linewidth(0.8)
    ax.set_xlim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_ylim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    _draw_scale_bar(ax)


def render_case(case):
    cmap = plt.cm.PiYG
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)

    # Both panels share one score; each is drawn at absolute MDS scale.
    panels = []
    for side in ("left", "right"):
        coords = mds_coords(build_sim_matrix(case[side]["sims"]))
        panels.append((coords, np.asarray(case[side]["aurocs"], dtype=float)))
    # Absolute scale (no rescaling): toy and case-study plots share one scale and
    # are directly comparable; the fixed reference bar makes that scale visible.
    scale = 1.0

    score = polysemanticity_score(case["left"]["sims"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (coords, aurocs), suffix in zip((ax1, ax2), panels, ("a", "b")):
        _draw_panel(ax, coords * scale, aurocs, cmap, norm)
        ax.set_title(f"Toy Neuron {case['num']}{suffix}",
                     fontsize=12, fontweight="bold", pad=8)

    # Shared AUROC colourbar with chance-level marker.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.035, pad=0.03)
    cbar.set_label("AUROC", rotation=270, labelpad=20, fontsize=11)
    cbar.ax.axhline(y=0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    fig.suptitle(f"Polysemanticity Score: {score:.3f}",
                 fontsize=14, fontweight="bold")

    base = OUTPUT_DIR / case["name"]
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return base, score


def _draw_framework_panel(ax, coords, aurocs, sim_matrix, decisions, cmap, norm):
    """Draw a toy neuron AFTER the reliability filter: survived = circle, removed
    = diamond, collapsed = blue-outlined diamond joined to its survivor by a blue
    edge. Same visual language as the case-study MDS plots."""
    marker = {"survived": "o", "collapsed": "D", "removed (low AUROC)": "D"}
    edge_c = {"survived": "black", "collapsed": COLLAPSED_BLUE, "removed (low AUROC)": "black"}
    label_c = {"survived": "black", "collapsed": COLLAPSED_BLUE, "removed (low AUROC)": "black"}

    # Collapse edges: each collapsed node to its surviving representative.
    for i, j in combinations(range(N_DESC), 2):
        di, dj = decisions[i], decisions[j]
        if sim_matrix[i, j] > SIM_THRESHOLD and (
                (di == "collapsed" and dj == "survived") or
                (dj == "collapsed" and di == "survived")):
            sat, rep = (i, j) if di == "collapsed" else (j, i)
            ax.plot([coords[sat, 0], coords[rep, 0]], [coords[sat, 1], coords[rep, 1]],
                    color=COLLAPSED_BLUE, linewidth=1.4, alpha=0.9, zorder=3)

    # Non-survivors first, survivors on top.
    for i in range(N_DESC):
        if decisions[i] == "survived":
            continue
        ax.scatter(coords[i, 0], coords[i, 1], color=cmap(norm(aurocs[i])), s=S_NODE,
                   marker=marker[decisions[i]], edgecolors=edge_c[decisions[i]],
                   linewidths=1.2, zorder=4)
    for i in range(N_DESC):
        if decisions[i] != "survived":
            continue
        ax.scatter(coords[i, 0], coords[i, 1], color=cmap(norm(aurocs[i])), s=S_NODE,
                   marker="o", edgecolors="black", linewidths=1.2, zorder=6)

    placed = []
    for i in range(N_DESC):
        angle = _best_label_angle(coords, i, LABEL_OFFSET, placed_labels=placed)
        lx = coords[i, 0] + LABEL_OFFSET * np.cos(angle)
        ly = coords[i, 1] + LABEL_OFFSET * np.sin(angle)
        placed.append((lx, ly))
        z = 7 if decisions[i] == "survived" else 5
        ax.text(lx, ly, f"D{i+1}", ha="center", va="center", fontsize=11,
                fontweight="bold", color=label_c[decisions[i]], zorder=z)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb"); spine.set_linewidth(0.8)
    ax.set_xlim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_ylim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    _draw_scale_bar(ax)


def render_case_framework(case):
    """Version 2: the same toy pair seen THROUGH the reliability framework.
    Filtering, WMS and evidence quality Q now distinguish the two neurons that
    the raw polysemanticity score alone cannot tell apart."""
    cmap = plt.cm.PiYG
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.8))
    for ax, side, suffix in zip((ax1, ax2), ("left", "right"), ("a", "b")):
        aurocs = np.asarray(case[side]["aurocs"], dtype=float)
        sim_matrix = build_sim_matrix(case[side]["sims"])
        coords = mds_coords(sim_matrix)
        decisions = filter_decisions(aurocs, sim_matrix)
        metrics = compute_metrics(aurocs, sim_matrix, decisions)
        _draw_framework_panel(ax, coords, aurocs, sim_matrix, decisions, cmap, norm)
        # nₛ is left off the subtitle -- the survivor count is read from the plot.
        if metrics["uncharacterized"]:
            wms_str, q_str = "—", "—"
        else:
            wms = metrics["weighted_mean_sim"]
            wms_str = f"{wms:.3f}" if not np.isnan(wms) else "—"
            q_str = f"{metrics['evidence_quality']:.3f}"
        ax.set_title(f"Toy Neuron {case['num']}{suffix}\nWMS = {wms_str}    Q = {q_str}",
                     fontsize=11, fontweight="bold", pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.035, pad=0.03)
    cbar.set_label("AUROC", rotation=270, labelpad=20, fontsize=11)
    cbar.ax.axhline(y=0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    circle_h = mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
                             markeredgecolor="black", markersize=9, markeredgewidth=1.2, label="Survived")
    diamond_h = mlines.Line2D([], [], marker="D", linestyle="None", markerfacecolor="white",
                              markeredgecolor="black", markersize=8, markeredgewidth=1.2, label="Removed")
    collapsed_h = mlines.Line2D([], [], color=COLLAPSED_BLUE, linewidth=1.8, label="Collapsed")
    fig.legend(handles=[circle_h, diamond_h, collapsed_h], fontsize=10, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), framealpha=0.85)

    base = OUTPUT_DIR / f"{case['name']}_framework"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return base


def main():
    for case in CASES:
        base, score = render_case(case)
        print(f"{case['name']:34s}  score={score:.3f}  -> {base.name}.png/.svg")
        fbase = render_case_framework(case)
        print(f"{'  (framework)':34s}                  -> {fbase.name}.png/.svg")


if __name__ == "__main__":
    main()

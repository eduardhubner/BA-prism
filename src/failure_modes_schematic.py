"""
Failure-modes schematic (paper), before/after the reliability framework.

Two rows x three columns of failure modes; within each mode, neurons A and B are
SEPARATE MDS plots. Top row = raw toy neurons (same PRISM_poly). Bottom row = the
same neurons after the framework (AUROC filter + similarity collapse), using the
MDS-plot convention (circle = survived, diamond = removed, blue-outline diamond +
blue link = collapsed), annotated with sigma-tilde and Q.

  (a) Quality blindness        - unreliable descriptions (below chance) are filtered;
                                 no collapse.
  (b) Redundancy               - redundant descriptions collapse into fewer survivors.
  (c) Characterization Failure - no reliable description survives (sigma = Q = -).

Encoding (PiYG/AUROC): fill = AUROC (green reliable, pink below chance, white =
chance); proximity = description similarity; axes = schematic MDS coordinates.
"""

from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import thesis_style as ts

_CMAP = ts.AUROC_CMAP
_NORM = ts.auroc_norm()
BLUE = ts.COLLAPSED_BLUE
SCALE = 0.5
D0 = 1.7
HI, LO = 0.93, 0.40
AUROC_THR, SIM_THR = 0.50, 0.70
S_NODE = 115

MARKER = {"survived": "o", "removed": "D", "collapsed": "D"}
EDGE = {"survived": "black", "removed": "black", "collapsed": BLUE}

# toy geometries. (a) reliable pair spaced so it does NOT collapse (sim < 0.70).
QUAL = np.array([(-0.56, 0.52), (0.56, 0.52),                    # reliable pair (no collapse)
                 (-1.05, -0.30), (1.05, -0.30), (0.0, -1.10)])   # the other three
A_RED = np.array([(-0.98, 0.58), (-0.40, 0.82), (-0.50, 0.30),   # concept 1 (3, collapse)
                  (0.78, -0.56), (0.78, -1.06)])                 # concept 2 (2, collapse)
PENT = np.array([(0.0, 0.92), (-0.87, 0.29), (-0.54, -0.76), (0.54, -0.76), (0.87, 0.29)])


def sim_matrix(coords):
    P = coords * SCALE
    n = len(P)
    S = np.eye(n)
    for i, j in combinations(range(n), 2):
        S[i, j] = S[j, i] = max(0.0, 1.0 - np.hypot(*(P[i] - P[j])) / D0)
    return S


def prism_poly(coords):
    S = sim_matrix(coords)
    return float(np.mean([S[i, j] for i, j in combinations(range(len(coords)), 2)]))


def filter_collapse(aurocs, S):
    kept = [i for i in range(len(aurocs)) if aurocs[i] > AUROC_THR]
    adj = {i: set() for i in kept}
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            i, j = kept[a], kept[b]
            if S[i, j] > SIM_THR:
                adj[i].add(j); adj[j].add(i)
    visited, comps = set(), []
    for s in kept:
        if s not in visited:
            c, st = [], [s]
            while st:
                x = st.pop()
                if x not in visited:
                    visited.add(x); c.append(x); st.extend(adj[x] - visited)
            comps.append(c)
    rep = {max(c, key=lambda i: aurocs[i]): c for c in comps}
    dec = ["removed" if aurocs[i] <= AUROC_THR else ("survived" if i in rep else "collapsed")
           for i in range(len(aurocs))]
    coll_to_rep = {m: r for r, c in rep.items() for m in c if m != r}
    return dec, coll_to_rep


def metrics(aurocs, S, dec):
    surv = [i for i, d in enumerate(dec) if d == "survived"]
    n_s = len(surv)
    if n_s == 0:
        return 0, None, None
    Q = 2 * np.mean([aurocs[i] for i in surv]) - 1
    if n_s == 1:
        return 1, 1.0, Q
    num = den = 0.0
    for i, j in combinations(surv, 2):
        w = (2 * aurocs[i] - 1) * (2 * aurocs[j] - 1)
        num += w * S[i, j]; den += w
    return n_s, (num / den if den > 0 else None), Q


def _after_label(n_s, sig, Q):
    if n_s == 0:
        return r"$\tilde{\sigma}$ = —,  Q = —"
    return rf"$\tilde{{\sigma}}$ = {sig:.2f},  Q = {Q:.2f}"


def _frame(ax, lim, xlab=False, ylab=False):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb"); s.set_linewidth(0.8)
    if xlab:
        ax.set_xlabel("MDS1", fontsize=10, color="#666", labelpad=2)
    if ylab:
        ax.set_ylabel("MDS2", fontsize=10, color="#666", labelpad=2)


def _draw_raw(ax, C, aurocs):
    P = C * SCALE
    for (x, y), a in zip(P, aurocs):
        ax.scatter(x, y, s=S_NODE, color=_CMAP(_NORM(a)), edgecolors="black", linewidths=0.8, zorder=3)


def _draw_framework(ax, C, aurocs, dec, coll_to_rep):
    P = C * SCALE
    for i, r in coll_to_rep.items():
        ax.plot([P[i, 0], P[r, 0]], [P[i, 1], P[r, 1]], color=BLUE, lw=1.0, alpha=0.8, zorder=2)
    for i in sorted(range(len(P)), key=lambda i: dec[i] == "survived"):
        ax.scatter(P[i, 0], P[i, 1], s=S_NODE, color=_CMAP(_NORM(aurocs[i])), marker=MARKER[dec[i]],
                   edgecolors=EDGE[dec[i]], linewidths=1.3 if dec[i] == "collapsed" else 0.9,
                   zorder=4 if dec[i] == "survived" else 3)


def make():
    ts.apply()
    # (a) uses the pentagon (all nodes mutually far => no collapse); A has two
    # reliable + three below-chance (filtered), B has five reliable.
    modes = [
        ("(a) Quality blindness", (PENT, [HI, HI, LO, LO, LO]), (PENT, [HI] * 5)),
        ("(b) Redundancy", (A_RED, [HI] * 5), (PENT, [HI] * 5)),
        ("(c) Characterization Failure", (PENT, [LO] * 5), (PENT, [HI] * 5)),
    ]
    lim = max(np.abs((c - c.mean(0)) * SCALE).max()
              for _, A, B in modes for c, _a in (A, B)) * 1.30

    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(2, 6, left=0.055, right=0.915, top=0.80, bottom=0.15,
                          wspace=0.28, hspace=0.14)
    top_ax = {}
    for mi, (title, A, B) in enumerate(modes):
        for ni, ((coords, aurocs), lab) in enumerate([(A, "A"), (B, "B")]):
            col = 2 * mi + ni
            C = coords - coords.mean(0)
            ax0 = fig.add_subplot(gs[0, col]); _frame(ax0, lim, ylab=(col == 0))
            _draw_raw(ax0, C, aurocs); ax0.set_title(lab, fontsize=12, fontweight="bold", pad=3)
            top_ax[(mi, ni)] = ax0

            ax1 = fig.add_subplot(gs[1, col]); _frame(ax1, lim, xlab=True, ylab=(col == 0))
            S = sim_matrix(coords)
            dec, c2r = filter_collapse(aurocs, S)
            _draw_framework(ax1, C, aurocs, dec, c2r)
            n_s, sig, Q = metrics(aurocs, S, dec)
            ax1.text(0.5, -0.17, _after_label(n_s, sig, Q), transform=ax1.transAxes,
                     ha="center", va="top", fontsize=10, color="#333")

    # mode titles + shared PRISM_poly, centred over each A/B pair
    for mi, (title, A, B) in enumerate(modes):
        p0 = top_ax[(mi, 0)].get_position(); p1 = top_ax[(mi, 1)].get_position()
        cx = (p0.x0 + p1.x1) / 2
        fig.text(cx, 0.95, title, ha="center", fontsize=13)
        fig.text(cx, 0.885, f"PRISM_poly = {prism_poly(A[0]):.2f}", ha="center",
                 fontsize=10, style="italic", color="#555")

    fig.text(0.012, 0.60, "before framework", rotation=90, va="center", fontsize=12, color="#444")
    fig.text(0.012, 0.32, "after framework", rotation=90, va="center", fontsize=12, color="#444")

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=_NORM); sm.set_array([])
    cax = fig.add_axes([0.935, 0.27, 0.013, 0.40])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=[0.0, 0.5, 1.0])
    cbar.set_label("AUROC", fontsize=11)
    cbar.ax.axhline(0.5, color="black", ls="--", lw=0.9, alpha=0.7)
    cbar.ax.set_yticklabels(["0.0", "0.5 (chance)", "1.0"], fontsize=9)

    handles = [
        mlines.Line2D([], [], marker="o", ls="None", mfc="white", mec="black", ms=7, mew=1.1, label="survived"),
        mlines.Line2D([], [], marker="D", ls="None", mfc="white", mec="black", ms=6, mew=1.0, label="filtered"),
        mlines.Line2D([], [], marker="D", ls="None", mfc="white", mec=BLUE, ms=6, mew=1.4, label="redundant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.48, 0.0))

    ts.save(fig, "fig_failure_modes")
    plt.close(fig)
    for mi, (title, A, B) in enumerate(modes):
        for (coords, aurocs), lab in [(A, "A"), (B, "B")]:
            S = sim_matrix(coords); dec, _ = filter_collapse(aurocs, S)
            n_s, sig, Q = metrics(aurocs, S, dec)
            print(f"  ({'abc'[mi]}){lab} poly={prism_poly(coords):.3f} n_s={n_s} "
                  f"sig={sig if sig is None else round(sig,3)} Q={Q if Q is None else round(Q,3)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    make()

"""
Cross-method complementary coverage.

A method *characterizes* a feature if it provides at least one description with
AUROC > 0.50 (a supported, above-chance characterization); otherwise it leaves the
feature *uncharacterized*. (Collapse cannot turn a non-empty supported set into an
empty one, so for this binary diagnostic characterization depends only on AUROC — no
embeddings are used.) For every ordered pair of methods (A, B) evaluated under the
same CoSy regime and sharing features:

    C(A -> B) = P(B characterizes | A does not characterize)
              = #{uncharacterized by A but characterized by B} / #{uncharacterized by A, B available}

i.e. when method A does not support a characterization, how often does method B cover
that feature (informally: "B rescues a feature A leaves uncharacterized").

This diagnostic is not represented by the original PRISM polysemanticity score. PRISM
already produces AUROC evidence, so a similar binary quantity could be derived post
hoc; the revised framework makes absence of supported descriptions an explicit
outcome, which makes this comparison natural and well-defined.

The diagnostic is DIRECTIONAL — C(A->B) != C(B->A) in general — but no method is a
privileged baseline: both directions are computed.

CAVEAT (keep prominent): methods do not have equal proposal budgets — PRISM has 5
descriptions per feature, the others 1 — so a method with more attempts has more
chances at one supported description, and the directional asymmetry can arise from
both genuine complementary coverage and unequal budgets. These rates measure
COMPLEMENTARY COVERAGE, not relative method quality.

Regimes are never mixed; cells are populated only where the two methods co-occur.
Aggregate cells pool over shared targets, so per-target rates are also saved (a pooled
rate can be dominated by one target — e.g. SAE, where PRISM's uncharacterized rate is
high). Pure analysis over the pooled artifacts; no new embedding computation.

RUN:  python src/cross_method_rescue.py
Outputs: results/cross_method_rescue/stats.json; figures/fig_cross_method_rescue.{png,pdf,svg}
"""

import sys
import json
from itertools import permutations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import thesis_style as ts

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "cross_method_rescue"
OUT.mkdir(parents=True, exist_ok=True)

OLD_METHODS = ["PRISM", "output-centric", "Neuronpedia", "GPT-explain"]
NEW_METHODS = ["Transluce", "max-act"]
BUDGET = {"PRISM": 5, "output-centric": 1, "Neuronpedia": 1, "GPT-explain": 1,
          "Transluce": 1, "max-act": 1}
TARGET_SHORT = {"Llama-3.1-8B-Instruct": "Llama", "gemma-scope-2b": "Gemma",
                "gpt2-small-sae": "GPT-2 SAE", "gpt2-xl": "GPT-2 XL"}
AUC_THR = 0.50


def _add_source(char, desc, method_masks):
    """char[(target,layer,unit)][method] = bool. Groups by (_target, layer, unit) so
    identical coordinates across models do not collide."""
    a = desc["auc"].to_numpy()
    for (target, layer, unit), g in desc.groupby(["_target", "layer", "unit"]):
        feat = char.setdefault((target, int(layer), int(unit)), {})
        for method, mask in method_masks.items():
            idx = g.index[mask(g)].to_numpy()
            if len(idx):
                feat[method] = bool((a[idx] > AUC_THR).any())


def gather_characterization():
    old, new = {}, {}
    oc = pd.read_csv(RESULTS / "pooled_outputcentric" / "descriptions.csv")
    oc["_target"] = oc["model"]
    _add_source(old, oc, {
        "PRISM": lambda g: (g["method"] == "input-centric").values,
        "output-centric": lambda g: (g["method"] == "output-centric").values,
        "Neuronpedia": lambda g: (g["method"] == "neuronpedia").values,
    })
    gx = pd.read_csv(RESULTS / "pooled_gpt2xl" / "descriptions.csv")
    gx["_target"] = "gpt2-xl"
    _add_source(old, gx, {
        "PRISM": lambda g: ((g["method"] == "input-centric") & (g["generator"] == "gemini-1.5-pro")).values,
        "GPT-explain": lambda g: (g["method"] == "GPT-explain").values,
    })
    nr = pd.read_csv(RESULTS / "pooled_newregime_llama" / "descriptions.csv")
    nr["_target"] = nr["model"]
    _add_source(new, nr, {
        "Transluce": lambda g: (g["method"] == "Transluce").values,
        "max-act": lambda g: (g["method"] == "max_act").values,
    })
    return old, new


def _rate(den, num):
    return {"n_uncharacterized_by_A": den, "n_covered_by_B": num,
            "coverage_pct": round(100 * num / den, 1) if den else None}


def coverage(char, A, B):
    """Per-target and pooled C(A->B) over features where both A and B are available."""
    per = {}
    tot = [0, 0]
    for f, m in char.items():
        if A in m and B in m and not m[A]:            # A uncharacterized, B available
            t = f[0]
            d = per.setdefault(t, [0, 0]); d[0] += 1; tot[0] += 1
            if m[B]:
                d[1] += 1; tot[1] += 1
    if tot[0] == 0:
        return None
    out = {t: _rate(den, num) for t, (den, num) in per.items()}
    out["pooled"] = _rate(tot[0], tot[1])
    return out


def coverage_matrix(char, methods):
    M = {A: {} for A in methods}
    for A, B in permutations(methods, 2):
        c = coverage(char, A, B)
        if c is not None:
            M[A][B] = c
    return M


def _draw_matrix(ax, M, methods, title):
    n = len(methods)
    grid = np.full((n, n), np.nan)
    for i, A in enumerate(methods):
        for j, B in enumerate(methods):
            if i != j and B in M.get(A, {}):
                grid[i, j] = M[A][B]["pooled"]["coverage_pct"]
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(methods, fontsize=8, rotation=30, ha="left")
    ax.set_yticklabels(methods, fontsize=8)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    for i, A in enumerate(methods):
        for j, B in enumerate(methods):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=9)
            elif B in M.get(A, {}):
                p = M[A][B]["pooled"]; col = "white" if p["coverage_pct"] > 55 else "#222"
                ax.text(j, i, f"{p['coverage_pct']:.0f}%\n{p['n_covered_by_B']}/{p['n_uncharacterized_by_A']}",
                        ha="center", va="center", color=col, fontsize=7)
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#ccc", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=22)
    ax.set_ylabel("A: does not characterize", fontsize=8)
    return im


def figure(old_M, new_M):
    ts.apply()
    fig = plt.figure(figsize=(11.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(OLD_METHODS), len(NEW_METHODS)],
                          left=0.10, right=0.92, top=0.78, bottom=0.12, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0]); im = _draw_matrix(ax0, old_M, OLD_METHODS, "old regime (gemini-1.5-pro)")
    ax1 = fig.add_subplot(gs[0, 1]); _draw_matrix(ax1, new_M, NEW_METHODS, "new regime (gemini-2.0-flash-lite)")
    fig.text(0.5, 0.90, "Cross-method complementary coverage", ha="center", fontsize=12)
    fig.text(0.5, 0.03, "C(A→B) = P(B characterizes | A does not).  rows A = does not characterize · columns B = characterizes · "
             "cell = C(A→B), covered/uncharacterized.\nRates measure complementary coverage, not method quality; proposal budgets differ across methods.",
             ha="center", fontsize=7.5, color="#555")
    cax = fig.add_axes([0.94, 0.30, 0.012, 0.40])
    fig.colorbar(im, cax=cax).set_label("coverage %", fontsize=8)
    ts.save(fig, "fig_cross_method_rescue")
    plt.close(fig)


def main():
    old, new = gather_characterization()
    old_M, new_M = coverage_matrix(old, OLD_METHODS), coverage_matrix(new, NEW_METHODS)
    stats = {
        "definition": "C(A->B) = P(B characterizes | A does not) over shared features",
        "characterization": "method characterizes a feature if >=1 of its descriptions has AUROC > 0.50",
        "directional": "C(A->B) != C(B->A) in general; both computed, no privileged baseline",
        "caveat": "complementary coverage, not method quality; unequal proposal budgets",
        "proposal_budget_desc_per_feature": BUDGET,
        "old_regime": {"eval": "gemini-1.5-pro", "matrix": old_M},
        "new_regime": {"eval": "gemini-2.0-flash-lite", "matrix": new_M},
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    figure(old_M, new_M)

    for name, M, methods in [("OLD (gemini-1.5-pro)", old_M, OLD_METHODS),
                             ("NEW (gemini-2.0-flash-lite)", new_M, NEW_METHODS)]:
        print(f"\n{name} — C(A does not characterize -> B characterizes):")
        for A in methods:
            for B in methods:
                if B in M.get(A, {}):
                    p = M[A][B]["pooled"]
                    bt = "  ".join(f"{TARGET_SHORT.get(t, t)} {v['coverage_pct']}% ({v['n_covered_by_B']}/{v['n_uncharacterized_by_A']})"
                                   for t, v in M[A][B].items() if t != "pooled")
                    print(f"  {A:14s} -> {B:14s}  pooled {p['coverage_pct']:5.1f}% ({p['n_covered_by_B']}/{p['n_uncharacterized_by_A']})   [{bt}]")
    print(f"\nSaved → {OUT}/stats.json; figures/fig_cross_method_rescue.*")


if __name__ == "__main__":
    main()

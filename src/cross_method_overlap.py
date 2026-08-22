"""
Cross-method semantic overlap (operationalised via redundancy collapse).

Companion to cross_method_rescue.py (complementary coverage). Conditioning on both
methods characterizing a feature, this asks whether they recover the SAME concept.

Per method and feature the framework is applied first: AUROC filter (> 0.50) then
within-method redundancy collapse (connected components at cosine > 0.70, keep the
highest-AUROC representative). This yields each method's *distinct supported
representatives*. Then, for an unordered pair (A, B):

    R(A, B) = P( exists a representative of A and a representative of B with
                 cosine > 0.70 | A and B both characterize )

i.e. among features both methods characterize, the fraction for which at least one
supported characterization from each method is semantically redundant. Collapsing
within-method first removes duplicate proposals as an artificial source of matching
opportunities. R is SYMMETRIC, in contrast to the directional coverage diagnostic.

CAVEAT (parallel to coverage): a method may legitimately retain more distinct
representatives than another (PRISM up to 5, others 1), which still gives more matching
opportunities. Unequal-budget rates remain valid DESCRIPTIVE overlap rates but are NOT
controlled comparisons of method similarity. The cleanest comparisons are the
equal-budget single-description pairs (Transluce ~ max-act; output-centric ~
Neuronpedia), where each method starts with one description, so R is a single-pair
redundancy rate.

Combined interpretation:
  coverage      — do methods characterize different features?
  overlap (R)   — when both characterize, do they recover the same concept?
  corroboration — supported, semantically distinct descriptions from independent
                  methods provide stronger evidence for multiple characterizations;
                  supported + redundant indicate cross-method agreement on a shared one.

Regimes are never mixed; cells populated only where the two methods co-occur.

RUN:  python src/cross_method_overlap.py
Outputs: results/cross_method_overlap/stats.json; figures/fig_cross_method_overlap.{png,pdf,svg}
"""

import sys
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pool_gpt2xl_analyze import filter_pool
import thesis_style as ts

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "cross_method_overlap"
OUT.mkdir(parents=True, exist_ok=True)

OLD_METHODS = ["PRISM", "output-centric", "Neuronpedia", "GPT-explain"]
NEW_METHODS = ["Transluce", "max-act"]
TARGET_SHORT = {"Llama-3.1-8B-Instruct": "Llama", "gemma-scope-2b": "Gemma",
                "gpt2-small-sae": "GPT-2 SAE", "gpt2-xl": "GPT-2 XL"}
SIM_THR = 0.70


def _representatives(a, emb, idx):
    """Distinct supported representatives: AUROC filter + within-method collapse
    (filter_pool 'survived' = highest-AUROC representative per component)."""
    if len(idx) == 0:
        return None
    sub = emb[idx]
    dec, _ = filter_pool(a[idx], sub @ sub.T)
    reps = [idx[k] for k in range(len(idx)) if dec[k] == "survived"]
    return emb[reps]                                        # (n_reps, d), possibly empty


def _add_source(char, desc, emb, method_masks):
    a = desc["auc"].to_numpy()
    for (target, layer, unit), g in desc.groupby(["_target", "layer", "unit"]):
        feat = char.setdefault((target, int(layer), int(unit)), {})
        for method, mask in method_masks.items():
            idx = g.index[mask(g)].to_numpy()
            reps = _representatives(a, emb, idx)
            if reps is not None:
                feat[method] = reps


def gather():
    old, new = {}, {}
    oc = pd.read_csv(RESULTS / "pooled_outputcentric" / "descriptions.csv"); oc["_target"] = oc["model"]
    oce = np.load(RESULTS / "pooled_outputcentric" / "embeddings.npy")
    _add_source(old, oc, oce, {
        "PRISM": lambda g: (g["method"] == "input-centric").values,
        "output-centric": lambda g: (g["method"] == "output-centric").values,
        "Neuronpedia": lambda g: (g["method"] == "neuronpedia").values,
    })
    gx = pd.read_csv(RESULTS / "pooled_gpt2xl" / "descriptions.csv"); gx["_target"] = "gpt2-xl"
    gxe = np.load(RESULTS / "pooled_gpt2xl" / "embeddings.npy")
    _add_source(old, gx, gxe, {
        "PRISM": lambda g: ((g["method"] == "input-centric") & (g["generator"] == "gemini-1.5-pro")).values,
        "GPT-explain": lambda g: (g["method"] == "GPT-explain").values,
    })
    nr = pd.read_csv(RESULTS / "pooled_newregime_llama" / "descriptions.csv"); nr["_target"] = nr["model"]
    nre = np.load(RESULTS / "pooled_newregime_llama" / "embeddings.npy")
    _add_source(new, nr, nre, {
        "Transluce": lambda g: (g["method"] == "Transluce").values,
        "max-act": lambda g: (g["method"] == "max_act").values,
    })
    return old, new


def _redundant(RA, RB):
    """A mixed collapse component exists iff some cross-method representative pair
    exceeds the threshold (internal edges cannot bridge two methods on their own)."""
    return bool((RA @ RB.T > SIM_THR).any())


def overlap(char, A, B):
    per = {}; tot = [0, 0]
    for f, m in char.items():
        if A in m and B in m and len(m[A]) and len(m[B]):     # both characterize
            t = f[0]; d = per.setdefault(t, [0, 0]); d[0] += 1; tot[0] += 1
            if _redundant(m[A], m[B]):
                d[1] += 1; tot[1] += 1
    if tot[0] == 0:
        return None
    def rate(den, num): return {"n_both_characterize": den, "n_overlap": num,
                                "overlap_pct": round(100 * num / den, 1) if den else None}
    out = {t: rate(den, num) for t, (den, num) in per.items()}
    out["pooled"] = rate(tot[0], tot[1])
    return out


def overlap_matrix(char, methods):
    M = {A: {} for A in methods}
    for A, B in combinations(methods, 2):
        r = overlap(char, A, B)
        if r is not None:
            M[A][B] = r; M[B][A] = r          # symmetric
    return M


def _draw(ax, M, methods, title):
    n = len(methods)
    grid = np.full((n, n), np.nan)
    for i, A in enumerate(methods):
        for j, B in enumerate(methods):
            if i != j and B in M.get(A, {}):
                grid[i, j] = M[A][B]["pooled"]["overlap_pct"]
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="Purples", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(methods, fontsize=8, rotation=30, ha="left")
    ax.set_yticklabels(methods, fontsize=8)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    for i, A in enumerate(methods):
        for j, B in enumerate(methods):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=9)
            elif B in M.get(A, {}):
                p = M[A][B]["pooled"]; col = "white" if p["overlap_pct"] > 55 else "#222"
                ax.text(j, i, f"{p['overlap_pct']:.0f}%\n{p['n_overlap']}/{p['n_both_characterize']}",
                        ha="center", va="center", color=col, fontsize=7)
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#ccc", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=22)
    return im


def figure(old_M, new_M):
    ts.apply()
    fig = plt.figure(figsize=(11.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(OLD_METHODS), len(NEW_METHODS)],
                          left=0.10, right=0.92, top=0.78, bottom=0.12, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0]); im = _draw(ax0, old_M, OLD_METHODS, "old regime (gemini-1.5-pro)")
    ax1 = fig.add_subplot(gs[0, 1]); _draw(ax1, new_M, NEW_METHODS, "new regime (gemini-2.0-flash-lite)")
    fig.text(0.5, 0.90, "Cross-method semantic overlap", ha="center", fontsize=12)
    fig.text(0.5, 0.03, "R(A,B) = P(a supported representative of A is redundant (cosine>0.70) with one of B | both characterize).  "
             "Symmetric; cell = R and overlap/both-characterize.\nWithin-method collapse applied first; unequal proposal budgets are not controlled — read the equal-budget single-description pairs.",
             ha="center", fontsize=7.5, color="#555")
    cax = fig.add_axes([0.94, 0.30, 0.012, 0.40])
    fig.colorbar(im, cax=cax).set_label("overlap %", fontsize=8)
    ts.save(fig, "fig_cross_method_overlap")
    plt.close(fig)


def main():
    old, new = gather()
    old_M, new_M = overlap_matrix(old, OLD_METHODS), overlap_matrix(new, NEW_METHODS)
    stats = {
        "definition": "R(A,B) = P(a supported representative of A is redundant (cosine>0.70) with one of B | both characterize); symmetric",
        "pipeline": "AUROC>0.5 filter -> within-method collapse (highest-AUROC representative) -> cross-method semantic overlap",
        "caveat": "unequal-budget rates are valid descriptive overlap rates but not controlled comparisons; cleanest are equal-budget single-description pairs (Transluce~max-act, output-centric~Neuronpedia)",
        "sim_threshold": SIM_THR,
        "old_regime": {"eval": "gemini-1.5-pro", "matrix": old_M},
        "new_regime": {"eval": "gemini-2.0-flash-lite", "matrix": new_M},
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    figure(old_M, new_M)

    for name, M, methods in [("OLD (gemini-1.5-pro)", old_M, OLD_METHODS),
                             ("NEW (gemini-2.0-flash-lite)", new_M, NEW_METHODS)]:
        print(f"\n{name} — R(A,B) semantic overlap when both characterize (symmetric):")
        for A, B in combinations(methods, 2):
            if B in M.get(A, {}):
                p = M[A][B]["pooled"]
                bt = "  ".join(f"{TARGET_SHORT.get(t, t)} {v['overlap_pct']}% ({v['n_overlap']}/{v['n_both_characterize']})"
                               for t, v in M[A][B].items() if t != "pooled")
                print(f"  {A:14s} ~ {B:14s}  pooled {p['overlap_pct']:5.1f}% ({p['n_overlap']}/{p['n_both_characterize']})   [{bt}]")
    print(f"\nSaved → {OUT}/stats.json; figures/fig_cross_method_overlap.*")


if __name__ == "__main__":
    main()

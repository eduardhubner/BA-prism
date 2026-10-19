"""
New-regime cross-method pool (Llama) — Stage 2 (filter + metrics + figure).

Runs the canonical pipeline (AUROC filter > 0.50, cosine-collapse > 0.70) on the
Transluce + max-act pool of 60 Llama features, all evaluated under the post-rebuttal
gemini-2.0-flash-lite regime. This is the new-regime parallel to Panel (c) of Figure 4.

There is no PRISM baseline in this pool, so the two methods are peers: a description's
outcome is filtered (removed by the AUROC gate), redundant (survives and collapses
together with the OTHER method — the two describe the same concept), or distinct
(survives on its own). Redundancy is therefore symmetric between the two methods.

RUN (default venv):
    python src/pool_newregime_llama_analyze.py

Outputs:
    results/pooled_newregime_llama/{profile.csv, stats.json}
    figures/fig_newregime_llama.{png,pdf,svg}
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pool_gpt2xl_analyze import filter_pool, metrics   # canonical filter + metrics
import thesis_style as ts

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "pooled_newregime_llama"
METHODS = ["Transluce", "max_act"]
METHOD_LABEL = {"Transluce": "Transluce", "max_act": "max-act"}
CAT = ["filtered", "retained as distinct", "redundant"]
CAT_COLOR = {"filtered": "#c9c9c9", "retained as distinct": "#4c9f70", "redundant": ts.COLLAPSED_BLUE}


def analyse(desc, emb):
    aurocs = desc["auroc"].to_numpy()
    fate = {m: {"filtered": 0, "converges": 0, "distinct": 0} for m in METHODS}
    both_survive = converged = 0
    rows = []
    for (model, layer, unit), g in desc.groupby(["model", "layer", "unit"]):
        idx = g.index.to_numpy()
        a = aurocs[idx]
        E = emb[idx]
        dec, comps = filter_pool(a, E @ E.T)
        methods = g["method"].to_numpy()
        row = {"layer": int(layer), "unit": int(unit)}
        for k in range(len(idx)):
            if dec[k] == "removed":
                outcome = "filtered"
            else:
                comp = next((c for c in comps if k in c), [k])
                other = any(methods[j] != methods[k] for j in comp)
                outcome = "converges" if other else "distinct"
            fate[methods[k]][outcome] += 1
            row[f"{methods[k]}_outcome"] = outcome
            row[f"{methods[k]}_auroc"] = float(a[k])
        surv = [k for k in range(len(idx)) if dec[k] != "removed"]
        if len(surv) == 2:
            both_survive += 1
            if len(comps) == 1:            # both survivors in one component -> agree
                converged += 1
        rows.append(row)
    agreement = {"n_both_survive": both_survive, "n_converged": converged,
                 "pct_converged_when_both_survive": round(100 * converged / both_survive, 1) if both_survive else None}
    return pd.DataFrame(rows), fate, agreement


def figure(fate, n_per_method):
    ts.apply()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    x = np.arange(len(METHODS)); bottom = np.zeros(len(METHODS))
    key = {"filtered": "filtered", "retained as distinct": "distinct", "redundant": "converges"}
    for cat in CAT:
        vals = np.array([100 * fate[m][key[cat]] / n_per_method[m] for m in METHODS])
        ax.bar(x, vals, bottom=bottom, width=0.6, color=CAT_COLOR[cat], edgecolor="white",
               linewidth=0.8, label=cat)
        bottom += vals
    for xi, m in zip(x, METHODS):
        pct = 100 * fate[m]["converges"] / n_per_method[m]
        ax.text(xi, 102, f"redundant: {pct:.0f}%", ha="center", va="bottom", fontsize=7,
                color=CAT_COLOR["redundant"] if pct > 0 else "#999")
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=9)
    ax.set_ylabel("% of all descriptions")
    ax.set_ylim(0, 108)
    ax.set_title("New-regime pool (Llama, gemini-2.0-flash-lite)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5, frameon=False)
    ts.despine(ax)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.20)
    ts.save(fig, "fig_newregime_llama")
    plt.close(fig)


def main():
    desc = pd.read_csv(OUT / "descriptions.csv")
    emb = np.load(OUT / "embeddings.npy")
    profile, fate, agreement = analyse(desc, emb)
    profile.to_csv(OUT / "profile.csv", index=False)
    n_per_method = {m: int((desc["method"] == m).sum()) for m in METHODS}

    stats = {"params": {"auroc_threshold": 0.50, "sim_threshold": 0.70,
                        "eval_regime": "gemini-2-0-flash-lite",
                        "n_features": int(profile.shape[0])},
             "n_per_method": n_per_method, "fate": fate, "agreement": agreement}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    figure(fate, n_per_method)

    print(f"New-regime Llama pool ({profile.shape[0]} features, gemini-2.0-flash-lite):")
    for m in METHODS:
        f = fate[m]; n = n_per_method[m]
        print(f"  {METHOD_LABEL[m]:10s} filtered {100*f['filtered']/n:4.1f}%  "
              f"distinct {100*f['distinct']/n:4.1f}%  redundant {100*f['converges']/n:4.1f}%  (n={n})")
    a = agreement
    print(f"  agreement: of {a['n_both_survive']} features where both survive, "
          f"{a['n_converged']} converge ({a['pct_converged_when_both_survive']}%)")
    print(f"\nSaved → {OUT}/stats.json, profile.csv; figures/fig_newregime_llama.*")


if __name__ == "__main__":
    main()

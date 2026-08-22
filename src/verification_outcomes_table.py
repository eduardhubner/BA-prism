"""
Table 1 / Table A2 (verification outcomes) -- single source of truth.

Emits the paper's verification-outcomes table as CSV and paste-ready LaTeX, so the
table is regenerable rather than hand-maintained.

DENOMINATORS
------------
The two description-level columns use *different* denominators by design. This is
correct and is stated in the appendix ("redundancy rates are relative to descriptions
entering the redundancy-collapse stage"), but it is easy to misread, so it is spelled
out here and asserted against the published values in `PAPER_REFERENCE` below.

  auroc_filtered_pct
      n_filtered / n_descriptions
      -- all candidate descriptions for that model.

  redundant_pct                                        <- the subtle one
      n_collapsed / n_eligible
      where n_eligible counts post-AUROC-filter descriptions belonging to features
      that have >= 2 of them. A feature left with a single description after
      filtering has nothing to collapse against, so its lone survivor is not
      *eligible* for redundancy collapse and must not sit in the denominator.
      Using all post-filter descriptions instead understates the rate
      (7.6% vs 8.1% overall); using all descriptions understates it badly (3.2%).

Because the denominators differ, the columns do not sum to 100%. The three-way
partition over all descriptions is emitted alongside (`*_pct_of_all`) for anyone who
wants a decomposition:

    n_filtered + n_collapsed + n_surviving = n_descriptions   (692 + 38 + 465 = 1195)

Feature-level columns (n_s = 0 / 1 / >=2) are percentages of features.

Output:
  results/verification_outcomes.csv
  results/verification_outcomes.tex

Run with: python src/verification_outcomes_table.py
"""

import sys
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAIRWISE = ROOT / "results" / "pairwise_similarities_dataset.csv"
OUT_CSV = ROOT / "results" / "verification_outcomes.csv"
OUT_TEX = ROOT / "results" / "verification_outcomes.tex"

SIM_THRESHOLD = 0.70
AUC_THRESHOLD = 0.50
N_DESC = 5

DISPLAY = {
    "gpt2-xl": "GPT-2 XL",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-scope-2b": "Gemma-Scope-2B",
    "gpt2-small-sae": "GPT-2-small SAE",
}
ORDER = ["gpt2-xl", "Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae"]

# Published values (auroc_filtered_pct, redundant_pct) -- regression guard.
PAPER_REFERENCE = {
    "GPT-2 XL": (31.0, 6.1),
    "Llama-3.1-8B": (48.0, 0.7),
    "Gemma-Scope-2B": (69.0, 22.5),
    "GPT-2-small SAE": (84.1, 12.8),
    "All": (57.9, 8.1),
}


def _sim_matrix(row):
    sim = np.eye(N_DESC)
    for i, j in combinations(range(1, N_DESC + 1), 2):
        sim[i - 1, j - 1] = sim[j - 1, i - 1] = row[f"sim_{i}{j}"]
    return sim


def _collapse(indices, sim, aucs):
    """Connected components at sim > threshold; keep highest-AUROC per component."""
    adj = {i: set() for i in indices}
    for i, j in combinations(indices, 2):
        if sim[i, j] > SIM_THRESHOLD:
            adj[i].add(j)
            adj[j].add(i)
    visited, reps = set(), set()
    for start in indices:
        if start in visited:
            continue
        comp, queue = [], [start]
        while queue:
            node = queue.pop()
            if node not in visited:
                visited.add(node)
                comp.append(node)
                queue.extend(adj[node] - visited)
        reps.add(max(comp, key=lambda i: aucs[i]))
    return reps


def _per_feature(row):
    sim = _sim_matrix(row)
    aucs = np.array([row[f"auc_{i}"] for i in range(1, N_DESC + 1)])
    kept = [i for i in range(N_DESC) if aucs[i] > AUC_THRESHOLD]   # stage 1
    surv = _collapse(kept, sim, aucs)                              # stage 2
    return {
        "n_descriptions": N_DESC,
        "n_filtered": N_DESC - len(kept),
        "n_reliable": len(kept),
        # a lone post-filter description cannot collapse with anything
        "n_eligible": len(kept) if len(kept) >= 2 else 0,
        "n_collapsed": len(kept) - len(surv),
        "n_surviving": len(surv),
    }


def _row(label, g):
    n_feat = len(g)
    nd, nf = int(g.n_descriptions.sum()), int(g.n_filtered.sum())
    nc, ne = int(g.n_collapsed.sum()), int(g.n_eligible.sum())
    ns_desc = int(g.n_surviving.sum())
    return {
        "model": label,
        "n_features": n_feat,
        "n_descriptions": nd,
        "n_auroc_filtered": nf,
        "n_eligible_for_collapse": ne,
        "n_redundant": nc,
        "n_surviving_desc": ns_desc,
        "auroc_filtered_pct": round(100 * nf / nd, 1),          # of all descriptions
        "redundant_pct": round(100 * nc / ne, 1) if ne else None,  # of eligible
        # three-way partition over all descriptions, for decomposition
        "auroc_filtered_pct_of_all": round(100 * nf / nd, 1),
        "redundant_pct_of_all": round(100 * nc / nd, 1),
        "surviving_pct_of_all": round(100 * ns_desc / nd, 1),
        "ns_eq_0_pct": round(100 * (g.n_surviving == 0).sum() / n_feat, 1),
        "ns_eq_1_pct": round(100 * (g.n_surviving == 1).sum() / n_feat, 1),
        "ns_ge_2_pct": round(100 * (g.n_surviving >= 2).sum() / n_feat, 1),
    }


def main():
    src = pd.read_csv(PAIRWISE)
    per = pd.DataFrame([{**{"model": r["model"]}, **_per_feature(r)} for _, r in src.iterrows()])

    rows = [_row(DISPLAY[m], per[per.model == m]) for m in ORDER if (per.model == m).any()]
    rows.append(_row("All", per))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # partition identity holds by construction
    for r in rows:
        tot = r["n_auroc_filtered"] + r["n_redundant"] + r["n_surviving_desc"]
        assert tot == r["n_descriptions"], f"{r['model']}: {tot} != {r['n_descriptions']}"

    # regression guard against the published table
    drift = [
        (r["model"], (r["auroc_filtered_pct"], r["redundant_pct"]), PAPER_REFERENCE[r["model"]])
        for r in rows
        if r["model"] in PAPER_REFERENCE
        and (r["auroc_filtered_pct"], r["redundant_pct"]) != PAPER_REFERENCE[r["model"]]
    ]

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{",
        r"Full verification outcomes. AUROC-filtered rates are relative to all generated",
        r"descriptions; redundancy rates are relative to descriptions entering the",
        r"redundancy-collapse stage (post-filter descriptions in features retaining at",
        r"least two, since a lone survivor cannot collapse). The two columns therefore",
        r"use different denominators and do not sum.",
        r"}",
        r"\label{tab:verification-outcomes-full}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & AUROC-filtered & Redundant & \(n_s=0\) & \(n_s=1\) & \(n_s\geq2\) \\",
        r"\midrule",
    ]
    for r in rows:
        if r["model"] == "All":
            lines.append(r"\midrule")
        lines.append(
            f"{r['model']:<14s} & {r['auroc_filtered_pct']}\\% & {r['redundant_pct']}\\% "
            f"& {r['ns_eq_0_pct']}\\% & {r['ns_eq_1_pct']}\\% & {r['ns_ge_2_pct']}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    cols = ["model", "n_descriptions", "n_auroc_filtered", "n_eligible_for_collapse",
            "n_redundant", "auroc_filtered_pct", "redundant_pct", "redundant_pct_of_all"]
    print(out[cols].to_string(index=False))
    a = rows[-1]
    print(f"\nPartition over all descriptions: {a['n_auroc_filtered']} filtered + "
          f"{a['n_redundant']} redundant + {a['n_surviving_desc']} surviving = {a['n_descriptions']}")
    print(f"Redundancy denominator (All): {a['n_eligible_for_collapse']} eligible, "
          f"not {a['n_descriptions'] - a['n_auroc_filtered']} post-filter "
          f"({a['n_descriptions'] - a['n_auroc_filtered'] - a['n_eligible_for_collapse']} "
          f"lone survivors excluded)")
    print(f"\nSaved {OUT_CSV}\nSaved {OUT_TEX}")

    if drift:
        print("\n!! DRIFT vs published table:")
        for m, got, exp in drift:
            print(f"   {m}: got {got}, paper {exp}")
        sys.exit(1)
    print("\nRegression guard: matches the published table exactly on all 5 rows.")


if __name__ == "__main__":
    main()

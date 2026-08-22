"""
Table 1 (verification outcomes) -- single source of truth.

Emits the paper's verification-outcomes table from results/reliability_profile.csv,
as CSV and as paste-ready LaTeX. Written because the table's redundancy column could
not be reproduced from the committed artifacts under any natural denominator, which
suggested it was carried over from an earlier pipeline state (pre filter/collapse
reordering). This script pins the definitions so the table is regenerable.

DENOMINATOR CONVENTION (the fix)
--------------------------------
Both description-level columns use the SAME denominator: all N candidate
descriptions for that model. This makes the row a partition:

    AUROC-filtered  +  redundant  +  surviving  =  100%   (692 + 38 + 465 = 1195)

The previous table mixed denominators -- AUROC-filtered over all descriptions but
redundancy over some post-filter base -- which is why "38 descriptions" and "8.1%"
disagreed (38/1195 = 3.2%).

The conditional rate (redundant among descriptions that survived the AUROC filter)
is a different and also-interesting quantity, so it is emitted as an extra column
`redundant_pct_of_reliable` for use in prose. Do NOT put it in the same table column
as the filtered rate without saying so explicitly.

Feature-level columns (n_s = 0 / 1 / >=2) are percentages of features, not
descriptions -- unchanged from the current table, and already verified exact.

Output:
  results/verification_outcomes.csv
  results/verification_outcomes.tex

Run with: python src/verification_outcomes_table.py
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROFILE = ROOT / "results" / "reliability_profile.csv"
OUT_CSV = ROOT / "results" / "verification_outcomes.csv"
OUT_TEX = ROOT / "results" / "verification_outcomes.tex"

DISPLAY = {
    "gpt2-xl": "GPT-2 XL",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-scope-2b": "Gemma-Scope-2B",
    "gpt2-small-sae": "GPT-2-small SAE",
}
ORDER = ["gpt2-xl", "Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae"]


def _row(label, g):
    n_feat = len(g)
    nd = int(g.n_descriptions.sum())
    nf = int(g.n_filtered.sum())
    nc = int(g.n_collapsed.sum())
    ns_desc = int(g.n_surviving.sum())
    reliable = nd - nf                      # survived the AUROC filter
    return {
        "model": label,
        "n_features": n_feat,
        "n_descriptions": nd,
        "n_auroc_filtered": nf,
        "n_redundant": nc,
        "n_surviving_desc": ns_desc,
        # description-level, common denominator = all descriptions
        "auroc_filtered_pct": round(100 * nf / nd, 1),
        "redundant_pct": round(100 * nc / nd, 1),
        "surviving_pct": round(100 * ns_desc / nd, 1),
        # conditional rate, for prose only
        "redundant_pct_of_reliable": round(100 * nc / reliable, 1) if reliable else None,
        # feature-level
        "ns_eq_0_pct": round(100 * (g.n_surviving == 0).sum() / n_feat, 1),
        "ns_eq_1_pct": round(100 * (g.n_surviving == 1).sum() / n_feat, 1),
        "ns_ge_2_pct": round(100 * (g.n_surviving >= 2).sum() / n_feat, 1),
    }


def main():
    d = pd.read_csv(PROFILE)
    rows = [_row(DISPLAY[m], d[d.model == m]) for m in ORDER if (d.model == m).any()]
    rows.append(_row("All", d))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # partition sanity check -- must hold by construction
    for r in rows:
        tot = r["n_auroc_filtered"] + r["n_redundant"] + r["n_surviving_desc"]
        assert tot == r["n_descriptions"], f"{r['model']}: {tot} != {r['n_descriptions']}"

    lines = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Verification outcomes across models. AUROC-filtered, redundant, and",
        r"    surviving are description-level percentages of all candidate descriptions and",
        r"    partition to 100\%; survivor categories are feature-level percentages.}",
        r"    \label{tab:verification-outcomes}",
        r"    \begin{tabular}{lcccccc}", r"        \toprule",
        r"        \textbf{Model (\(n\))}",
        r"        & \textbf{AUROC-filtered} & \textbf{Redundant} & \textbf{Surviving}",
        r"        & \(\mathbf{n_s=0}\) & \(\mathbf{n_s=1}\) & \(\mathbf{n_s\geq2}\) \\",
        r"        \midrule",
    ]
    for i, r in enumerate(rows):
        if r["model"] == "All":
            lines.append(r"        \midrule")
        lines.append(
            f"        {r['model']} (\\({r['n_features']}\\))\n"
            f"        & \\({r['auroc_filtered_pct']}\\%\\) & \\({r['redundant_pct']}\\%\\)"
            f" & \\({r['surviving_pct']}\\%\\)\n"
            f"        & \\({r['ns_eq_0_pct']}\\%\\) & \\({r['ns_eq_1_pct']}\\%\\)"
            f" & \\({r['ns_ge_2_pct']}\\%\\) \\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print(out.to_string(index=False))
    print(f"\nSaved {OUT_CSV}")
    print(f"Saved {OUT_TEX}")
    a = rows[-1]
    print(f"\nPartition (All): {a['n_auroc_filtered']} filtered + {a['n_redundant']} redundant "
          f"+ {a['n_surviving_desc']} surviving = {a['n_descriptions']}")
    print(f"Prose-only conditional rate (All): {a['redundant_pct_of_reliable']}% of the "
          f"{a['n_descriptions'] - a['n_auroc_filtered']} descriptions that passed the AUROC filter")


if __name__ == "__main__":
    main()

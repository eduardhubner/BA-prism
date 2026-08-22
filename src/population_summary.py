"""
Population summary table: per-model breakdown of the main descriptive metrics.

Aggregates filtered_evaluation.csv (the two-stage filter's per-neuron output,
see filtered_evaluation.py) by model, reporting the headline population-level
numbers: survivor-count distribution, mean n_s, mean evidence quality Q, and mean
weighted mean similarity (sigma-tilde) restricted to neurons with n_surviving >= 2.

Q = 2*mean(survivor AUROC) - 1 is the reported quality metric (paper, Method).
It is defined for n_s >= 1 and undefined at n_s = 0. `confidence_C` is retained as
`mean_confidence_C_legacy` only because the k=10 ablation's dilution result refers
to it; it is NOT a reported metric. The two differ substantively -- C averages over
ALL candidates so it is dragged down by failed ones, which reorders the models
(gemma and Llama swap rank).

Note on the restriction: sigma-tilde is *defined* for n_s >= 1 (it is 1.0 by
convention at n_s = 1, see filtered_evaluation.py). Restricting to n_s >= 2 is a
conditioning choice -- it is where the value is informative rather than definitional --
not the domain of the metric. Hence the column name `_ns_ge_2`, not `_where_defined`.

Outputs:
  results/population_summary_by_model.csv

Run with: python src/population_summary.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FILTERED_PATH = ROOT / "results" / "filtered_evaluation.csv"
OUT_CSV = ROOT / "results" / "population_summary_by_model.csv"


def main():
    df = pd.read_csv(FILTERED_PATH)

    rows = []
    for model, grp in df.groupby("model"):
        n = len(grp)
        n_uncharacterized = int(grp["uncharacterized"].sum())
        n_single = int((grp["n_surviving"] == 1).sum())
        n_multi = int((grp["n_surviving"] > 1).sum())
        sigma_tilde_ns_ge_2 = grp[grp["n_surviving"] > 1]["weighted_mean_similarity"]
        q_defined = grp[grp["n_surviving"] >= 1]["evidence_quality"]

        rows.append({
            "model": model,
            "n_neurons": n,
            "mean_n_surviving": round(float(grp["n_surviving"].mean()), 4),
            "n_uncharacterized": n_uncharacterized,
            "pct_uncharacterized": round(100 * n_uncharacterized / n, 1),
            "n_single_survivor": n_single,
            "pct_single_survivor": round(100 * n_single / n, 1),
            "n_multi_survivor": n_multi,
            "pct_multi_survivor": round(100 * n_multi / n, 1),
            "mean_evidence_quality_Q": round(float(q_defined.mean()), 4) if len(q_defined) else None,
            "n_Q_defined": int(len(q_defined)),
            "mean_confidence_C_legacy": round(float(grp["confidence_C"].mean()), 4),
            "mean_sigma_tilde_ns_ge_2": round(float(sigma_tilde_ns_ge_2.mean()), 4) if len(sigma_tilde_ns_ge_2) else None,
            "n_sigma_tilde_ns_ge_2": int(len(sigma_tilde_ns_ge_2)),
        })

    overall = {
        "model": "ALL",
        "n_neurons": len(df),
        "mean_n_surviving": round(float(df["n_surviving"].mean()), 4),
        "n_uncharacterized": int(df["uncharacterized"].sum()),
        "pct_uncharacterized": round(100 * df["uncharacterized"].sum() / len(df), 1),
        "n_single_survivor": int((df["n_surviving"] == 1).sum()),
        "pct_single_survivor": round(100 * (df["n_surviving"] == 1).sum() / len(df), 1),
        "n_multi_survivor": int((df["n_surviving"] > 1).sum()),
        "pct_multi_survivor": round(100 * (df["n_surviving"] > 1).sum() / len(df), 1),
        "mean_evidence_quality_Q": round(float(df[df["n_surviving"] >= 1]["evidence_quality"].mean()), 4),
        "n_Q_defined": int((df["n_surviving"] >= 1).sum()),
        "mean_confidence_C_legacy": round(float(df["confidence_C"].mean()), 4),
        "mean_sigma_tilde_ns_ge_2": round(float(df[df["n_surviving"] > 1]["weighted_mean_similarity"].mean()), 4),
        "n_sigma_tilde_ns_ge_2": int((df["n_surviving"] > 1).sum()),
    }
    rows.append(overall)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved population summary to {OUT_CSV}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

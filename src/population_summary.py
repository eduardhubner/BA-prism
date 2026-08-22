"""
Population summary table: per-model breakdown of the main descriptive metrics.

Aggregates filtered_evaluation.csv (the two-stage filter's per-neuron output,
including confidence_C -- see filtered_evaluation.py) by model, reporting the
headline population-level numbers: survivor-count distribution, mean n_s,
mean confidence C, and mean weighted mean similarity (sigma-tilde) among
neurons where it's defined (n_surviving > 1).

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
        sigma_tilde_defined = grp[grp["n_surviving"] > 1]["weighted_mean_similarity"]

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
            "mean_confidence_C": round(float(grp["confidence_C"].mean()), 4),
            "mean_sigma_tilde_where_defined": round(float(sigma_tilde_defined.mean()), 4) if len(sigma_tilde_defined) else None,
            "n_sigma_tilde_defined": int(len(sigma_tilde_defined)),
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
        "mean_confidence_C": round(float(df["confidence_C"].mean()), 4),
        "mean_sigma_tilde_where_defined": round(float(df[df["n_surviving"] > 1]["weighted_mean_similarity"].mean()), 4),
        "n_sigma_tilde_defined": int((df["n_surviving"] > 1).sum()),
    }
    rows.append(overall)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved population summary to {OUT_CSV}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

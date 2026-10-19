"""
Embedding-model robustness matrix.

src/embedding_model_comparison.py computes, for each of 4 alternative
embedding models, how well that model's pairwise-similarity rankings agree
with the baseline (gte-Qwen2-1.5B-instruct, the model used throughout the
rest of the thesis pipeline) -- but only baseline-vs-alternative, never
alternative-vs-alternative. This script aggregates those per-model CSVs
(results/embedding_comparison/*_within.csv and cross_neuron_pairs.csv) into
the full 5x5 pairwise Spearman correlation matrix across all 5 models:
gte-Qwen2-1.5B (baseline), MiniLM-L6-v2, Nomic-Embed-v1.5, Qwen3-Embedding-4B,
Qwen3-Embedding-8B.

Two pair sets, both already computed and saved by embedding_model_comparison.py:
  - Within-neuron: all 10 pairwise combinations of the 5 descriptions for
    each of 239 neurons (2390 pairs). Covers the 0.3-1.0 similarity range
    typical of same-neuron descriptions.
  - Cross-neuron: 500 pairs sampled from different neurons (seed=42), giving
    a broader range down to near-zero similarity.

Outputs:
  results/embedding_robustness_matrix.json   -- both 5x5 Spearman matrices
                                                 + per-model agreement stats
  results/embedding_robustness_within.csv    -- merged wide within-neuron
                                                 table (one column per model)

Run with: python src/embedding_robustness_matrix.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
COMPARISON_DIR = ROOT / "results" / "embedding_comparison"
CROSS_PATH = COMPARISON_DIR / "cross_neuron_pairs.csv"
OUT_JSON = ROOT / "results" / "embedding_robustness_matrix.json"
OUT_WITHIN_CSV = ROOT / "results" / "embedding_robustness_within.csv"

MODEL_SLUGS = {
    "text-embedding-all-minilm-l6-v2-embedding": "MiniLM-L6-v2",
    "text-embedding-nomic-embed-text-v1.5": "Nomic-Embed-v1.5",
    "text-embedding-qwen3-embedding-4b": "Qwen3-Embedding-4B",
    "text-embedding-qwen3-embedding-8b": "Qwen3-Embedding-8B",
}
BASELINE_NAME = "gte-Qwen2-1.5B (baseline)"
MODEL_ORDER = [BASELINE_NAME] + list(MODEL_SLUGS.values())


def build_within_matrix():
    merged = None
    for slug, name in MODEL_SLUGS.items():
        path = COMPARISON_DIR / f"{slug}_within.csv"
        df = pd.read_csv(path)[["neuron_id", "pair", "baseline", "new"]]
        df = df.rename(columns={"new": name})
        if merged is None:
            merged = df.rename(columns={"baseline": BASELINE_NAME})
        else:
            # sanity check: baseline column should match across files
            check = merged.merge(df[["neuron_id", "pair", "baseline"]], on=["neuron_id", "pair"])
            max_baseline_diff = (check[BASELINE_NAME] - check["baseline"]).abs().max()
            merged = merged.merge(df[["neuron_id", "pair", name]], on=["neuron_id", "pair"])
            if max_baseline_diff > 1e-6:
                print(f"WARNING: baseline mismatch vs {name}'s file, max diff={max_baseline_diff}")

    merged.to_csv(OUT_WITHIN_CSV, index=False)
    print(f"Saved merged within-neuron table to {OUT_WITHIN_CSV}  (n={len(merged)})")
    return merged


def correlation_matrix(df, columns):
    n = len(columns)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearmanr(df[columns[i]], df[columns[j]]).statistic
            mat[i, j] = mat[j, i] = r
    return mat


def matrix_to_dict(mat, labels):
    return {labels[i]: {labels[j]: round(float(mat[i, j]), 4) for j in range(len(labels))}
            for i in range(len(labels))}


def main():
    within = build_within_matrix()
    within_mat = correlation_matrix(within, MODEL_ORDER)

    cross = pd.read_csv(CROSS_PATH)
    cross_cols_map = {"baseline_sim": BASELINE_NAME}
    for slug, name in MODEL_SLUGS.items():
        cross_cols_map[f"sim_{slug}"] = name
    cross = cross.rename(columns=cross_cols_map).dropna(subset=MODEL_ORDER)
    cross_mat = correlation_matrix(cross, MODEL_ORDER)

    print("\n=== Within-neuron 5x5 Spearman matrix (n={}) ===".format(len(within)))
    print(pd.DataFrame(within_mat, index=MODEL_ORDER, columns=MODEL_ORDER).round(3).to_string())

    print("\n=== Cross-neuron 5x5 Spearman matrix (n={}) ===".format(len(cross)))
    print(pd.DataFrame(cross_mat, index=MODEL_ORDER, columns=MODEL_ORDER).round(3).to_string())

    # Agreement stats: baseline vs each alternative, off-diagonal-only summary
    baseline_idx = MODEL_ORDER.index(BASELINE_NAME)
    within_vs_baseline = {MODEL_ORDER[j]: round(float(within_mat[baseline_idx, j]), 4)
                           for j in range(len(MODEL_ORDER)) if j != baseline_idx}
    cross_vs_baseline = {MODEL_ORDER[j]: round(float(cross_mat[baseline_idx, j]), 4)
                          for j in range(len(MODEL_ORDER)) if j != baseline_idx}

    summary = {
        "models": MODEL_ORDER,
        "within_neuron": {
            "n_pairs": int(len(within)),
            "n_neurons": int(within["neuron_id"].nunique()),
            "spearman_matrix": matrix_to_dict(within_mat, MODEL_ORDER),
            "vs_baseline": within_vs_baseline,
        },
        "cross_neuron": {
            "n_pairs": int(len(cross)),
            "spearman_matrix": matrix_to_dict(cross_mat, MODEL_ORDER),
            "vs_baseline": cross_vs_baseline,
        },
        "note": "within-neuron pairs (sim 0.3-1.0 range, same-neuron descriptions) and "
                "cross-neuron pairs (broader range incl. near-zero) are reported separately "
                "since embedding-model agreement can differ across similarity ranges.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full matrices + summary to {OUT_JSON}")


if __name__ == "__main__":
    main()

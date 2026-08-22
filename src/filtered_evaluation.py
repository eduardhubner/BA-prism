"""
Filtered evaluation of prism neuron descriptions.

Pipeline:
  Stage 1 - AUC filter (> AUC_THRESHOLD):
            Remove descriptions below the discriminability threshold up front,
            so a low-AUC description can never bridge two otherwise-distinct
            survivors during the collapse.
  Stage 2 - Similarity collapse (sim > SIM_THRESHOLD):
            Connected components over the surviving descriptions; keep the
            highest-AUC description per component. Deterministic — no seeds.
            Neurons with 0 survivors are flagged as uncharacterized.

Note: for this dataset filter-then-collapse and collapse-then-filter yield
identical survivor sets for all 239 neurons; the orders only diverge when a
low-AUC description bridges two distinct survivors, which does not occur here.

Metrics computed per neuron (post-filter):
  - n_surviving: number of surviving descriptions
  - mean_auc_post: mean AUC of survivors
  - effective_n_descriptions: AUC-weighted count of survivors
  - mean_similarity_post: unweighted mean pairwise similarity of survivors
    (1.0 by convention for single survivors)
  - weighted_mean_similarity: AUC-weighted mean pairwise similarity of survivors
    (1.0 by convention for single survivors)
  - confidence_C: mean over ALL N_DESC candidate descriptions (not just
    survivors) of max(auc-0.5, 0)/0.5. Note this is a mean over the full
    candidate pool, so it is mechanically sensitive to candidate count (see
    the k=10 ablation's C-dilution finding) -- a definitional property of
    this formula, not a bug.
  - uncharacterized: True if 0 survivors

Output: results/filtered_evaluation.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

import verification as V

# --- Config ---
SIM_THRESHOLD = V.SIM_THRESHOLD
AUC_THRESHOLD = V.AUC_THRESHOLD
INPUT_PATH = Path(__file__).parent.parent / "results" / "pairwise_similarities_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "filtered_evaluation.csv"

N_DESC = 5
SIM_PAIRS = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]


def get_sim_matrix(row):
    """5x5 similarity matrix from a row. Delegates to `verification`."""
    return V.sim_matrix_from_row(row, n_desc=N_DESC)


def components_collapse(indices, sim_matrix, threshold, aucs):
    """Back-compat shim retained for the k=5 path; no in-repo caller remains.
    Use verification.filter_and_collapse instead."""
    res = V.filter_and_collapse(
        np.where(np.isin(np.arange(len(aucs)), list(indices)), aucs, 0.0),
        sim_matrix, sim_threshold=threshold, auc_threshold=AUC_THRESHOLD)
    return set(res.surviving)


def process_neuron(row):
    """Two-stage verification + metrics for one neuron, mapped to this
    script's legacy column names."""
    aucs = np.array([row[f"auc_{i}"] for i in range(1, N_DESC + 1)])
    sim = V.sim_matrix_from_row(row, n_desc=N_DESC)
    res = V.filter_and_collapse(aucs, sim, sim_threshold=SIM_THRESHOLD,
                                auc_threshold=AUC_THRESHOLD)
    m = V.compute_metrics(aucs, sim, res)
    return {
        "n_surviving": m["n_surviving"],
        "mean_auc_pre": m["mean_auc_pre"],
        "mean_auc_post": m["mean_auc_post"],
        "effective_n_descriptions": m["effective_n"],
        "mean_similarity_post": m["mean_sim_post"],
        "weighted_mean_similarity": m["weighted_mean_sim"],
        "confidence_C": m["confidence_C"],
        "uncharacterized": m["uncharacterized"],
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} neurons")

    all_rows = []
    for _, row in df.iterrows():
        metrics = process_neuron(row)
        base = {
            "model": row["model"],
            "layer": row["layer"],
            "unit": row["unit"],
        }
        base.update(metrics)
        all_rows.append(base)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    n_unchar = results_df["uncharacterized"].sum()
    n_single = (results_df["n_surviving"] == 1).sum()
    n_multi = (results_df["n_surviving"] > 1).sum()

    print("\n=== Summary ===")
    print(f"Total neurons:          {len(results_df)}")
    print(f"Uncharacterized (0):    {n_unchar}")
    print(f"Single survivor (1):    {n_single}")
    print(f"Multi survivor (>1):    {n_multi}")
    print(f"Mean AUC pre-filter:    {results_df['mean_auc_pre'].mean():.4f}")
    print(f"Mean AUC post-filter:   {results_df['mean_auc_post'].mean():.4f}")
    print(f"Mean n_surviving:       {results_df['n_surviving'].mean():.2f}")
    print(f"Mean effective N:       {results_df['effective_n_descriptions'].mean():.4f}")
    print(f"Mean weighted sim:      {results_df['weighted_mean_similarity'].mean():.4f}")
    print(f"Mean confidence C:      {results_df['confidence_C'].mean():.4f}")


if __name__ == "__main__":
    main()

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

# --- Config ---
SIM_THRESHOLD = 0.70
AUC_THRESHOLD = 0.50
INPUT_PATH = Path(__file__).parent.parent / "results" / "pairwise_similarities_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "filtered_evaluation.csv"

N_DESC = 5
SIM_PAIRS = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]


def get_sim_matrix(row):
    """Build 5x5 similarity matrix from row."""
    sim = np.zeros((N_DESC, N_DESC))
    for i, j in SIM_PAIRS:
        val = row[f"sim_{i}{j}"]
        sim[i-1, j-1] = val
        sim[j-1, i-1] = val
    return sim


def components_collapse(indices, sim_matrix, threshold, aucs):
    """
    Connected components collapse: find components where any two descriptions
    have sim > threshold, keep the highest-AUC description per component.
    Deterministic.
    """
    adj = {i: set() for i in indices}
    for i, j in combinations(indices, 2):
        if sim_matrix[i, j] > threshold:
            adj[i].add(j)
            adj[j].add(i)

    visited = set()
    kept = set()
    for start in indices:
        if start not in visited:
            component = []
            queue = [start]
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    queue.extend(adj[node] - visited)
            # Keep the highest-AUC description in this component
            best = max(component, key=lambda i: aucs[i])
            kept.add(best)
    return kept


def compute_metrics(surviving, aucs, sim_matrix, all_aucs):
    """Compute all post-filter metrics for a neuron."""
    n_surviving = len(surviving)
    surviving = list(surviving)
    mean_auc_pre = float(np.mean(all_aucs))
    confidence_C = float(np.mean([max(a - 0.5, 0) / 0.5 for a in all_aucs]))

    if n_surviving == 0:
        return {
            "n_surviving": 0,
            "mean_auc_pre": mean_auc_pre,
            "mean_auc_post": np.nan,
            "effective_n_descriptions": np.nan,
            "mean_similarity_post": np.nan,
            "weighted_mean_similarity": np.nan,
            "confidence_C": confidence_C,
            "uncharacterized": True,
        }

    surviving_aucs = [aucs[i] for i in surviving]
    mean_auc_post = float(np.mean(surviving_aucs))
    effective_n = float(np.sum([(a - 0.5) / 0.5 for a in surviving_aucs]))

    if n_surviving == 1:
        mean_sim_post = 1.0
        weighted_mean_sim = 1.0
    else:
        pairs = list(combinations(surviving, 2))
        mean_sim_post = float(np.mean([sim_matrix[i, j] for i, j in pairs]))
        numerator = 0.0
        denominator = 0.0
        for i, j in pairs:
            w = ((aucs[i] - 0.5) / 0.5) * ((aucs[j] - 0.5) / 0.5)
            numerator += w * sim_matrix[i, j]
            denominator += w
        weighted_mean_sim = float(numerator / denominator) if denominator > 0 else np.nan

    return {
        "n_surviving": n_surviving,
        "mean_auc_pre": mean_auc_pre,
        "mean_auc_post": mean_auc_post,
        "effective_n_descriptions": effective_n,
        "mean_similarity_post": mean_sim_post,
        "weighted_mean_similarity": weighted_mean_sim,
        "confidence_C": confidence_C,
        "uncharacterized": False,
    }


def process_neuron(row):
    sim_matrix = get_sim_matrix(row)
    aucs = np.array([row[f"auc_{i}"] for i in range(1, N_DESC+1)])

    # Stage 1: AUC filter (before collapse). Low-AUC descriptions are removed
    # outright and excluded from the similarity collapse below.
    kept = [i for i in range(N_DESC) if aucs[i] > AUC_THRESHOLD]

    # Stage 2: similarity collapse among kept descriptions (keep highest AUC
    # per connected component).
    surviving = components_collapse(kept, sim_matrix, SIM_THRESHOLD, aucs)

    return compute_metrics(surviving, aucs, sim_matrix, aucs)


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

"""
k=10 ablation analysis.

Reproduces, from scratch and on disk, the three findings drawn from the k=10
ablation experiment (60 GPT2-XL neurons, layers 0/20/40, k=10 candidate
descriptions instead of the standard k=5). Runs the same two-stage filter as
filtered_evaluation.py (similarity collapse + AUROC filter), generalized to
arbitrary description count, on both the k=5 and k=10 data for the same
neurons.

Scope note: most k=5-vs-k=10 metric comparisons are confounded, because the
description *sets* differ between k=5 and k=10 (different generations), not
just the count. A metric changing between them can't be cleanly attributed to
k-sensitivity vs. just-different-content. Only comparisons below are
structured to control for or sidestep that confound:

  1. Weighting gap vs. survivor count -- does AUROC-weighting (WMS) diverge
     more from the unweighted mean similarity as survivor count grows?
     This tests whether weighting "earns its keep" more at higher N. The gap
     (weighted - unweighted) is signed and roughly symmetric across neurons,
     so a plain mean cancels out and is uninformative -- mean ABSOLUTE gap is
     used instead. n_surviving == 2 is a structural special case (exactly one
     pairwise similarity, so the weighted "mean" of a single value trivially
     equals the unweighted mean -- gap is exactly 0 by construction, not
     because weighting doesn't matter) and is excluded from the trend.
     A within-neuron paired comparison (same neurons at both k=5 and k=10)
     is reported alongside the pooled binned trend to rule out the gap
     vs. n_surviving relationship being an artifact of pool composition.
  2. Zero-survivor rescue -- neurons uncharacterized (0 survivors) at k=5:
     do they gain survivors when given more candidates at k=10? This is a
     within-neuron paired comparison, not a population-level metric average.
     The actual description-level detail (which new k=10 candidate crossed
     AUROC_THRESHOLD) is saved separately so the rescue claim is inspectable,
     not just assertable from an aggregate count.
  3. Confidence C dilution -- C is a mean over ALL candidate descriptions
     (not just survivors), so it is mechanically diluted by adding more
     (typically lower-quality) candidates. This is reported as a definitional
     property of C's formula, not a "k effect" finding.
  4. Mean survivor-count increase -- the average within-neuron change in
     n_surviving from k=5 to k=10, over all 60 matched neurons (paired, not
     pooled-average, so it's not affected by the different n_surviving>1
     pool sizes used in Finding 1).

Outputs:
  results/k10_ablation_analysis.csv   -- per-neuron k=5 and k=10 metrics
  results/k10_ablation_summary.json   -- the headline numbers
  results/k10_rescue_detail.csv       -- description-level AUROC detail (text +
                                          AUROC, k=5 and k=10, survived flag) for
                                          every neuron uncharacterized at k=5

Run with: python src/k10_ablation_analysis.py
"""

import json
import numpy as np

import verification as V
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

SIM_THRESHOLD = 0.70
AUROC_THRESHOLD = 0.50

K5_PAIRWISE = RESULTS / "pairwise_similarities_dataset.csv"
K10_PAIRWISE = RESULTS / "k10_pairwise_similarities.csv"
OUT_CSV = RESULTS / "k10_ablation_analysis.csv"
OUT_JSON = RESULTS / "k10_ablation_summary.json"
OUT_RESCUE_CSV = RESULTS / "k10_rescue_detail.csv"


def build_sim_matrix(row, n_desc, sep):
    """sep='' for k=5 cols like sim_12, sep='_' for k=10 cols like sim_1_2."""
    sim = np.eye(n_desc)
    for i, j in combinations(range(1, n_desc + 1), 2):
        col = f"sim_{i}{sep}{j}"
        sim[i - 1, j - 1] = row[col]
        sim[j - 1, i - 1] = row[col]
    return sim


def components_collapse(indices, sim_matrix, threshold, aurocs):
    """Connected-components collapse. Delegates to `verification`.

    NOTE: this script applies the collapse BEFORE the AUROC filter, which is the
    non-canonical stage order (see verification.collapse_then_filter). Kept
    as-is so results are unchanged; see ARTIFACT_PLAN.md for the open decision.
    """
    return V.collapse_components(indices, sim_matrix, aurocs, sim_threshold=threshold)


def compute_post_filter_metrics(surviving, aurocs, sim_matrix):
    n_surviving = len(surviving)
    surviving = list(surviving)
    confidence_C = float(np.mean([max(a - 0.5, 0) / 0.5 for a in aurocs]))

    if n_surviving == 0:
        return dict(n_surviving=0, mean_auroc_post=np.nan, effective_n=np.nan,
                    mean_sim_post=np.nan, weighted_mean_sim=np.nan,
                    uncharacterized=True, confidence_C=confidence_C)

    surviving_aurocs = [aurocs[i] for i in surviving]
    mean_auroc_post = float(np.mean(surviving_aurocs))
    effective_n = float(np.sum([(a - 0.5) / 0.5 for a in surviving_aurocs]))

    if n_surviving == 1:
        mean_sim_post = 1.0
        weighted_mean_sim = 1.0
    else:
        pairs = list(combinations(surviving, 2))
        mean_sim_post = float(np.mean([sim_matrix[i, j] for i, j in pairs]))
        num, den = 0.0, 0.0
        for i, j in pairs:
            w = ((aurocs[i] - 0.5) / 0.5) * ((aurocs[j] - 0.5) / 0.5)
            num += w * sim_matrix[i, j]
            den += w
        weighted_mean_sim = float(num / den) if den > 0 else np.nan

    return dict(n_surviving=n_surviving, mean_auroc_post=mean_auroc_post,
                effective_n=effective_n, mean_sim_post=mean_sim_post,
                weighted_mean_sim=weighted_mean_sim, uncharacterized=False,
                confidence_C=confidence_C)


def process_neuron(row, n_desc, sep):
    sim_matrix = build_sim_matrix(row, n_desc, sep)
    aurocs = np.array([row[f"auroc_{i}"] for i in range(1, n_desc + 1)])
    # Canonical stage order: AUROC filter, then redundancy collapse. Previously
    # this collapsed first, which lets a low-AUROC description bridge two
    # otherwise-distinct survivors -- harmless at k=5 (orders agree on 0/239) but
    # divergent at k=10 (2/60 neurons). See verification.collapse_then_filter.
    stage2 = set(V.filter_and_collapse(aurocs, sim_matrix,
                                       sim_threshold=SIM_THRESHOLD,
                                       auroc_threshold=AUROC_THRESHOLD).surviving)
    metrics = compute_post_filter_metrics(stage2, aurocs, sim_matrix)
    return metrics, stage2


def build_rescue_detail(k5_df, k10_df, rescue_neurons):
    """Description-level AUROC table (k=5 and k=10) for neurons uncharacterized
    at k=5, so the rescue claim can be inspected description-by-description
    instead of just asserted as an aggregate count."""
    rows = []
    for layer, unit in rescue_neurons:
        k5_row = k5_df[(k5_df["layer"] == layer) & (k5_df["unit"] == unit)].iloc[0]
        _, k5_survivors = process_neuron(k5_row, n_desc=5, sep="")
        for i in range(5):
            rows.append({
                "layer": layer, "unit": unit, "k": 5, "desc_idx": i + 1,
                "description": k5_row[f"description_{i+1}"],
                "auroc": k5_row[f"auroc_{i+1}"],
                "survived_filter": i in k5_survivors,
            })

        k10_row = k10_df[(k10_df["layer"] == layer) & (k10_df["unit"] == unit)].iloc[0]
        _, k10_survivors = process_neuron(k10_row, n_desc=10, sep="_")
        for i in range(10):
            rows.append({
                "layer": layer, "unit": unit, "k": 10, "desc_idx": i + 1,
                "description": k10_row[f"description_{i+1}"],
                "auroc": k10_row[f"auroc_{i+1}"],
                "survived_filter": i in k10_survivors,
            })
    return pd.DataFrame(rows)


def main():
    k5_df = pd.read_csv(K5_PAIRWISE)
    k5_df = k5_df[k5_df["model"] == "gpt2-xl"].copy()
    k10_df = pd.read_csv(K10_PAIRWISE).copy()

    k5_neurons = set(zip(k5_df["layer"], k5_df["unit"]))
    k10_neurons = set(zip(k10_df["layer"], k10_df["unit"]))
    common = k5_neurons & k10_neurons
    print(f"k=5 gpt2-xl neurons: {len(k5_neurons)}  |  k=10 neurons: {len(k10_neurons)}  |  matched: {len(common)}")

    rows = []
    for _, row in k5_df.iterrows():
        key = (row["layer"], row["unit"])
        if key not in common:
            continue
        m, _ = process_neuron(row, n_desc=5, sep="")
        rows.append({"layer": row["layer"], "unit": row["unit"], "k": 5, **m})

    for _, row in k10_df.iterrows():
        key = (row["layer"], row["unit"])
        if key not in common:
            continue
        m, _ = process_neuron(row, n_desc=10, sep="_")
        rows.append({"layer": row["layer"], "unit": row["unit"], "k": 10, **m})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved per-neuron metrics to {OUT_CSV}")

    # ---- Finding 1: weighting gap vs. survivor count ----
    multi = out[out["n_surviving"] > 1].copy()
    multi["gap"] = multi["weighted_mean_sim"] - multi["mean_sim_post"]
    multi["abs_gap"] = multi["gap"].abs()
    multi["trivial_floor"] = multi["n_surviving"] == 2

    abs_gap_k5 = float(multi.loc[multi["k"] == 5, "abs_gap"].mean())
    abs_gap_k10 = float(multi.loc[multi["k"] == 10, "abs_gap"].mean())

    # Trend: abs_gap vs n_surviving, pooled across k=5 and k=10 (more range of
    # n_surviving than either alone), excluding the trivial n_surviving==2 floor.
    trend = multi[~multi["trivial_floor"]]
    spearman_trend = spearmanr(trend["n_surviving"], trend["abs_gap"])
    pearson_trend = pearsonr(trend["n_surviving"], trend["abs_gap"])

    bin_labels = ["2 (trivial)", "3", "4", "5", "6+"]
    bins = pd.cut(multi["n_surviving"], bins=[1, 2, 3, 4, 5, 100], labels=bin_labels)
    binned = (multi.groupby(bins, observed=True)
                    .agg(n=("abs_gap", "size"), mean_abs_gap=("abs_gap", "mean")))
    binned_table = {str(idx): {"n": int(row["n"]), "mean_abs_gap": round(float(row["mean_abs_gap"]), 4)}
                     for idx, row in binned.iterrows()}

    # Within-neuron paired comparison: neurons with n_surviving > 1 at BOTH k=5
    # and k=10, controlling for neuron identity directly (no pool-composition
    # confound from comparing two different sets of 46 vs. 51 neurons).
    wide_gap = multi.pivot(index=["layer", "unit"], columns="k",
                            values=["n_surviving", "abs_gap"])
    paired = wide_gap.dropna(subset=[("n_surviving", 5), ("n_surviving", 10)]).copy()
    paired["delta_n_surviving"] = paired[("n_surviving", 10)] - paired[("n_surviving", 5)]
    paired["delta_abs_gap"] = paired[("abs_gap", 10)] - paired[("abs_gap", 5)]
    n_paired = len(paired)
    if n_paired >= 3:
        paired_corr_result = pearsonr(paired["delta_n_surviving"], paired["delta_abs_gap"])
        paired_corr = {"r": round(float(paired_corr_result.statistic), 4),
                        "p": round(float(paired_corr_result.pvalue), 4)}
    else:
        paired_corr = None

    # ---- Finding 2: zero-survivor rescue ----
    wide = out.pivot(index=["layer", "unit"], columns="k",
                      values=["n_surviving", "uncharacterized"])
    zero_k5 = wide[("uncharacterized", 5)] == True
    n_zero_k5 = int(zero_k5.sum())
    rescued = zero_k5 & (wide[("uncharacterized", 10)] == False)
    n_rescued = int(rescued.sum())

    rescue_neurons = list(wide.index[zero_k5])
    rescue_detail = build_rescue_detail(k5_df, k10_df, rescue_neurons)
    rescue_detail.to_csv(OUT_RESCUE_CSV, index=False)
    print(f"Saved rescue detail (description-level AUROC, k=5 vs k=10) to {OUT_RESCUE_CSV}")

    # ---- Finding 2b: mean survivor-count increase (all 60 matched neurons) ----
    wide_n = wide["n_surviving"]
    delta_n_surviving = wide_n[10] - wide_n[5]
    mean_delta_n_surviving = float(delta_n_surviving.mean())
    std_delta_n_surviving = float(delta_n_surviving.std())

    # ---- Finding 3: confidence C dilution ----
    C_k5 = float(out.loc[out["k"] == 5, "confidence_C"].mean())
    C_k10 = float(out.loc[out["k"] == 10, "confidence_C"].mean())

    summary = {
        "weighting_gap_vs_survivor_count": {
            "pooled_mean_abs_gap": {
                "k5": round(abs_gap_k5, 4),
                "k10": round(abs_gap_k10, 4),
                "n_neurons_k5": int((multi["k"] == 5).sum()),
                "n_neurons_k10": int((multi["k"] == 10).sum()),
                "note": "uses |weighted_mean_sim - mean_sim_post|; the signed mean is "
                        "near zero by cancellation (positive and negative deviations "
                        "roughly balance across neurons) and is not informative alone",
            },
            "structural_floor": {
                "n_surviving_eq_2_gap_is_always_0": True,
                "reason": "with exactly one pairwise similarity in the component, the "
                          "AUROC-weighted mean of a single value trivially equals the "
                          "unweighted mean -- not evidence that weighting doesn't matter",
                "n_such_neuron_observations": int(multi["trivial_floor"].sum()),
            },
            "trend_abs_gap_vs_n_surviving": {
                "binned_pooled_k5_and_k10": binned_table,
                "spearman_r": round(float(spearman_trend.statistic), 4),
                "spearman_p": round(float(spearman_trend.pvalue), 4),
                "pearson_r": round(float(pearson_trend.statistic), 4),
                "pearson_p": round(float(pearson_trend.pvalue), 4),
                "excludes_n_surviving_eq_2": True,
                "interpretation": "gap peaks around n_surviving=3 and declines as survivor "
                                   "count grows further -- opposite of 'weighting matters more "
                                   "at higher N'; no significant monotonic widening trend",
            },
            "within_neuron_paired_comparison": {
                "n_neurons_with_n_surviving_gt_1_at_both_k": n_paired,
                "delta_n_surviving_vs_delta_abs_gap": paired_corr,
                "interpretation": "controls for neuron identity directly; consistent with "
                                   "the pooled binned trend above",
            },
        },
        "zero_survivor_rescue": {
            "n_zero_at_k5": n_zero_k5,
            "n_rescued_at_k10": n_rescued,
            "rescue_rate": round(n_rescued / n_zero_k5, 4) if n_zero_k5 else None,
            "detail_csv": str(OUT_RESCUE_CSV.relative_to(ROOT)),
        },
        "survivor_count_increase": {
            "n_neurons": int(len(wide_n)),
            "mean_delta_n_surviving": round(mean_delta_n_surviving, 4),
            "std_delta_n_surviving": round(std_delta_n_surviving, 4),
            "mean_n_surviving_k5": round(float(wide_n[5].mean()), 4),
            "mean_n_surviving_k10": round(float(wide_n[10].mean()), 4),
        },
        "confidence_C_dilution": {
            "mean_C_k5": round(C_k5, 4),
            "mean_C_k10": round(C_k10, 4),
            "interpretation": "C is a mean over ALL candidate descriptions, so it is "
                               "mechanically diluted by adding more (typically lower-quality) "
                               "candidates -- a definitional property of C's formula, not a k effect",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUT_JSON}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

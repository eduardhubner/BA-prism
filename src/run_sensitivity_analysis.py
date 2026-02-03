"""
Minimal, high-value sensitivity analysis for polysemanticity scoring.

This script runs 7 experiments total:
A) 3 weighting schemes (auc, auc_centered, auc_sigmoid)
B) 2 reliability filters (0.0, 0.5)
C) 2 redundancy thresholds (0.9, 0.95)

Usage:
    python3 src/run_sensitivity_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from polysemanticity_scoring import PolysemanticityScoringExperiment

RESULTS_DIR = Path(__file__).parent.parent / "results" / "polysemanticity_experiments"


def run_experiment_a():
    """
    A) Weighting schemes comparison

    Question: Is the score definition arbitrary?
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: WEIGHTING SCHEMES")
    print("="*80)

    schemes = ["auc", "auc_centered", "auc_sigmoid"]
    results_by_scheme = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scheme in schemes:
        print(f"\n{'='*40}")
        print(f"Running: {scheme}")
        print(f"{'='*40}")

        # Create fresh experiment instance per run
        experiment = PolysemanticityScoringExperiment()

        results = experiment.compute_all_metrics(
            weight_scheme=scheme,
            reliability_filter=0.0,
            redundancy_threshold=0.9
        )

        results_by_scheme[scheme] = results
        experiment.save_results(results, experiment_name=f"A_weight_{scheme}_{timestamp}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: WEIGHTING SCHEMES")
    print("="*80)

    base = results_by_scheme["auc"]

    for scheme in ["auc_centered", "auc_sigmoid"]:
        comp = results_by_scheme[scheme]

        # Merge on neuron identifiers
        merged = base[['model', 'layer', 'unit', 'diversity_score']].merge(
            comp[['model', 'layer', 'unit', 'diversity_score']],
            on=['model', 'layer', 'unit'],
            suffixes=('_auc', f'_{scheme}')
        )

        # Create neuron ID for proper tracking
        merged['neuron_id'] = (
            merged['model'].astype(str) + "|L" +
            merged['layer'].astype(int).astype(str) + "|U" +
            merged['unit'].astype(int).astype(str)
        )

        # Rank correlation (drop NaNs for safety)
        valid_mask = merged['diversity_score_auc'].notna() & merged[f'diversity_score_{scheme}'].notna()
        if valid_mask.sum() > 2:
            rho, p = spearmanr(
                merged.loc[valid_mask, 'diversity_score_auc'],
                merged.loc[valid_mask, f'diversity_score_{scheme}']
            )
            print(f"\n{scheme} vs auc:")
            print(f"  Spearman ρ: {rho:.4f} (p={p:.2e})")
        else:
            print(f"\n{scheme} vs auc:")
            print(f"  Spearman ρ: insufficient data")
            rho = np.nan

        # Top 10 overlap (use nlargest to avoid tie issues)
        top10_auc = set(merged.nlargest(10, 'diversity_score_auc')['neuron_id'])
        top10_comp = set(merged.nlargest(10, f'diversity_score_{scheme}')['neuron_id'])
        overlap = len(top10_auc & top10_comp)

        print(f"  Top-10 overlap: {overlap}/10")

        # Biggest rank changes (using fractional ranks with 'first' tie-breaking)
        merged['rank_auc'] = merged['diversity_score_auc'].rank(ascending=False, method='first')
        merged[f'rank_{scheme}'] = merged[f'diversity_score_{scheme}'].rank(ascending=False, method='first')
        merged['rank_change'] = abs(merged['rank_auc'] - merged[f'rank_{scheme}'])

        top_changers = merged.nlargest(5, 'rank_change')

        print(f"  Biggest rank changes:")
        for idx, row in top_changers.iterrows():
            print(f"    {row['model']} L{int(row['layer'])}_U{int(row['unit'])}: "
                  f"rank {int(row['rank_auc'])} → {int(row[f'rank_{scheme}'])} "
                  f"(Δ={int(row['rank_change'])})")

    return results_by_scheme


def run_experiment_b():
    """
    B) Reliability filter comparison

    Question: Are diagnostics stable or artifacts of filtering?
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: RELIABILITY FILTER")
    print("="*80)

    filters = [0.0, 0.5]
    results_by_filter = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for filt in filters:
        print(f"\n{'='*40}")
        print(f"Running: reliability_filter={filt}")
        print(f"{'='*40}")

        # Create fresh experiment instance per run
        experiment = PolysemanticityScoringExperiment()

        results = experiment.compute_all_metrics(
            weight_scheme="auc",
            reliability_filter=filt,
            redundancy_threshold=0.9
        )

        results_by_filter[filt] = results
        experiment.save_results(results, experiment_name=f"B_filter_{filt}_{timestamp}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: RELIABILITY FILTER")
    print("="*80)

    no_filter = results_by_filter[0.0]
    with_filter = results_by_filter[0.5]

    # Merge on neuron identifiers for proper alignment
    merged = no_filter[['model', 'layer', 'unit', 'median_sim_unweighted', 'min_sim_unweighted', 'range_gap_unweighted', 'redundancy_fraction_unweighted']].merge(
        with_filter[['model', 'layer', 'unit', 'median_sim_unweighted', 'min_sim_unweighted', 'range_gap_unweighted', 'redundancy_fraction_unweighted']],
        on=['model', 'layer', 'unit'],
        suffixes=('_no_filter', '_with_filter')
    )

    # Count undefined diagnostics with filtering
    undefined_with_filter = merged['median_sim_unweighted_with_filter'].isna().sum()
    print(f"\nNeurons with undefined diagnostics when filtering:")
    print(f"  With filter=0.5: {undefined_with_filter}/{len(merged)}")

    # Compare diagnostics for neurons that have both
    print(f"\nDiagnostic changes (for {len(merged) - undefined_with_filter} neurons with both):")
    for metric in ['median_sim_unweighted', 'min_sim_unweighted', 'range_gap_unweighted', 'redundancy_fraction_unweighted']:
        # Drop NaNs for this metric comparison
        valid_mask = merged[f'{metric}_no_filter'].notna() & merged[f'{metric}_with_filter'].notna()
        valid_merged = merged[valid_mask]

        if len(valid_merged) < 2:
            print(f"  {metric}: insufficient data")
            continue

        diff = (valid_merged[f'{metric}_no_filter'] - valid_merged[f'{metric}_with_filter']).abs()
        print(f"  {metric}:")
        print(f"    Mean absolute diff: {diff.mean():.4f}")
        print(f"    Max diff: {diff.max():.4f}")

        # Correlation (drop NaNs for safety)
        rho, p = spearmanr(
            valid_merged[f'{metric}_no_filter'],
            valid_merged[f'{metric}_with_filter']
        )
        print(f"    Spearman ρ: {rho:.4f}")

    return results_by_filter


def run_experiment_c():
    """
    C) Redundancy threshold comparison

    Question: Is redundancy detection robust?
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: REDUNDANCY THRESHOLD")
    print("="*80)

    thresholds = [0.9, 0.95]
    results_by_threshold = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for thresh in thresholds:
        print(f"\n{'='*40}")
        print(f"Running: redundancy_threshold={thresh}")
        print(f"{'='*40}")

        # Create fresh experiment instance per run
        experiment = PolysemanticityScoringExperiment()

        results = experiment.compute_all_metrics(
            weight_scheme="auc",
            reliability_filter=0.0,
            redundancy_threshold=thresh
        )

        results_by_threshold[thresh] = results
        experiment.save_results(results, experiment_name=f"C_redund_{thresh}_{timestamp}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: REDUNDANCY THRESHOLD")
    print("="*80)

    low = results_by_threshold[0.9]
    high = results_by_threshold[0.95]

    # Merge on neuron identifiers for proper alignment
    merged = low[['model', 'layer', 'unit', 'redundancy_fraction_unweighted']].merge(
        high[['model', 'layer', 'unit', 'redundancy_fraction_unweighted']],
        on=['model', 'layer', 'unit'],
        suffixes=('_low', '_high')
    )

    # Correlation of redundancy_fraction (drop NaNs for safety)
    valid_mask = merged['redundancy_fraction_unweighted_low'].notna() & merged['redundancy_fraction_unweighted_high'].notna()
    valid_merged = merged[valid_mask]

    if len(valid_merged) >= 2:
        rho, p = spearmanr(
            valid_merged['redundancy_fraction_unweighted_low'],
            valid_merged['redundancy_fraction_unweighted_high']
        )
        print(f"\nredundancy_fraction_unweighted correlation:")
        print(f"  Spearman ρ: {rho:.4f} (p={p:.2e})")
    else:
        print(f"\nredundancy_fraction_unweighted correlation: insufficient data")

    # Classification flips (using 0.3 as cutoff for "redundant")
    low_redundant = merged['redundancy_fraction_unweighted_low'] > 0.3
    high_redundant = merged['redundancy_fraction_unweighted_high'] > 0.3

    flips = (low_redundant != high_redundant).sum()

    print(f"\nRedundancy classification (>30% pairs redundant):")
    print(f"  threshold=0.9: {low_redundant.sum()} neurons classified as redundant")
    print(f"  threshold=0.95: {high_redundant.sum()} neurons classified as redundant")
    print(f"  Classification flips: {flips}/{len(merged)} neurons")

    # Mean redundancy fraction
    print(f"\nMean redundancy_fraction_unweighted:")
    print(f"  threshold=0.9:  {low['redundancy_fraction_unweighted'].mean():.4f} ± {low['redundancy_fraction_unweighted'].std():.4f}")
    print(f"  threshold=0.95: {high['redundancy_fraction_unweighted'].mean():.4f} ± {high['redundancy_fraction_unweighted'].std():.4f}")

    return results_by_threshold


def create_summary_table(results_a, results_b, results_c):
    """Create summary CSV of all experiments."""

    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)

    summary = []

    # Experiment A: Weighting schemes
    for scheme, results in results_a.items():
        summary.append({
            'experiment': 'A_weighting',
            'parameter': 'weight_scheme',
            'value': scheme,
            'n_neurons': len(results),
            'mean_diversity': results['diversity_score'].mean(),
            'std_diversity': results['diversity_score'].std(),
            'mean_n_eff': results['n_eff'].mean(),
            'mean_sum_weights': results['sum_weights'].mean(),
        })

    # Experiment B: Reliability filters
    for filt, results in results_b.items():
        summary.append({
            'experiment': 'B_reliability_filter',
            'parameter': 'reliability_filter',
            'value': filt,
            'n_neurons': len(results),
            'mean_diversity': results['diversity_score'].mean(),
            'std_diversity': results['diversity_score'].std(),
            'mean_n_eff': results['n_eff'].mean(),
            'mean_sum_weights': results['sum_weights'].mean(),
        })

    # Experiment C: Redundancy thresholds
    for thresh, results in results_c.items():
        summary.append({
            'experiment': 'C_redundancy_threshold',
            'parameter': 'redundancy_threshold',
            'value': thresh,
            'n_neurons': len(results),
            'mean_diversity': results['diversity_score'].mean(),
            'std_diversity': results['diversity_score'].std(),
            'mean_n_eff': results['n_eff'].mean(),
            'mean_sum_weights': results['sum_weights'].mean(),
        })

    summary_df = pd.DataFrame(summary)
    summary_file = RESULTS_DIR / "sensitivity_analysis_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✓ Saved summary to: {summary_file}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def main():
    """Run all sensitivity experiments."""

    print("="*80)
    print("MINIMAL HIGH-VALUE SENSITIVITY ANALYSIS")
    print("="*80)
    print("\nRunning 7 experiments:")
    print("  A) 3 weighting schemes")
    print("  B) 2 reliability filters")
    print("  C) 2 redundancy thresholds")

    # Run experiments
    results_a = run_experiment_a()
    results_b = run_experiment_b()
    results_c = run_experiment_c()

    # Create summary
    create_summary_table(results_a, results_b, results_c)

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print("1. Check Spearman ρ for weighting schemes:")
    print("   - High ρ (>0.9) indicates robustness")
    print("   - Lower ρ for centered vs auc expected (gating changes rankings)")
    print("   - Inspect rank changers to understand differences")
    print("2. Check top-10 overlap:")
    print("   - High overlap (8-10/10) indicates stable top neurons")
    print("   - Low overlap reveals where methods disagree")
    print("3. Check diagnostic stability with filtering:")
    print("   - High ρ (>0.9) means diagnostics are robust")
    print("   - Count how many neurons lose data with filtering")
    print("4. Check redundancy classification robustness:")
    print("   - Few flips indicate stable redundancy detection")
    print("\nAll results saved to:")
    print(f"  {RESULTS_DIR}")


if __name__ == "__main__":
    main()

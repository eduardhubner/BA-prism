"""
Integrated Polysemanticity Scoring System

This script combines:
1. AUC-weighted scoring formulas
2. Experiment framework for testing multiple metrics
3. Loading and processing pairwise similarities data
4. Saving results and generating comparisons

Usage:
    python3 src/polysemanticity_scoring.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import json
import warnings
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"
PAIRWISE_FILE = RESULTS_DIR / "pairwise_similarities_dataset.csv"
BASELINE_FILE = RESULTS_DIR / "random_baseline_similarities.csv"
OUTPUT_DIR = RESULTS_DIR / "polysemanticity_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

# Pair ordering for 5 descriptions (consistent across all functions)
PAIRS = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]

# Analysis parameters
REDUNDANCY_THRESHOLD = 0.9  # Similarity threshold for redundancy detection
AUC_THRESHOLD = 0.5  # AUC threshold for reliability diagnostics (not default weighting)
WEIGHT_SCHEME = "auc"  # Default: use AUC directly. Options: "auc", "auc_centered", "auc_sigmoid"
RELIABILITY_FILTER_AUC = 0.5  # AUC threshold for reliability-based filtering in diagnostics
HAS_RELIABLE_DESC_AUC = 0.55  # AUC threshold for "has reliable description" flag (slightly above chance)


# ============================================================================
# AUC WEIGHTING FUNCTIONS
# ============================================================================

def compute_weights(aucs: np.ndarray,
                   scheme: str = "auc",
                   threshold: float = 0.5) -> np.ndarray:
    """
    Compute AUC-based confidence weights for descriptions.

    Schemes:
    - "auc": Weight = AUC (default, most principled)
    - "auc_centered": Weight = max(0, AUC - threshold) (hard gating)
    - "auc_sigmoid": Weight = sigmoid around threshold (smooth gating)

    Args:
        aucs: Array of AUC scores (shape: n_descriptions)
        scheme: Weighting scheme to use
        threshold: Threshold parameter for centering/sigmoid (default: 0.5)

    Returns:
        Array of weights (shape: n_descriptions)
    """
    if scheme == "auc":
        # Direct AUC (recommended default)
        return aucs
    elif scheme == "auc_centered":
        # Hard threshold at AUC = threshold
        return np.maximum(0.0, aucs - threshold)
    elif scheme == "auc_sigmoid":
        # Smooth sigmoid around threshold (steepness = 10)
        return 1.0 / (1.0 + np.exp(-10 * (aucs - threshold)))
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")


# ============================================================================
# LEGACY / DEPRECATED (kept for backward compatibility)
# ============================================================================

def compute_auc_weights(aucs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    DEPRECATED: Use compute_weights() instead.

    Legacy function for backward compatibility.
    Implements hard gating: Weight = max(0, AUC - threshold)

    This is no longer recommended - use compute_weights(scheme="auc") for principled weighting.
    """
    return compute_weights(aucs, scheme="auc_centered", threshold=threshold)


def compute_effective_n(weights: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute effective number of descriptions based on weight distribution.

    n_eff = (sum of weights)^2 / sum of squared weights

    Interpretation:
    - Uniform weights: n_eff ≈ n (all descriptions equally reliable)
    - One dominant weight: n_eff ≈ 1 (only one reliable description)
    - Zero weights: n_eff = 0 (no reliable descriptions)

    Args:
        weights: Array of AUC-based weights
        epsilon: Small constant to avoid division by zero

    Returns:
        Effective number of descriptions (float)
    """
    sum_weights = np.sum(weights)
    sum_squared_weights = np.sum(weights ** 2)

    if sum_weights < epsilon:
        return 0.0

    return (sum_weights ** 2) / (sum_squared_weights + epsilon)


# ============================================================================
# WEIGHTED SIMILARITY SCORES
# ============================================================================

def compute_weighted_mean_similarity(pairwise_sims: np.ndarray,
                                     weights: np.ndarray) -> float:
    """
    Compute weighted mean pairwise similarity.

    For 5 descriptions with weights w_1,...,w_5:
    weighted_mean_sim = sum(w_i * w_j * sim_ij) / sum(w_i * w_j)

    where the sums are over all pairs i < j.

    Args:
        pairwise_sims: Array of 10 pairwise similarities (ordered by PAIRS)
        weights: Array of 5 AUC-based weights [w_1, w_2, w_3, w_4, w_5]

    Returns:
        Weighted mean similarity (float)
    """
    if len(pairwise_sims) != 10 or len(weights) != 5:
        return np.nan

    # Compute pairwise weight products using PAIRS ordering
    weight_products = np.array([weights[i-1] * weights[j-1] for i, j in PAIRS])

    # Weighted sum
    weighted_sum = np.sum(weight_products * pairwise_sims)
    total_weight = np.sum(weight_products)

    if total_weight < 1e-10:
        return np.nan

    return weighted_sum / total_weight


# ============================================================================
# DISTRIBUTION DIAGNOSTICS
# ============================================================================

def compute_similarity_range(pairwise_sims: np.ndarray,
                             aucs: np.ndarray,
                             reliability_filter: float = 0.5) -> Tuple[float, float, float, int]:
    """
    Compute simple similarity range diagnostic: (median - min).

    This measures the "spread" of descriptions, focusing on how different
    the most dissimilar pair is from the typical pair.

    Args:
        pairwise_sims: Array of 10 pairwise similarities
        aucs: Array of 5 AUC values (for reliability filtering)
        reliability_filter: Optional AUC threshold for filtering pairs (0 = no filter)

    Returns:
        Tuple of (median_sim, min_sim, range_gap, n_valid_pairs)
        where range_gap = median - min
    """
    if len(pairwise_sims) != 10:
        return np.nan, np.nan, np.nan, 0

    # Optional: filter pairs where both descriptions have AUC >= threshold
    if reliability_filter > 0:
        valid_mask = np.array([(aucs[i-1] >= reliability_filter) and (aucs[j-1] >= reliability_filter)
                               for i, j in PAIRS])
        n_valid_pairs = int(np.sum(valid_mask))

        if n_valid_pairs < 2:
            return np.nan, np.nan, np.nan, n_valid_pairs

        valid_sims = pairwise_sims[valid_mask]
    else:
        # No filtering: use all pairs
        valid_sims = pairwise_sims
        n_valid_pairs = len(pairwise_sims)

    median_sim = float(np.median(valid_sims))
    min_sim = float(np.min(valid_sims))
    range_gap = median_sim - min_sim

    return median_sim, min_sim, range_gap, n_valid_pairs


def compute_redundancy_fraction(pairwise_sims: np.ndarray,
                                 aucs: np.ndarray,
                                 sim_threshold: float = 0.9,
                                 reliability_filter: float = 0.5) -> float:
    """
    Compute fraction of high-similarity pairs (redundant descriptions).

    This is an unweighted diagnostic: we optionally filter by AUC reliability,
    then compute the fraction of pairs above the similarity threshold.

    Args:
        pairwise_sims: Array of 10 pairwise similarities
        aucs: Array of 5 AUC values (for optional reliability filtering)
        sim_threshold: Similarity threshold for redundancy (default: 0.9)
        reliability_filter: Optional AUC threshold for filtering (0 = no filter)

    Returns:
        Redundancy fraction (float in [0, 1])
    """
    if len(pairwise_sims) != 10:
        return np.nan

    # Optional: filter pairs where both descriptions have AUC >= threshold
    if reliability_filter > 0:
        valid_mask = np.array([(aucs[i-1] >= reliability_filter) and (aucs[j-1] >= reliability_filter)
                               for i, j in PAIRS])

        if np.sum(valid_mask) == 0:
            return np.nan

        valid_sims = pairwise_sims[valid_mask]
    else:
        # No filtering: use all pairs
        valid_sims = pairwise_sims

    return float(np.mean(valid_sims > sim_threshold))


# ============================================================================
# DATA LOADING
# ============================================================================

class PolysemanticityScoringExperiment:
    """
    Framework for experimenting with polysemanticity scoring methods.
    """

    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize the experiment framework.

        Args:
            models: List of model names to analyze (None = all models)
        """
        print("="*80)
        print("POLYSEMANTICITY SCORING EXPERIMENT")
        print("="*80)

        # Load data
        self._load_data()

        # Filter by models if specified
        if models is not None:
            self.pairwise_df = self.pairwise_df[self.pairwise_df['model'].isin(models)]
            self.baseline_df = self.baseline_df[self.baseline_df['model'].isin(models)]

        print(f"\nLoaded {len(self.pairwise_df)} neurons from {self.pairwise_df['model'].nunique()} models")
        print(f"Models: {sorted(self.pairwise_df['model'].unique())}")

    def _load_data(self):
        """Load pairwise similarities and baseline data."""
        print("\nLoading data...")

        if not PAIRWISE_FILE.exists():
            raise FileNotFoundError(
                f"Pairwise similarities not found: {PAIRWISE_FILE}\n"
                "Run step2_compute_pairwise_similarities.py first!"
            )

        if not BASELINE_FILE.exists():
            raise FileNotFoundError(
                f"Baseline not found: {BASELINE_FILE}\n"
                "Run step2_compute_pairwise_similarities.py first!"
            )

        self.pairwise_df = pd.read_csv(PAIRWISE_FILE)
        self.baseline_df = pd.read_csv(BASELINE_FILE)

        print(f"  ✓ Loaded {len(self.pairwise_df)} neurons")
        print(f"  ✓ Loaded {len(self.baseline_df)} baseline samples")

    def _extract_neuron_data(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pairwise similarities and AUCs from a neuron row."""
        # Extract pairwise similarities using PAIRS ordering
        sim_cols = [f'sim_{i}{j}' for i, j in PAIRS]
        pairwise_sims = np.array([row[col] for col in sim_cols])

        # Extract AUCs
        aucs = np.array([row[f'auc_{i}'] for i in range(1, 6)])

        return pairwise_sims, aucs

    def compute_all_metrics(self,
                           weight_scheme: str = WEIGHT_SCHEME,
                           auc_threshold: float = AUC_THRESHOLD,
                           redundancy_threshold: float = REDUNDANCY_THRESHOLD,
                           reliability_filter: float = RELIABILITY_FILTER_AUC) -> pd.DataFrame:
        """
        Compute all polysemanticity metrics for all neurons.

        Args:
            weight_scheme: Weighting scheme ("auc", "auc_centered", "auc_sigmoid")
            auc_threshold: AUC threshold parameter (only used for centered/sigmoid schemes)
            redundancy_threshold: Similarity threshold for redundancy detection
            reliability_filter: AUC threshold for filtering diagnostics (0 = no filter)

        Returns:
            DataFrame with computed metrics
        """
        print(f"\nComputing metrics...")
        print(f"  Weight scheme: {weight_scheme}")
        print(f"  AUC threshold: {auc_threshold}")
        print(f"  Redundancy threshold: {redundancy_threshold}")
        print(f"  Reliability filter: {reliability_filter}")

        # Store parameters for saving later
        self.last_run_params = {
            'weight_scheme': weight_scheme,
            'auc_threshold': auc_threshold,
            'redundancy_threshold': redundancy_threshold,
            'reliability_filter': reliability_filter,
        }

        # Pre-compute baseline statistics by model (and cache values for percentile)
        baseline_stats = {}
        baseline_values = {}
        for model in self.pairwise_df['model'].unique():
            model_baseline = self.baseline_df[self.baseline_df['model'] == model]
            baseline_stats[model] = {
                'mean': model_baseline['mean_similarity'].mean(),
                'std': model_baseline['mean_similarity'].std(),
            }
            baseline_values[model] = model_baseline['mean_similarity'].values

        results = []

        for idx, row in self.pairwise_df.iterrows():
            pairwise_sims, aucs = self._extract_neuron_data(row)

            # Guard against NaNs in data
            if np.isnan(pairwise_sims).any() or np.isnan(aucs).any():
                print(f"Warning: Skipping neuron {row['model']} L{row['layer']}_U{row['unit']} due to NaN values")
                continue

            # Compute weights using selected scheme
            weights = compute_weights(aucs, scheme=weight_scheme, threshold=auc_threshold)
            sum_weights = np.sum(weights)  # Total reliability mass
            max_auc = np.max(aucs)

            # Primary metric: AUC-weighted mean similarity
            weighted_mean_sim = compute_weighted_mean_similarity(pairwise_sims, weights)

            # Primary score: diversity (1 - similarity)
            diversity_score = 1.0 - weighted_mean_sim if not np.isnan(weighted_mean_sim) else np.nan

            # Reliability metrics
            n_eff = compute_effective_n(weights)  # Effective number of descriptions (entropy-based)

            # Auxiliary reliability indicators (interpretable thresholds)
            n_desc_auc_ge_05 = int(np.sum(aucs >= 0.5))  # Count above random baseline
            has_reliable_desc = max_auc >= HAS_RELIABLE_DESC_AUC  # At least one prediction above chance

            # Diagnostics (using AUC-based filtering if reliability_filter > 0)
            median_sim, min_sim, range_gap, n_valid_pairs = compute_similarity_range(
                pairwise_sims, aucs, reliability_filter=reliability_filter
            )
            redundancy_frac = compute_redundancy_fraction(
                pairwise_sims, aucs,
                sim_threshold=redundancy_threshold,
                reliability_filter=reliability_filter
            )

            # Baseline normalization
            # FIX: Use unweighted similarity for z-score (apples-to-apples comparison)
            # The baseline is computed from unweighted random similarities, so we compare
            # the unweighted mean similarity to maintain consistency.
            baseline = baseline_stats[row['model']]
            baseline_mean = baseline['mean']
            baseline_std = baseline['std']

            # Get unweighted mean similarity for this neuron
            unweighted_mean_sim = row['mean_similarity']

            # Z-score normalization (unweighted, for consistency with baseline)
            if not np.isnan(unweighted_mean_sim) and baseline_std > 1e-8:
                unweighted_similarity_zscore = (unweighted_mean_sim - baseline_mean) / baseline_std
            else:
                unweighted_similarity_zscore = np.nan

            # Percentile under baseline (unweighted)
            if not np.isnan(unweighted_mean_sim):
                model_baseline_sims = baseline_values[row['model']]
                baseline_percentile = np.mean(model_baseline_sims < unweighted_mean_sim) * 100
            else:
                baseline_percentile = np.nan

            # EXPERIMENTAL: Confidence-adjusted diversity (requires justification)
            confidence_adjusted_diversity = diversity_score * (n_eff / 5.0) if not np.isnan(diversity_score) else np.nan

            # Sanity metric: unweighted diversity (PRISM baseline)
            unweighted_diversity = 1.0 - row['mean_similarity']

            results.append({
                'model': row['model'],
                'layer': row['layer'],
                'unit': row['unit'],

                # Primary metrics
                'weighted_mean_sim': weighted_mean_sim,
                'diversity_score': diversity_score,

                # Reliability
                'n_eff': n_eff,
                'sum_weights': sum_weights,
                'max_auc': max_auc,
                'n_desc_auc_ge_05': n_desc_auc_ge_05,  # Count of descriptions with AUC >= 0.5
                'has_reliable_desc': has_reliable_desc,  # Boolean: max_auc >= 0.55

                # Diagnostics (unweighted)
                'median_sim_unweighted': median_sim,
                'min_sim_unweighted': min_sim,
                'range_gap_unweighted': range_gap,
                'n_valid_pairs': n_valid_pairs,
                'redundancy_fraction_unweighted': redundancy_frac,

                # Baseline normalization (unweighted, for apples-to-apples comparison)
                'unweighted_similarity_zscore': unweighted_similarity_zscore,
                'baseline_percentile': baseline_percentile,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,

                # Experimental (requires justification)
                'exp_confidence_adjusted_diversity': confidence_adjusted_diversity,

                # For comparison / sanity checks
                'unweighted_mean_sim': row['mean_similarity'],
                'unweighted_diversity': unweighted_diversity,
            })

        results_df = pd.DataFrame(results)

        # Print summary
        self._print_metric_summary(results_df)

        return results_df

    def _print_metric_summary(self, results_df: pd.DataFrame):
        """Print summary statistics for computed metrics."""
        print("\n" + "="*80)
        print("METRIC SUMMARY")
        print("="*80)

        # Primary metrics
        print("\nPRIMARY METRICS:")
        for metric in ['weighted_mean_sim', 'diversity_score']:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  Mean: {values.mean():.4f} ± {values.std():.4f}")
                print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

        # Reliability
        print("\nRELIABILITY:")
        for metric in ['n_eff', 'sum_weights', 'max_auc', 'n_desc_auc_ge_05']:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  Mean: {values.mean():.4f} ± {values.std():.4f}")
                print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

        print(f"\nNeurons with reliable descriptions (max_auc >= {HAS_RELIABLE_DESC_AUC}): {results_df['has_reliable_desc'].sum()}/{len(results_df)}")

        # Diagnostics
        print("\nDIAGNOSTICS (unweighted):")
        for metric in ['median_sim_unweighted', 'min_sim_unweighted', 'range_gap_unweighted', 'n_valid_pairs', 'redundancy_fraction_unweighted']:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  Mean: {values.mean():.4f} ± {values.std():.4f}")
                print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

        # Baseline normalization
        print("\nBASELINE NORMALIZATION (unweighted):")
        for metric in ['unweighted_similarity_zscore', 'baseline_percentile']:
            values = results_df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  Mean: {values.mean():.4f} ± {values.std():.4f}")
                print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

    def compare_metrics(self, results_df: pd.DataFrame,
                       metric1: str, metric2: str) -> Dict:
        """
        Compare two metrics using correlation analysis.

        Args:
            results_df: DataFrame with computed metrics
            metric1: First metric name
            metric2: Second metric name

        Returns:
            Dictionary with comparison statistics
        """
        valid_mask = results_df[metric1].notna() & results_df[metric2].notna()
        valid_df = results_df[valid_mask]

        if len(valid_df) < 2:
            return {'error': 'Not enough valid data points'}

        x = valid_df[metric1].values
        y = valid_df[metric2].values

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        return {
            'metric1': metric1,
            'metric2': metric2,
            'n_samples': len(valid_df),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }

    def save_results(self, results_df: pd.DataFrame,
                    experiment_name: str = None):
        """
        Save experiment results to CSV and metadata to JSON.

        Args:
            results_df: DataFrame with computed metrics
            experiment_name: Name for this experiment (defaults to timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save CSV
        csv_path = OUTPUT_DIR / f"polysemanticity_scores_{experiment_name}.csv"
        results_df.to_csv(csv_path, index=False)

        # Create metadata
        metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'n_neurons': len(results_df),
            'models': sorted(results_df['model'].unique().tolist()),
            'parameters': {
                **self.last_run_params,  # Use actual run parameters
                'n_descriptions': 5,
                'n_pairs': 10,
            },
            'metrics_computed': {
                'primary': [
                    'weighted_mean_sim',
                    'diversity_score',
                ],
                'reliability': [
                    'n_eff',
                    'sum_weights',
                    'max_auc',
                    'n_desc_auc_ge_05',
                    'has_reliable_desc',
                ],
                'diagnostics': [
                    'median_sim_unweighted',
                    'min_sim_unweighted',
                    'range_gap_unweighted',
                    'n_valid_pairs',
                    'redundancy_fraction_unweighted',
                ],
                'baseline_normalization': [
                    'unweighted_similarity_zscore',
                    'baseline_percentile',
                ],
                'experimental': [
                    'exp_confidence_adjusted_diversity',
                ],
                'sanity_checks': [
                    'unweighted_mean_sim',
                    'unweighted_diversity',
                ],
            },
            'summary_statistics': {}
        }

        # Add summary stats (flatten metrics_computed structure)
        all_metrics = []
        for category in metadata['metrics_computed'].values():
            all_metrics.extend(category)

        for metric in all_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    metadata['summary_statistics'][metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                    }

        # Save metadata
        json_path = OUTPUT_DIR / f"polysemanticity_scores_{experiment_name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*80)
        print("RESULTS SAVED")
        print("="*80)
        print(f"CSV:      {csv_path}")
        print(f"Metadata: {json_path}")
        print("="*80)

    def print_summary_by_model(self, results_df: pd.DataFrame):
        """Print summary statistics grouped by model."""
        print("\n" + "="*80)
        print("SUMMARY BY MODEL")
        print("="*80)

        for model in sorted(results_df['model'].unique()):
            model_data = results_df[results_df['model'] == model]

            print(f"\n{model}:")
            print(f"  Neurons: {len(model_data)}")
            print(f"  Neurons with reliable desc: {model_data['has_reliable_desc'].sum()}")
            print(f"  Weighted mean sim: {model_data['weighted_mean_sim'].mean():.4f} ± {model_data['weighted_mean_sim'].std():.4f}")
            print(f"  Diversity score:   {model_data['diversity_score'].mean():.4f} ± {model_data['diversity_score'].std():.4f}")
            print(f"  Effective n:       {model_data['n_eff'].mean():.4f} ± {model_data['n_eff'].std():.4f}")
            print(f"  Desc. AUC >= 0.5:  {model_data['n_desc_auc_ge_05'].mean():.2f}")
            print(f"  Unweighted similarity z-score: {model_data['unweighted_similarity_zscore'].mean():.4f} ± {model_data['unweighted_similarity_zscore'].std():.4f}")
            print(f"  Baseline:          {model_data['baseline_mean'].mean():.4f} ± {model_data['baseline_std'].mean():.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the polysemanticity scoring experiment."""

    # Initialize experiment
    experiment = PolysemanticityScoringExperiment()

    # Compute all metrics with default scheme (direct AUC weighting)
    results = experiment.compute_all_metrics(
        weight_scheme="auc",  # Default: direct AUC weighting (most principled)
        reliability_filter=0.0  # No filtering in diagnostics
    )

    # Print summary by model
    experiment.print_summary_by_model(results)

    # Compare metrics
    print("\n" + "="*80)
    print("METRIC CORRELATIONS")
    print("="*80)

    comparisons = [
        ('weighted_mean_sim', 'unweighted_mean_sim'),
        ('diversity_score', 'unweighted_diversity'),
        ('diversity_score', 'n_eff'),
        ('redundancy_fraction_unweighted', 'weighted_mean_sim'),
        ('range_gap_unweighted', 'diversity_score'),
        ('unweighted_similarity_zscore', 'baseline_percentile'),
    ]

    for metric1, metric2 in comparisons:
        comp = experiment.compare_metrics(results, metric1, metric2)
        if 'error' not in comp:
            print(f"\n{metric1} vs {metric2}:")
            print(f"  Pearson r:  {comp['pearson_r']:.4f} (p={comp['pearson_p']:.2e})")
            print(f"  Spearman ρ: {comp['spearman_r']:.4f} (p={comp['spearman_p']:.2e})")

    # Save results
    experiment.save_results(results, experiment_name="auc_direct")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nKey metrics:")
    print("  PRIMARY: diversity_score = 1 - weighted_mean_sim")
    print("           (Default: direct AUC weighting, no gating)")
    print("  RELIABILITY: n_eff (effective number of descriptions)")
    print("               sum_weights (total reliability mass)")
    print("  DIAGNOSTICS: median_sim_unweighted, min_sim_unweighted, range_gap_unweighted")
    print("  BASELINE: unweighted_similarity_zscore (preferred normalization)")
    print("\nWeighting schemes:")
    print("  'auc' (default): w = AUC (principled, no gating)")
    print("  'auc_centered': w = max(0, AUC - 0.5) (hard gating)")
    print("  'auc_sigmoid': smooth sigmoid around threshold")
    print("\nNote: exp_confidence_adjusted_diversity is EXPERIMENTAL")
    print("      (requires justification before use)")
    print("\nNext steps:")
    print("1. Analyze the saved results CSV")
    print("2. Create visualizations (scatter plots, histograms)")
    print("3. Compare scores across different models")
    print("4. Sensitivity analysis:")
    print("   - Compare weight_scheme='auc' vs 'auc_centered'")
    print("   - Vary redundancy_threshold (0.9 vs 0.95)")
    print("   - Vary reliability_filter (0.0 vs 0.5)")
    print("5. Rank-change analysis: correlation with PRISM baseline")
    print("6. Identify neurons with interesting polysemanticity patterns")


if __name__ == "__main__":
    main()

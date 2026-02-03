"""
Compare clustering results between Qwen2-1.5B and Qwen3-0.5B embedding models.

Analyzes robustness of clustering-based polysemanticity detection across
different embedding models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

# Load both result files
QWEN2_RESULTS = Path("clustering_analysis_qwen2")
QWEN3_RESULTS = Path("clustering_analysis_cosine")

def load_latest_results(results_dir):
    """Load the most recent results.json file from a directory."""
    # Try both naming patterns: results_*.json and results.json
    results_files = list(results_dir.glob("results_*.json"))

    if not results_files:
        # Try plain results.json
        plain_file = results_dir / "results.json"
        if plain_file.exists():
            results_files = [plain_file]
        else:
            raise FileNotFoundError(f"No results files found in {results_dir}")

    # Sort by timestamp and get latest
    latest_file = sorted(results_files)[-1]
    print(f"Loading: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def extract_metrics_by_neuron(results):
    """Extract key metrics organized by neuron."""
    metrics = {}

    for neuron in results['neurons']:
        layer = neuron['layer']
        unit = neuron['unit']
        key = (layer, unit)

        # Handle both old format (no prefix) and new format (euclidean_/cosine_ prefix)
        kmeans = neuron['kmeans_results']

        # Check which format we're dealing with
        if 'cosine_silhouette' in kmeans:
            # New format with metric prefixes
            sil_key = 'cosine_silhouette'
            bic_key = 'cosine_bic'
            db_key = 'cosine_davies_bouldin'
            k5_key = 'cosine_fixed-k5'
        else:
            # Old format without prefixes (assumes cosine was used)
            sil_key = 'silhouette'
            bic_key = 'bic'
            db_key = 'davies_bouldin'
            k5_key = 'fixed-k5'

        metrics[key] = {
            # Clusterability
            'hopkins_euclidean': neuron['clusterability'].get('hopkins_euclidean'),
            'hopkins_cosine': neuron['clusterability'].get('hopkins_cosine'),

            # K-means (cosine, silhouette-selected)
            'cosine_sil_k': kmeans[sil_key]['selected_k'],
            'cosine_sil_score': kmeans[sil_key]['silhouette_score'],
            'cosine_sil_db': kmeans[sil_key].get('davies_bouldin_score'),

            # K-means (cosine, BIC-selected)
            'cosine_bic_k': kmeans[bic_key]['selected_k'],
            'cosine_bic_sil': kmeans[bic_key]['silhouette_score'],

            # K-means (cosine, Davies-Bouldin-selected)
            'cosine_db_k': kmeans[db_key]['selected_k'],
            'cosine_db_sil': kmeans[db_key]['silhouette_score'],

            # Fixed k=5 baseline
            'fixed_k5_sil': kmeans[k5_key]['silhouette_score'],
        }

    return metrics


def compare_models(qwen2_metrics, qwen3_metrics):
    """Compare metrics between the two embedding models."""

    # Find common neurons
    common_neurons = sorted(set(qwen2_metrics.keys()) & set(qwen3_metrics.keys()))
    print(f"\nComparing {len(common_neurons)} neurons present in both datasets")

    if len(common_neurons) == 0:
        print("ERROR: No common neurons found!")
        return

    # Organize data for comparison
    comparison_data = []

    for neuron in common_neurons:
        layer, unit = neuron
        q2 = qwen2_metrics[neuron]
        q3 = qwen3_metrics[neuron]

        comparison_data.append({
            'layer': layer,
            'unit': unit,
            # Hopkins
            'qwen2_hopkins_cos': q2['hopkins_cosine'],
            'qwen3_hopkins_cos': q3['hopkins_cosine'],
            # Silhouette scores
            'qwen2_sil': q2['cosine_sil_score'],
            'qwen3_sil': q3['cosine_sil_score'],
            'sil_diff': q2['cosine_sil_score'] - q3['cosine_sil_score'],
            'sil_diff_pct': (q2['cosine_sil_score'] - q3['cosine_sil_score']) / q3['cosine_sil_score'] * 100,
            # Selected k values
            'qwen2_k_sil': q2['cosine_sil_k'],
            'qwen3_k_sil': q3['cosine_sil_k'],
            'k_agrees': q2['cosine_sil_k'] == q3['cosine_sil_k'],
            'k_diff': abs(q2['cosine_sil_k'] - q3['cosine_sil_k']),
            # BIC k-selection
            'qwen2_k_bic': q2['cosine_bic_k'],
            'qwen3_k_bic': q3['cosine_bic_k'],
            'k_bic_agrees': q2['cosine_bic_k'] == q3['cosine_bic_k'],
        })

    df = pd.DataFrame(comparison_data)

    # Print summary statistics
    print("\n" + "="*80)
    print("EMBEDDING MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Qwen2-1.5B vs Qwen3-0.5B")
    print(f"Common neurons: {len(common_neurons)}")
    print("="*80)

    # Hopkins statistic correlation
    print("\n1. HOPKINS STATISTIC (Clustering Tendency)")
    print("-" * 60)
    hopkins_corr = pearsonr(df['qwen2_hopkins_cos'], df['qwen3_hopkins_cos'])
    print(f"  Pearson correlation: r={hopkins_corr[0]:.4f}, p={hopkins_corr[1]:.4e}")
    print(f"  Mean Hopkins (Qwen2): {df['qwen2_hopkins_cos'].mean():.4f}")
    print(f"  Mean Hopkins (Qwen3): {df['qwen3_hopkins_cos'].mean():.4f}")
    print(f"  Mean difference: {(df['qwen2_hopkins_cos'] - df['qwen3_hopkins_cos']).mean():.4f}")

    # Silhouette score correlation
    print("\n2. SILHOUETTE SCORES (Clustering Quality)")
    print("-" * 60)
    sil_corr = pearsonr(df['qwen2_sil'], df['qwen3_sil'])
    print(f"  Pearson correlation: r={sil_corr[0]:.4f}, p={sil_corr[1]:.4e}")
    print(f"  Mean Silhouette (Qwen2): {df['qwen2_sil'].mean():.4f} ± {df['qwen2_sil'].std():.4f}")
    print(f"  Mean Silhouette (Qwen3): {df['qwen3_sil'].mean():.4f} ± {df['qwen3_sil'].std():.4f}")
    print(f"  Mean absolute difference: {df['sil_diff'].abs().mean():.4f}")
    print(f"  Mean % difference: {df['sil_diff_pct'].mean():.1f}%")

    # K-selection agreement
    print("\n3. K-SELECTION AGREEMENT")
    print("-" * 60)
    print(f"  Silhouette k-selection:")
    k_agreement = (df['k_agrees'].sum() / len(df) * 100)
    print(f"    Exact agreement: {df['k_agrees'].sum()}/{len(df)} ({k_agreement:.1f}%)")
    print(f"    Mean |k_diff|: {df['k_diff'].mean():.2f}")
    print(f"    Qwen2 mean k: {df['qwen2_k_sil'].mean():.2f} ± {df['qwen2_k_sil'].std():.2f}")
    print(f"    Qwen3 mean k: {df['qwen3_k_sil'].mean():.2f} ± {df['qwen3_k_sil'].std():.2f}")

    print(f"\n  BIC k-selection:")
    k_bic_agreement = (df['k_bic_agrees'].sum() / len(df) * 100)
    print(f"    Exact agreement: {df['k_bic_agrees'].sum()}/{len(df)} ({k_bic_agreement:.1f}%)")
    print(f"    Qwen2 mean k: {df['qwen2_k_bic'].mean():.2f} ± {df['qwen2_k_bic'].std():.2f}")
    print(f"    Qwen3 mean k: {df['qwen3_k_bic'].mean():.2f} ± {df['qwen3_k_bic'].std():.2f}")

    # K-selection distribution
    print("\n4. K-SELECTION DISTRIBUTION (Silhouette)")
    print("-" * 60)
    print("  Qwen2-1.5B:")
    for k in range(2, 11):
        count = (df['qwen2_k_sil'] == k).sum()
        pct = count / len(df) * 100
        if count > 0:
            print(f"    k={k}: {count:2d} ({pct:4.1f}%)")

    print("\n  Qwen3-0.5B:")
    for k in range(2, 11):
        count = (df['qwen3_k_sil'] == k).sum()
        pct = count / len(df) * 100
        if count > 0:
            print(f"    k={k}: {count:2d} ({pct:4.1f}%)")

    # Layer-wise analysis
    print("\n5. LAYER-WISE COMPARISON")
    print("-" * 60)
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer]
        print(f"\n  Layer {layer} ({len(layer_df)} neurons):")
        print(f"    Hopkins corr: r={pearsonr(layer_df['qwen2_hopkins_cos'], layer_df['qwen3_hopkins_cos'])[0]:.3f}")
        print(f"    Silhouette corr: r={pearsonr(layer_df['qwen2_sil'], layer_df['qwen3_sil'])[0]:.3f}")
        print(f"    K-agreement: {(layer_df['k_agrees'].sum() / len(layer_df) * 100):.1f}%")
        print(f"    Mean Sil (Q2/Q3): {layer_df['qwen2_sil'].mean():.3f} / {layer_df['qwen3_sil'].mean():.3f}")

    # Neurons with largest differences
    print("\n6. NEURONS WITH LARGEST SILHOUETTE DIFFERENCES")
    print("-" * 60)
    top_diff = df.nlargest(5, 'sil_diff_pct')
    print("  Top 5 (Qwen2 better):")
    for idx, row in top_diff.iterrows():
        print(f"    L{row['layer']}_U{row['unit']}: {row['sil_diff_pct']:+.1f}% "
              f"(Q2={row['qwen2_sil']:.3f}, Q3={row['qwen3_sil']:.3f})")

    bottom_diff = df.nsmallest(5, 'sil_diff_pct')
    print("\n  Top 5 (Qwen3 better):")
    for idx, row in bottom_diff.iterrows():
        print(f"    L{row['layer']}_U{row['unit']}: {row['sil_diff_pct']:+.1f}% "
              f"(Q2={row['qwen2_sil']:.3f}, Q3={row['qwen3_sil']:.3f})")

    # Save detailed comparison to CSV
    output_file = Path("embedding_model_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed comparison saved to: {output_file}")
    print(f"{'='*80}")

    return df


def main():
    print("Loading clustering results...")
    print("-" * 60)

    try:
        qwen2_results = load_latest_results(QWEN2_RESULTS)
        qwen3_results = load_latest_results(QWEN3_RESULTS)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print("\nExtracting metrics...")
    qwen2_metrics = extract_metrics_by_neuron(qwen2_results)
    qwen3_metrics = extract_metrics_by_neuron(qwen3_results)

    print(f"  Qwen2-1.5B: {len(qwen2_metrics)} neurons")
    print(f"  Qwen3-0.5B: {len(qwen3_metrics)} neurons")

    # Run comparison
    comparison_df = compare_models(qwen2_metrics, qwen3_metrics)

    if comparison_df is not None:
        print("\nInterpretation:")
        print("-" * 60)

        # Interpret correlations
        hopkins_r = pearsonr(comparison_df['qwen2_hopkins_cos'], comparison_df['qwen3_hopkins_cos'])[0]
        sil_r = pearsonr(comparison_df['qwen2_sil'], comparison_df['qwen3_sil'])[0]
        k_agreement = (comparison_df['k_agrees'].sum() / len(comparison_df) * 100)

        if hopkins_r > 0.7 and sil_r > 0.7:
            print("✓ STRONG correlation in both Hopkins and Silhouette scores")
            print("  → Clustering tendency is robust across embedding models")
        elif hopkins_r > 0.5 and sil_r > 0.5:
            print("✓ MODERATE correlation in clustering metrics")
            print("  → Some embedding-specific effects, but general structure preserved")
        else:
            print("⚠ WEAK correlation in clustering metrics")
            print("  → Embedding model choice significantly affects clustering")

        if k_agreement > 70:
            print(f"\n✓ HIGH k-selection agreement ({k_agreement:.0f}%)")
            print("  → Optimal cluster count is robust to embedding choice")
        elif k_agreement > 50:
            print(f"\n✓ MODERATE k-selection agreement ({k_agreement:.0f}%)")
            print("  → Some neurons show different optimal k across models")
        else:
            print(f"\n⚠ LOW k-selection agreement ({k_agreement:.0f}%)")
            print("  → Optimal cluster count varies significantly with embedding model")

        mean_sil_diff = abs(comparison_df['sil_diff'].mean())
        if mean_sil_diff < 0.05:
            print(f"\n✓ Small mean silhouette difference ({mean_sil_diff:.3f})")
            print("  → Clustering quality is comparable across models")
        else:
            print(f"\n⚠ Notable mean silhouette difference ({mean_sil_diff:.3f})")
            print("  → One embedding model produces consistently better clusters")


if __name__ == "__main__":
    main()

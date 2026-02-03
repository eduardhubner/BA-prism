"""
Clustering Analysis Pipeline - Dual Metric Version

Analyzes clusterability and clustering quality using BOTH Euclidean and Cosine
distance metrics for comparison between embedding models.

This version runs:
- Standard K-Means with Euclidean distance
- Spherical K-Means with Cosine distance
- All adaptive-k methods (Silhouette, BIC, Davies-Bouldin) + fixed-k5
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from utils import cache, clustering, clusterability, cluster_cache, config
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Check for optional dependencies
try:
    from utils.spherical_kmeans import SphericalKMeans
    SPHERICAL_KMEANS_AVAILABLE = True
except ImportError:
    SPHERICAL_KMEANS_AVAILABLE = False
    print("WARNING: SphericalKMeans not available. Cosine methods will be skipped.")

# All neurons with cached embeddings
NEURONS = [
    (0, 1149),(0, 1749),(0, 2725),(0, 3057),(0, 3124),(0, 3279),
    (0, 3533),(0, 3696),(0, 3943),(0, 4297),(0, 4405),(0, 440),
    (0, 4679),(0, 4842),(0, 5085),(0, 5551),(0, 5781),(0, 5960),
    (0, 6114),(0, 6314),
    (20, 1004),(20, 1406),(20, 1424),(20, 1790),(20, 2325),(20, 2679),
    (20, 2700),(20, 2869),(20, 2988),(20, 328),(20, 3313),(20, 3885),
    (20, 4268),(20, 4447),(20, 4683),(20, 5278),(20, 5662),(20, 5741),
    (20, 5789),(20, 6045),
    (40, 1516),(40, 1555),(40, 1823),(40, 183),(40, 3254),(40, 3515),
    (40, 3612),(40, 3636),(40, 364),(40, 3948),(40, 4055),(40, 4244),
    (40, 4808),(40, 4965),(40, 5557),(40, 556),(40, 6067),(40, 6106),
    (40, 6364),(40, 824)
]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "clustering_analysis_dual_metric"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_kmeans_analysis(embeddings, normalized, metric='euclidean', method_prefix=''):
    """
    Run k-means clustering with all adaptive-k methods + fixed-k5.

    Args:
        embeddings: Raw embeddings (for Euclidean)
        normalized: Normalized embeddings (for Cosine)
        metric: 'euclidean' or 'cosine'
        method_prefix: Prefix for method names (e.g., 'euclidean_' or 'cosine_')

    Returns:
        Dictionary with clustering results for all methods
    """
    print(f"\n{'='*60}")
    print(f"K-Means Clustering ({metric.upper()} distance)")
    print(f"{'='*60}")

    results = {}
    data = normalized if metric == 'cosine' else embeddings

    # Run k-means for k=2 to k=10
    print("  Running k-means for k=2 to k=10...")
    all_kmeans_results = {}

    for k in range(2, 11):
        if metric == 'cosine' and SPHERICAL_KMEANS_AVAILABLE:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = kmeans.fit_predict(data)

        all_kmeans_results[k] = {
            'labels': labels,
            'inertia': kmeans.inertia_
        }

    # Compute scores for all adaptive-k methods
    for method in ['silhouette', 'bic', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_kmeans_results[k]['labels']
            inertia = all_kmeans_results[k]['inertia']

            if method == "silhouette":
                scores[k] = silhouette_score(data, labels, metric=metric)
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(data, labels)
            elif method == "bic":
                n_parameters = k * data.shape[1]
                variance = inertia / (len(data) - k)
                log_likelihood = -len(data) * np.log(max(variance, 1e-10)) - inertia / (2 * variance)
                bic = -2 * log_likelihood + n_parameters * np.log(len(data))
                scores[k] = bic

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # bic or davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_kmeans_results[optimal_k]['labels']

        # Compute all metrics for this clustering
        sil_score = silhouette_score(data, labels, metric=metric)
        db_score = davies_bouldin_score(data, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        results[f'{method_prefix}{method}'] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()},
            'metric': metric
        }

    # Fixed k=5 baseline
    print("\n  Fixed k=5 (baseline):")
    labels = all_kmeans_results[5]['labels']
    sil_score = silhouette_score(data, labels, metric=metric)
    db_score = davies_bouldin_score(data, labels)

    print(f"    k=5 (fixed): Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

    results[f'{method_prefix}fixed-k5'] = {
        'selected_k': 5,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'all_k_scores': {},
        'metric': metric
    }

    return results


def analyze_neuron(layer, unit):
    """
    Run complete clustering analysis for one neuron with BOTH metrics.

    Args:
        layer: Layer ID
        unit: Unit ID

    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Layer {layer}, Unit {unit}")
    print(f"{'='*80}")

    # Load cached embeddings
    print("Loading embeddings...")
    cached_data = cache.load_cached_embeddings(layer, unit)
    if cached_data is None:
        print(f"  ERROR: No cached embeddings found!")
        return None

    embeddings, texts = cached_data
    print(f"  Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normalized = embeddings / norms

    # Clusterability Assessment
    print("\nClusterability Assessment:")
    print("  Computing Hopkins statistic (Euclidean)...")
    hopkins_euc = clusterability.hopkins_statistic(embeddings, metric='euclidean')
    print(f"    Hopkins (Euclidean): {hopkins_euc:.4f}")

    print("  Computing Hopkins statistic (Cosine)...")
    hopkins_cos = clusterability.hopkins_statistic(normalized, metric='cosine')
    print(f"    Hopkins (Cosine): {hopkins_cos:.4f}")

    # Run K-Means with EUCLIDEAN distance
    euclidean_results = run_kmeans_analysis(
        embeddings, normalized,
        metric='euclidean',
        method_prefix='euclidean_'
    )

    # Run K-Means with COSINE distance (if available)
    if SPHERICAL_KMEANS_AVAILABLE:
        cosine_results = run_kmeans_analysis(
            embeddings, normalized,
            metric='cosine',
            method_prefix='cosine_'
        )
    else:
        cosine_results = {}
        print("\nSkipping cosine methods - SphericalKMeans not available")

    # Combine all results
    all_kmeans_results = {**euclidean_results, **cosine_results}

    # Compile final results
    results = {
        'layer': layer,
        'unit': unit,
        'clusterability': {
            'hopkins_euclidean': float(hopkins_euc),
            'hopkins_cosine': float(hopkins_cos)
        },
        'kmeans_results': all_kmeans_results
    }

    return results


def main():
    """Run clustering analysis on all neurons with both metrics."""
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS PIPELINE - DUAL METRIC")
    print("="*80)
    print(f"Metrics: Euclidean + Cosine")
    print(f"Total neurons: {len(NEURONS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    all_results = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'embedding_model': config.CLUSTER_EMBEDDING_MODEL_NAME,
            'metrics': ['euclidean', 'cosine'],
            'total_neurons': len(NEURONS),
            'spherical_kmeans_available': SPHERICAL_KMEANS_AVAILABLE
        },
        'neurons': []
    }

    # Run analysis for each neuron
    for idx, (layer, unit) in enumerate(NEURONS, 1):
        print(f"\n[{idx}/{len(NEURONS)}]")
        results = analyze_neuron(layer, unit)

        if results is not None:
            all_results['neurons'].append(results)
        else:
            print(f"  Skipped due to error")

    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = OUTPUT_DIR / f"results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\nSummary Statistics:")

    # Hopkins statistics
    hopkins_euc_scores = [n['clusterability']['hopkins_euclidean']
                          for n in all_results['neurons']]
    hopkins_cos_scores = [n['clusterability']['hopkins_cosine']
                          for n in all_results['neurons']]
    print(f"  Mean Hopkins (Euclidean): {np.mean(hopkins_euc_scores):.4f}")
    print(f"  Mean Hopkins (Cosine): {np.mean(hopkins_cos_scores):.4f}")

    # Silhouette scores for best method per metric
    euc_sil = [n['kmeans_results']['euclidean_silhouette']['silhouette_score']
               for n in all_results['neurons']]
    print(f"\n  Euclidean K-Means (Silhouette-selected k):")
    print(f"    Mean Silhouette: {np.mean(euc_sil):.4f}")

    if SPHERICAL_KMEANS_AVAILABLE:
        cos_sil = [n['kmeans_results']['cosine_silhouette']['silhouette_score']
                   for n in all_results['neurons']]
        print(f"\n  Cosine K-Means (Silhouette-selected k):")
        print(f"    Mean Silhouette: {np.mean(cos_sil):.4f}")
        print(f"    Improvement: {(np.mean(cos_sil) - np.mean(euc_sil)) / np.mean(euc_sil) * 100:.1f}%")


if __name__ == "__main__":
    main()

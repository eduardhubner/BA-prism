"""
Clustering Analysis Pipeline

Analyzes clusterability and clustering quality for all cached neurons
without generating descriptions. Results are saved for later analysis.

This is a standalone analysis pipeline separate from the main
feature_description.py workflow.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from utils import cache, clustering, clusterability, cluster_cache, config
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Check for optional dependencies
try:
    from utils.spherical_kmeans import SphericalKMeans
    SPHERICAL_KMEANS_AVAILABLE = True
except ImportError:
    SPHERICAL_KMEANS_AVAILABLE = False
    print("WARNING: SphericalKMeans not available. Will use standard K-Means.")

# Check UMAP availability
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: UMAP not installed. UMAP visualizations will be skipped.")
    print("Install with: pip install umap-learn")

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

# Output directories - relative to project root
# Go up from src/ to project root, then to clustering_analysis_cosine
OUTPUT_DIR = Path(__file__).parent.parent / "clustering_analysis_cosine"
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def analyze_neuron(layer, unit):
    """
    Run complete clusterability and clustering analysis for one neuron.

    Args:
        layer: Layer ID
        unit: Unit ID

    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Layer {layer}, Unit {unit}")
    print(f"{'='*60}")

    # Load cached embeddings
    print("Loading embeddings...")
    cached_data = cache.load_cached_embeddings(layer, unit)
    if cached_data is None:
        print(f"  ERROR: No cached embeddings found!")
        return None

    embeddings, texts = cached_data
    print(f"  Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Normalize for cosine similarity (guard against zero vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)  # Prevent division by zero
    normalized = embeddings / norms

    # Clusterability Assessment
    print("\nClusterability Assessment:")
    print("  Computing Hopkins statistic (Euclidean)...")
    hopkins_euc = clusterability.hopkins_statistic(embeddings, metric='euclidean')
    print(f"    Hopkins (Euclidean): {hopkins_euc:.4f}")

    print("  Computing Hopkins statistic (Cosine)...")
    hopkins_cos = clusterability.hopkins_statistic(normalized, metric='cosine')
    print(f"    Hopkins (Cosine): {hopkins_cos:.4f} - {interpret_hopkins(hopkins_cos)}")

    print("  Creating PCA visualization...")
    neuron_id = f"layer-{layer}_unit-{unit}"
    pca_stats = clusterability.create_pca_plot(
        normalized,
        save_path=VIZ_DIR / f"{neuron_id}_pca.png",
        title=f"PCA: Layer {layer}, Unit {unit}"
    )
    print(f"    PCA variance explained: {pca_stats['cumulative_variance']:.1%}")

    if UMAP_AVAILABLE:
        print("  Creating UMAP visualization...")
        umap_stats = clusterability.create_umap_plot(
            normalized,
            metric='cosine',
            save_path=VIZ_DIR / f"{neuron_id}_umap.png",
            title=f"UMAP: Layer {layer}, Unit {unit}"
        )
    else:
        umap_stats = {}

    # 3. K-Means Clustering (3 adaptive-k methods + fixed_k(=5))
    print("\nK-Means Clustering (Spherical + Cosine):")
    kmeans_results = {}

    # Run k-means once for each k, store all results
    print("  Running k-means for k=2 to k=10...")
    all_kmeans_results = {}
    for k in range(2, 11):
        if config.SPHERICAL and SPHERICAL_KMEANS_AVAILABLE:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = kmeans.fit_predict(normalized)

        all_kmeans_results[k] = {
            'labels': labels,
            'inertia': kmeans.inertia_
        }

    # Compute all 3 metrics from stored results
    for method in ['silhouette', 'bic', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_kmeans_results[k]['labels']
            inertia = all_kmeans_results[k]['inertia']

            if method == "silhouette":
                scores[k] = silhouette_score(normalized, labels, metric='cosine')
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(normalized, labels)
            elif method == "bic":
                n_parameters = k * normalized.shape[1]
                variance = inertia / (len(normalized) - k)
                log_likelihood = -len(normalized) * np.log(max(variance, 1e-10)) - inertia / (2 * variance)
                bic = -2 * log_likelihood + n_parameters * np.log(len(normalized))
                scores[k] = bic

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # bic or davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_kmeans_results[optimal_k]['labels']

        # Compute all metrics for this clustering
        sil_score = silhouette_score(normalized, labels, metric='cosine')
        db_score = davies_bouldin_score(normalized, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, method, labels, texts)

        kmeans_results[method] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()}
        }

    # Fixed k=5 k-means (baseline) - reuse k=5 from above
    print("\nFixed k=5 k-means (baseline):")
    labels = all_kmeans_results[5]['labels']

    # Compute metrics
    sil_score = silhouette_score(normalized, labels, metric='cosine')
    db_score = davies_bouldin_score(normalized, labels)

    print(f"  k=5 (fixed): Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

    # Save cluster assignments
    cluster_cache.save_clusters(layer, unit, 'fixed-k5', labels, texts)

    kmeans_results['fixed-k5'] = {
        'selected_k': 5,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'all_k_scores': {}  # Empty since we didn't test multiple k values
    }

    # 4. Agglomerative Clustering
    print("\nAgglomerative Clustering (cosine):")
    agglomerative_results = {}

    # Build dendrogram once and compute all metrics from it
    print("  Building dendrogram...")

    # Compute distance matrix and build dendrogram
    distances = pdist(normalized, metric='cosine')
    Z = linkage(distances, method='average')

    # Get labels for all k values
    all_agg_labels = {}
    for k in range(2, 11):
        labels = fcluster(Z, k, criterion='maxclust') - 1  # 0-indexed
        all_agg_labels[k] = labels

    # Compute both metrics from the same clustering results
    for method in ['silhouette', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_agg_labels[k]

            if method == "silhouette":
                scores[k] = silhouette_score(normalized, labels, metric='cosine')
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(normalized, labels)

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_agg_labels[optimal_k]

        # Compute all metrics for this clustering
        sil_score = silhouette_score(normalized, labels, metric='cosine')
        db_score = davies_bouldin_score(normalized, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, f'agglomerative_{method}', labels, texts)

        agglomerative_results[method] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()}
        }

    # 5. HDBSCAN Clustering
    print("\nHDBSCAN Clustering (Cosine):")
    hdbscan_labels, n_clusters, n_noise = clustering.run_hdbscan_clustering(
        normalized, metric='cosine', min_cluster_size=5
    )

    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # Compute metrics for HDBSCAN (excluding noise points)
    if n_clusters > 1 and n_noise < len(hdbscan_labels):
        non_noise_mask = hdbscan_labels != -1
        hdbscan_sil = silhouette_score(
            normalized[non_noise_mask],
            hdbscan_labels[non_noise_mask],
            metric='cosine'
        )
        hdbscan_db = davies_bouldin_score(
            normalized[non_noise_mask],
            hdbscan_labels[non_noise_mask]
        )
        print(f"  Silhouette: {hdbscan_sil:.4f}, Davies-Bouldin: {hdbscan_db:.4f}")
    else:
        hdbscan_sil = None
        hdbscan_db = None
        print(f"  Not enough clusters for quality metrics")

    # Save HDBSCAN clusters
    cluster_cache.save_clusters(layer, unit, 'hdbscan', hdbscan_labels, texts)

    # Create cluster-colored UMAP for HDBSCAN
    if UMAP_AVAILABLE:
        print("  Creating cluster-colored UMAP...")
        clusterability.create_umap_plot(
            normalized,
            labels=hdbscan_labels,
            metric='cosine',
            save_path=VIZ_DIR / f"{neuron_id}_umap_hdbscan.png",
            title=f"UMAP + HDBSCAN: Layer {layer}, Unit {unit}"
        )

    hdbscan_results = {
        'n_clusters_found': int(n_clusters),
        'n_noise_points': int(n_noise),
        'silhouette_score': float(hdbscan_sil) if hdbscan_sil is not None else None,
        'davies_bouldin_score': float(hdbscan_db) if hdbscan_db is not None else None
    }

    # 6. Compile all results
    results = {
        'layer': layer,
        'unit': unit,
        'clusterability': {
            'hopkins_euclidean': float(hopkins_euc),
            'hopkins_cosine': float(hopkins_cos),
            'pca_variance_explained': pca_stats['variance_explained'],
            'pca_cumulative_variance': pca_stats['cumulative_variance']
        },
        'kmeans_results': kmeans_results,
        'agglomerative_results': agglomerative_results,
        'hdbscan_results': hdbscan_results,
        'visualizations': {
            'pca': f"visualizations/{neuron_id}_pca.png",
            'umap': f"visualizations/{neuron_id}_umap.png",
            'umap_hdbscan': f"visualizations/{neuron_id}_umap_hdbscan.png"
        }
    }

    return results


def interpret_hopkins(h):
    """Interpret Hopkins statistic value."""
    if h > 0.7:
        return "STRONG clustering tendency"
    elif h > 0.5:
        return "MODERATE clustering tendency"
    else:
        return "WEAK clustering tendency"


def main():
    """Run clustering analysis on all neurons."""
    # Set SPHERICAL mode for this analysis
    original_spherical = config.SPHERICAL
    config.SPHERICAL = True

    try:
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS PIPELINE")
        print("="*60)
        print(f"Mode: SPHERICAL = {config.SPHERICAL} (Cosine similarity)")
        print(f"Total neurons: {len(NEURONS)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("="*60)

        all_results = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'spherical': config.SPHERICAL,
                'embedding_model': config.CLUSTER_EMBEDDING_MODEL_NAME,
                'distance_metric': 'cosine',
                'total_neurons': len(NEURONS)
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
        output_file = OUTPUT_DIR / "results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Analysis complete!")
        print(f"Results saved to: {output_file}")
        print(f"Visualizations saved to: {VIZ_DIR}")
        print(f"Cluster assignments saved to: {cluster_cache.CLUSTER_CACHE_DIR}")
        print(f"{'='*60}")

        # Print summary statistics
        print("\nSummary Statistics:")
        hopkins_cosine_scores = [n['clusterability']['hopkins_cosine']
                                 for n in all_results['neurons']]
        print(f"  Mean Hopkins (Cosine): {np.mean(hopkins_cosine_scores):.4f}")
        print(f"  Neurons with good clusterability (>0.7): "
              f"{sum(1 for h in hopkins_cosine_scores if h > 0.7)}/{len(hopkins_cosine_scores)}")

        sil_scores = [n['kmeans_results']['silhouette']['silhouette_score']
                      for n in all_results['neurons']]
        print(f"  Mean Silhouette (K-Means, best k): {np.mean(sil_scores):.4f}")
        print(f"  Neurons with good clustering (>0.1): "
              f"{sum(1 for s in sil_scores if s > 0.1)}/{len(sil_scores)}")

    finally:
        # Restore original config
        config.SPHERICAL = original_spherical


if __name__ == "__main__":
    main()

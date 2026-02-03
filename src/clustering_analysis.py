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
from utils import clustering, clusterability, cluster_cache, config
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

# Directories
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings_qwen2"  # Load from Qwen2 embeddings
DATA_DIR = Path(__file__).parent.parent / "data" / "candidate_inputs_decoded"
OUTPUT_DIR = Path(__file__).parent.parent / "clustering_analysis_qwen2"  # Save to separate directory
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

    # Load Qwen2 embeddings from file
    print("Loading embeddings...")
    embedding_file = EMBEDDINGS_DIR / f"gpt2-xl_layer-{layer}_unit-{unit}.npy"
    text_file = DATA_DIR / f"layer{layer}_{unit}.json"

    if not embedding_file.exists():
        print(f"  ERROR: Embedding file not found: {embedding_file}")
        return None
    if not text_file.exists():
        print(f"  ERROR: Text file not found: {text_file}")
        return None

    embeddings = np.load(embedding_file)
    with open(text_file, 'r') as f:
        texts = json.load(f)

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

    # 3. K-Means Clustering - BOTH Euclidean and Cosine
    kmeans_results = {}

    # === EUCLIDEAN K-MEANS ===
    print("\n" + "="*60)
    print("K-Means Clustering (EUCLIDEAN distance)")
    print("="*60)
    print("  Running Euclidean k-means for k=2 to k=10...")

    from sklearn.cluster import KMeans
    all_kmeans_euclidean = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        all_kmeans_euclidean[k] = {
            'labels': labels,
            'inertia': kmeans.inertia_
        }

    # === COSINE K-MEANS (Spherical) ===
    print("\n" + "="*60)
    print("K-Means Clustering (COSINE distance - Spherical)")
    print("="*60)
    print("  Running Spherical k-means for k=2 to k=10...")

    all_kmeans_cosine = {}
    for k in range(2, 11):
        if SPHERICAL_KMEANS_AVAILABLE:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = kmeans.fit_predict(normalized)
        all_kmeans_cosine[k] = {
            'labels': labels,
            'inertia': kmeans.inertia_
        }

    # Compute metrics for EUCLIDEAN K-Means
    for method in ['silhouette', 'bic', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_kmeans_euclidean[k]['labels']
            inertia = all_kmeans_euclidean[k]['inertia']

            if method == "silhouette":
                scores[k] = silhouette_score(embeddings, labels, metric='euclidean')
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(embeddings, labels)
            elif method == "bic":
                n_parameters = k * embeddings.shape[1]
                variance = inertia / (len(embeddings) - k)
                log_likelihood = -len(embeddings) * np.log(max(variance, 1e-10)) - inertia / (2 * variance)
                bic = -2 * log_likelihood + n_parameters * np.log(len(embeddings))
                scores[k] = bic

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # bic or davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_kmeans_euclidean[optimal_k]['labels']

        # Compute all metrics for this clustering
        sil_score = silhouette_score(embeddings, labels, metric='euclidean')
        db_score = davies_bouldin_score(embeddings, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, f'euclidean_{method}', labels, texts)

        kmeans_results[f'euclidean_{method}'] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()},
            'metric': 'euclidean'
        }

    # Fixed k=5 Euclidean k-means (baseline)
    print("\n  Fixed k=5 (baseline):")
    labels = all_kmeans_euclidean[5]['labels']
    sil_score = silhouette_score(embeddings, labels, metric='euclidean')
    db_score = davies_bouldin_score(embeddings, labels)
    print(f"    k=5 (fixed): Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")
    cluster_cache.save_clusters(layer, unit, 'euclidean_fixed-k5', labels, texts)

    kmeans_results['euclidean_fixed-k5'] = {
        'selected_k': 5,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'all_k_scores': {},
        'metric': 'euclidean'
    }

    # Compute metrics for COSINE K-Means
    for method in ['silhouette', 'bic', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_kmeans_cosine[k]['labels']
            inertia = all_kmeans_cosine[k]['inertia']

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

        labels = all_kmeans_cosine[optimal_k]['labels']

        # Compute all metrics for this clustering
        sil_score = silhouette_score(normalized, labels, metric='cosine')
        db_score = davies_bouldin_score(normalized, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, f'cosine_{method}', labels, texts)

        kmeans_results[f'cosine_{method}'] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()},
            'metric': 'cosine'
        }

    # Fixed k=5 Cosine k-means (baseline)
    print("\n  Fixed k=5 (baseline):")
    labels = all_kmeans_cosine[5]['labels']
    sil_score = silhouette_score(normalized, labels, metric='cosine')
    db_score = davies_bouldin_score(normalized, labels)
    print(f"    k=5 (fixed): Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")
    cluster_cache.save_clusters(layer, unit, 'cosine_fixed-k5', labels, texts)

    kmeans_results['cosine_fixed-k5'] = {
        'selected_k': 5,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'all_k_scores': {},
        'metric': 'cosine'
    }

    # 4. Agglomerative Clustering - BOTH Euclidean and Cosine
    agglomerative_results = {}

    # === EUCLIDEAN AGGLOMERATIVE ===
    print("\n" + "="*60)
    print("Agglomerative Clustering (EUCLIDEAN distance)")
    print("="*60)
    print("  Building dendrogram...")

    # Compute distance matrix and build dendrogram
    distances_euc = pdist(embeddings, metric='euclidean')
    Z_euc = linkage(distances_euc, method='average')

    # Get labels for all k values
    all_agg_euclidean = {}
    for k in range(2, 11):
        labels = fcluster(Z_euc, k, criterion='maxclust') - 1  # 0-indexed
        all_agg_euclidean[k] = labels

    # Compute both metrics from the same clustering results
    for method in ['silhouette', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_agg_euclidean[k]

            if method == "silhouette":
                scores[k] = silhouette_score(embeddings, labels, metric='euclidean')
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(embeddings, labels)

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_agg_euclidean[optimal_k]

        # Compute all metrics for this clustering
        sil_score = silhouette_score(embeddings, labels, metric='euclidean')
        db_score = davies_bouldin_score(embeddings, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, f'euclidean_agglomerative_{method}', labels, texts)

        agglomerative_results[f'euclidean_{method}'] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()},
            'metric': 'euclidean'
        }

    # === COSINE AGGLOMERATIVE ===
    print("\n" + "="*60)
    print("Agglomerative Clustering (COSINE distance)")
    print("="*60)
    print("  Building dendrogram...")

    # Compute distance matrix and build dendrogram
    distances_cos = pdist(normalized, metric='cosine')
    Z_cos = linkage(distances_cos, method='average')

    # Get labels for all k values
    all_agg_cosine = {}
    for k in range(2, 11):
        labels = fcluster(Z_cos, k, criterion='maxclust') - 1  # 0-indexed
        all_agg_cosine[k] = labels

    # Compute both metrics from the same clustering results
    for method in ['silhouette', 'davies_bouldin']:
        print(f"  Computing {method} scores...")
        scores = {}

        for k in range(2, 11):
            labels = all_agg_cosine[k]

            if method == "silhouette":
                scores[k] = silhouette_score(normalized, labels, metric='cosine')
            elif method == "davies_bouldin":
                scores[k] = davies_bouldin_score(normalized, labels)

        # Select optimal k
        if method == "silhouette":
            optimal_k = max(scores, key=scores.get)
        else:  # davies_bouldin
            optimal_k = min(scores, key=scores.get)

        labels = all_agg_cosine[optimal_k]

        # Compute all metrics for this clustering
        sil_score = silhouette_score(normalized, labels, metric='cosine')
        db_score = davies_bouldin_score(normalized, labels)

        print(f"    {method}: selected k={optimal_k}, Silhouette: {sil_score:.4f}, DB: {db_score:.4f}")

        # Save cluster assignments
        cluster_cache.save_clusters(layer, unit, f'cosine_agglomerative_{method}', labels, texts)

        agglomerative_results[f'cosine_{method}'] = {
            'selected_k': int(optimal_k),
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'all_k_scores': {int(k): float(v) for k, v in scores.items()},
            'metric': 'cosine'
        }

    # 5. HDBSCAN Clustering - BOTH Euclidean and Cosine
    hdbscan_results = {}

    # === EUCLIDEAN HDBSCAN ===
    print("\n" + "="*60)
    print("HDBSCAN Clustering (EUCLIDEAN distance)")
    print("="*60)
    hdbscan_labels_euc, n_clusters_euc, n_noise_euc = clustering.run_hdbscan_clustering(
        embeddings, metric='euclidean', min_cluster_size=5
    )

    print(f"  Found {n_clusters_euc} clusters, {n_noise_euc} noise points")

    # Compute metrics for HDBSCAN (excluding noise points)
    if n_clusters_euc > 1 and n_noise_euc < len(hdbscan_labels_euc):
        non_noise_mask = hdbscan_labels_euc != -1
        hdbscan_sil_euc = silhouette_score(
            embeddings[non_noise_mask],
            hdbscan_labels_euc[non_noise_mask],
            metric='euclidean'
        )
        hdbscan_db_euc = davies_bouldin_score(
            embeddings[non_noise_mask],
            hdbscan_labels_euc[non_noise_mask]
        )
        print(f"  Silhouette: {hdbscan_sil_euc:.4f}, Davies-Bouldin: {hdbscan_db_euc:.4f}")
    else:
        hdbscan_sil_euc = None
        hdbscan_db_euc = None
        print(f"  Not enough clusters for quality metrics")

    # Save HDBSCAN clusters
    cluster_cache.save_clusters(layer, unit, 'euclidean_hdbscan', hdbscan_labels_euc, texts)

    hdbscan_results['euclidean'] = {
        'n_clusters_found': int(n_clusters_euc),
        'n_noise_points': int(n_noise_euc),
        'silhouette_score': float(hdbscan_sil_euc) if hdbscan_sil_euc is not None else None,
        'davies_bouldin_score': float(hdbscan_db_euc) if hdbscan_db_euc is not None else None,
        'metric': 'euclidean'
    }

    # === COSINE HDBSCAN ===
    print("\n" + "="*60)
    print("HDBSCAN Clustering (COSINE distance)")
    print("="*60)
    hdbscan_labels_cos, n_clusters_cos, n_noise_cos = clustering.run_hdbscan_clustering(
        normalized, metric='cosine', min_cluster_size=5
    )

    print(f"  Found {n_clusters_cos} clusters, {n_noise_cos} noise points")

    # Compute metrics for HDBSCAN (excluding noise points)
    if n_clusters_cos > 1 and n_noise_cos < len(hdbscan_labels_cos):
        non_noise_mask = hdbscan_labels_cos != -1
        hdbscan_sil_cos = silhouette_score(
            normalized[non_noise_mask],
            hdbscan_labels_cos[non_noise_mask],
            metric='cosine'
        )
        hdbscan_db_cos = davies_bouldin_score(
            normalized[non_noise_mask],
            hdbscan_labels_cos[non_noise_mask]
        )
        print(f"  Silhouette: {hdbscan_sil_cos:.4f}, Davies-Bouldin: {hdbscan_db_cos:.4f}")
    else:
        hdbscan_sil_cos = None
        hdbscan_db_cos = None
        print(f"  Not enough clusters for quality metrics")

    # Save HDBSCAN clusters
    cluster_cache.save_clusters(layer, unit, 'cosine_hdbscan', hdbscan_labels_cos, texts)

    # Create cluster-colored UMAP for HDBSCAN (cosine version)
    if UMAP_AVAILABLE:
        print("  Creating cluster-colored UMAP...")
        clusterability.create_umap_plot(
            normalized,
            labels=hdbscan_labels_cos,
            metric='cosine',
            save_path=VIZ_DIR / f"{neuron_id}_umap_hdbscan.png",
            title=f"UMAP + HDBSCAN: Layer {layer}, Unit {unit}"
        )

    hdbscan_results['cosine'] = {
        'n_clusters_found': int(n_clusters_cos),
        'n_noise_points': int(n_noise_cos),
        'silhouette_score': float(hdbscan_sil_cos) if hdbscan_sil_cos is not None else None,
        'davies_bouldin_score': float(hdbscan_db_cos) if hdbscan_db_cos is not None else None,
        'metric': 'cosine'
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
    # Override cluster cache directory for this analysis
    cluster_cache.CLUSTER_CACHE_DIR = OUTPUT_DIR / "clusters"
    cluster_cache.CLUSTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Set SPHERICAL mode for this analysis
    original_spherical = config.SPHERICAL
    config.SPHERICAL = True

    try:
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS PIPELINE - DUAL METRIC")
        print("="*60)
        print(f"Distance metrics: Euclidean + Cosine")
        print(f"Total neurons: {len(NEURONS)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("="*60)

        all_results = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'spherical': config.SPHERICAL,
                'embedding_model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',  # Original PRISM model
                'embedding_source': 'embeddings_qwen2/',
                'distance_metrics': ['euclidean', 'cosine'],
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

        # Hopkins statistics
        hopkins_euc_scores = [n['clusterability']['hopkins_euclidean']
                              for n in all_results['neurons']]
        hopkins_cos_scores = [n['clusterability']['hopkins_cosine']
                              for n in all_results['neurons']]
        print(f"  Mean Hopkins (Euclidean): {np.mean(hopkins_euc_scores):.4f}")
        print(f"  Mean Hopkins (Cosine): {np.mean(hopkins_cos_scores):.4f}")

        # K-Means comparison
        euc_sil = [n['kmeans_results']['euclidean_silhouette']['silhouette_score']
                   for n in all_results['neurons']]
        cos_sil = [n['kmeans_results']['cosine_silhouette']['silhouette_score']
                   for n in all_results['neurons']]

        print(f"\n  K-Means Clustering Quality:")
        print(f"    Euclidean - Mean Silhouette: {np.mean(euc_sil):.4f}")
        print(f"    Cosine - Mean Silhouette: {np.mean(cos_sil):.4f}")
        print(f"    Improvement: {(np.mean(cos_sil) - np.mean(euc_sil)) / np.mean(euc_sil) * 100:+.1f}%")

    finally:
        # Restore original config
        config.SPHERICAL = original_spherical


if __name__ == "__main__":
    main()

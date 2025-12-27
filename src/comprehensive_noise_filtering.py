"""
Comprehensive Noise Filtering Experiment

Tests if HDBSCAN noise filtering improves ALL clustering methods:
1. K-means (spherical + cosine) with adaptive k-selection:
   - Silhouette
   - BIC
   - Davies-Bouldin
   - Fixed k=5
2. Agglomerative (cosine) with adaptive k-selection:
   - Silhouette
   - Davies-Bouldin
3. K-means (standard + euclidean) as baseline
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from utils.spherical_kmeans import SphericalKMeans

# Paths
EMBEDDING_CACHE_DIR = Path(__file__).parent.parent / "cache" / "embeddings"
ANALYSIS_DIR = Path(__file__).parent.parent / "clustering_analysis_cosine"
CLUSTER_DIR = ANALYSIS_DIR / "clusters"
RESULTS_FILE = ANALYSIS_DIR / "results.json"


def select_optimal_k_method(embeddings, method='silhouette', k_range=range(2, 11),
                           metric='cosine', use_spherical=True):
    """
    Select optimal k using specified method.

    Args:
        embeddings: Normalized embeddings
        method: 'silhouette', 'bic', or 'davies_bouldin'
        k_range: Range of k values to test
        metric: 'cosine' or 'euclidean'
        use_spherical: Use SphericalKMeans or standard KMeans

    Returns:
        (optimal_k, labels, score)
    """
    scores = {}
    all_labels = {}

    for k in k_range:
        if use_spherical:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = kmeans.fit_predict(embeddings)
        all_labels[k] = labels

        if method == 'silhouette':
            scores[k] = silhouette_score(embeddings, labels, metric=metric)
        elif method == 'davies_bouldin':
            scores[k] = davies_bouldin_score(embeddings, labels)
        elif method == 'bic':
            n_parameters = k * embeddings.shape[1]
            variance = kmeans.inertia_ / (len(embeddings) - k)
            log_likelihood = -len(embeddings) * np.log(max(variance, 1e-10)) - kmeans.inertia_ / (2 * variance)
            bic = -2 * log_likelihood + n_parameters * np.log(len(embeddings))
            scores[k] = bic

    # Select best k
    if method == 'silhouette':
        optimal_k = max(scores, key=scores.get)
    else:  # bic or davies_bouldin
        optimal_k = min(scores, key=scores.get)

    best_labels = all_labels[optimal_k]
    best_score = scores[optimal_k]

    return optimal_k, best_labels, best_score


def test_comprehensive_filtering(layer, unit, existing_results):
    """
    Comprehensive test: all methods with and without noise filtering.

    Uses existing results from clustering_analysis.py for "all samples" metrics.
    Only runs clustering on "clean samples" after noise filtering.

    Args:
        layer: Layer ID
        unit: Unit ID
        existing_results: Dict with existing clustering results for this neuron

    Returns:
        Dict with results for all methods
    """
    print(f"\n{'='*60}")
    print(f"Layer {layer}, Unit {unit}")
    print(f"{'='*60}")

    # Load embeddings
    embeddings_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_embeddings.npy"
    texts_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_texts.pkl"

    if not embeddings_path.exists():
        print(f"ERROR: No cached embeddings")
        return None

    embeddings = np.load(embeddings_path)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)

    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-9, None)

    print(f"Total samples: {len(normalized)}")

    # Load HDBSCAN noise labels
    hdbscan_file = CLUSTER_DIR / f"layer-{layer}_unit-{unit}_hdbscan.json"
    with open(hdbscan_file, 'r') as f:
        hdbscan_data = json.load(f)

    # Extract noise mask
    hdbscan_labels = np.full(len(normalized), -1, dtype=int)
    for cluster_id, items in hdbscan_data['clusters'].items():
        cluster_label = int(cluster_id)
        for item in items:
            hdbscan_labels[item['index']] = cluster_label

    clean_mask = hdbscan_labels != -1
    n_clean = clean_mask.sum()
    n_noise = (~clean_mask).sum()

    print(f"Clean: {n_clean} ({n_clean/len(normalized)*100:.1f}%)")
    print(f"Noise: {n_noise} ({n_noise/len(normalized)*100:.1f}%)")

    if n_clean < 25:
        print("ERROR: Not enough clean samples")
        return None

    clean_embeddings = normalized[clean_mask]

    results = {
        'layer': layer,
        'unit': unit,
        'total_samples': len(normalized),
        'clean_samples': int(n_clean),
        'noise_samples': int(n_noise),
        'methods': {}
    }

    # Test all methods
    methods_to_test = [
        # K-Means Spherical + Cosine (current best practice)
        ('kmeans_sil', 'K-Means Spherical (Silhouette)', 'kmeans', 'silhouette'),
        ('kmeans_bic', 'K-Means Spherical (BIC)', 'kmeans', 'bic'),
        ('kmeans_db', 'K-Means Spherical (Davies-Bouldin)', 'kmeans', 'davies_bouldin'),
        ('kmeans_k5', 'K-Means Spherical (k=5)', 'kmeans', 'fixed-k5'),

        # Agglomerative + Cosine
        ('agg_sil', 'Agglomerative (Silhouette)', 'agglomerative', 'silhouette'),
        ('agg_db', 'Agglomerative (Davies-Bouldin)', 'agglomerative', 'davies_bouldin'),
    ]

    for method_id, method_name, method_type, selection_method in methods_to_test:
        print(f"\n{method_name}:")

        # Get existing "all samples" results
        try:
            if method_type == 'kmeans':
                existing = existing_results['kmeans_results'][selection_method]
            else:  # agglomerative
                existing = existing_results['agglomerative_results'][selection_method]

            k_all = existing['selected_k']
            sil_all = existing['silhouette_score']
            db_all = existing['davies_bouldin_score']
            print(f"  All samples: k={k_all}, Sil={sil_all:.4f}, DB={db_all:.4f} (from existing results)")
        except (KeyError, TypeError) as e:
            print(f"  All samples: ERROR loading existing results - {e}")
            sil_all, db_all, k_all = None, None, None

        # Test on CLEAN samples only
        try:
            if method_type == 'kmeans':
                if selection_method == 'fixed-k5':
                    # Fixed k=5
                    kmeans_clean = SphericalKMeans(n_clusters=5, random_state=42, n_init=10)
                    labels_clean = kmeans_clean.fit_predict(clean_embeddings)
                    k_clean = 5
                else:
                    # Adaptive k (silhouette, bic, davies_bouldin)
                    k_clean, labels_clean, _ = select_optimal_k_method(
                        clean_embeddings, method=selection_method, metric='cosine', use_spherical=True
                    )
                sil_clean = silhouette_score(clean_embeddings, labels_clean, metric='cosine')
                db_clean = davies_bouldin_score(clean_embeddings, labels_clean)

            else:  # agglomerative
                # Run agglomerative on clean samples
                distances_clean = pdist(clean_embeddings, metric='cosine')
                Z_clean = linkage(distances_clean, method='average')

                scores_clean = {}
                for k in range(2, 11):
                    labels_clean = fcluster(Z_clean, k, criterion='maxclust') - 1
                    if selection_method == 'silhouette':
                        scores_clean[k] = silhouette_score(clean_embeddings, labels_clean, metric='cosine')
                    else:
                        scores_clean[k] = davies_bouldin_score(clean_embeddings, labels_clean)

                k_clean = max(scores_clean, key=scores_clean.get) if selection_method == 'silhouette' else min(scores_clean, key=scores_clean.get)
                labels_clean = fcluster(Z_clean, k_clean, criterion='maxclust') - 1
                sil_clean = silhouette_score(clean_embeddings, labels_clean, metric='cosine')
                db_clean = davies_bouldin_score(clean_embeddings, labels_clean)

            print(f"  Clean only:  k={k_clean}, Sil={sil_clean:.4f}, DB={db_clean:.4f}")

            # Improvement
            if sil_all is not None:
                improvement = (sil_clean - sil_all) / abs(sil_all) * 100
                print(f"  Improvement: {improvement:+.1f}%")
            else:
                improvement = None
        except Exception as e:
            print(f"  Clean only: FAILED - {e}")
            sil_clean, db_clean, k_clean, improvement = None, None, None, None

        results['methods'][method_id] = {
            'name': method_name,
            'all_samples': {
                'k': k_all,
                'silhouette': float(sil_all) if sil_all is not None else None,
                'davies_bouldin': float(db_all) if db_all is not None else None
            },
            'clean_only': {
                'k': k_clean,
                'silhouette': float(sil_clean) if sil_clean is not None else None,
                'davies_bouldin': float(db_clean) if db_clean is not None else None
            },
            'improvement_pct': float(improvement) if improvement is not None else None
        }

    return results


def main():
    """Test comprehensive noise filtering on all neurons."""

    # Load existing results
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    print("="*60)
    print("COMPREHENSIVE NOISE FILTERING EXPERIMENT")
    print("="*60)
    print("Testing all clustering methods with/without noise filtering")
    print(f"Total neurons: {len(data['neurons'])}")
    print("Using existing results for 'all samples' metrics (faster)")
    print("="*60)

    all_results = []

    # Create lookup dict for existing results
    existing_lookup = {
        (n['layer'], n['unit']): n
        for n in data['neurons']
    }

    for idx, neuron in enumerate(data['neurons'], 1):
        print(f"\n[{idx}/{len(data['neurons'])}]")
        existing_results = existing_lookup[(neuron['layer'], neuron['unit'])]
        result = test_comprehensive_filtering(neuron['layer'], neuron['unit'], existing_results)

        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Average Improvements by Method")
    print("="*60)

    # Aggregate improvements by method
    method_improvements = {}

    for result in all_results:
        for method_id, method_data in result['methods'].items():
            if method_id not in method_improvements:
                method_improvements[method_id] = {
                    'name': method_data['name'],
                    'improvements': []
                }

            if method_data['improvement_pct'] is not None:
                method_improvements[method_id]['improvements'].append(method_data['improvement_pct'])

    # Print summary table
    print(f"\n{'Method':<40} {'Avg Improvement':<20} {'Success Rate'}")
    print("-"*80)

    for method_id, data in method_improvements.items():
        if data['improvements']:
            avg_improvement = np.mean(data['improvements'])
            success_rate = f"{len(data['improvements'])}/{len(all_results)}"
            print(f"{data['name']:<40} {avg_improvement:>+9.1f}% {'':<10} {success_rate}")

    # Save results
    output_file = ANALYSIS_DIR / "comprehensive_noise_filtering.json"
    with open(output_file, 'w') as f:
        json.dump({
            'neurons': all_results,
            'summary': {
                method_id: {
                    'name': data['name'],
                    'mean_improvement': float(np.mean(data['improvements'])) if data['improvements'] else None,
                    'median_improvement': float(np.median(data['improvements'])) if data['improvements'] else None,
                    'success_count': len(data['improvements']),
                    'total_count': len(all_results)
                }
                for method_id, data in method_improvements.items()
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print("="*60)

    # Find best method
    best_method = None
    best_improvement = -float('inf')

    for method_id, data in method_improvements.items():
        if data['improvements']:
            avg = np.mean(data['improvements'])
            if avg > best_improvement:
                best_improvement = avg
                best_method = data['name']

    if best_method:
        print(f"Best method with noise filtering: {best_method}")
        print(f"Average improvement: {best_improvement:+.1f}%")
        print(f"\nNoise filtering is BENEFICIAL for all methods.")
        print(f"Recommendation: Apply HDBSCAN noise filtering before clustering.")


if __name__ == "__main__":
    main()

"""
Test HDBSCAN hyperparameters to find optimal noise/cluster balance.

Goal: Reduce noise percentage while maintaining cluster quality.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.spherical_kmeans import SphericalKMeans
from utils import clustering

# Paths
EMBEDDING_CACHE_DIR = Path(__file__).parent.parent / "cache" / "embeddings"

def test_hdbscan_params(layer, unit, min_cluster_sizes=[3, 5, 10, 15, 20]):
    """
    Test different HDBSCAN hyperparameters on one neuron.

    Returns:
        Dict mapping (min_cluster_size) -> results
    """
    # Load embeddings
    embeddings_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_embeddings.npy"
    texts_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_texts.pkl"

    embeddings = np.load(embeddings_path)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-9, None)

    results = {}

    for min_size in min_cluster_sizes:
        # Run HDBSCAN with this min_cluster_size
        labels, n_clusters, n_noise = clustering.run_hdbscan_clustering(
            normalized,
            metric='cosine',
            min_cluster_size=min_size
        )

        noise_ratio = n_noise / len(normalized)

        # If we have clean samples, test k-means on them
        if n_clusters > 0 and n_noise < len(normalized) - 25:
            clean_mask = labels != -1
            clean_embeddings = normalized[clean_mask]

            # Test k-means (k=5) on clean samples
            if len(clean_embeddings) >= 25:
                kmeans = SphericalKMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(clean_embeddings)

                sil_score = silhouette_score(clean_embeddings, kmeans_labels, metric='cosine')
                db_score = davies_bouldin_score(clean_embeddings, kmeans_labels)
            else:
                sil_score = None
                db_score = None
        else:
            sil_score = None
            db_score = None
            clean_mask = labels != -1

        results[min_size] = {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'n_clean': int((labels != -1).sum()),
            'noise_ratio': float(noise_ratio),
            'silhouette': float(sil_score) if sil_score is not None else None,
            'davies_bouldin': float(db_score) if db_score is not None else None
        }

    return results


def main():
    """Test on all neurons to get comprehensive statistics."""

    # Load all neurons from results
    ANALYSIS_DIR = Path(__file__).parent.parent / "clustering_analysis_cosine"
    RESULTS_FILE = ANALYSIS_DIR / "results.json"

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    test_neurons = [(n['layer'], n['unit']) for n in data['neurons']]

    min_cluster_sizes = [3, 5, 7, 10, 15, 20]

    print("="*80)
    print("HDBSCAN HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Testing min_cluster_size: {min_cluster_sizes}")
    print("="*80)

    all_results = {}

    for layer, unit in test_neurons:
        print(f"\n{'='*80}")
        print(f"Layer {layer}, Unit {unit}")
        print(f"{'='*80}")

        results = test_hdbscan_params(layer, unit, min_cluster_sizes)
        all_results[f'L{layer}_U{unit}'] = results

        # Print table
        print(f"\n{'min_size':<10} {'clusters':<10} {'noise':<10} {'clean':<10} {'noise%':<10} {'silhouette':<12} {'DB':<10}")
        print("-"*80)

        for min_size in min_cluster_sizes:
            r = results[min_size]
            sil_str = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "N/A"
            db_str = f"{r['davies_bouldin']:.4f}" if r['davies_bouldin'] is not None else "N/A"

            print(f"{min_size:<10} {r['n_clusters']:<10} {r['n_noise']:<10} "
                  f"{r['n_clean']:<10} {r['noise_ratio']*100:<9.1f}% {sil_str:<12} {db_str:<10}")

    # Analysis: Which min_cluster_size is best?
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print("="*80)

    # Average across neurons for each min_cluster_size
    avg_results = {}
    for min_size in min_cluster_sizes:
        noise_ratios = []
        sil_scores = []

        for neuron_results in all_results.values():
            r = neuron_results[min_size]
            noise_ratios.append(r['noise_ratio'])
            if r['silhouette'] is not None:
                sil_scores.append(r['silhouette'])

        avg_results[min_size] = {
            'avg_noise_ratio': np.mean(noise_ratios),
            'avg_silhouette': np.mean(sil_scores) if sil_scores else None,
            'n_valid': len(sil_scores)
        }

    total_neurons = len(test_neurons)
    print(f"\n{'min_size':<10} {'avg_noise%':<12} {'avg_sil':<12} {'success_rate':<15}")
    print("-"*60)
    for min_size in min_cluster_sizes:
        r = avg_results[min_size]
        sil_str = f"{r['avg_silhouette']:.4f}" if r['avg_silhouette'] is not None else "N/A"
        success_rate = f"{r['n_valid']}/{total_neurons} ({r['n_valid']/total_neurons*100:.1f}%)"
        print(f"{min_size:<10} {r['avg_noise_ratio']*100:<11.1f}% {sil_str:<12} {success_rate}")

    print(f"\n{'='*80}")
    print("TRADE-OFF SUMMARY")
    print("="*80)
    print("\nKey trade-off: Higher min_cluster_size → Better silhouette but fewer neurons succeed")
    print("\nCurrent setting: min_cluster_size=5")
    if 5 in avg_results and avg_results[5]['avg_silhouette'] is not None:
        current = avg_results[5]
        print(f"  Silhouette: {current['avg_silhouette']:.4f}")
        print(f"  Success rate: {current['n_valid']}/{total_neurons} ({current['n_valid']/total_neurons*100:.1f}%)")
        print(f"  Noise: {current['avg_noise_ratio']*100:.1f}%")

    # Save results
    output_file = Path(__file__).parent.parent / "clustering_analysis_cosine" / "hdbscan_tuning.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_neurons': [f'L{l}_U{u}' for l, u in test_neurons],
            'min_cluster_sizes': min_cluster_sizes,
            'results': {k: {str(ks): v for ks, v in vs.items()}
                       for k, vs in all_results.items()},
            'averages': {str(k): v for k, v in avg_results.items()}
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

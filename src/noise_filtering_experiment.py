"""
Experiment: Does filtering noise improve clustering?

Strategy:
1. Load existing HDBSCAN results
2. Remove "noise" samples (label=-1)
3. Re-cluster only the "clean" samples
4. Check if silhouette scores improve

This tests if the low silhouette scores are caused by noise.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.spherical_kmeans import SphericalKMeans

# Direct cache path (use 0.6B embeddings that were used for clustering analysis)
EMBEDDING_CACHE_DIR = Path(__file__).parent.parent / "cache" / "embeddings"

# Paths
ANALYSIS_DIR = Path(__file__).parent.parent / "clustering_analysis_cosine"
CLUSTER_DIR = ANALYSIS_DIR / "clusters"
RESULTS_FILE = ANALYSIS_DIR / "results.json"

def test_noise_filtering(layer, unit, k=5):
    """
    Test if removing HDBSCAN noise improves k-means clustering quality.

    Args:
        layer: Layer ID
        unit: Unit ID
        k: Number of clusters for k-means

    Returns:
        Dict with before/after metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing Noise Filtering: Layer {layer}, Unit {unit}")
    print(f"{'='*60}")

    # Load embeddings directly from cache
    embeddings_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_embeddings.npy"
    texts_path = EMBEDDING_CACHE_DIR / f"layer{layer}_{unit}_texts.pkl"

    if not embeddings_path.exists() or not texts_path.exists():
        print(f"ERROR: No cached embeddings at {embeddings_path}")
        return None

    embeddings = np.load(embeddings_path)
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normalized = embeddings / norms

    print(f"Total samples: {len(normalized)}")

    # Load HDBSCAN results
    hdbscan_file = CLUSTER_DIR / f"layer-{layer}_unit-{unit}_hdbscan.json"
    with open(hdbscan_file, 'r') as f:
        hdbscan_data = json.load(f)

    # Extract labels from cluster structure
    # File format: {"clusters": {"-1": [...], "0": [...], "1": [...]}}
    # Each cluster contains list of {"index": i, "text": "..."} dicts
    hdbscan_labels = np.full(len(normalized), -1, dtype=int)

    for cluster_id, items in hdbscan_data['clusters'].items():
        cluster_label = int(cluster_id)
        for item in items:
            idx = item['index']
            hdbscan_labels[idx] = cluster_label

    # Identify noise vs clean samples
    noise_mask = hdbscan_labels == -1
    clean_mask = ~noise_mask

    n_noise = noise_mask.sum()
    n_clean = clean_mask.sum()

    print(f"Noise samples: {n_noise} ({n_noise/len(normalized)*100:.1f}%)")
    print(f"Clean samples: {n_clean} ({n_clean/len(normalized)*100:.1f}%)")

    # Test 1: K-means on ALL samples
    print(f"\n[1/2] K-means on ALL samples (k={k})...")
    kmeans_all = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
    labels_all = kmeans_all.fit_predict(normalized)
    sil_all = silhouette_score(normalized, labels_all, metric='cosine')
    db_all = davies_bouldin_score(normalized, labels_all)
    print(f"  Silhouette: {sil_all:.4f}")
    print(f"  Davies-Bouldin: {db_all:.4f}")

    # Test 2: K-means on CLEAN samples only
    if n_clean >= k * 5:  # Need at least 5 samples per cluster
        print(f"\n[2/2] K-means on CLEAN samples only (k={k})...")
        clean_embeddings = normalized[clean_mask]

        kmeans_clean = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        labels_clean = kmeans_clean.fit_predict(clean_embeddings)
        sil_clean = silhouette_score(clean_embeddings, labels_clean, metric='cosine')
        db_clean = davies_bouldin_score(clean_embeddings, labels_clean)
        print(f"  Silhouette: {sil_clean:.4f}")
        print(f"  Davies-Bouldin: {db_clean:.4f}")

        # Improvement
        sil_improvement = (sil_clean - sil_all) / abs(sil_all) * 100
        print(f"\n  Improvement: {sil_improvement:+.1f}%")

        if sil_clean > sil_all:
            print(f"  ✓ Filtering noise IMPROVES clustering quality")
        else:
            print(f"  ✗ Filtering noise does NOT improve clustering")

        return {
            'layer': layer,
            'unit': unit,
            'total_samples': len(normalized),
            'noise_samples': int(n_noise),
            'clean_samples': int(n_clean),
            'noise_ratio': float(n_noise / len(normalized)),
            'all_samples': {
                'silhouette': float(sil_all),
                'davies_bouldin': float(db_all)
            },
            'clean_only': {
                'silhouette': float(sil_clean),
                'davies_bouldin': float(db_clean)
            },
            'improvement_pct': float(sil_improvement)
        }
    else:
        print(f"\n  ✗ Not enough clean samples ({n_clean}) for k={k} clustering")
        return None


def main():
    """Test noise filtering on all neurons."""

    # Load results
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    print("="*60)
    print("NOISE FILTERING EXPERIMENT")
    print("="*60)
    print("Testing if removing HDBSCAN noise improves k-means clustering")
    print(f"Total neurons: {len(data['neurons'])}")
    print("="*60)

    results = []
    for idx, neuron in enumerate(data['neurons'], 1):
        print(f"\n[{idx}/{len(data['neurons'])}]")
        result = test_noise_filtering(
            neuron['layer'],
            neuron['unit'],
            k=5
        )
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    improvements = [r['improvement_pct'] for r in results]
    print(f"Average silhouette improvement: {np.mean(improvements):+.1f}%")
    print(f"Neurons improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    if np.mean(improvements) > 10:
        print("\n✓ Filtering noise SIGNIFICANTLY improves clustering")
        print("  Recommendation: Modify pipeline to exclude HDBSCAN noise")
    elif np.mean(improvements) > 0:
        print("\n~ Filtering noise slightly improves clustering")
        print("  Recommendation: Consider filtering as optional preprocessing")
    else:
        print("\n✗ Filtering noise does NOT improve clustering")
        print("  Recommendation: Problem is not noise, but weak cluster structure")

    # Save results
    output_file = ANALYSIS_DIR / "noise_filtering_test.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_neurons': results,
            'summary': {
                'mean_improvement_pct': float(np.mean(improvements)),
                'neurons_improved': int(sum(1 for i in improvements if i > 0)),
                'total_tested': len(results)
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

"""
Cluster assignment caching utilities.

Saves cluster assignments to JSON files for later description generation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# Cluster cache directory - relative to project root
# Go up from src/utils/ to project root, then to clustering_analysis_cosine/clusters
CLUSTER_CACHE_DIR = Path(__file__).parent.parent.parent / "clustering_analysis_cosine" / "clusters"


def save_clusters(layer: int, unit: int, method: str, labels: np.ndarray, texts: List[str]):
    """
    Save cluster assignments to JSON file.

    Args:
        layer: Layer ID
        unit: Unit ID
        method: Clustering method ('silhouette', 'bic', 'davies_bouldin', 'hdbscan')
        labels: Cluster labels array
        texts: List of text strings corresponding to labels
    """
    CLUSTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Group texts by cluster
    clusters = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        label = int(label)  # Convert numpy int to Python int
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'index': idx,
            'text': text
        })

    # Count noise points (for HDBSCAN)
    n_noise = int((labels == -1).sum()) if -1 in labels else 0

    data = {
        'layer': layer,
        'unit': unit,
        'method': method,
        'n_clusters': len([k for k in clusters.keys() if k != -1]),
        'n_noise': n_noise,
        'n_samples': len(texts),
        'clusters': clusters
    }

    filepath = CLUSTER_CACHE_DIR / f"layer-{layer}_unit-{unit}_{method}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Saved clusters to {filepath.name}")


def load_clusters(layer: int, unit: int, method: str) -> Optional[Dict]:
    """
    Load cluster assignments from JSON file.

    Args:
        layer: Layer ID
        unit: Unit ID
        method: Clustering method

    Returns:
        Dictionary with cluster data, or None if not found
    """
    filepath = CLUSTER_CACHE_DIR / f"layer-{layer}_unit-{unit}_{method}.json"

    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def get_cluster_summary(layer: int, unit: int, method: str) -> Optional[Dict]:
    """
    Get summary statistics for cached clusters.

    Args:
        layer: Layer ID
        unit: Unit ID
        method: Clustering method

    Returns:
        Dictionary with summary stats, or None if not found
    """
    data = load_clusters(layer, unit, method)
    if data is None:
        return None

    cluster_sizes = {}
    for cluster_id, items in data['clusters'].items():
        cluster_sizes[cluster_id] = len(items)

    return {
        'n_clusters': data['n_clusters'],
        'n_noise': data['n_noise'],
        'n_samples': data['n_samples'],
        'cluster_sizes': cluster_sizes,
        'mean_cluster_size': np.mean(list(cluster_sizes.values())),
        'min_cluster_size': min(cluster_sizes.values()),
        'max_cluster_size': max(cluster_sizes.values())
    }


def clear_cluster_cache(layer: Optional[int] = None, unit: Optional[int] = None):
    """
    Clear cluster cache.

    Args:
        layer: If provided, only clear this layer
        unit: If provided (with layer), only clear this neuron
    """
    if not CLUSTER_CACHE_DIR.exists():
        return

    if layer is not None and unit is not None:
        # Clear specific neuron
        pattern = f"layer-{layer}_unit-{unit}_*.json"
        for filepath in CLUSTER_CACHE_DIR.glob(pattern):
            filepath.unlink()
        print(f"Cleared cluster cache for layer {layer}, unit {unit}")

    elif layer is not None:
        # Clear all neurons in layer
        pattern = f"layer-{layer}_*.json"
        for filepath in CLUSTER_CACHE_DIR.glob(pattern):
            filepath.unlink()
        print(f"Cleared cluster cache for layer {layer}")

    else:
        # Clear all cache
        for filepath in CLUSTER_CACHE_DIR.glob("*.json"):
            filepath.unlink()
        print("Cleared all cluster cache")

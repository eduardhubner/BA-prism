"""
Clusterability assessment metrics and visualizations.

Provides tools to evaluate whether embeddings form natural clusters
before applying clustering algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")


def hopkins_statistic(X, metric='cosine', sampling_size=None, random_state=42):
    """
    Compute Hopkins statistic for clustering tendency.

    The Hopkins statistic tests the spatial randomness of data. It compares
    distances between real data points to distances between random points
    and real data.

    Args:
        X: Data matrix (n_samples, n_features), should be normalized for cosine
        metric: 'cosine' or 'euclidean'
        sampling_size: Number of points to sample (default: 10% of data)
        random_state: Random seed for reproducibility

    Returns:
        H: Hopkins statistic
            ~0.5 = uniformly random (no clustering tendency)
            >0.7 = strong clustering tendency
            <0.3 = regularly spaced (rare in practice)
    """
    np.random.seed(random_state)

    if sampling_size is None:
        sampling_size = min(int(0.1 * len(X)), 100)  # Cap at 100 for efficiency

    n_samples, n_features = X.shape

    if sampling_size >= n_samples:
        sampling_size = max(n_samples // 2, 1)

    # Sample random real points
    sample_indices = np.random.choice(n_samples, sampling_size, replace=False)
    X_sample = X[sample_indices]

    # Distances for real points to nearest neighbor (excluding self)
    nbrs_real = NearestNeighbors(n_neighbors=2, metric=metric).fit(X)
    distances_real, _ = nbrs_real.kneighbors(X_sample)
    u = distances_real[:, 1]  # Exclude self (index 0 is the point itself)

    # Generate random points
    if metric == 'cosine':
        # Random points on unit sphere
        X_random = np.random.randn(sampling_size, n_features)
        X_random = X_random / np.linalg.norm(X_random, axis=1, keepdims=True)
    else:
        # Random points in bounding box
        X_random = np.random.uniform(
            X.min(axis=0), X.max(axis=0),
            size=(sampling_size, n_features)
        )

    # Distances for random points to nearest real point
    nbrs_random = NearestNeighbors(n_neighbors=1, metric=metric).fit(X)
    distances_random, _ = nbrs_random.kneighbors(X_random)
    w = distances_random[:, 0]

    # Hopkins statistic
    H = w.sum() / (w.sum() + u.sum())

    return H


def create_pca_plot(embeddings, labels=None, save_path=None, title='PCA Visualization'):
    """
    Create PCA visualization of embeddings.

    Args:
        embeddings: numpy array (n_samples, n_features)
        labels: Optional cluster labels for coloring
        save_path: If provided, save plot to this path
        title: Plot title

    Returns:
        dict with variance_explained and cumulative_variance
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Color by cluster
        unique_labels = np.unique(labels)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
    else:
        # No labels - just show distribution
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   alpha=0.6, s=30, color='steelblue')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(title)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        'variance_explained': pca.explained_variance_ratio_[:2].tolist(),
        'cumulative_variance': float(pca.explained_variance_ratio_[:2].sum())
    }


def create_umap_plot(embeddings, labels=None, metric='cosine', save_path=None,
                    title='UMAP Visualization', random_state=42):
    """
    Create UMAP visualization of embeddings.

    Args:
        embeddings: numpy array (n_samples, n_features)
        labels: Optional cluster labels for coloring
        metric: Distance metric ('cosine' or 'euclidean')
        save_path: If provided, save plot to this path
        title: Plot title
        random_state: Random seed for reproducibility

    Returns:
        dict (currently empty, could add UMAP-specific metrics)
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping UMAP visualization.")
        return {}

    umap_model = UMAP(n_components=2, metric=metric, random_state=random_state,
                     n_neighbors=15, min_dist=0.1)
    embeddings_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Color by cluster
        unique_labels = np.unique(labels)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
    else:
        # No labels - just show distribution
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   alpha=0.6, s=30, color='steelblue')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'{title} (metric={metric})')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {}


def assess_clusterability(embeddings, metric='cosine', create_plots=True, save_dir=None, neuron_id=None):
    """
    Comprehensive clusterability assessment.

    Args:
        embeddings: numpy array (n_samples, n_features), normalized for cosine
        metric: 'cosine' or 'euclidean'
        create_plots: Whether to generate PCA and UMAP plots
        save_dir: Directory to save plots
        neuron_id: String identifier for plot filenames (e.g., "layer-0_unit-440")

    Returns:
        dict with all clusterability metrics
    """
    results = {}

    # Hopkins statistic
    hopkins = hopkins_statistic(embeddings, metric=metric)
    results['hopkins_statistic'] = float(hopkins)

    # Interpretation
    if hopkins > 0.7:
        results['hopkins_interpretation'] = 'Strong clustering tendency'
    elif hopkins > 0.5:
        results['hopkins_interpretation'] = 'Moderate clustering tendency'
    else:
        results['hopkins_interpretation'] = 'Weak/no clustering tendency'

    # PCA analysis
    if create_plots and save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pca_path = save_dir / f'{neuron_id}_pca.png' if neuron_id else None
        pca_stats = create_pca_plot(embeddings, save_path=pca_path)
        results['pca'] = pca_stats

        # UMAP (if available)
        if UMAP_AVAILABLE:
            umap_path = save_dir / f'{neuron_id}_umap.png' if neuron_id else None
            umap_stats = create_umap_plot(embeddings, metric=metric, save_path=umap_path)
            results['umap'] = umap_stats
    else:
        # Just compute PCA stats without plotting
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        results['pca'] = {
            'variance_explained': pca.explained_variance_ratio_[:2].tolist(),
            'cumulative_variance': float(pca.explained_variance_ratio_[:2].sum())
        }

    return results

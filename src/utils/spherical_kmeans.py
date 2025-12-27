"""
Simple implementation of Spherical K-Means clustering.

This is a clean implementation that uses cosine similarity (via dot product on normalized vectors)
instead of Euclidean distance. Perfect for clustering embeddings.
"""

import numpy as np
from sklearn.utils import check_random_state


class SphericalKMeans:
    """Spherical K-Means clustering using cosine similarity.

    This algorithm is designed for normalized data (unit sphere). It maintains
    normalized centroids throughout the clustering process.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form
    max_iter : int, default=300
        Maximum number of iterations
    n_init : int, default=10
        Number of times to run k-means with different initializations
    random_state : int or RandomState, default=None
        Random seed for reproducibility
    tol : float, default=1e-4
        Convergence tolerance

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers (normalized)
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center
    """

    def __init__(self, n_clusters, max_iter=300, n_init=10, random_state=None, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol

    def _initialize_centers(self, X, random_state):
        """Initialize centers using k-means++ adapted for spherical space."""
        n_samples = X.shape[0]

        # Choose first center randomly
        center_indices = [random_state.randint(n_samples)]

        # Choose remaining centers
        for _ in range(1, self.n_clusters):
            # Compute cosine distances to nearest existing center
            centers = X[center_indices]
            similarities = X @ centers.T  # Cosine similarity (already normalized)
            max_similarities = similarities.max(axis=1)
            distances = 1 - max_similarities  # Convert to cosine distance

            # Choose next center with probability proportional to distance squared
            probabilities = distances ** 2
            probabilities /= probabilities.sum()

            next_center = random_state.choice(n_samples, p=probabilities)
            center_indices.append(next_center)

        return X[center_indices].copy()

    def _assign_labels(self, X, centers):
        """Assign each point to nearest center using cosine similarity."""
        # Compute cosine similarities (dot product on unit sphere)
        similarities = X @ centers.T
        # Assign to cluster with highest similarity
        labels = similarities.argmax(axis=1)
        return labels

    def _update_centers(self, X, labels):
        """Update centers as normalized mean of assigned points."""
        new_centers = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Mean of assigned points
                new_center = cluster_points.mean(axis=0)
                # Normalize to unit sphere
                norm = np.linalg.norm(new_center)
                if norm > 0:
                    new_centers[k] = new_center / norm
                else:
                    # If cluster is empty or mean is zero, keep previous center
                    new_centers[k] = self.cluster_centers_[k]
            else:
                # Empty cluster - keep previous center
                new_centers[k] = self.cluster_centers_[k]

        return new_centers

    def _compute_inertia(self, X, labels, centers):
        """Compute inertia as sum of cosine distances to assigned centers."""
        assigned_centers = centers[labels]
        # Cosine distance = 1 - cosine similarity
        similarities = (X * assigned_centers).sum(axis=1)
        distances = 1 - similarities
        return distances.sum()

    def _fit_single(self, X, random_state):
        """Run single k-means iteration."""
        # Initialize centers
        centers = self._initialize_centers(X, random_state)

        # Iterative refinement
        for iteration in range(self.max_iter):
            # Assign labels
            labels = self._assign_labels(X, centers)

            # Update centers
            new_centers = self._update_centers(X, labels)

            # Check convergence
            center_shift = np.abs(new_centers - centers).max()
            if center_shift < self.tol:
                break

            centers = new_centers

        # Compute final inertia
        inertia = self._compute_inertia(X, labels, centers)

        return labels, centers, inertia

    def fit(self, X):
        """Compute spherical k-means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (should be normalized to unit sphere)

        Returns
        -------
        self
        """
        # Normalize input (safety check)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        random_state = check_random_state(self.random_state)

        # Run multiple initializations
        best_inertia = None
        best_labels = None
        best_centers = None

        for _ in range(self.n_init):
            labels, centers, inertia = self._fit_single(X, random_state)

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        # Store results
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (should be normalized to unit sphere)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data (should be normalized to unit sphere)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        # Normalize input
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Assign to nearest center
        return self._assign_labels(X, self.cluster_centers_)

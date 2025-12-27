from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score #new
from utils import config
from utils.spherical_kmeans import SphericalKMeans #new
import numpy as np #new
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def run_agglomerative_clustering(corpus_embeddings, n_clusters, metric = 'cosine', linkage = 'average'):
    """
    Run Agglomerative clustering.
    
    Args:
        corpus_embeddings: Normalized embeddings
        n_clusters: Number of clusters
        metric: Distance metric ('cosine' or 'euclidean')
        linkage: Linkage method ('average', 'complete', 'single')
    
    Returns:
        labels: Cluster labels
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)

    labels = clustering.fit_predict(corpus_embeddings)

    return labels

def select_optimal_k_agglomerative(corpus_embeddings, k_min, k_max, method, metric = 'cosine'):
    """
    Select optimal k using specified method.

    Args:
        corpus_embeddings: np.ndarray of embeddings
        k_min: Minimum k to test
        k_max: Maximum k to test
        method: "davies_bouldin" or "silhouette"

    Returns:
        optimal_k: int
        scores: dict mapping k -> score
        labels: Cluster labels for optimal k
    """
    n_samples, n_features = corpus_embeddings.shape

    # Don't test k larger than number of samples
    k_max_adjusted = min(k_max, n_samples - 1)

    if k_max_adjusted < k_min:
        return k_min, {}, np.zeros(n_samples, dtype=int)

    corpus_embeddings = corpus_embeddings/ np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    #Build dedrogram
    #compute pairwise distance
    distance = pdist(corpus_embeddings, metric =metric)
    #build linkage matrix (dendrogram)
    Z = linkage(distance, method ='average')

    scores = {}
    all_labels = {}  # Store labels for each k

    #cutting dedrogram
    for k in range(k_min, k_max_adjusted+1):
        labels = fcluster(Z, k, criterion= 'maxclust')
        labels = labels -1 # convert to 0-indexed
        all_labels[k] = labels  # Store labels

        #compute metrics
        if method == "silhouette":
            #higher is better
            scores[k]= silhouette_score(corpus_embeddings, labels, metric =metric)
        elif method == "davies_bouldin":
            #lower is better
            scores[k] = davies_bouldin_score(corpus_embeddings, labels)
        else:
            raise ValueError(f"Unknown method: {method}")

    if method == "silhouette":
        optimal_k = max(scores, key= scores.get)
    else:
        optimal_k = min(scores, key= scores.get) #davies_bouldin

    return optimal_k, scores, all_labels[optimal_k]
        
    

def select_optimal_k(corpus_embeddings, k_min, k_max, method):
    """
    Select optimal k using specified method.

    Args:
        corpus_embeddings: np.ndarray of embeddings
        k_min: Minimum k to test
        k_max: Maximum k to test
        method: "davies_bouldin", "bic", or "silhouette"

    Returns:
        optimal_k: int
        scores: dict mapping k -> score
        labels: Cluster labels for optimal k
    """
    n_samples, n_features = corpus_embeddings.shape

    # Don't test k larger than number of samples
    k_max_adjusted = min(k_max, n_samples - 1)

    if k_max_adjusted < k_min:
        return k_min, {}, np.zeros(n_samples, dtype=int)

    scores = {}
    all_labels = {}  # Store labels for each k

    # Normalize embeddings if using spherical k-means
    if config.SPHERICAL:
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )

    for k in range(k_min, k_max_adjusted + 1):
        # Choose clustering algorithm based on SPHERICAL flag
        if config.SPHERICAL:
            kmeans = SphericalKMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = kmeans.fit_predict(corpus_embeddings)
        all_labels[k] = labels  # Store labels

        if method == "silhouette":
            # Higher is better
            # Use cosine metric if SPHERICAL, otherwise euclidean
            metric = 'cosine' if config.SPHERICAL else 'euclidean'
            scores[k] = silhouette_score(corpus_embeddings, labels, metric=metric)

        elif method == "davies_bouldin":
            # Lower is better
            scores[k] = davies_bouldin_score(corpus_embeddings, labels)

        elif method == "bic":
            # Lower is better
            n_parameters = k * n_features
            inertia = max(kmeans.inertia_, 1e-10)
            variance = inertia / (n_samples - k)
            log_likelihood = -n_samples * np.log(variance) - inertia / (2 * variance)
            bic = -2 * log_likelihood + n_parameters * np.log(n_samples)

            scores[k] = bic

        else:
            raise ValueError(f"Unknown method: {method}")

    # Select optimal k based on method
    if method == "silhouette":
        optimal_k = max(scores, key=scores.get)  # Higher is better
    else:  # davies_bouldin or bic
        optimal_k = min(scores, key=scores.get)  # Lower is better

    return optimal_k, scores, all_labels[optimal_k]

def apply(texts: list[str], activations: list[float] | None = None) -> list[list[dict]]:
    """Apply clustering to a list of texts using the specified method.

    This function encodes the input texts into embeddings and applies clustering
    to group similar texts together.

    Args:
        texts: A list of strings to be clustered.
        embedder: A SentenceTransformer object used to encode the texts into embeddings.
        method: The clustering algorithm to use. Options are:
            - "KMeans": Uses KMeans clustering
            - "HDBSCAN": Uses HDBSCAN clustering

    Returns:
        clustered_sentences: A list of clusters, where each cluster is a list of strings from the
          original texts list that belong to the same cluster.
        new: metadata: Dict with clustering information (including adaptive k if used)

    Raises:
        ValueError: If an unknown clustering method is specified.

    Note:
        Adapted from:
        https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/kmeans.py
    """
    method = config.CLUSTER_METHOD

    #encode texts (with caching)
    if config.CLUSTER_EMBEDDING_MODEL_NAME == None:
        corpus_embeddings = activations.cpu()
    else:
        # Use caching to avoid regenerating embeddings for the same neuron
        from . import cache
        corpus_embeddings = cache.get_or_generate_embeddings(
            texts=texts,
            layer_id=config.LAYER_ID,
            unit_id=config.UNIT_ID,
            embedding_model_name=config.CLUSTER_EMBEDDING_MODEL_NAME,
            batch_size=config.BATCH_SIZE,
            max_seq_length=config.CLUSTER_MAX_SEQ_LEN,
            force_regenerate=False
        )

    # Normalize embeddings if using spherical k-means
    if config.SPHERICAL:
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )

    # Select k (new)
    metadata = {}
    if config.ADAPTIVE_K:
        optimal_k, k_scores = select_optimal_k(
            corpus_embeddings,
            k_min=config.ADAPTIVE_K_MIN,
            k_max=config.ADAPTIVE_K_MAX,
            method=config.ADAPTIVE_K_METHOD
        )
        n_cluster = optimal_k
        metadata["adaptive_k_selected"] = optimal_k
        metadata["adaptive_k_method"] = config.ADAPTIVE_K_METHOD
        metadata["k_selection_scores"] = k_scores
    else:
        n_cluster = config.N_CLUSTERS
        metadata["adaptive_k_selected"] = None

    # Apply clustering
    if method == "KMeans":
        # Choose algorithm based on SPHERICAL flag
        if config.SPHERICAL:
            clustering_model = SphericalKMeans(n_clusters=n_cluster, random_state=42)
        else:
            clustering_model = KMeans(n_clusters=n_cluster, random_state=42)

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
    elif method == "HDBSCAN":
        clustering_model = HDBSCAN(min_cluster_size=2)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        n_cluster = len(set(cluster_assignment)) - (
            1 if -1 in cluster_assignment else 0
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clustered_sentences: list[list[dict]] = [[] for _ in range(n_cluster)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(
            {
                "sentence_id": sentence_id,
                "text": texts[sentence_id],
            }
        )

    return clustered_sentences, metadata #(new)


def run_hdbscan_clustering(corpus_embeddings, metric='cosine', min_cluster_size=5):
    """
    Run HDBSCAN clustering.

    Args:
        corpus_embeddings: Normalized embeddings (for cosine)
        metric: Distance metric ('cosine' or 'euclidean')
        min_cluster_size: Minimum cluster size

    Returns:
        labels: Cluster labels (-1 = noise)
        n_clusters: Number of clusters found
        n_noise: Number of noise points
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(corpus_embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    return labels, n_clusters, n_noise


def get_cosine_similarity(descriptions, embedder):
    """Compute the cosine similarity matrix and average similarity score for a list of n descriptions.

    Args:
        descriptions (list): List of n strings.

    Returns:
        tuple: (similarity_matrix (torch.Tensor), average_similarity (float))
    """
    corpus_embeddings = embedder.encode(
        descriptions, batch_size=config.BATCH_SIZE, convert_to_tensor=True
    )

    # Compute pairwise cosine similarity matrix
    similarity_matrix = util.pytorch_cos_sim(corpus_embeddings, corpus_embeddings)

    # Compute average similarity excluding the diagonal
    n = similarity_matrix.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity_matrix[i, j].item()
            count += 1
    average_similarity = total / count

    return similarity_matrix, average_similarity

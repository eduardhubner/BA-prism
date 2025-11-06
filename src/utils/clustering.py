from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score #new
from utils import config
import numpy as np #new

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
    """
    n_samples, n_features = corpus_embeddings.shape

    # Don't test k larger than number of samples
    k_max_adjusted = min(k_max, n_samples - 1)

    if k_max_adjusted < k_min:
        return k_min, {}

    scores = {}

    for k in range(k_min, k_max_adjusted + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(corpus_embeddings)
        
        if method == "silhouette":
            # Higher is better
            scores[k] = silhouette_score(corpus_embeddings, labels)
        
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
    
    return optimal_k, scores

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

    #encode texts
    if config.CLUSTER_EMBEDDING_MODEL_NAME == None:
        corpus_embeddings = activations.cpu()
    else:
        embedder = SentenceTransformer(
            config.CLUSTER_EMBEDDING_MODEL_NAME, trust_remote_code=True
        )
        embedder.max_seq_length = config.CLUSTER_MAX_SEQ_LEN
        corpus_embeddings = embedder.encode(texts, batch_size=config.BATCH_SIZE)

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

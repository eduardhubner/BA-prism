"""
Caching utilities for experiment acceleration.

Caches embeddings so they can be reused across multiple k-selection methods
for the same neuron, ensuring fair comparison and faster execution.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Cache directory - relative to project root
# Go up from src/utils/ to project root, then to cache/embeddings
# NOTE: Change to "embeddings_4B" to use 4B model embeddings
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "embeddings_4B"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_paths(layer_id: int, unit_id: int) -> Tuple[Path, Path]:
    """
    Get cache file paths for a specific neuron.

    Args:
        layer_id: Layer ID
        unit_id: Unit ID

    Returns:
        (embeddings_path, texts_path): Paths to cache files
    """
    base_name = f"layer{layer_id}_{unit_id}"
    embeddings_path = CACHE_DIR / f"{base_name}_embeddings.npy"
    texts_path = CACHE_DIR / f"{base_name}_texts.pkl"

    return embeddings_path, texts_path


def load_cached_embeddings(
    layer_id: int,
    unit_id: int
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Load cached embeddings and texts for a neuron.

    Args:
        layer_id: Layer ID
        unit_id: Unit ID

    Returns:
        (embeddings, texts) if cache exists, None otherwise
    """
    embeddings_path, texts_path = get_cache_paths(layer_id, unit_id)

    # Check if both cache files exist
    if not embeddings_path.exists() or not texts_path.exists():
        logger.info(f"No cache found for layer {layer_id}, unit {unit_id}")
        return None

    try:
        # Load embeddings
        embeddings = np.load(embeddings_path)

        # Load texts
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)

        logger.info(
            f"✓ Loaded cached embeddings for layer {layer_id}, unit {unit_id} "
            f"(shape: {embeddings.shape})"
        )

        return embeddings, texts

    except Exception as e:
        logger.warning(f"Error loading cache: {e}. Will regenerate embeddings.")
        return None


def save_embeddings_to_cache(
    layer_id: int,
    unit_id: int,
    embeddings: np.ndarray,
    texts: List[str]
) -> None:
    """
    Save embeddings and texts to cache.

    Args:
        layer_id: Layer ID
        unit_id: Unit ID
        embeddings: Numpy array of embeddings (n_samples, embedding_dim)
        texts: List of text strings
    """
    embeddings_path, texts_path = get_cache_paths(layer_id, unit_id)

    try:
        # Save embeddings as numpy array
        np.save(embeddings_path, embeddings)

        # Save texts as pickle
        with open(texts_path, 'wb') as f:
            pickle.dump(texts, f)

        logger.info(
            f"✓ Cached embeddings for layer {layer_id}, unit {unit_id} "
            f"(shape: {embeddings.shape})"
        )

    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def get_or_generate_embeddings(
    texts: List[str],
    layer_id: int,
    unit_id: int,
    embedding_model_name: str,
    batch_size: int = 32,
    max_seq_length: int = 8192,
    force_regenerate: bool = False
) -> np.ndarray:
    """
    Get embeddings from cache or generate them.

    This is the main caching function that should be used in the experiment.

    Args:
        texts: List of text strings to embed
        layer_id: Layer ID (for cache naming)
        unit_id: Unit ID (for cache naming)
        embedding_model_name: Name of embedding model
        batch_size: Batch size for encoding
        max_seq_length: Max sequence length for model
        force_regenerate: If True, ignore cache and regenerate embeddings

    Returns:
        embeddings: Numpy array of shape (n_texts, embedding_dim)
    """
    # Try to load from cache (unless forced to regenerate)
    if not force_regenerate:
        cached = load_cached_embeddings(layer_id, unit_id)
        if cached is not None:
            cached_embeddings, cached_texts = cached

            # Verify texts match (safety check)
            if len(cached_texts) == len(texts):
                # Spot check: compare first and last texts
                if cached_texts[0] == texts[0] and cached_texts[-1] == texts[-1]:
                    logger.info("Cache validated. Using cached embeddings.")
                    return cached_embeddings
                else:
                    logger.warning(
                        "Cached texts don't match current texts. Regenerating embeddings."
                    )
            else:
                logger.warning(
                    f"Cached text count ({len(cached_texts)}) doesn't match "
                    f"current count ({len(texts)}). Regenerating embeddings."
                )

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    logger.info(f"Using model: {embedding_model_name}")

    embedder = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    embedder.max_seq_length = max_seq_length

    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)

    logger.info(f"✓ Generated embeddings with shape: {embeddings.shape}")

    # Save to cache
    save_embeddings_to_cache(layer_id, unit_id, embeddings, texts)

    return embeddings


def clear_cache(layer_id: Optional[int] = None, unit_id: Optional[int] = None) -> None:
    """
    Clear embedding cache.

    Args:
        layer_id: If provided, only clear cache for this layer
        unit_id: If provided (with layer_id), only clear cache for this neuron
    """
    if layer_id is not None and unit_id is not None:
        # Clear specific neuron
        embeddings_path, texts_path = get_cache_paths(layer_id, unit_id)
        if embeddings_path.exists():
            embeddings_path.unlink()
        if texts_path.exists():
            texts_path.unlink()
        logger.info(f"Cleared cache for layer {layer_id}, unit {unit_id}")

    elif layer_id is not None:
        # Clear all neurons in layer
        pattern = f"layer{layer_id}_*"
        for path in CACHE_DIR.glob(pattern):
            path.unlink()
        logger.info(f"Cleared cache for layer {layer_id}")

    else:
        # Clear all cache
        for path in CACHE_DIR.glob("*"):
            path.unlink()
        logger.info("Cleared all cache")


def get_cache_stats() -> dict:
    """
    Get statistics about cache contents.

    Returns:
        Dictionary with cache statistics
    """
    embedding_files = list(CACHE_DIR.glob("*_embeddings.npy"))
    text_files = list(CACHE_DIR.glob("*_texts.pkl"))

    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*"))

    return {
        'num_neurons_cached': len(embedding_files),
        'num_embedding_files': len(embedding_files),
        'num_text_files': len(text_files),
        'total_size_mb': total_size / (1024 * 1024),
        'cache_dir': str(CACHE_DIR)
    }

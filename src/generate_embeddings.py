"""
Generate embeddings for all cached neurons with a specified embedding model.

This script loads the high-activation text samples for each neuron and generates
embeddings using the specified model, caching them to a separate directory for
comparison with the original embeddings.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from utils import cache
from sentence_transformers import SentenceTransformer

# All neurons with cached text samples
# TEST MODE: Only testing with one neuron first
NEURONS = [
    (0, 440)  # Test with first neuron
]

# Full list (uncomment to run all 60):
# NEURONS = [
#     (0, 1149),(0, 1749),(0, 2725),(0, 3057),(0, 3124),(0, 3279),
#     (0, 3533),(0, 3696),(0, 3943),(0, 4297),(0, 4405),(0, 440),
#     (0, 4679),(0, 4842),(0, 5085),(0, 5551),(0, 5781),(0, 5960),
#     (0, 6114),(0, 6314),
#     (20, 1004),(20, 1406),(20, 1424),(20, 1790),(20, 2325),(20, 2679),
#     (20, 2700),(20, 2869),(20, 2988),(20, 328),(20, 3313),(20, 3885),
#     (20, 4268),(20, 4447),(20, 4683),(20, 5278),(20, 5662),(20, 5741),
#     (20, 5789),(20, 6045),
#     (40, 1516),(40, 1555),(40, 1823),(40, 183),(40, 3254),(40, 3515),
#     (40, 3612),(40, 3636),(40, 364),(40, 3948),(40, 4055),(40, 4244),
#     (40, 4808),(40, 4965),(40, 5557),(40, 556),(40, 6067),(40, 6106),
#     (40, 6364),(40, 824)
# ]

# Embedding model configuration
EMBEDDING_MODEL = "dunzhang/stella_en_1.5B_v5"  # 1.5B params - best realistic option
BATCH_SIZE = 8  # Larger batch with VSCode closed (more RAM available)
MAX_SEQ_LENGTH = 512

# New cache directory for Stella 1.5B embeddings (separate from 0.6B embeddings)
NEW_CACHE_DIR = Path(__file__).parent.parent / "cache" / "embeddings_stella_1.5B"
NEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_embeddings_for_neuron(layer, unit, embedder, force_regenerate=False):
    """
    Generate and cache embeddings for one neuron.

    Args:
        layer: Layer ID
        unit: Unit ID
        embedder: SentenceTransformer model
        force_regenerate: If True, regenerate even if cache exists

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Layer {layer}, Unit {unit}")
    print(f"{'='*60}")

    # Check if embeddings already exist in new cache directory
    new_embeddings_path = NEW_CACHE_DIR / f"layer{layer}_{unit}_embeddings.npy"
    new_texts_path = NEW_CACHE_DIR / f"layer{layer}_{unit}_texts.pkl"

    if not force_regenerate and new_embeddings_path.exists():
        embeddings = np.load(new_embeddings_path)
        print(f"  ✓ Embeddings already cached ({embeddings.shape[0]} samples, dim={embeddings.shape[1]})")
        return True

    # Load texts from the original 0.6B cache
    # Note: We hardcode the path here since cache.CACHE_DIR now points to embeddings_4B
    old_cache_dir = Path(__file__).parent.parent / "cache" / "embeddings"
    old_texts_path = old_cache_dir / f"layer{layer}_{unit}_texts.pkl"

    if not old_texts_path.exists():
        print(f"  ERROR: No text samples found at {old_texts_path}")
        print(f"         Make sure original 0.6B embeddings were generated first")
        return False

    # Load texts
    try:
        with open(old_texts_path, 'rb') as f:
            texts = pickle.load(f)
    except Exception as e:
        print(f"  ERROR: Failed to load texts: {e}")
        return False

    # Validate text count
    if len(texts) == 0:
        print(f"  ERROR: No texts found in cache file")
        return False

    print(f"  Loaded {len(texts)} text samples")
    print(f"  Generating embeddings with {EMBEDDING_MODEL}...")

    # Generate embeddings
    try:
        embeddings = embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"  ✓ Generated embeddings: {embeddings.shape}")

        # Save to new cache directory
        np.save(new_embeddings_path, embeddings)
        with open(new_texts_path, 'wb') as f:
            pickle.dump(texts, f)

        print(f"  ✓ Cached to {NEW_CACHE_DIR.name}/")

        return True

    except Exception as e:
        print(f"  ERROR: Failed to generate embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Generate embeddings for all neurons."""
    print("\n" + "="*60)
    print("EMBEDDING GENERATION PIPELINE")
    print("="*60)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"Total Neurons: {len(NEURONS)}")
    print("="*60)

    # Load embedding model
    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embedder.max_seq_length = MAX_SEQ_LENGTH
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {embedder.get_sentence_embedding_dimension()}")

    # Track statistics
    start_time = datetime.now()
    successful = 0
    failed = 0
    skipped = 0

    # Generate embeddings for each neuron
    for idx, (layer, unit) in enumerate(NEURONS, 1):
        print(f"\n[{idx}/{len(NEURONS)}]")

        result = generate_embeddings_for_neuron(
            layer, unit, embedder, force_regenerate=False
        )

        if result:
            successful += 1
        else:
            failed += 1

    # Print summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total neurons: {len(NEURONS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed}")
    print(f"Average time per neuron: {elapsed / len(NEURONS)}")
    print("="*60)

    # Show cache stats
    cache_stats = cache.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Neurons cached: {cache_stats['num_neurons_cached']}")
    print(f"  Total cache size: {cache_stats['total_size_mb']:.1f} MB")
    print(f"  Cache directory: {cache_stats['cache_dir']}")

    if failed > 0:
        print(f"\n⚠️  {failed} neurons failed - check errors above")
    else:
        print(f"\n✓ All embeddings generated successfully!")
        print(f"\nNext step: Run clustering analysis with:")
        print(f"  python src/clustering_analysis.py")


if __name__ == "__main__":
    main()

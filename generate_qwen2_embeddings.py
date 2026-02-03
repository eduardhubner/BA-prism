"""
Generate embeddings using the original PRISM embedding model (Qwen2-1.5B).

This script:
1. Loads pre-sampled text data for all neurons from data/candidate_inputs_decoded/
2. Generates embeddings using Alibaba-NLP/gte-Qwen2-1.5B-instruct
3. Saves embeddings to embeddings_qwen2/ directory for later clustering analysis

Usage (in venv_qwen2):
    python generate_qwen2_embeddings.py
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configuration
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MAX_SEQ_LEN = 8192  # Original PRISM config
OUTPUT_DIR = Path("embeddings_qwen2")
DATA_DIR = Path("data/candidate_inputs_decoded")
BATCH_SIZE = 32

# Neurons to process (59 remaining neurons - excluding L0_U440 already processed in test)
NEURONS = [
    # Layer 0 (excluding 440 - already processed)
    (0, 1149), (0, 1749), (0, 2725), (0, 3057), (0, 3124), (0, 3279),
    (0, 3533), (0, 3696), (0, 3943), (0, 4297), (0, 4405), (0, 4679), (0, 4842),
    (0, 5085), (0, 5551), (0, 5781), (0, 5960), (0, 6114), (0, 6314),
    # Layer 20
    (20, 328), (20, 1004), (20, 1406), (20, 1424), (20, 1790), (20, 2325), (20, 2679),
    (20, 2700), (20, 2869), (20, 2988), (20, 3313), (20, 3885), (20, 4268), (20, 4447),
    (20, 4683), (20, 5278), (20, 5662), (20, 5741), (20, 5789), (20, 6045),
    # Layer 40
    (40, 183), (40, 364), (40, 556), (40, 824), (40, 1516), (40, 1555), (40, 1823),
    (40, 3254), (40, 3515), (40, 3612), (40, 3636), (40, 3948), (40, 4055), (40, 4244),
    (40, 4808), (40, 4965), (40, 5557), (40, 6067), (40, 6106), (40, 6364)
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text_samples(layer_id: int, unit_id: int) -> list[str]:
    """Load pre-sampled text data for a neuron."""
    file_path = DATA_DIR / f"layer{layer_id}_{unit_id}.json"

    if not file_path.exists():
        logger.warning(f"Data file not found: {file_path}")
        return []

    with open(file_path, 'r') as f:
        texts = json.load(f)

    # Data files contain a list of text samples directly
    logger.info(f"Loaded {len(texts)} text samples for L{layer_id}_U{unit_id}")

    return texts


def generate_embeddings(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Generate embeddings for text samples."""
    if not texts:
        return np.array([])

    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def save_embeddings(embeddings: np.ndarray, layer_id: int, unit_id: int):
    """Save embeddings to disk."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / f"gpt2-xl_layer-{layer_id}_unit-{unit_id}.npy"
    np.save(output_file, embeddings)

    logger.info(f"Saved embeddings to {output_file} - shape: {embeddings.shape}")


def main():
    """Generate and save Qwen2-1.5B embeddings for all neurons."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("Qwen2-1.5B Embedding Generation")
    logger.info("=" * 80)
    logger.info(f"Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"Neurons to process: {len(NEURONS)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")

    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.error("Make sure pre-sampled data exists before generating embeddings.")
        return

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    embedder.max_seq_length = MAX_SEQ_LEN
    logger.info(f"Model loaded - max_seq_length: {MAX_SEQ_LEN}")
    logger.info("")

    # Process each neuron
    results = []
    for i, (layer_id, unit_id) in enumerate(NEURONS, 1):
        logger.info(f"[{i}/{len(NEURONS)}] Processing L{layer_id}_U{unit_id}")

        # Load text samples
        texts = load_text_samples(layer_id, unit_id)
        if not texts:
            logger.warning(f"Skipping L{layer_id}_U{unit_id} - no text samples found")
            results.append({
                'layer': layer_id,
                'unit': unit_id,
                'status': 'skipped',
                'num_samples': 0,
                'embedding_shape': None
            })
            continue

        # Generate embeddings
        try:
            embeddings = generate_embeddings(embedder, texts)

            # Save embeddings
            save_embeddings(embeddings, layer_id, unit_id)

            results.append({
                'layer': layer_id,
                'unit': unit_id,
                'status': 'success',
                'num_samples': len(texts),
                'embedding_shape': embeddings.shape
            })

        except Exception as e:
            logger.error(f"Error processing L{layer_id}_U{unit_id}: {e}")
            results.append({
                'layer': layer_id,
                'unit': unit_id,
                'status': 'error',
                'error': str(e)
            })

        logger.info("")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    logger.info(f"Total neurons: {len(NEURONS)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
    logger.info("")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_file = OUTPUT_DIR / f"embedding_generation_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")

    if error_count > 0:
        logger.warning(f"\n{error_count} neurons had errors. Check log above for details.")


if __name__ == "__main__":
    main()

"""
STEP 1: Generate embeddings for original PRISM descriptions.

This script:
1. Loads all descriptions from original PRISM evaluation files
2. Creates a mapping file with unique IDs for each description
3. Generates embeddings using Alibaba-NLP/gte-Qwen2-1.5B-instruct
4. Saves embeddings with traceability to original descriptions

Run this in venv_qwen2 (with sentence_transformers installed):
    source venv_qwen2/bin/activate
    python src/step1_generate_description_embeddings.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MAX_SEQ_LEN = 512
BATCH_SIZE = 32

MODELS = [
    "gpt2-xl",
    "Llama-3.1-8B-Instruct",
    "gemma-scope-2b",
    "gpt2-small-sae"
]

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "description_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_description_manifest():
    """
    Create a manifest file mapping each description to a unique ID.

    Returns:
        DataFrame with columns:
        - desc_id: Unique ID (e.g., "gpt2-xl_L0_U440_D1")
        - model: Model name
        - layer: Layer number
        - unit: Unit number
        - desc_idx: Description index (1-5)
        - description: Full text
        - auc: AUC score
        - mad: MAD score
        - p_value: Statistical p-value
    """
    all_descriptions = []

    for model_name in MODELS:
        eval_file = (
            RESULTS_DIR /
            f"cosy-evaluation_target-{model_name}_textgen-gemini-1-5-pro_"
            f"mean_evalgen-gemini-1-5-pro_cosmopedia_1000.csv"
        )

        if not eval_file.exists():
            print(f"Warning: File not found for {model_name}: {eval_file}")
            continue

        print(f"Loading {model_name}...")
        df = pd.read_csv(eval_file)

        # Group by neuron and assign description indices
        for (layer, unit), group in df.groupby(['layer', 'unit']):
            # Sort by AUC (descending) for consistent ordering
            group = group.sort_values('AUC', ascending=False).reset_index(drop=True)

            if len(group) != 5:
                print(f"  Warning: Neuron {model_name} L{layer}_U{unit} has {len(group)} descriptions (expected 5)")
                continue

            for idx, row in group.iterrows():
                desc_id = f"{model_name}_L{layer}_U{unit}_D{idx+1}"

                all_descriptions.append({
                    'desc_id': desc_id,
                    'model': model_name,
                    'layer': layer,
                    'unit': unit,
                    'desc_idx': idx + 1,  # 1-indexed
                    'description': row['explanation'],
                    'auc': row['AUC'],
                    'mad': row['MAD'],
                    'p_value': row['p']
                })

    manifest_df = pd.DataFrame(all_descriptions)
    print(f"\nCreated manifest with {len(manifest_df)} descriptions from {manifest_df['model'].nunique()} models")

    return manifest_df


def generate_embeddings(manifest_df: pd.DataFrame, embedder) -> dict:
    """
    Generate embeddings for all descriptions.

    Args:
        manifest_df: DataFrame with description manifest
        embedder: SentenceTransformer model

    Returns:
        Dictionary mapping desc_id -> embedding array
    """
    print(f"\nGenerating embeddings for {len(manifest_df)} descriptions...")

    # Extract descriptions in order
    descriptions = manifest_df['description'].tolist()
    desc_ids = manifest_df['desc_id'].tolist()

    # Generate embeddings in batches with progress bar
    embeddings = {}

    for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Embedding"):
        batch_descs = descriptions[i:i+BATCH_SIZE]
        batch_ids = desc_ids[i:i+BATCH_SIZE]

        # Generate embeddings
        batch_embeddings = embedder.encode(
            batch_descs,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Store with IDs
        for desc_id, embedding in zip(batch_ids, batch_embeddings):
            embeddings[desc_id] = embedding

    return embeddings


def save_embeddings(embeddings: dict, manifest_df: pd.DataFrame):
    """
    Save embeddings and manifest.

    Creates:
    - description_manifest.csv: Full manifest with all metadata
    - embeddings.npz: All embeddings in a compressed numpy archive
    - embedding_index.csv: Mapping from desc_id to array index (for loading)
    """
    # Save manifest
    manifest_file = OUTPUT_DIR / "description_manifest.csv"
    manifest_df.to_csv(manifest_file, index=False)
    print(f"\n✓ Saved manifest to: {manifest_file}")

    # Prepare embeddings for saving
    # Store in consistent order matching manifest
    desc_ids = manifest_df['desc_id'].tolist()
    embedding_matrix = np.array([embeddings[desc_id] for desc_id in desc_ids])

    # Save embeddings as compressed numpy archive
    embeddings_file = OUTPUT_DIR / "embeddings.npz"
    np.savez_compressed(
        embeddings_file,
        embeddings=embedding_matrix,
        desc_ids=desc_ids
    )
    print(f"✓ Saved embeddings to: {embeddings_file}")
    print(f"  Shape: {embedding_matrix.shape}")
    print(f"  Size: {embeddings_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Save index for easy lookup
    index_df = pd.DataFrame({
        'desc_id': desc_ids,
        'array_index': range(len(desc_ids))
    })
    index_file = OUTPUT_DIR / "embedding_index.csv"
    index_df.to_csv(index_file, index=False)
    print(f"✓ Saved embedding index to: {index_file}")


def print_summary(manifest_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nTotal descriptions: {len(manifest_df)}")
    print(f"Models: {manifest_df['model'].nunique()}")

    for model in manifest_df['model'].unique():
        model_data = manifest_df[manifest_df['model'] == model]
        n_neurons = len(model_data.groupby(['layer', 'unit']))
        print(f"  {model}: {len(model_data)} descriptions from {n_neurons} neurons")

    print(f"\nLayers: {sorted(manifest_df['layer'].unique())}")
    print(f"Descriptions per neuron: {manifest_df.groupby(['model', 'layer', 'unit']).size().value_counts().to_dict()}")

    print(f"\nAUC statistics:")
    print(f"  Mean: {manifest_df['auc'].mean():.4f}")
    print(f"  Median: {manifest_df['auc'].median():.4f}")
    print(f"  Range: [{manifest_df['auc'].min():.4f}, {manifest_df['auc'].max():.4f}]")


def main():
    print("="*80)
    print("STEP 1: GENERATE DESCRIPTION EMBEDDINGS")
    print("="*80)
    print(f"\nEmbedding model: {EMBEDDING_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Step 1: Create manifest
    print("\n" + "-"*80)
    print("Creating description manifest...")
    print("-"*80)
    manifest_df = create_description_manifest()

    # Step 2: Load embedding model
    print("\n" + "-"*80)
    print("Loading embedding model...")
    print("-"*80)
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embedder.max_seq_length = MAX_SEQ_LEN
    print(f"✓ Model loaded: {embedder.get_sentence_embedding_dimension()} dimensions")

    # Step 3: Generate embeddings
    print("\n" + "-"*80)
    print("Generating embeddings...")
    print("-"*80)
    embeddings = generate_embeddings(manifest_df, embedder)

    # Step 4: Save everything
    print("\n" + "-"*80)
    print("Saving results...")
    print("-"*80)
    save_embeddings(embeddings, manifest_df)

    # Step 5: Print summary
    print_summary(manifest_df)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print("\nNext step: Run step2_compute_pairwise_similarities.py")
    print("(Can be run in any environment with numpy/pandas)")


if __name__ == "__main__":
    main()

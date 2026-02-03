"""
STEP 2: Compute pairwise similarities from embeddings.

This script:
1. Loads the description manifest and embeddings from Step 1
2. Computes all pairwise similarities for each neuron (10 pairs per neuron)
3. Computes random baseline similarities
4. Saves unified dataset ready for polysemanticity scoring

Run this in any environment with numpy/pandas/scipy:
    python src/step2_compute_pairwise_similarities.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "description_embeddings"
OUTPUT_FILE = RESULTS_DIR / "pairwise_similarities_dataset.csv"
BASELINE_FILE = RESULTS_DIR / "random_baseline_similarities.csv"


def load_embeddings():
    """
    Load embeddings and manifest from Step 1.

    Returns:
        manifest_df: DataFrame with description metadata
        embeddings: dict mapping desc_id -> embedding array
    """
    print("Loading embeddings and manifest...")

    # Load manifest
    manifest_file = EMBEDDINGS_DIR / "description_manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}\nRun step1_generate_description_embeddings.py first!")

    manifest_df = pd.read_csv(manifest_file)
    print(f"  ✓ Loaded manifest: {len(manifest_df)} descriptions")

    # Load embeddings
    embeddings_file = EMBEDDINGS_DIR / "embeddings.npz"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}\nRun step1_generate_description_embeddings.py first!")

    data = np.load(embeddings_file)
    embedding_matrix = data['embeddings']
    desc_ids = data['desc_ids']

    # Create dict for easy lookup
    embeddings = {desc_id: embedding_matrix[i] for i, desc_id in enumerate(desc_ids)}
    print(f"  ✓ Loaded embeddings: {embedding_matrix.shape}")

    # Verify consistency
    if len(embeddings) != len(manifest_df):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(manifest_df)} descriptions!")

    return manifest_df, embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    # Using 1 - cosine_distance for consistency with sentence_transformers
    return 1.0 - cosine(vec1, vec2)


def compute_pairwise_similarities_for_neuron(desc_ids, embeddings):
    """
    Compute all 10 pairwise similarities for a neuron's 5 descriptions.

    Args:
        desc_ids: List of 5 description IDs
        embeddings: Dict mapping desc_id -> embedding

    Returns:
        Array of 10 similarities (upper triangle, excluding diagonal)
    """
    similarities = []

    for i in range(len(desc_ids)):
        for j in range(i + 1, len(desc_ids)):
            sim = cosine_similarity(
                embeddings[desc_ids[i]],
                embeddings[desc_ids[j]]
            )
            similarities.append(sim)

    return np.array(similarities)


def create_unified_dataset(manifest_df, embeddings):
    """
    Create unified dataset with all pairwise similarities.

    Structure per row:
        - model, layer, unit (identifiers)
        - desc_id_1..5 (unique IDs for traceability)
        - description_1..5 (text)
        - auc_1..5 (quality scores)
        - mad_1..5, p_1..5 (other scores)
        - sim_12, sim_13, sim_14, sim_15, sim_23, sim_24, sim_25, sim_34, sim_35, sim_45
          (10 pairwise similarities)
        - mean_similarity, min_similarity, max_similarity, std_similarity
    """
    all_rows = []

    # Group by neuron
    grouped = manifest_df.groupby(['model', 'layer', 'unit'])

    print(f"\nComputing pairwise similarities for {len(grouped)} neurons...")

    for (model, layer, unit), group in tqdm(grouped, desc="Neurons"):
        # Should have exactly 5 descriptions
        if len(group) != 5:
            print(f"  Warning: Neuron {model} L{layer}_U{unit} has {len(group)} descriptions, expected 5")
            continue

        # Sort by desc_idx for consistent ordering
        group = group.sort_values('desc_idx').reset_index(drop=True)

        # Extract IDs and data
        desc_ids = group['desc_id'].tolist()
        descriptions = group['description'].tolist()
        aucs = group['auc'].tolist()
        mads = group['mad'].tolist()
        p_values = group['p_value'].tolist()

        # Compute pairwise similarities
        pairwise_sims = compute_pairwise_similarities_for_neuron(desc_ids, embeddings)

        # Create row
        row = {
            'model': model,
            'layer': layer,
            'unit': unit,
        }

        # Add description IDs (for traceability)
        for i, desc_id in enumerate(desc_ids, 1):
            row[f'desc_id_{i}'] = desc_id

        # Add descriptions
        for i, desc in enumerate(descriptions, 1):
            row[f'description_{i}'] = desc

        # Add scores
        for i, (auc, mad, p) in enumerate(zip(aucs, mads, p_values), 1):
            row[f'auc_{i}'] = auc
            row[f'mad_{i}'] = mad
            row[f'p_{i}'] = p

        # Add pairwise similarities
        # Naming: sim_12 means similarity between description 1 and 2
        sim_names = [
            'sim_12', 'sim_13', 'sim_14', 'sim_15',
            'sim_23', 'sim_24', 'sim_25',
            'sim_34', 'sim_35',
            'sim_45'
        ]
        for name, sim in zip(sim_names, pairwise_sims):
            row[name] = sim

        # Add summary statistics
        row['mean_similarity'] = pairwise_sims.mean()
        row['min_similarity'] = pairwise_sims.min()
        row['max_similarity'] = pairwise_sims.max()
        row['std_similarity'] = pairwise_sims.std()

        all_rows.append(row)

    return pd.DataFrame(all_rows)


def compute_random_baselines(manifest_df, embeddings, n_samples=1000):
    """
    Compute random baseline similarities.

    For each model, randomly sample 5 descriptions from different neurons,
    compute their pairwise similarities, repeat n_samples times.
    """
    print(f"\nComputing random baselines ({n_samples} samples per model)...")

    baselines = []

    for model in manifest_df['model'].unique():
        model_descs = manifest_df[manifest_df['model'] == model]
        desc_ids = model_descs['desc_id'].tolist()

        print(f"  {model}...")

        for _ in tqdm(range(n_samples), desc=f"  {model}", leave=False):
            # Sample 5 random description IDs
            sample_ids = np.random.choice(desc_ids, size=5, replace=False)

            # Compute pairwise similarities
            pairwise_sims = compute_pairwise_similarities_for_neuron(sample_ids.tolist(), embeddings)

            baselines.append({
                'model': model,
                'mean_similarity': pairwise_sims.mean(),
                'min_similarity': pairwise_sims.min(),
                'max_similarity': pairwise_sims.max(),
                'std_similarity': pairwise_sims.std(),
            })

    return pd.DataFrame(baselines)


def print_summary(unified_df, baseline_df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for model in unified_df['model'].unique():
        model_data = unified_df[unified_df['model'] == model]
        baseline_data = baseline_df[baseline_df['model'] == model]

        print(f"\n{model.upper()}:")
        print(f"  Neurons: {len(model_data)}")
        print(f"  Mean similarity: {model_data['mean_similarity'].mean():.4f} ± {model_data['mean_similarity'].std():.4f}")
        print(f"  Random baseline: {baseline_data['mean_similarity'].mean():.4f} ± {baseline_data['mean_similarity'].std():.4f}")
        print(f"  Range: [{model_data['min_similarity'].min():.4f}, {model_data['max_similarity'].max():.4f}]")


def main():
    print("="*80)
    print("STEP 2: COMPUTE PAIRWISE SIMILARITIES")
    print("="*80)

    # Step 1: Load embeddings and manifest
    print("\n" + "-"*80)
    manifest_df, embeddings = load_embeddings()

    # Step 2: Create unified dataset
    print("\n" + "-"*80)
    print("Creating unified dataset...")
    print("-"*80)
    unified_df = create_unified_dataset(manifest_df, embeddings)
    print(f"\n✓ Created dataset with {len(unified_df)} neurons")
    print(f"  Columns: {len(unified_df.columns)}")

    # Step 3: Compute random baselines
    print("\n" + "-"*80)
    print("Computing random baselines...")
    print("-"*80)
    baseline_df = compute_random_baselines(manifest_df, embeddings, n_samples=1000)
    print(f"\n✓ Computed {len(baseline_df)} baseline samples")

    # Step 4: Save results
    print("\n" + "-"*80)
    print("Saving results...")
    print("-"*80)

    unified_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved pairwise similarities to:\n  {OUTPUT_FILE}")

    baseline_df.to_csv(BASELINE_FILE, index=False)
    print(f"✓ Saved random baselines to:\n  {BASELINE_FILE}")

    # Step 5: Print summary
    print_summary(unified_df, baseline_df)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print("\nYou can now use this dataset for polysemanticity scoring experiments.")
    print(f"\nMain dataset: {OUTPUT_FILE}")
    print(f"Baseline dataset: {BASELINE_FILE}")
    print("\nTraceability:")
    print("  - Each row has desc_id_1..5 columns linking to description_manifest.csv")
    print("  - Use desc_id to trace back to original evaluations")


if __name__ == "__main__":
    main()

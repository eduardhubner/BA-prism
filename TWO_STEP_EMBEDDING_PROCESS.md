# Two-Step Pairwise Similarity Preparation

## Overview

To handle compatibility issues with sentence_transformers, we split the process into two independent steps:

1. **Step 1**: Generate embeddings (requires sentence_transformers in venv_qwen2)
2. **Step 2**: Compute similarities (only needs numpy/pandas, any environment)

## Why Two Steps?

- `venv_qwen2` was created for a specific version of sentence_transformers
- After embeddings are generated, we don't need sentence_transformers anymore
- Similarities can be computed with basic numpy (more portable)

---

## Step 1: Generate Description Embeddings

**Script**: `src/step1_generate_description_embeddings.py`

### What It Does

1. Loads all descriptions from original PRISM evaluation files (4 models)
2. Creates a **manifest** with unique IDs for each description
3. Generates embeddings using `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
4. Saves everything with full traceability

### Environment

**MUST use venv_qwen2** (has sentence_transformers 5.1.2 already installed):
```bash
cd /Users/eduardhubner/Desktop/AI/Bachelor\ MechInterp/prism

# Use venv_qwen2's python directly (activation sometimes doesn't work in scripts)
venv_qwen2/bin/python3 src/step1_generate_description_embeddings.py
```

**Important**: Don't upgrade sentence_transformers! Version 5.1.2 works with gte-Qwen2-1.5B-instruct. Newer versions may have compatibility issues.

### Output Files

All saved to `results/description_embeddings/`:

1. **`description_manifest.csv`** - Full metadata for all descriptions
   ```
   desc_id, model, layer, unit, desc_idx, description, auc, mad, p_value
   gpt2-xl_L0_U440_D1, gpt2-xl, 0, 440, 1, "Text...", 0.8, 1.2, 0.001
   ```

2. **`embeddings.npz`** - Compressed numpy archive with all embeddings
   - Contains: `embeddings` (numpy array) and `desc_ids` (list)
   - Shape: (N_descriptions, embedding_dim)

3. **`embedding_index.csv`** - Quick lookup from desc_id to array index
   ```
   desc_id, array_index
   gpt2-xl_L0_U440_D1, 0
   gpt2-xl_L0_U440_D2, 1
   ```

### Traceability

Each description gets a unique ID: `{model}_L{layer}_U{unit}_D{desc_idx}`

Example: `gpt2-xl_L0_U440_D3` = GPT2-XL, Layer 0, Unit 440, Description 3

---

## Step 2: Compute Pairwise Similarities

**Script**: `src/step2_compute_pairwise_similarities.py`

### What It Does

1. Loads embeddings from Step 1
2. For each neuron (5 descriptions), computes all 10 pairwise similarities
3. Computes random baseline similarities (1000 samples per model)
4. Saves unified dataset ready for scoring experiments

### Environment

**Can use ANY environment** with numpy/pandas/scipy:
```bash
cd /Users/eduardhubner/Desktop/AI/Bachelor\ MechInterp/prism

# Use main venv OR venv_qwen2, doesn't matter
source venv/bin/activate
# OR
source venv_qwen2/bin/activate

python3 src/step2_compute_pairwise_similarities.py
```

### Output Files

Saved to `results/`:

1. **`pairwise_similarities_dataset.csv`** - Main dataset for scoring

   **Structure (one row per neuron)**:
   ```
   Identifiers:
     model, layer, unit

   Traceability:
     desc_id_1, desc_id_2, desc_id_3, desc_id_4, desc_id_5

   Descriptions:
     description_1, description_2, description_3, description_4, description_5

   Quality Scores:
     auc_1, auc_2, auc_3, auc_4, auc_5
     mad_1, mad_2, mad_3, mad_4, mad_5
     p_1, p_2, p_3, p_4, p_5

   Pairwise Similarities (10 per neuron):
     sim_12, sim_13, sim_14, sim_15    # Description 1 with 2,3,4,5
     sim_23, sim_24, sim_25            # Description 2 with 3,4,5
     sim_34, sim_35                    # Description 3 with 4,5
     sim_45                            # Description 4 with 5

   Summary Stats:
     mean_similarity, min_similarity, max_similarity, std_similarity
   ```

2. **`random_baseline_similarities.csv`** - Random baselines
   ```
   model, mean_similarity, min_similarity, max_similarity, std_similarity
   (1000 samples per model)
   ```

---

## Complete Workflow

### First Time Setup

```bash
# 1. Activate venv_qwen2
cd /Users/eduardhubner/Desktop/AI/Bachelor\ MechInterp/prism
source venv_qwen2/bin/activate

# 2. Run Step 1 (generates embeddings)
# sentence_transformers 5.1.2 is already installed in venv_qwen2
python3 src/step1_generate_description_embeddings.py
# Estimated time: 2-5 minutes
```

### Every Time After

```bash
# Run Step 2 (computes similarities)
# Can use any environment with numpy/pandas
python3 src/step2_compute_pairwise_similarities.py
# Estimated time: < 1 minute
```

---

## Data Scope

- **Models**: 4 (gpt2-xl, Llama-3.1-8B-Instruct, gemma-scope-2b, gpt2-small-sae)
- **Descriptions per model**: ~300 (60 neurons × 5 descriptions)
- **Total descriptions**: ~1,200
- **Pairwise similarities**: ~600 per model (60 neurons × 10 pairs)
- **Total similarities**: ~2,400

---

## Traceability System

### From Similarity to Original Evaluation

1. **Start with**: Row in `pairwise_similarities_dataset.csv`
   - Has `desc_id_1` through `desc_id_5`

2. **Lookup**: In `description_manifest.csv`
   - Find rows matching the desc_ids
   - Get full metadata (model, layer, unit, AUC, etc.)

3. **Trace back**: To original evaluation file
   - File: `cosy-evaluation_target-{model}_textgen-gemini-1-5-pro_...csv`
   - Match by model, layer, unit, and description text

### Example

```python
# Load datasets
similarities = pd.read_csv("results/pairwise_similarities_dataset.csv")
manifest = pd.read_csv("results/description_embeddings/description_manifest.csv")

# Get a neuron
neuron = similarities[(similarities['model'] == 'gpt2-xl') &
                     (similarities['layer'] == 0) &
                     (similarities['unit'] == 440)].iloc[0]

# Get its descriptions
desc_ids = [neuron['desc_id_1'], neuron['desc_id_2'], neuron['desc_id_3'],
            neuron['desc_id_4'], neuron['desc_id_5']]

# Lookup full metadata
descriptions = manifest[manifest['desc_id'].isin(desc_ids)]
print(descriptions[['desc_id', 'auc', 'description']])
```

---

## Validation

Both scripts include validation checks:

✓ Each neuron has exactly 5 descriptions
✓ Number of embeddings matches number of descriptions
✓ All desc_ids are unique
✓ Embeddings and manifest are synchronized
✓ Similarity computation produces exactly 10 values per neuron

---

## Troubleshooting

### Step 1 Issues

**"ModuleNotFoundError: sentence_transformers"**
- This shouldn't happen! sentence_transformers 5.1.2 is pre-installed in venv_qwen2
- Make sure you activated venv_qwen2: `source venv_qwen2/bin/activate`
- Check: `which python` should point to venv_qwen2

**"Model not found: Alibaba-NLP/gte-Qwen2-1.5B-instruct"**
- First run downloads the model (~1.5GB)
- Requires internet connection

**CUDA out of memory**
- Reduce BATCH_SIZE in script (default: 32)
- Or run on CPU (slower but works)

### Step 2 Issues

**"Manifest not found"**
- Must run Step 1 first!

**"Mismatch in number of embeddings"**
- Re-run Step 1 completely
- Don't manually edit output files

---

## Next Steps

Once both steps complete, you have:

✅ `pairwise_similarities_dataset.csv` - Ready for scoring experiments
✅ `random_baseline_similarities.csv` - For normalization
✅ Full traceability via desc_ids

Now you can implement your polysemanticity scoring formulas!

# Experiments: Adaptive K-Selection for PRISM

This document summarizes the experiments conducted for Eduard Hübner's Bachelor's Thesis at TU Berlin, extending the PRISM framework with adaptive k-selection methods.

## Overview

**Research Question:** Can adaptive k-selection methods automatically determine the optimal number of semantic clusters in polysemantic neurons?

**Test Setup:**
- **Model:** GPT2-XL
- **Neurons:** 60 neurons (20 per layer from layers 0, 20, 40)
- **Embedding Model:** Qwen3-Embedding-0.6B (768 dimensions)
- **Samples per neuron:** 1000 top-activation texts

---

## Experiment 1: Full PRISM Pipeline with Adaptive K-Selection

**Objective:** Test adaptive k-selection methods in the complete PRISM pipeline including description generation and AUC evaluation.

### Methods Tested

1. **Silhouette Score** - Maximizes cluster separation
2. **BIC (Bayesian Information Criterion)** - Model selection criterion
3. **Davies-Bouldin Index** - Minimizes cluster overlap
4. **Fixed k=5** - Baseline for comparison

### Experiment Structure

**Phase 1 (First 15 neurons):**
- Tested all 4 methods on neurons across all three layers
- Generated feature descriptions for each method

**Phase 2 (Remaining 45 neurons):**
- Continued with only Silhouette and Fixed k=5
- **Reason:** BIC and Davies-Bouldin showed no sensitivity to k-selection (consistently chose similar values)
- Calculated AUC scores for evaluation

### Results Location

**Feature Descriptions:**
```
descriptions/gemini-2-5-flash/gpt2-xl/
├── gpt2-xl_layer-{L}_unit-{U}_{method}_{timestamp}.csv
```

**AUC Evaluations:**
```
assets/explanations/GPT-explain/
├── eval_neuron_batch_L0_L20_L40.csv      # First 15 neurons, all methods
├── eval_neuron_batch_layer0.csv          # Layer 0 results
├── eval_neuron_batch_layer20.csv         # Layer 20 results
├── eval_neuron_batch_layer40.csv         # Layer 40 results
└── eval_neuron_batch_remaining.csv       # Remaining 45 neurons (Silhouette + Fixed-k5)
```

### Key Findings

1. **BIC and Davies-Bouldin lack sensitivity** - showed minimal variation in k-selection across neurons
2. **Silhouette provides more adaptive behavior** - selected varying k values based on data structure
3. **Fixed k=5 serves as reasonable baseline** - comparable performance without complexity

---

## Experiment 2: Clustering Analysis (Distance Metric Investigation)

**Objective:** Systematically compare clustering quality across methods and investigate whether cosine similarity improves results compared to Euclidean distance.

### Why This Experiment?

After Experiment 1 showed weak performance, we hypothesized that:
- Euclidean distance may be inappropriate for 768-dimensional normalized embeddings
- Cosine similarity better captures semantic relationships
- Noise filtering (HDBSCAN) could improve cluster quality

### Methods Tested

1. **K-Means Silhouette** (cosine distance, spherical k-means)
2. **K-Means BIC** (cosine distance)
3. **K-Means Davies-Bouldin** (cosine distance)
4. **Agglomerative Silhouette** (cosine linkage)
5. **Agglomerative Davies-Bouldin** (cosine linkage)
6. **HDBSCAN** (cosine metric) - noise filtering
7. **Fixed k=5** (baseline)

### Noise Filtering Strategy

Used HDBSCAN to identify and filter noise points before applying other clustering methods:

1. Run HDBSCAN with `min_cluster_size=5`
2. Remove samples labeled as noise (label -1)
3. Re-cluster remaining "clean" samples
4. Compare before/after noise filtering

### Hyperparameter Tuning

Tested HDBSCAN `min_cluster_size` values: 3, 5, 7, 10

Results (see `clustering_analysis_cosine/hdbscan_tuning.json`):
- **min_cluster_size=3:** 31.6% noise, silhouette 0.0251
- **min_cluster_size=5:** 73.5% noise, silhouette 0.0303 ✓ (best balance)
- **min_cluster_size=7:** 94.1% noise, silhouette 0.0912
- **min_cluster_size=10:** 99.8% noise (too aggressive)

### Results Location

**Main Results:**
```
clustering_analysis_cosine/
├── results.json                           # All 60 neurons, 7 methods, clustering metrics
├── comprehensive_noise_filtering.json     # Before/after noise filtering comparison
├── hdbscan_tuning.json                   # Hyperparameter tuning results
└── clusters/                             # Individual cluster assignments (420 JSONs)
    └── layer-{L}_unit-{U}_{method}.json
```

### Key Findings

#### Distance Metric Comparison

| Metric | Euclidean (Old) | Cosine (Current) | Improvement |
|--------|----------------|------------------|-------------|
| **Silhouette Score** | ~0.015 | ~0.032 | **+113%** |
| **Hopkins Statistic** | 0.644 | 0.620 | Comparable |

**Conclusion:** Cosine similarity provides significant improvement but clustering remains weak.

#### Noise Filtering Impact

**Before noise filtering:**
- Mean silhouette: 0.0321 (K-Means Silhouette)
- Range: 0.0266 - 0.0583

**After noise filtering (73.5% removed):**
- Mean silhouette: 0.0642 (K-Means Silhouette) - **100% improvement**
- Range: 0.0529 - 0.1201

**Conclusion:** Noise filtering doubles clustering quality but absolute scores remain weak.

#### K-Selection Patterns

**K-Means methods:**
- Silhouette: Typically selects k=2-4
- BIC: Typically selects k=2
- Davies-Bouldin: Typically selects k=2

**Agglomerative methods:**
- Silhouette: 100% select k=2
- Davies-Bouldin: 68% select k=2

**Conclusion:** Methods consistently favor low k values, suggesting weak multi-cluster structure.

#### Hopkins Statistic

**Mean: 0.620** (range: 0.5-0.7)

**Interpretation:**
- < 0.3: Regular/uniform distribution
- ~0.5: Random distribution
- > 0.7: Strong clustering tendency

**Conclusion:** Data shows moderate to weak clustering tendency, closer to random than strongly clustered.

---

## Overall Conclusions

### What We Learned

1. **Distance metric matters:** Cosine similarity provides 2x improvement over Euclidean for semantic embeddings

2. **Noise filtering helps:** Removing 73.5% of noisy samples doubles clustering quality

3. **Clustering remains weak:** Even after improvements, silhouette scores (~0.06-0.09) indicate substantial overlap between clusters

4. **Adaptive k-selection is challenging:** Methods show low sensitivity and consistently favor low k values

5. **Fixed k=5 is competitive:** Performs comparably to adaptive methods with less complexity

### Possible Explanations for Weak Clustering

1. **Embedding model quality:** Qwen3-0.6B (0.6B parameters) may be insufficient to capture semantic nuances
2. **Sample size:** 1000 texts per neuron may not capture full range of activation patterns
3. **Inherent polysemanticity with fuzzy boundaries:** Neurons may respond to semantically related/overlapping concepts rather than distinct categories, resulting in gradual transitions between activation patterns

---

## Reproducibility

All experiment results are version-controlled:
- Clustering results: `clustering_analysis_cosine/`
- Feature descriptions: `descriptions/gemini-2-5-flash/gpt2-xl/`
- AUC evaluations: `assets/explanations/GPT-explain/`

Analysis scripts and plots are reproducible from these results.

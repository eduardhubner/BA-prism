# PRISM Extension: Description Verification

**Fork of the original PRISM repository, adding a verification stage for feature descriptions.**

Part of Eduard Hübner's Bachelor's Thesis at TU Berlin.

**Original repository:** [lkopf/prism](https://github.com/lkopf/automated-interpretability)

**Original paper:** [Capturing Polysemanticity with PRISM](https://arxiv.org/abs/2506.15538) by Kopf et al., 2025

## Description verification

PRISM scores a feature by the mean pairwise similarity of several candidate
descriptions. That score says nothing about whether those descriptions are
*supported* by the feature's behaviour. This fork checks them first:

1. **AUROC filter** — drop descriptions performing at or below chance (τ_auroc = 0.50)
2. **Redundancy collapse** — among survivors, merge descriptions whose embeddings
   exceed τ_sim = 0.70, keeping the best-supported one per group

The result is a profile rather than a single number: `n_s` surviving descriptions,
evidence quality `Q`, and semantic diversity `D = 1 − σ̃`.

**Entry point:** [`src/verification.py`](src/verification.py) — the whole method in
one module, numpy and stdlib only:

```python
from verification import filter_and_collapse, compute_metrics

result  = filter_and_collapse(aurocs, sim_matrix)
metrics = compute_metrics(aurocs, sim_matrix, result)   # n_s, sigma-tilde, D, Q
```

Nothing in it assumes the descriptions are text, so it transfers to other
modalities: supply AUROCs from your own evaluation and a cosine-similarity matrix
from whatever embedding you use.

Two things to know before reusing it:

- **τ_sim = 0.70 is calibrated for `gte-Qwen2-1.5B` text embeddings.** The absolute
  cosine scale does not transfer across embedding models — recalibrate rather than
  inherit the value.
- **σ̃ = 1 and D = 0 at a single survivor are conventions, not measurements**, and
  both are undefined when no description survives.

Per-feature profiles for the 239 features in the study:
[`results/reliability_profile.csv`](results/reliability_profile.csv).

## Also in this fork: adaptive k-selection

Clustering extensions explored during the thesis, kept for reference:

- Adaptive k-selection using Silhouette Score, BIC and Davies-Bouldin Index
- Cosine similarity via Spherical K-Means for high-dimensional embeddings
- HDBSCAN-based noise filtering; agglomerative clustering with cosine linkage
- Hopkins statistic for clustering tendency

**Modified relative to upstream:** `src/utils/clustering.py` (`select_optimal_k()`,
spherical k-means, HDBSCAN, cosine distance), `src/utils/config.py` (`ADAPTIVE_K`,
`SPHERICAL`, `USE_HDBSCAN_FILTERING`), `src/feature_description.py`,
`src/evaluation.py`, `src/utils/helper_modules.py`, `src/utils/models.py`.

---

<br/><br/>
<p align="center">
  <img width="250" src="./prism_logo.svg">
<h3 align="center"><b>Original PRISM Framework</b></h3>
<p align="center">
  PyTorch

  </p>

This repository contains the code and experiments for the paper **[Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework](https://arxiv.org/abs/2506.15538)** by Kopf et al., 2025. 

## Table of Contents

- [About](#-about)
- [Repository Overview](#-repository-overview)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
  - [1 Feature Descriptions](#feature-descriptions)
  - [2 Evaluation](#evaluation)
  - [3 Meta-labels](#metalabels)

## About

</p>
<p align="center">
  <img width="800" src="./prism_framework.jpg">
</p>

Unlike prior approaches that assign a single description per feature, PRISM (Polysemantic FeatuRe Identification and Scoring Method) provides more nuanced descriptions for both polysemantic and monosemantic features. PRISM samples sentences from the top percentile activation distribution, clusters them in embedding space, and uses an LLM to generate labels for each concept cluster. We benchmark PRISM across various layers and architectures, showing how polysemanticity and interpretability shift through the model. In exploring the concept space, we use PRISM to characterize more complex components, finding and interpreting patterns that specific attention heads or groups of neurons respond to. Our findings show that the PRISM framework not only provides multiple human interpretable descriptions for neurons but also aligns with the human interpretation of polysemanticity.

## Repository Overview

The repository is organized for ease of use:
- **`assets/explanations/`** – Pre-computed feature descriptions from various feature description methods.  
- **`descriptions/`** – Feature descriptions generated with **PRISM**.  
- **`generated_text/`** – Concept text samples generated for evaluation purposes.  
- **`notebooks/`** – Contains a Jupyter notebook for reproducing the benchmark table and plots shown in the paper.  
- **`src/`** – Core source code, including all necessary functions for running feature description and evaluation.


## Installation

Install the necessary packages using the provided [requirements.txt](https://github.com/lkopf/prism/blob/main/src/requirements.txt):

```bash
pip install -r requirements.txt
```

## Running Experiments

First, set paramters in [src/utils/config.py](https://github.com/lkopf/prism/blob/main/src/utils/config.py) or use default parameters.

### 1. Feature Descriptions

This script outputs multiple feature descriptions based on percentile sampling and clustering for one feature.

```bash
python src/feature_description.py
```

To generate descriptions for multiple features define `EXPLAIN_FILE` in [config.py](https://github.com/lkopf/prism/blob/main/src/utils/config.py) and run: 

```bash
python src/run_feature_description.py
```

Generated feature descriptions can be found in [descriptions](https://github.com/lkopf/prism/tree/main/descriptions) folder.

### 2. Evaluation

Evaluate feature descriptions with CoSy scores:

```bash
python src/evaluation.py
```

Generated concept samples can be found in [generated_text](https://github.com/lkopf/prism/tree/main/generated_text) folder.

Evaluate all feature descriptions per feature with polysemanticity score (cosine similarity), max AUROC, and max MAD.

> **Terminology:** the metric is **AUROC** throughout — prose, code, and CSV columns. Imported CoSy evaluation CSVs keep their upstream `AUC` column name.

```bash
python src/meta_evaluation.py
```

All evaluation scores can be found in [results](https://github.com/lkopf/prism/tree/main/results) folder.

### 3. Meta-labels

To generate meta-labels for concepts found in feature descriptions run:

```bash
python src/run_concept_summary.py
```

All meta-label results can be found in **`metalabels/`** folder.


## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@misc{kopf2025prism,
      title={Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework}, 
      author={Laura Kopf and Nils Feldhus and Kirill Bykov and Philine Lou Bommer and Anna Hedström and Marina M. -C. Höhne and Oliver Eberle},
      year={2025},
      eprint={2506.15538},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.15538}, 
}
```
This work is in review.

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for more in-depth discussions or suggestions.

📧 Contact us:
- Laura Kopf: kopf[at]tu-berlin.de

Thank you for your interest and support!
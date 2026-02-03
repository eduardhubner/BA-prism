# Polysemanticity Scoring Experiment Framework

## Overview

This framework provides infrastructure for experimenting with different polysemanticity scoring methods using the original PRISM results.

## File Structure

```
src/
  polysemanticity_scoring_experiments.py  # Main experiment framework

results/
  polysemanticity_experiments/             # Experiment results directory
    {model}_{experiment_name}.csv          # Scores for all neurons
    {model}_{experiment_name}_metadata.json # Experiment metadata
```

## Quick Start

```python
from polysemanticity_scoring_experiments import PolysemanticityScoringExperiment

# Initialize with a model
experiment = PolysemanticityScoringExperiment(model_name="gpt2-xl")

# Print data summary
experiment.print_summary()

# Define your scoring function
def my_scoring_function(neuron_data: dict) -> float:
    """
    Compute polysemanticity score for a neuron.

    neuron_data contains:
        - descriptions: list of description strings
        - aucs: list of AUC scores
        - mads: list of MAD scores
        - cosine_similarity: pre-computed average similarity
        - cosine_similarity_random: random baseline
        - max_auc, max_mad: maximum values
    """
    # Your scoring logic here
    return score

# Run experiment
scoring_functions = {
    'my_metric': (my_scoring_function, "Description of what this measures")
}

results = experiment.run_experiment(
    scoring_functions=scoring_functions,
    experiment_name="my_experiment"
)

# Compare metrics
experiment.compare_metrics(results, 'my_metric', 'baseline_cosine')
```

## Available Data

### Models Available
- `gpt2-xl` (60 neurons, 300 descriptions)
- `Llama-3.1-8B-Instruct` (60 neurons, 300 descriptions)
- `gemma-scope-2b` (60 neurons, 300 descriptions)
- `gpt2-small-sae` (59 neurons, 295 descriptions)

### Data Per Neuron
- **Exactly 5 descriptions** (from fixed k=5 clustering)
- **Per-description AUC scores** (quality metric)
- **Per-description MAD scores** (activation difference)
- **Pre-computed cosine similarity** (average pairwise similarity of descriptions)
- **Random baseline** (cosine similarity of 5 random descriptions)

## Key Features

### 1. Easy Data Access
```python
# Get all data for a specific neuron
neuron = experiment.get_neuron_data(layer=0, unit=1149)
print(neuron['descriptions'])
print(neuron['aucs'])
print(neuron['cosine_similarity'])
```

### 2. Apply Custom Scoring Functions
```python
# Apply a single scoring function
result = experiment.apply_scoring_function(
    scoring_func=my_function,
    metric_name="my_metric",
    description="What this measures"
)
```

### 3. Batch Experiments
```python
# Run multiple scoring functions at once
scoring_functions = {
    'metric1': (func1, "description1"),
    'metric2': (func2, "description2"),
    'metric3': (func3, "description3"),
}
results = experiment.run_experiment(scoring_functions)
```

### 4. Automatic Result Saving
- Results saved to CSV with all scores
- Metadata saved to JSON with statistics
- Timestamped filenames for version control

### 5. Metric Comparison
```python
# Compute correlations between metrics
comparison = experiment.compare_metrics(results, 'metric1', 'metric2')
print(f"Pearson r = {comparison['pearson_r']:.3f}")
print(f"Spearman ρ = {comparison['spearman_r']:.3f}")
```

## Example Scoring Functions

The file includes placeholder scoring functions to get you started:

### 1. `baseline_cosine_similarity`
Simply returns the pre-computed average cosine similarity.
- Lower = more polysemantic (diverse descriptions)
- Higher = less polysemantic (similar descriptions)

### 2. `inverted_cosine_similarity`
Returns `1 - cosine_similarity` for intuitive interpretation.
- Higher = more polysemantic
- Lower = less polysemantic

### 3. `auc_weighted_diversity_placeholder`
Combines quality (mean AUC) and diversity (1 - cosine_similarity).
- Placeholder for more sophisticated weighting

### 4. `effective_concept_count_placeholder`
Counts descriptions with AUC > 0.5.
- Simple discrete metric

## Next Steps: Implementing Real Scoring Logic

The framework is ready for experimentation. Now you can:

1. **Design scoring metrics** that combine:
   - Distribution of AUC scores (quality)
   - Diversity of descriptions (similarity)
   - Statistical properties (entropy, etc.)

2. **Test different approaches**:
   - AUC-weighted diversity
   - Effective number of concepts
   - Distribution entropy
   - Distinct concept counting

3. **Compare across models**:
   - Run same metrics on all 4 models
   - Analyze cross-model consistency

4. **Visualize results**:
   - Create plots from saved CSV files
   - Compare new metrics to baseline

## Data Structure Details

### Neuron Data Dictionary
```python
{
    'layer': int,
    'unit': int,
    'descriptions': [str, str, str, str, str],  # Always 5 descriptions
    'aucs': [float, float, float, float, float],  # Quality scores
    'mads': [float, float, float, float, float],  # Activation differences
    'p_values': [float, float, float, float, float],  # Statistical significance
    'n_descriptions': 5,  # Always 5 for original PRISM
    'cosine_similarity': float,  # Pre-computed average pairwise similarity
    'cosine_similarity_random': float,  # Random baseline for comparison
    'max_auc': float,  # Best single description
    'max_mad': float,  # Highest activation difference
}
```

## Example Output

```
================================================================================
RUNNING EXPERIMENT: baseline_example
Model: gpt2-xl
Scoring functions: ['baseline_cosine', 'inverted_cosine', ...]
================================================================================

baseline_cosine:
  Pre-computed average cosine similarity (lower = more polysemantic)
  Range: 0.3060 to 0.6755
  Mean: 0.4175 ± 0.0749

inverted_cosine:
  1 - cosine_similarity (higher = more polysemantic)
  Range: 0.3245 to 0.6940
  Mean: 0.5825 ± 0.0749

================================================================================
Results saved to: results/polysemanticity_experiments/gpt2-xl_baseline_example.csv
Metadata saved to: results/polysemanticity_experiments/gpt2-xl_baseline_example_metadata.json
================================================================================
```

## Tips for Designing Scoring Functions

1. **Use available data**: Don't need to compute new embeddings - use pre-computed cosine_similarity
2. **Consider edge cases**: What if all AUCs are low? What if similarity is very high?
3. **Normalize appropriately**: Consider scale of your scores (0-1, unbounded, etc.)
4. **Test on examples**: Use `get_neuron_data()` to check specific neurons
5. **Compare to baseline**: Always compare new metrics to simple cosine similarity
6. **Think about interpretation**: What does a high/low score mean conceptually?

## Current Status

✅ Framework created and tested
✅ Data loading works for all 4 models
✅ Result saving and metadata tracking works
✅ Placeholder scoring functions provided
⏳ **Next: Design and implement actual scoring logic**

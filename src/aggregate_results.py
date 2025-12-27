"""
Results aggregator for adaptive k-selection experiment.

This script compiles results from multiple neuron runs, extracting:
- K-selection choices (k values chosen by each method)
- Cluster descriptions
- Evaluation metrics (AUC, MAD) if available
- Metadata from logs

Outputs summary CSV files and statistics for thesis analysis.

Usage:
    python src/aggregate_results.py
"""

import csv
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aggregate_results.log'),
        logging.StreamHandler()
    ]
)

# Paths
LOGS_DIR = Path("logs")
DESCRIPTIONS_DIR = Path("descriptions")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/aggregated")


def parse_log_file(log_path: Path) -> Optional[Dict]:
    """
    Extract metadata from a feature_description log file.

    Looks for:
    - Layer ID, Unit ID
    - Adaptive k selected (if applicable)
    - K-selection method
    - K-selection scores
    - Number of clusters/descriptions generated

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with extracted metadata, or None if parsing fails
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {}

        # Extract layer and unit
        layer_match = re.search(r'Layer:\s*(\d+)', content, re.IGNORECASE)
        unit_match = re.search(r'Unit:\s*(\d+)', content, re.IGNORECASE)

        if not layer_match or not unit_match:
            # Try alternative format from log filename
            name_match = re.search(r'layer-(\d+)_unit-(\d+)', log_path.name)
            if name_match:
                metadata['layer'] = int(name_match.group(1))
                metadata['unit'] = int(name_match.group(2))
            else:
                logger.warning(f"Could not extract layer/unit from {log_path.name}")
                return None
        else:
            metadata['layer'] = int(layer_match.group(1))
            metadata['unit'] = int(unit_match.group(1))

        # Check if adaptive k was used
        adaptive_k_match = re.search(r'Adaptive k selected:\s*(\d+)', content)
        if adaptive_k_match:
            metadata['adaptive_k'] = True
            metadata['k_selected'] = int(adaptive_k_match.group(1))

            # Extract method
            method_match = re.search(r'K-selection method:\s*(\w+)', content)
            if method_match:
                metadata['method'] = method_match.group(1)

            # Extract k-selection scores
            scores_match = re.search(r'K-selection scores:\s*(\{[^\}]+\})', content)
            if scores_match:
                try:
                    scores_str = scores_match.group(1)
                    # Parse dictionary string
                    scores_dict = eval(scores_str)
                    metadata['k_scores'] = scores_dict
                except:
                    pass
        else:
            # Fixed k
            fixed_k_match = re.search(r'Using fixed k:\s*(\d+)', content)
            if fixed_k_match:
                metadata['adaptive_k'] = False
                metadata['k_selected'] = int(fixed_k_match.group(1))
                metadata['method'] = 'fixed'

        # Count clusters generated
        cluster_matches = re.findall(r'CLUSTER #(\d+)', content)
        if cluster_matches:
            metadata['num_clusters'] = len(cluster_matches)

        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', log_path.name)
        if timestamp_match:
            metadata['timestamp'] = timestamp_match.group(1)

        return metadata

    except Exception as e:
        logger.error(f"Error parsing {log_path}: {e}")
        return None


def load_descriptions_csv(csv_path: Path) -> List[Dict]:
    """
    Load cluster descriptions from CSV file.

    Args:
        csv_path: Path to descriptions CSV

    Returns:
        List of dictionaries with description data
    """
    descriptions = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                descriptions.append({
                    'layer': int(row['layer']),
                    'unit': int(row['unit']),
                    'description': row['description'],
                    'mean_activation': float(row['mean_activation']) if row['mean_activation'] else None,
                    'highlights': row.get('highlights', '')
                })
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")

    return descriptions


def load_evaluation_results(results_dir: Path) -> pd.DataFrame:
    """
    Load evaluation results (CoSy scores) if available.

    Args:
        results_dir: Directory containing evaluation CSV files

    Returns:
        DataFrame with evaluation metrics
    """
    all_results = []

    # Look for evaluation CSV files
    eval_files = list(results_dir.glob("*evaluation*.csv"))

    for eval_file in eval_files:
        try:
            df = pd.read_csv(eval_file)
            all_results.append(df)
        except Exception as e:
            logger.warning(f"Could not load {eval_file}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def aggregate_k_selection_results() -> pd.DataFrame:
    """
    Aggregate k-selection results from all log files.

    Returns:
        DataFrame with columns: layer, unit, method, k_selected, k_scores, timestamp
    """
    results = []

    # Find all log files
    log_files = list(LOGS_DIR.glob("*.log"))
    logger.info(f"Found {len(log_files)} log files")

    for log_file in log_files:
        metadata = parse_log_file(log_file)
        if metadata:
            results.append(metadata)

    df = pd.DataFrame(results)

    if not df.empty:
        # Sort by layer, unit, method
        df = df.sort_values(['layer', 'unit', 'method'])

    logger.info(f"Extracted metadata from {len(results)} runs")

    return df


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics across all neurons and methods.

    Args:
        df: DataFrame with k-selection results

    Returns:
        Dictionary with summary statistics
    """
    stats = {}

    if df.empty:
        return stats

    # K-selection statistics per method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        stats[method] = {
            'num_neurons': len(method_df),
            'k_mean': method_df['k_selected'].mean(),
            'k_std': method_df['k_selected'].std(),
            'k_median': method_df['k_selected'].median(),
            'k_min': method_df['k_selected'].min(),
            'k_max': method_df['k_selected'].max(),
            'k_mode': method_df['k_selected'].mode().iloc[0] if not method_df['k_selected'].mode().empty else None,
        }

    # Overall comparison
    if 'fixed' in df['method'].values:
        adaptive_methods = df[df['method'] != 'fixed']
        fixed_df = df[df['method'] == 'fixed']

        stats['comparison'] = {
            'adaptive_k_mean': adaptive_methods['k_selected'].mean(),
            'fixed_k_mean': fixed_df['k_selected'].mean(),
            'adaptive_lower_than_fixed': (adaptive_methods['k_selected'] < 5).sum(),
            'adaptive_equal_to_fixed': (adaptive_methods['k_selected'] == 5).sum(),
            'adaptive_higher_than_fixed': (adaptive_methods['k_selected'] > 5).sum(),
        }

    return stats


def create_k_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table comparing k values across methods for each neuron.

    Args:
        df: DataFrame with k-selection results

    Returns:
        Pivot table with neurons as rows, methods as columns
    """
    if df.empty:
        return pd.DataFrame()

    # Create pivot table
    pivot = df.pivot_table(
        index=['layer', 'unit'],
        columns='method',
        values='k_selected',
        aggfunc='first'
    )

    return pivot


def aggregate_descriptions() -> pd.DataFrame:
    """
    Aggregate all cluster descriptions from CSV files.

    Returns:
        DataFrame with all descriptions
    """
    all_descriptions = []

    # Find all description CSV files
    desc_files = list(DESCRIPTIONS_DIR.glob("**/*.csv"))
    logger.info(f"Found {len(desc_files)} description CSV files")

    for desc_file in desc_files:
        descriptions = load_descriptions_csv(desc_file)

        # Add source file info
        for desc in descriptions:
            desc['source_file'] = desc_file.name

        all_descriptions.extend(descriptions)

    df = pd.DataFrame(all_descriptions)
    logger.info(f"Loaded {len(all_descriptions)} cluster descriptions")

    return df


def main():
    """Main aggregation workflow."""
    logger.info("=" * 80)
    logger.info("AGGREGATING EXPERIMENT RESULTS")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate k-selection results
    logger.info("\n1. Aggregating k-selection results from logs...")
    k_results_df = aggregate_k_selection_results()

    if not k_results_df.empty:
        # Save full results
        output_path = OUTPUT_DIR / "k_selection_results.csv"
        k_results_df.to_csv(output_path, index=False)
        logger.info(f"Saved k-selection results to: {output_path}")

        # Create comparison table
        comparison_table = create_k_comparison_table(k_results_df)
        if not comparison_table.empty:
            comp_path = OUTPUT_DIR / "k_comparison_by_neuron.csv"
            comparison_table.to_csv(comp_path)
            logger.info(f"Saved k-comparison table to: {comp_path}")

        # Compute and save summary statistics
        stats = compute_summary_statistics(k_results_df)
        stats_path = OUTPUT_DIR / "summary_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("ADAPTIVE K-SELECTION EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for method, method_stats in stats.items():
                if method != 'comparison':
                    f.write(f"\n{method.upper()}:\n")
                    for key, value in method_stats.items():
                        f.write(f"  {key}: {value}\n")

            if 'comparison' in stats:
                f.write("\n\nCOMPARISON TO FIXED K=5:\n")
                for key, value in stats['comparison'].items():
                    f.write(f"  {key}: {value}\n")

        logger.info(f"Saved summary statistics to: {stats_path}")

        # Print summary to console
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        for method in ['davies_bouldin', 'bic', 'silhouette', 'fixed']:
            if method in stats:
                logger.info(f"\n{method.upper()}:")
                logger.info(f"  k (mean ± std): {stats[method]['k_mean']:.2f} ± {stats[method]['k_std']:.2f}")
                logger.info(f"  k (median): {stats[method]['k_median']:.1f}")
                logger.info(f"  k (range): [{stats[method]['k_min']}, {stats[method]['k_max']}]")

    else:
        logger.warning("No k-selection results found in logs")

    # 2. Aggregate cluster descriptions
    logger.info("\n2. Aggregating cluster descriptions...")
    descriptions_df = aggregate_descriptions()

    if not descriptions_df.empty:
        desc_path = OUTPUT_DIR / "all_descriptions.csv"
        descriptions_df.to_csv(desc_path, index=False)
        logger.info(f"Saved all descriptions to: {desc_path}")

        # Count descriptions per neuron
        desc_counts = descriptions_df.groupby(['layer', 'unit']).size().reset_index(name='num_descriptions')
        desc_counts_path = OUTPUT_DIR / "descriptions_per_neuron.csv"
        desc_counts.to_csv(desc_counts_path, index=False)
        logger.info(f"Saved description counts to: {desc_counts_path}")

    else:
        logger.warning("No descriptions found")

    # 3. Load evaluation results if available
    logger.info("\n3. Checking for evaluation results...")
    eval_df = load_evaluation_results(RESULTS_DIR)

    if not eval_df.empty:
        eval_path = OUTPUT_DIR / "evaluation_metrics.csv"
        eval_df.to_csv(eval_path, index=False)
        logger.info(f"Saved evaluation metrics to: {eval_path}")

        # Merge with k-selection results if possible
        if not k_results_df.empty and 'layer' in eval_df.columns and 'unit' in eval_df.columns:
            merged = k_results_df.merge(
                eval_df,
                on=['layer', 'unit'],
                how='left'
            )
            merged_path = OUTPUT_DIR / "combined_results.csv"
            merged.to_csv(merged_path, index=False)
            logger.info(f"Saved combined results to: {merged_path}")
    else:
        logger.info("No evaluation results found (run evaluation.py to generate)")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"\nOutput files saved to: {OUTPUT_DIR}/")
    logger.info("\nGenerated files:")
    for file in OUTPUT_DIR.glob("*"):
        logger.info(f"  - {file.name}")


if __name__ == "__main__":
    main()

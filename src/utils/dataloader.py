"""
Data loading utilities for pre-sampled activation data.

This module handles loading pre-sampled top-activating text excerpts
provided by supervisors, bypassing the percentile sampling step.
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def load_presampled_json(
    data_dir: str,
    layer_id: int,
    unit_id: int,
    expected_samples: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load pre-sampled data from JSON files provided by supervisors.

    Expected format:
    - Directory with files named: layer{LAYER_ID}_{UNIT_ID}.json
    - Each JSON file contains a list of text strings
    - Text strings may have [highlighted] tokens in brackets

    Args:
        data_dir: Directory containing JSON files
        layer_id: Target layer ID
        unit_id: Target unit/neuron ID
        expected_samples: Expected number of samples (for validation)

    Returns:
        candidate_inputs_decoded: List of text excerpts
        cluster_activations: Array of activation values (dummy values since not provided)

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If data format is invalid
    """
    # Construct filename
    filename = f"layer{layer_id}_{unit_id}.json"
    file_path = Path(data_dir) / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Pre-sampled data file not found: {file_path}\n"
            f"Expected filename format: layer{{LAYER_ID}}_{{UNIT_ID}}.json"
        )

    logger.info(f"Loading pre-sampled data from {file_path}")

    # Load JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate format
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON to contain a list, got {type(data)}")

    if not all(isinstance(item, str) for item in data):
        raise ValueError("Expected all items in JSON list to be strings")

    candidate_inputs_decoded = data

    # Since activation values are not provided, use dummy values
    # These won't be used since we're clustering with embedding model (not activations)
    cluster_activations = np.ones(len(candidate_inputs_decoded), dtype=np.float32)

    # Validate sample count
    actual_samples = len(candidate_inputs_decoded)
    if expected_samples is not None and abs(actual_samples - expected_samples) > 10:
        logger.warning(
            f"Expected ~{expected_samples} samples but got {actual_samples}. "
            "Proceeding with available data."
        )

    logger.info(
        f"Successfully loaded {actual_samples} pre-sampled text excerpts "
        f"for layer {layer_id}, unit {unit_id}"
    )

    return candidate_inputs_decoded, cluster_activations


def load_presampled_data(
    file_path: str,
    layer_id: int,
    unit_id: int,
    expected_samples: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load pre-sampled activation data from CSV file.

    Expected CSV format (to be confirmed with actual dataset):
    - Columns: 'layer', 'unit', 'text', 'activation', ...
    OR
    - Columns: 'text', 'activation' (if file is already filtered for specific neuron)

    Args:
        file_path: Path to pre-sampled data CSV
        layer_id: Target layer ID
        unit_id: Target unit/neuron ID
        expected_samples: Expected number of samples (for validation)

    Returns:
        candidate_inputs_decoded: List of text excerpts
        cluster_activations: Array of activation values

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid or neuron not found
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pre-sampled data file not found: {file_path}")

    logger.info(f"Loading pre-sampled data from {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Check if data needs filtering by layer/unit
    if 'layer' in df.columns and 'unit' in df.columns:
        # Filter for specific neuron
        df_filtered = df[(df['layer'] == layer_id) & (df['unit'] == unit_id)]
        if len(df_filtered) == 0:
            raise ValueError(
                f"No data found for layer={layer_id}, unit={unit_id}. "
                f"Available combinations: {df[['layer', 'unit']].drop_duplicates().to_dict('records')[:5]}"
            )
        df = df_filtered
        logger.info(f"Filtered to {len(df)} samples for layer={layer_id}, unit={unit_id}")

    # Validate required columns
    if 'text' not in df.columns:
        raise ValueError(f"Required column 'text' not found. Available columns: {df.columns.tolist()}")

    if 'activation' not in df.columns:
        raise ValueError(f"Required column 'activation' not found. Available columns: {df.columns.tolist()}")

    # Extract data
    candidate_inputs_decoded = df['text'].tolist()
    cluster_activations = df['activation'].values

    # Validate sample count
    actual_samples = len(candidate_inputs_decoded)
    if expected_samples is not None and actual_samples != expected_samples:
        logger.warning(
            f"Expected {expected_samples} samples but got {actual_samples}. "
            "Proceeding with available data."
        )

    # Validate activations
    if len(cluster_activations) != len(candidate_inputs_decoded):
        raise ValueError(
            f"Mismatch between text samples ({len(candidate_inputs_decoded)}) "
            f"and activation values ({len(cluster_activations)})"
        )

    logger.info(
        f"Successfully loaded {actual_samples} pre-sampled text excerpts "
        f"with activation range [{cluster_activations.min():.4f}, {cluster_activations.max():.4f}]"
    )

    return candidate_inputs_decoded, cluster_activations


def validate_presampled_format(file_path: str) -> Dict[str, any]:
    """
    Validate and inspect pre-sampled data file format.

    Useful for checking supervisor's dataset before running full experiment.

    Args:
        file_path: Path to pre-sampled data CSV

    Returns:
        Dictionary with format information:
        - columns: List of column names
        - num_rows: Total number of rows
        - neurons: List of (layer, unit) tuples if applicable
        - sample_counts: Samples per neuron if applicable
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    info = {
        'columns': df.columns.tolist(),
        'num_rows': len(df),
        'has_layer_unit': 'layer' in df.columns and 'unit' in df.columns,
        'has_text': 'text' in df.columns,
        'has_activation': 'activation' in df.columns,
    }

    # If data includes layer/unit, get neuron list
    if info['has_layer_unit']:
        neurons = df[['layer', 'unit']].drop_duplicates()
        info['neurons'] = neurons.to_dict('records')
        info['num_neurons'] = len(neurons)

        # Count samples per neuron
        sample_counts = df.groupby(['layer', 'unit']).size()
        info['sample_counts'] = {
            'min': sample_counts.min(),
            'max': sample_counts.max(),
            'mean': sample_counts.mean(),
            'median': sample_counts.median(),
        }

    return info


def get_neuron_list(file_path: str) -> List[Tuple[int, int]]:
    """
    Extract list of (layer_id, unit_id) tuples from pre-sampled data.

    Args:
        file_path: Path to pre-sampled data CSV OR directory with JSON files

    Returns:
        List of (layer_id, unit_id) tuples
    """
    path = Path(file_path)

    # Check if it's a directory (JSON format) or CSV file
    if path.is_dir():
        # JSON format: parse filenames
        neurons = []
        json_files = list(path.glob("layer*.json"))

        import re
        for json_file in json_files:
            # Parse filename: layer{LAYER_ID}_{UNIT_ID}.json
            match = re.search(r'layer(\d+)_(\d+)\.json', json_file.name)
            if match:
                layer_id = int(match.group(1))
                unit_id = int(match.group(2))
                neurons.append((layer_id, unit_id))
            else:
                logger.warning(f"Could not parse filename: {json_file.name}")

        neurons.sort()  # Sort by layer, then unit
        logger.info(f"Found {len(neurons)} unique neurons in {path}")
        return neurons

    else:
        # CSV format: read from columns
        df = pd.read_csv(file_path)

        if 'layer' not in df.columns or 'unit' not in df.columns:
            raise ValueError(
                "Cannot extract neuron list: 'layer' and 'unit' columns not found. "
                "File may contain data for single neuron only."
            )

        neurons = df[['layer', 'unit']].drop_duplicates().values.tolist()
        neurons = [(int(layer), int(unit)) for layer, unit in neurons]

        logger.info(f"Found {len(neurons)} unique neurons in dataset")

        return neurons

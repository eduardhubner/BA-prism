"""
Batch experiment runner for adaptive k-selection evaluation.

This script runs the full adaptive k experiment across multiple neurons
and multiple k-selection methods (Davies-Bouldin, BIC, Silhouette).

Usage:
    python src/run_full_experiment.py

Configuration:
    - Set USE_PRESAMPLED_DATA = True in config.py
    - Set PRESAMPLED_DATA_PATH to your supervisor's dataset
    - Optionally adjust PARALLEL_PROCESSES for multiprocessing
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

# Import config and dataloader
sys.path.insert(0, str(Path(__file__).parent))
from utils import config, dataloader, helper_modules

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment_runner.log'),
        logging.StreamHandler()
    ]
)

# Experiment configuration
ADAPTIVE_K_METHODS = ["davies_bouldin", "bic", "silhouette"]
PARALLEL_PROCESSES = 1  # Set to 1 for safe sequential execution (4 hours total)
RUN_BASELINE = True  # Also run fixed k=5 baseline for comparison
USE_SUBSET = False  # Use all 60 neurons
SUBSET_SIZE_PER_LAYER = 10  # How many neurons to use per layer when USE_SUBSET=True


def check_result_exists(layer_id: int, unit_id: int, method: str, use_adaptive: bool) -> bool:
    """
    Check if a valid result file already exists for this neuron+method combination.

    Args:
        layer_id: Layer ID
        unit_id: Unit/neuron ID
        method: K-selection method
        use_adaptive: Whether using adaptive k

    Returns:
        True if valid result exists, False otherwise
    """
    import csv

    # Determine method suffix for filename
    if use_adaptive:
        method_suffix = f"_{method}"
    else:
        method_suffix = "_fixed-k5"

    # Look for CSV files in the descriptions directory
    # Using hardcoded path since config attributes may vary
    csv_dir = Path("descriptions") / "gemini-2-5-flash" / config.TARGET_MODEL_NAME

    # Find files matching this neuron and method (ignoring timestamp)
    pattern = f"{config.TARGET_MODEL_NAME}_layer-{layer_id}_unit-{unit_id}{method_suffix}_*.csv"
    matching_files = list(csv_dir.glob(pattern))

    if not matching_files:
        return False

    # Check if the most recent file has valid content (more than just header)
    most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(most_recent, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Valid file should have header + at least 1 description row
            if len(rows) > 1:
                logger.info(f"  ✓ Result already exists: {most_recent.name}")
                return True
            else:
                logger.info(f"  ⚠️  Empty result file found: {most_recent.name}")
                return False
    except Exception as e:
        logger.warning(f"  ⚠️  Error reading result file {most_recent.name}: {e}")
        return False


def validate_dataset(data_path: str) -> Tuple[bool, str]:
    """
    Validate that pre-sampled dataset directory exists and contains JSON files.

    Args:
        data_path: Path to directory containing JSON files

    Returns:
        (is_valid, message): Tuple of validation status and message
    """
    try:
        from pathlib import Path
        import json

        data_dir = Path(data_path)

        # Check directory exists
        if not data_dir.exists():
            return False, f"Directory does not exist: {data_path}"

        if not data_dir.is_dir():
            return False, f"Path is not a directory: {data_path}"

        # Find all JSON files
        json_files = list(data_dir.glob("layer*.json"))

        if len(json_files) == 0:
            return False, f"No JSON files found in {data_path}"

        # Validate a sample file
        sample_file = json_files[0]
        with open(sample_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return False, f"JSON file should contain a list of texts"

        if len(data) == 0:
            return False, f"JSON file is empty"

        logger.info(f"Dataset validation passed:")
        logger.info(f"  - Directory: {data_path}")
        logger.info(f"  - JSON files found: {len(json_files)}")
        logger.info(f"  - Sample file: {sample_file.name}")
        logger.info(f"  - Texts in sample: {len(data)}")

        return True, "Dataset valid"

    except Exception as e:
        return False, f"Validation error: {e}"


def run_single_experiment(
    layer_id: int,
    unit_id: int,
    method: str,
    use_adaptive: bool = True
) -> Tuple[bool, str]:
    """
    Run feature description for a single neuron with specified method.

    Args:
        layer_id: Layer ID
        unit_id: Unit/neuron ID
        method: K-selection method ("davies_bouldin", "bic", "silhouette", or "fixed")
        use_adaptive: Whether to use adaptive k (False for baseline)

    Returns:
        (success, message): Tuple indicating success and result message
    """
    # Check if result already exists
    if check_result_exists(layer_id, unit_id, method, use_adaptive):
        run_name = f"L{layer_id}_U{unit_id}_{method if use_adaptive else 'fixed_k5'}"
        return True, f"{run_name}: SKIPPED (already exists)"

    start_time = time.time()

    # Prepare environment variables to override config
    import os
    env = os.environ.copy()
    env['LAYER_ID'] = str(layer_id)
    env['UNIT_ID'] = str(unit_id)

    if use_adaptive:
        env['ADAPTIVE_K'] = 'True'
        env['ADAPTIVE_K_METHOD'] = method
        run_name = f"L{layer_id}_U{unit_id}_{method}"
    else:
        env['ADAPTIVE_K'] = 'False'
        run_name = f"L{layer_id}_U{unit_id}_fixed_k5"

    logger.info(f"Starting: {run_name}")

    try:
        # Get the absolute path to the src directory
        src_dir = Path(__file__).parent.absolute()

        # Create temporary config override (safer than env vars)
        config_overrides = f"""
import sys
from pathlib import Path
sys.path.insert(0, r'{src_dir}')
from utils import config

config.LAYER_ID = {layer_id}
config.UNIT_ID = {unit_id}
config.ADAPTIVE_K = {use_adaptive}
config.ADAPTIVE_K_METHOD = "{method}"
config.USE_PRESAMPLED_DATA = True

# Import and run feature_description
import feature_description
"""

        # Run as subprocess to ensure clean environment
        # Pass through necessary environment variables
        import os
        env = os.environ.copy()

        # Load .env file if it exists
        env_file = src_dir.parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env[key.strip()] = value.strip()

        result = subprocess.run(
            [sys.executable, '-c', config_overrides],
            cwd=src_dir.parent,  # Run from project root
            capture_output=True,
            text=True,
            env=env,  # Pass through environment variables
            timeout=7200  # 2 hour timeout per neuron
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"Completed: {run_name} in {elapsed:.1f}s")
            return True, f"{run_name}: SUCCESS ({elapsed:.1f}s)"
        else:
            logger.error(f"Failed: {run_name}")
            logger.error(f"STDERR: {result.stderr}")
            return False, f"{run_name}: FAILED - {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {run_name}")
        return False, f"{run_name}: TIMEOUT (>2h)"
    except Exception as e:
        logger.error(f"Error in {run_name}: {e}")
        return False, f"{run_name}: ERROR - {str(e)[:200]}"


def run_neuron_all_methods(
    neuron: Tuple[int, int],
    methods: List[str],
    run_baseline: bool
) -> List[Tuple[bool, str]]:
    """
    Run all k-selection methods for a single neuron.

    Args:
        neuron: (layer_id, unit_id) tuple
        methods: List of k-selection methods to run
        run_baseline: Whether to also run fixed k=5 baseline

    Returns:
        List of (success, message) tuples for each method
    """
    layer_id, unit_id = neuron
    results = []

    logger.info("=" * 80)
    logger.info(f"Processing neuron: Layer {layer_id}, Unit {unit_id}")
    logger.info("=" * 80)

    # Run adaptive k methods
    for method in methods:
        success, message = run_single_experiment(layer_id, unit_id, method, use_adaptive=True)
        results.append((success, message))

    # Run baseline if requested
    if run_baseline:
        success, message = run_single_experiment(layer_id, unit_id, "fixed", use_adaptive=False)
        results.append((success, message))

    return results


def run_experiment_sequential(
    neurons: List[Tuple[int, int]],
    methods: List[str],
    run_baseline: bool
):
    """Run experiment sequentially (one neuron at a time)."""
    all_results = []
    total_runs = len(neurons) * (len(methods) + (1 if run_baseline else 0))
    completed = 0

    logger.info("=" * 80)
    logger.info(f"STARTING SEQUENTIAL EXPERIMENT")
    logger.info(f"Total neurons: {len(neurons)}")
    logger.info(f"Methods per neuron: {methods} {'+ fixed_k5' if run_baseline else ''}")
    logger.info(f"Total runs: {total_runs}")
    logger.info("=" * 80)

    start_time = time.time()

    for neuron in neurons:
        results = run_neuron_all_methods(neuron, methods, run_baseline)
        all_results.extend(results)
        completed += len(results)

        # Progress update
        elapsed = time.time() - start_time
        avg_time_per_run = elapsed / completed
        remaining_runs = total_runs - completed
        eta_seconds = avg_time_per_run * remaining_runs
        eta_hours = eta_seconds / 3600

        logger.info(f"Progress: {completed}/{total_runs} runs completed")
        logger.info(f"Estimated time remaining: {eta_hours:.1f} hours")

    return all_results


def run_experiment_parallel(
    neurons: List[Tuple[int, int]],
    methods: List[str],
    run_baseline: bool,
    num_processes: int
):
    """Run experiment in parallel (multiple neurons at once)."""
    logger.info("=" * 80)
    logger.info(f"STARTING PARALLEL EXPERIMENT")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"Total neurons: {len(neurons)}")
    logger.info(f"Methods per neuron: {methods} {'+ fixed_k5' if run_baseline else ''}")
    logger.info("=" * 80)

    # Create partial function with fixed parameters
    process_func = partial(run_neuron_all_methods, methods=methods, run_baseline=run_baseline)

    # Run with multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        results_per_neuron = pool.map(process_func, neurons)

    # Flatten results
    all_results = [result for results in results_per_neuron for result in results]

    return all_results


def main():
    """Main experiment runner."""
    logger.info("=" * 80)
    logger.info("ADAPTIVE K-SELECTION EXPERIMENT RUNNER")
    logger.info("=" * 80)

    # Validate configuration
    if not config.USE_PRESAMPLED_DATA:
        logger.error("ERROR: USE_PRESAMPLED_DATA must be True in config.py")
        logger.error("Please update config.py before running experiment")
        return

    # Validate dataset
    logger.info(f"Validating dataset: {config.PRESAMPLED_DATA_PATH}")
    is_valid, message = validate_dataset(config.PRESAMPLED_DATA_PATH)

    if not is_valid:
        logger.error(f"Dataset validation failed: {message}")
        logger.error("Please check PRESAMPLED_DATA_PATH in config.py")
        return

    # Get list of neurons from dataset
    all_neurons = dataloader.get_neuron_list(config.PRESAMPLED_DATA_PATH)
    logger.info(f"Found {len(all_neurons)} neurons in dataset")

    # Optionally use subset for faster experimentation
    if USE_SUBSET:
        # Separate by layer
        layer0 = [n for n in all_neurons if n[0] == 0]
        layer20 = [n for n in all_neurons if n[0] == 20]
        layer40 = [n for n in all_neurons if n[0] == 40]

        # Take first N from each layer
        neurons = (
            layer0[:SUBSET_SIZE_PER_LAYER] +
            layer20[:SUBSET_SIZE_PER_LAYER] +
            layer40[:SUBSET_SIZE_PER_LAYER]
        )
        logger.info(f"Using subset: {len(neurons)} neurons ({SUBSET_SIZE_PER_LAYER} per layer)")
        logger.info(f"  Layer 0: {len([n for n in neurons if n[0] == 0])} neurons")
        logger.info(f"  Layer 20: {len([n for n in neurons if n[0] == 20])} neurons")
        logger.info(f"  Layer 40: {len([n for n in neurons if n[0] == 40])} neurons")
    else:
        neurons = all_neurons
        logger.info(f"Using all {len(neurons)} neurons")

    # Confirm before starting
    total_runs = len(neurons) * (len(ADAPTIVE_K_METHODS) + (1 if RUN_BASELINE else 0))
    logger.info(f"This will run {total_runs} total experiments")
    logger.info(f"Methods: {ADAPTIVE_K_METHODS} {'+ fixed_k5' if RUN_BASELINE else ''}")

    # Estimate time (assuming ~30-45 min per run with pre-sampled data)
    estimated_hours = (total_runs * 37.5) / 60  # 37.5 min average
    if PARALLEL_PROCESSES > 1:
        estimated_hours /= PARALLEL_PROCESSES

    logger.info(f"Estimated total time: {estimated_hours:.1f} hours")

    # Run experiment
    start_time = time.time()

    if PARALLEL_PROCESSES > 1:
        results = run_experiment_parallel(
            neurons, ADAPTIVE_K_METHODS, RUN_BASELINE, PARALLEL_PROCESSES
        )
    else:
        results = run_experiment_sequential(neurons, ADAPTIVE_K_METHODS, RUN_BASELINE)

    # Summary
    elapsed_hours = (time.time() - start_time) / 3600
    successes = sum(1 for success, _ in results if success)
    failures = len(results) - successes

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed_hours:.2f} hours")
    logger.info(f"Successful runs: {successes}/{len(results)}")
    logger.info(f"Failed runs: {failures}/{len(results)}")

    if failures > 0:
        logger.info("\nFailed runs:")
        for success, message in results:
            if not success:
                logger.info(f"  - {message}")

    logger.info("\nResults saved to: descriptions/ and logs/")
    logger.info("Run aggregate_results.py to compile analysis")


if __name__ == "__main__":
    main()

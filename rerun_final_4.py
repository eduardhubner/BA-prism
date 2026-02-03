"""
Re-run the final 4 failed experiments.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rerun_final_4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The 4 remaining failures
FAILED_RUNS = [
    (0, 3943, "silhouette", True),
    (20, 1790, "davies_bouldin", True),
    (20, 2700, "davies_bouldin", True),
    (0, 2725, "fixed-k5", False),
]

def run_single_experiment(layer_id, unit_id, method, use_adaptive=True):
    """Run feature description for a single neuron with specified method."""
    start_time = time.time()

    if use_adaptive:
        run_name = f"L{layer_id}_U{unit_id}_{method}"
    else:
        run_name = f"L{layer_id}_U{unit_id}_fixed_k5"

    logger.info(f"Starting: {run_name}")

    try:
        src_dir = Path(__file__).parent / "src"

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

        import os
        env = os.environ.copy()

        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env[key.strip()] = value.strip()

        result = subprocess.run(
            [sys.executable, '-c', config_overrides],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            env=env,
            timeout=7200
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ Completed: {run_name} in {elapsed:.1f}s")
            return True, f"{run_name}: SUCCESS ({elapsed:.1f}s)"
        else:
            logger.error(f"✗ Failed: {run_name}")
            logger.error(f"STDERR: {result.stderr[:500]}")
            return False, f"{run_name}: FAILED - {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout: {run_name}")
        return False, f"{run_name}: TIMEOUT (>2h)"
    except Exception as e:
        logger.error(f"✗ Error in {run_name}: {e}")
        return False, f"{run_name}: ERROR - {str(e)[:200]}"


def main():
    logger.info("=" * 80)
    logger.info("RE-RUNNING FINAL 4 FAILED EXPERIMENTS")
    logger.info("=" * 80)

    successes = []
    failures = []

    for i, (layer_id, unit_id, method, use_adaptive) in enumerate(FAILED_RUNS, 1):
        logger.info(f"\n[{i}/4] Layer {layer_id}, Unit {unit_id}, Method {method}")

        success, message = run_single_experiment(layer_id, unit_id, method, use_adaptive)

        if success:
            successes.append((layer_id, unit_id, method))
        else:
            failures.append((layer_id, unit_id, method, message))

        logger.info(f"Progress: {i}/4, Successes: {len(successes)}, Failures: {len(failures)}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RE-RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total: 4")
    logger.info(f"Successful: {len(successes)}")
    logger.info(f"Failed: {len(failures)}")

    if failures:
        logger.info("\nStill failed:")
        for layer_id, unit_id, method, message in failures:
            logger.info(f"  - Layer {layer_id}, Unit {unit_id}, Method {method}")


if __name__ == "__main__":
    main()

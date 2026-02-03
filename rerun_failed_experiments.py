"""
Re-run the 29 failed experiments identified by comprehensive validation.
Reads from truly_failed_runs.txt and re-runs each failed neuron+method combination.
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
        logging.FileHandler('logs/rerun_failed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_single_experiment(layer_id, unit_id, method, use_adaptive=True):
    """Run feature description for a single neuron with specified method."""
    start_time = time.time()

    if use_adaptive:
        run_name = f"L{layer_id}_U{unit_id}_{method}"
    else:
        run_name = f"L{layer_id}_U{unit_id}_fixed_k5"

    logger.info(f"Starting: {run_name}")

    try:
        # Get absolute path to src directory
        src_dir = Path(__file__).parent / "src"

        # Create config override
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

        # Load environment variables from .env file
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

        # Run subprocess
        result = subprocess.run(
            [sys.executable, '-c', config_overrides],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            env=env,
            timeout=7200  # 2 hour timeout
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
    """Read failed runs and re-run them."""

    # Read the failed runs file
    failed_runs_file = Path("truly_failed_runs.txt")

    if not failed_runs_file.exists():
        logger.error("truly_failed_runs.txt not found! Run comprehensive_validation.py first.")
        return

    # Parse failed runs
    failed_runs = []
    with open(failed_runs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) >= 3:
                layer_id = int(parts[0])
                unit_id = int(parts[1])
                method = parts[2]
                reason = parts[3] if len(parts) > 3 else "unknown"
                failed_runs.append((layer_id, unit_id, method, reason))

    total_runs = len(failed_runs)
    logger.info("=" * 80)
    logger.info(f"RE-RUNNING {total_runs} FAILED EXPERIMENTS")
    logger.info("=" * 80)

    successes = []
    failures = []

    for i, (layer_id, unit_id, method, reason) in enumerate(failed_runs, 1):
        logger.info(f"\n[{i}/{total_runs}] Re-running: Layer {layer_id}, Unit {unit_id}, Method {method}")
        logger.info(f"  Reason for re-run: {reason}")

        # Determine if adaptive or fixed
        use_adaptive = (method != "fixed-k5")

        # Run the experiment
        success, message = run_single_experiment(layer_id, unit_id, method, use_adaptive)

        if success:
            successes.append((layer_id, unit_id, method))
        else:
            failures.append((layer_id, unit_id, method, message))

        # Progress update
        logger.info(f"Progress: {i}/{total_runs} re-runs attempted")
        logger.info(f"  Successes: {len(successes)}, Failures: {len(failures)}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("RE-RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total re-runs attempted: {total_runs}")
    logger.info(f"Successful: {len(successes)}")
    logger.info(f"Failed: {len(failures)}")
    logger.info(f"Success rate: {len(successes)/total_runs*100:.1f}%")

    if failures:
        logger.info("\nStill failed after re-run:")
        for layer_id, unit_id, method, message in failures:
            logger.info(f"  - Layer {layer_id}, Unit {unit_id}, Method {method}")
            logger.info(f"    {message}")

    logger.info("\nRe-run complete! Run comprehensive_validation.py again to verify.")


if __name__ == "__main__":
    main()

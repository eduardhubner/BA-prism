#!/usr/bin/env python3
"""
Run evaluation for multiple neurons sequentially.
Usage: python run_multiple_neurons.py --layer 0 --neurons 440,1149,1749
"""
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_neuron_evaluation(layer, unit):
    """Run evaluation for a single neuron."""
    print("\n" + "="*80)
    print(f"EVALUATING: Layer {layer}, Unit {unit}")
    print("="*80)

    start_time = time.time()

    # Create CSV for this neuron
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Creating evaluation CSV...")
    result = subprocess.run(
        ['python', 'evaluate_one_neuron_simple.py', '--layer', str(layer), '--unit', str(unit)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to create CSV: {result.stderr}")
        return False

    print(result.stdout)

    # Update config.py with the correct CSV filename
    csv_filename = f"eval_neuron_L{layer}_U{unit}.csv"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating config.py to use {csv_filename}...")

    config_path = Path("src/utils/config.py")
    with open(config_path, 'r') as f:
        config_content = f.read()

    # Replace EXPLAIN_FILE line
    import re
    new_content = re.sub(
        r'EXPLAIN_FILE = "eval_neuron_L\d+_U\d+\.csv"',
        f'EXPLAIN_FILE = "{csv_filename}"',
        config_content
    )

    with open(config_path, 'w') as f:
        f.write(new_content)

    # Run evaluation
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation...")
    # Load environment and run evaluation.py
    result = subprocess.run(
        ['bash', '-c', 'set -a && source .env && set +a && python src/evaluation.py'],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n❌ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {result.stderr[-500:]}")  # Show last 500 chars of error
        return False

def main():
    parser = argparse.ArgumentParser(description='Run evaluation for multiple neurons')
    parser.add_argument('--layer', type=int, required=True, help='Layer ID (0, 20, or 40)')
    parser.add_argument('--neurons', type=str, required=True, help='Comma-separated list of unit IDs (e.g., 440,1149,1749)')
    args = parser.parse_args()

    # Parse neurons
    neurons = [int(u.strip()) for u in args.neurons.split(',')]

    print("="*80)
    print("BATCH EVALUATION")
    print("="*80)
    print(f"Layer: {args.layer}")
    print(f"Neurons: {neurons}")
    print(f"Total: {len(neurons)} neurons")
    print("="*80)

    # Confirm
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run evaluations
    results = []
    start_time = time.time()

    for i, unit in enumerate(neurons, 1):
        print(f"\n\n{'='*80}")
        print(f"PROGRESS: {i}/{len(neurons)} neurons")
        print(f"Elapsed: {(time.time() - start_time)/60:.1f} minutes")
        print(f"{'='*80}")

        success = run_neuron_evaluation(args.layer, unit)
        results.append((args.layer, unit, success))

        # Save progress
        with open('evaluation_progress.txt', 'a') as f:
            f.write(f"{datetime.now()}: Layer {args.layer}, Unit {unit} - {'SUCCESS' if success else 'FAILED'}\n")

    # Summary
    total_time = time.time() - start_time
    successes = sum(1 for _, _, success in results if success)

    print("\n" + "="*80)
    print("BATCH EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {successes}/{len(neurons)}")
    print(f"Failed: {len(neurons) - successes}/{len(neurons)}")

    if successes < len(neurons):
        print("\nFailed neurons:")
        for layer, unit, success in results:
            if not success:
                print(f"  - Layer {layer}, Unit {unit}")

if __name__ == "__main__":
    main()

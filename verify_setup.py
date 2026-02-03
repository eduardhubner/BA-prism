#!/usr/bin/env python3
"""
Verify that all necessary files and dependencies are in place for batch evaluation.
Run this before starting batch evaluation to catch any issues early.
"""
import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - NOT FOUND: {path}")
        return False

def check_env_var(var_name):
    """Check if an environment variable is set."""
    if os.getenv(var_name):
        print(f"✅ Environment variable {var_name} is set")
        return True
    else:
        print(f"❌ Environment variable {var_name} is NOT SET")
        return False

def check_directory(path, description):
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - NOT FOUND: {path}")
        return False

def count_descriptions(layer, unit):
    """Count description files for a neuron."""
    desc_dir = Path("descriptions/gemini-2-5-flash/gpt2-xl")
    pattern = f"gpt2-xl_layer-{layer}_unit-{unit}_*.csv"
    files = list(desc_dir.glob(pattern))
    return len(files)

def main():
    print("="*80)
    print("BATCH EVALUATION SETUP VERIFICATION")
    print("="*80)

    all_good = True

    # Check critical scripts
    print("\n--- Critical Scripts ---")
    all_good &= check_file("run_multiple_neurons.py", "Batch evaluation script")
    all_good &= check_file("evaluate_one_neuron_simple.py", "Single neuron CSV preparation")
    all_good &= check_file("run_evaluation_iterative.sh", "Evaluation runner script")
    all_good &= check_file("src/evaluation_iterative.py", "Main evaluation script")
    all_good &= check_file("src/utils/models_iterative.py", "Iterative generation module")

    # Check analysis tools
    print("\n--- Analysis Tools ---")
    all_good &= check_file("analysis_notebook.ipynb", "Analysis Jupyter notebook")

    # Check directories
    print("\n--- Required Directories ---")
    all_good &= check_directory("descriptions/gemini-2-5-flash/gpt2-xl", "Description files directory")
    all_good &= check_directory("assets/explanations/GPT-explain", "Evaluation CSV directory")
    all_good &= check_directory("results", "Results directory")
    all_good &= check_directory("activations/gpt2-xl", "Activations cache directory")

    # Check control activations
    print("\n--- Control Activation Cache ---")
    layer0_cache = check_file(
        "activations/gpt2-xl/control_gpt2-xl_layer0_mean_cosmopedia_1000.pt",
        "Layer 0 control activations (CACHED - saves 67 mins per neuron)"
    )

    if not Path("activations/gpt2-xl/control_gpt2-xl_layer20_mean_cosmopedia_1000.pt").exists():
        print("⏳ Layer 20 control activations (will be generated on first neuron - adds 67 mins)")
    else:
        print("✅ Layer 20 control activations (CACHED)")

    if not Path("activations/gpt2-xl/control_gpt2-xl_layer40_mean_cosmopedia_1000.pt").exists():
        print("⏳ Layer 40 control activations (will be generated on first neuron - adds 67 mins)")
    else:
        print("✅ Layer 40 control activations (CACHED)")

    # Check environment variables (requires .env to be sourced)
    print("\n--- Environment Variables ---")
    print("(These should be set when running via run_evaluation_iterative.sh)")
    env_ok = check_env_var("GEMINI_KEY")
    if not env_ok:
        print("   NOTE: run_evaluation_iterative.sh will source .env automatically")

    # Check description files for sample neurons
    print("\n--- Sample Description Files ---")
    layer0_neurons = [440, 1149, 1749, 2725, 3057]
    layer20_neurons = [1004, 1406, 1424, 1790, 2325]
    layer40_neurons = [1516, 1555, 1823, 183, 3254]

    print("\nLayer 0 neurons:")
    for unit in layer0_neurons:
        count = count_descriptions(0, unit)
        if count == 4:
            print(f"  ✅ Unit {unit}: {count} description files")
        else:
            print(f"  ❌ Unit {unit}: {count} description files (expected 4)")
            all_good = False

    print("\nLayer 20 neurons:")
    for unit in layer20_neurons:
        count = count_descriptions(20, unit)
        if count == 4:
            print(f"  ✅ Unit {unit}: {count} description files")
        else:
            print(f"  ❌ Unit {unit}: {count} description files (expected 4)")
            all_good = False

    print("\nLayer 40 neurons:")
    for unit in layer40_neurons:
        count = count_descriptions(40, unit)
        if count == 4:
            print(f"  ✅ Unit {unit}: {count} description files")
        else:
            print(f"  ❌ Unit {unit}: {count} description files (expected 4)")
            all_good = False

    # Check existing results
    print("\n--- Existing Evaluation Results ---")
    results_file = Path("results/cosy-evaluation_target-gpt2-xl_textgen-gemini-2-5-flash_mean_evalgen-gemini-2-5-flash_cosmopedia_1000.csv")
    if results_file.exists():
        with open(results_file, 'r') as f:
            num_lines = sum(1 for _ in f) - 1  # Subtract header
        print(f"✅ Results file exists with {num_lines} evaluations")
        print(f"   Expected after full run: ~270 evaluations (15 neurons × 18 descriptions)")
    else:
        print("⏳ Results file does not exist yet (will be created during evaluation)")

    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready for batch evaluation!")
        print("\nTo start evaluation, run:")
        print("  python run_multiple_neurons.py --layer 0 --neurons 1149,1749,2725,3057,3124")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before running batch evaluation")
        print("\nSee errors above for details")
    print("="*80)

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())

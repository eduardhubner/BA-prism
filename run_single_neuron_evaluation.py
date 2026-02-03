"""
Run evaluation for a single neuron with cost tracking.

Usage:
    python run_single_neuron_evaluation.py --layer 0 --unit 440
"""

import argparse
import csv
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single neuron')
    parser.add_argument('--layer', type=int, required=True, help='Layer ID')
    parser.add_argument('--unit', type=int, required=True, help='Unit/neuron ID')
    args = parser.parse_args()

    print("=" * 80)
    print(f"SINGLE NEURON EVALUATION: Layer {args.layer}, Unit {args.unit}")
    print("=" * 80)

    # Find all CSV files for this neuron
    csv_dir = Path("descriptions/gemini-2-5-flash/gpt2-xl")
    pattern = f"*_layer-{args.layer}_unit-{args.unit}_*.csv"
    csv_files = sorted(csv_dir.glob(pattern))

    if not csv_files:
        print(f"\n❌ No description files found for Layer {args.layer}, Unit {args.unit}")
        return

    print(f"\nFound {len(csv_files)} description files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Collect all descriptions
    all_descriptions = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_descriptions.append({
                    'layer': row['layer'],
                    'unit': row['unit'],
                    'description': row['description']
                })

    print(f"\nTotal descriptions: {len(all_descriptions)}")

    # Estimate cost
    input_tokens_per_desc = 150
    output_tokens_per_desc = 7000  # 10 samples × 700 tokens
    total_input = len(all_descriptions) * input_tokens_per_desc
    total_output = len(all_descriptions) * output_tokens_per_desc

    input_cost_usd = total_input * 0.30 / 1_000_000
    output_cost_usd = total_output * 2.50 / 1_000_000
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_eur = total_cost_usd * 0.95

    print(f"\n--- Cost Estimate ---")
    print(f"Input tokens: {total_input:,}")
    print(f"Output tokens: {total_output:,}")
    print(f"Estimated cost: ${total_cost_usd:.4f} (~€{total_cost_eur:.4f})")

    # Create temporary CSV for this neuron only
    temp_csv = Path("temp_single_neuron_eval.csv")
    with open(temp_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'unit', 'description'])
        writer.writeheader()
        writer.writerows(all_descriptions)

    print(f"\n✅ Created temporary file: {temp_csv}")
    print(f"\n--- Next Steps ---")
    print(f"1. Update src/utils/config.py:")
    print(f"   - Set EXPLAIN_FILE = '{temp_csv.name}'")
    print(f"   - Set MULTI_EVAL = True")
    print(f"   - Set EVALUATION_TEXT_GENERATOR_NAME = 'gemini-2.5-flash'")
    print(f"")
    print(f"2. Run evaluation:")
    print(f"   python src/evaluation.py")
    print(f"")
    print(f"3. Monitor costs in Google Cloud Console")
    print(f"")
    print(f"4. After evaluation, check your actual costs vs estimate")
    print(f"")
    print(f"⚠️  Important: The evaluation script will run on ALL neurons in the CSV.")
    print(f"   This temp file contains only Layer {args.layer}, Unit {args.unit}")


if __name__ == "__main__":
    main()

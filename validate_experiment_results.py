"""
Comprehensive validation of all experiment runs.
Checks for missing files, empty files, and partial completions.
"""

import csv
from pathlib import Path
import re

# Find all JSON files in the data directory to determine expected neurons
data_dir = Path("data/candidate_inputs_decoded")
json_files = list(data_dir.glob("layer*.json"))

# Extract (layer, unit) tuples from filenames
neurons = []
for json_file in json_files:
    match = re.match(r'layer(\d+)_(\d+)\.json', json_file.name)
    if match:
        layer_id = int(match.group(1))
        unit_id = int(match.group(2))
        neurons.append((layer_id, unit_id))

neurons = sorted(neurons)

methods = ["davies_bouldin", "bic", "silhouette", "fixed-k5"]
csv_dir = Path("descriptions/gemini-2-5-flash/gpt2-xl")

print("=" * 80)
print("VALIDATION OF ALL EXPERIMENT RUNS")
print("=" * 80)
print(f"Found {len(neurons)} neurons in dataset")
print(f"Expected {len(neurons) * len(methods)} total runs")
print("=" * 80)

missing = []
empty = []
partial_fixed_k5 = []
suspicious_adaptive = []
complete = []

for layer_id, unit_id in neurons:
    for method in methods:
        pattern = f"gpt2-xl_layer-{layer_id}_unit-{unit_id}_{method}_*.csv"
        files = list(csv_dir.glob(pattern))

        if not files:
            missing.append((layer_id, unit_id, method))
        else:
            # Check file content
            csv_file = files[0]
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    num_descriptions = len(rows) - 1  # Subtract header

                    if num_descriptions == 0:
                        empty.append((layer_id, unit_id, method, csv_file.name))
                    elif method == "fixed-k5":
                        if num_descriptions == 5:
                            complete.append((layer_id, unit_id, method, num_descriptions))
                        else:
                            partial_fixed_k5.append((layer_id, unit_id, method, num_descriptions, csv_file.name))
                    else:
                        # Adaptive methods: should have 2-10 descriptions
                        if 2 <= num_descriptions <= 10:
                            complete.append((layer_id, unit_id, method, num_descriptions))
                        else:
                            suspicious_adaptive.append((layer_id, unit_id, method, num_descriptions, csv_file.name))
            except Exception as e:
                empty.append((layer_id, unit_id, method, f"ERROR: {e}"))

print("\n1. MISSING FILES (no CSV exists)")
print("-" * 80)
if missing:
    for layer_id, unit_id, method in missing:
        print(f"   Layer {layer_id}, Unit {unit_id}, Method {method}")
else:
    print("   None!")

print(f"\n2. EMPTY FILES (CSV exists but no descriptions)")
print("-" * 80)
if empty:
    for item in empty:
        if len(item) == 4:
            layer_id, unit_id, method, filename = item
            print(f"   Layer {layer_id}, Unit {unit_id}, Method {method}")
            print(f"      File: {filename}")
        else:
            print(f"   {item}")
else:
    print("   None!")

print(f"\n3. PARTIAL FIXED-K5 (not exactly 5 descriptions)")
print("-" * 80)
if partial_fixed_k5:
    for layer_id, unit_id, method, count, filename in partial_fixed_k5:
        print(f"   Layer {layer_id}, Unit {unit_id}, Method {method} - {count}/5 descriptions")
        print(f"      File: {filename}")
else:
    print("   None!")

print(f"\n4. SUSPICIOUS ADAPTIVE (not in 2-10 range)")
print("-" * 80)
if suspicious_adaptive:
    for layer_id, unit_id, method, count, filename in suspicious_adaptive:
        print(f"   Layer {layer_id}, Unit {unit_id}, Method {method} - {count} descriptions")
        print(f"      File: {filename}")
else:
    print("   None!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
total_expected = len(neurons) * len(methods)
total_failed = len(missing) + len(empty) + len(partial_fixed_k5) + len(suspicious_adaptive)
print(f"Total expected runs: {total_expected}")
print(f"Complete runs: {len(complete)}")
print(f"Failed runs: {total_failed}")
print(f"  - Missing: {len(missing)}")
print(f"  - Empty: {len(empty)}")
print(f"  - Partial fixed-k5: {len(partial_fixed_k5)}")
print(f"  - Suspicious adaptive: {len(suspicious_adaptive)}")
print(f"Success rate: {len(complete)/total_expected*100:.1f}%")

# Save failed runs to a file for re-running
if total_failed > 0:
    print("\n" + "=" * 80)
    print("SAVING FAILED RUNS TO FILE")
    print("=" * 80)

    with open('failed_runs_to_repeat.txt', 'w') as f:
        f.write("# Failed runs that need to be repeated\n")
        f.write("# Format: layer_id,unit_id,method\n\n")

        all_failed = []
        all_failed.extend([(l, u, m) for l, u, m in missing])
        all_failed.extend([(l, u, m) for l, u, m, _ in empty])
        all_failed.extend([(l, u, m) for l, u, m, _, _ in partial_fixed_k5])
        all_failed.extend([(l, u, m) for l, u, m, _, _ in suspicious_adaptive])

        for layer_id, unit_id, method in sorted(all_failed):
            f.write(f"{layer_id},{unit_id},{method}\n")

    print(f"Saved {total_failed} failed runs to 'failed_runs_to_repeat.txt'")

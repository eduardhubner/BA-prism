"""
Rename existing result files to include k-selection method in filename.

This script reads the log files to identify which method was used,
then renames both the log and CSV files accordingly.
"""

import re
from pathlib import Path

# Paths
logs_dir = Path("logs")
csv_dir = Path("descriptions/gemini-2-5-flash/gpt2-xl")

def get_method_from_log(log_path):
    """Extract method name from log file content."""
    with open(log_path, 'r') as f:
        content = f.read(2000)  # Read first 2000 chars

    # Look for ADAPTIVE_K config
    adaptive_match = re.search(r'ADAPTIVE_K = (True|False)', content)
    method_match = re.search(r'ADAPTIVE_K_METHOD = (\w+)', content)

    if not adaptive_match:
        return None

    if adaptive_match.group(1) == 'False':
        return 'fixed-k5'
    elif method_match:
        return method_match.group(1)
    else:
        return None

def rename_files_for_today():
    """Rename today's experiment files to include method name."""

    # Find all log files from today's experiment
    today_logs = list(logs_dir.glob("gpt2-xl_layer-*_unit-*_2025-11-16_*.log"))

    print(f"Found {len(today_logs)} log files from today's experiment")

    renamed_count = 0

    for log_file in today_logs:
        # Extract method from log content
        method = get_method_from_log(log_file)

        if not method:
            print(f"  ⚠️  Could not determine method for {log_file.name}")
            continue

        # Parse filename
        match = re.search(r'(gpt2-xl_layer-\d+_unit-\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', log_file.name)
        if not match:
            print(f"  ⚠️  Could not parse filename: {log_file.name}")
            continue

        prefix = match.group(1)
        timestamp = match.group(2)

        # New filenames with method
        new_log_name = f"{prefix}_{method}_{timestamp}.log"
        new_csv_name = f"{prefix}_{method}_{timestamp}.csv"

        # Rename log file
        new_log_path = logs_dir / new_log_name
        if log_file.name != new_log_name:
            log_file.rename(new_log_path)
            print(f"  ✓ Renamed log: {log_file.name} → {new_log_name}")

        # Rename corresponding CSV file
        old_csv_name = log_file.name.replace('.log', '.csv')
        old_csv_path = csv_dir / old_csv_name
        new_csv_path = csv_dir / new_csv_name

        if old_csv_path.exists() and old_csv_path.name != new_csv_name:
            old_csv_path.rename(new_csv_path)
            print(f"  ✓ Renamed CSV: {old_csv_name} → {new_csv_name}")

        renamed_count += 1

    print(f"\n✅ Renamed {renamed_count} file pairs")

if __name__ == "__main__":
    print("=" * 80)
    print("RENAMING EXISTING EXPERIMENT RESULTS")
    print("=" * 80)
    rename_files_for_today()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)

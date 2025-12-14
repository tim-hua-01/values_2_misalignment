"""
Regrade all logs in the logs/ folder using the fixed grading script.
"""
import os
import subprocess
from pathlib import Path
import re

def extract_info_from_log_filename(filename: str) -> tuple[str, str, str, str] | None:
    """Extract value pair, model, and timestamp from log filename."""
    if not filename.endswith('.eval'):
        return None
    
    # Remove .eval extension
    name = filename[:-5]
    
    # Split by last underscore to separate timestamp
    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        return None
    
    value_pair_and_model = parts[0]
    timestamp = parts[1]
    
    # Split by last underscore again to separate model
    parts2 = value_pair_and_model.rsplit('_', 1)
    if len(parts2) != 2:
        return None
    
    value_pair = parts2[0]
    model_short = parts2[1]
    
    # Extract value1 and value2 from value_pair (format: value1_vs_value2)
    if '_vs_' not in value_pair:
        return None
    
    # Split by _vs_ and convert back to normal names
    value1_safe, value2_safe = value_pair.split('_vs_', 1)
    
    return value1_safe, value2_safe, model_short, timestamp

def denormalize_value_name(safe_name: str) -> str:
    """Convert sanitized filename back to value name (best effort)."""
    # Replace underscores with spaces
    return safe_name.replace('_', ' ')

# Read the CSV files to get the actual value names
import pandas as pd

# Load both CSVs
minitest_df = pd.read_csv('petri_values_run_minitest.csv')
run1_df = pd.read_csv('petri_values_run1.csv')

# Combine into one dataframe
all_pairs_df = pd.concat([minitest_df, run1_df], ignore_index=True)

# Create a mapping from normalized names to actual names and csv_source
value_pair_map = {}
for _, row in all_pairs_df.iterrows():
    if pd.notna(row['value1']) and pd.notna(row['value2']):
        v1 = row['value1']
        v2 = row['value2']
        csv_source = row['csv_source']
        
        # Normalize for matching
        v1_norm = v1.replace(' ', '_').replace('-', '_').lower()
        v2_norm = v2.replace(' ', '_').replace('-', '_').lower()
        
        # Store both orderings
        value_pair_map[f"{v1_norm}_vs_{v2_norm}"] = (v1, v2, csv_source)
        value_pair_map[f"{v2_norm}_vs_{v1_norm}"] = (v2, v1, csv_source)

# Get all log files
logs_dir = Path('logs')
log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.eval')])

print(f"Found {len(log_files)} log files to regrade")
print()

# Create intermediate_results directory
Path('intermediate_results').mkdir(exist_ok=True)

# Regrade each log
successful = 0
failed = 0
skipped = 0

for log_file in log_files:
    print(f"Processing: {log_file}")
    
    result = extract_info_from_log_filename(log_file)
    if not result:
        print(f"  ⚠️  Skipped: Could not parse filename")
        skipped += 1
        continue
    
    v1_safe, v2_safe, model_short, timestamp = result
    
    # Try to match the value pair
    value_pair_key = f"{v1_safe}_vs_{v2_safe}".lower()
    
    if value_pair_key not in value_pair_map:
        print(f"  ⚠️  Skipped: Value pair not found in CSVs")
        skipped += 1
        continue
    
    value1, value2, csv_source = value_pair_map[value_pair_key]
    
    # Generate output filename
    intermediate_csv = f"intermediate_results/{v1_safe}_vs_{v2_safe}_{model_short}.csv"
    
    # Run grading script
    cmd = [
        'uv', 'run', 'python', 'petri_grade_script.py',
        '--log', f'logs/{log_file}',
        '--csv', csv_source,
        '--value1', value1,
        '--value2', value2,
        '--output', intermediate_csv
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Graded successfully")
        print(f"    Output: {intermediate_csv}")
        
        # Show if any epochs were filtered
        if 'WARNING' in result.stdout:
            for line in result.stdout.split('\n'):
                if 'WARNING' in line or 'filtered' in line.lower():
                    print(f"    {line}")
        
        successful += 1
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        failed += 1
    
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total logs: {len(log_files)}")
print(f"Successfully graded: {successful}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print()

if failed > 0:
    print("⚠️  Some logs failed to grade. Check output above for details.")
    exit(1)

print("✓ All logs graded successfully!")

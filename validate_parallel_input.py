#!/usr/bin/env python3
"""
Validate parallel petri input CSV before running the orchestrator.
Checks for valid values, valid csv sources, and prints configuration summary.
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter


def load_values_from_csv(csv_path: str) -> set[str]:
    """Load valid value names from a csv_source file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV source file not found: {csv_path}")
    
    values = set()
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        if 'value_name' not in reader.fieldnames:
            raise ValueError(f"CSV source {csv_path} missing 'value_name' column")
        for row in reader:
            values.add(row['value_name'].strip())
    return values


def validate_input_csv(input_csv: str) -> tuple[list[dict], list[str]]:
    """Validate the input CSV and return (rows, errors)."""
    path = Path(input_csv)
    if not path.exists():
        return [], [f"Input CSV not found: {input_csv}"]
    
    errors = []
    rows = []
    
    # Cache for loaded csv_source files
    csv_source_values: dict[str, set[str]] = {}
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Check required columns
        required_cols = {'value1', 'value2', 'csv_source'}
        if not reader.fieldnames:
            return [], ["Input CSV is empty or has no headers"]
        
        missing_cols = required_cols - set(reader.fieldnames)
        if missing_cols:
            return [], [f"Missing required columns: {missing_cols}"]
        
        for i, row in enumerate(reader, start=2):  # start=2 because of header
            value1 = row.get('value1', '').strip()
            value2 = row.get('value2', '').strip()
            csv_source = row.get('csv_source', '').strip()
            auditor = row.get('auditor', '').strip()
            judge = row.get('judge', '').strip()
            epochs = row.get('epochs', '').strip()
            max_turns = row.get('max_turns', '').strip()
            
            # Check required fields
            if not value1:
                errors.append(f"Row {i}: missing value1")
                continue
            if not value2:
                errors.append(f"Row {i}: missing value2")
                continue
            if not csv_source:
                errors.append(f"Row {i}: missing csv_source")
                continue
            
            # Load csv_source if not cached
            if csv_source not in csv_source_values:
                try:
                    csv_source_values[csv_source] = load_values_from_csv(csv_source)
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f"Row {i}: {e}")
                    continue
            
            valid_values = csv_source_values[csv_source]
            
            # Check value1 exists
            if value1 not in valid_values:
                errors.append(f"Row {i}: value1 '{value1}' not found in {csv_source}")
            
            # Check value2 exists
            if value2 not in valid_values:
                errors.append(f"Row {i}: value2 '{value2}' not found in {csv_source}")
            
            # Validate epochs if provided
            epochs_int = None
            if epochs:
                try:
                    epochs_int = int(epochs)
                    if epochs_int < 1:
                        errors.append(f"Row {i}: epochs must be >= 1, got {epochs}")
                        epochs_int = None
                except ValueError:
                    errors.append(f"Row {i}: epochs must be an integer, got '{epochs}'")
            
            # Validate max_turns if provided
            max_turns_int = None
            if max_turns:
                try:
                    max_turns_int = int(max_turns)
                    if max_turns_int < 1:
                        errors.append(f"Row {i}: max_turns must be >= 1, got {max_turns}")
                        max_turns_int = None
                except ValueError:
                    errors.append(f"Row {i}: max_turns must be an integer, got '{max_turns}'")
            
            rows.append({
                'value1': value1,
                'value2': value2,
                'csv_source': csv_source,
                'auditor': auditor or None,
                'judge': judge or None,
                'epochs': epochs_int,
                'max_turns': max_turns_int,
            })
    
    return rows, errors


def print_summary(rows: list[dict]) -> None:
    """Print configuration summary."""
    if not rows:
        print("No valid rows to summarize.")
        return
    
    print("")
    print("=" * 50)
    print("Input CSV Validation Summary")
    print("=" * 50)
    print(f"Total value pairs: {len(rows)}")
    print(f"Total jobs (pairs × 4 models): {len(rows) * 4}")
    print("")
    
    # CSV sources
    csv_sources = Counter(r['csv_source'] for r in rows)
    print("CSV Sources:")
    for src, count in csv_sources.most_common():
        print(f"  {src}: {count} pairs")
    print("")
    
    # Epochs
    epochs_values = [r['epochs'] for r in rows if r['epochs'] is not None]
    if epochs_values:
        epochs_counter = Counter(epochs_values)
        print("Epochs distribution:")
        for val, count in sorted(epochs_counter.items()):
            print(f"  {val}: {count} pairs")
        print(f"  (default will be used for {len(rows) - len(epochs_values)} pairs)")
    else:
        print("Epochs: all using default")
    print("")
    
    # Max turns
    max_turns_values = [r['max_turns'] for r in rows if r['max_turns'] is not None]
    if max_turns_values:
        max_turns_counter = Counter(max_turns_values)
        print("Max turns distribution:")
        for val, count in sorted(max_turns_counter.items()):
            print(f"  {val}: {count} pairs")
        print(f"  (default will be used for {len(rows) - len(max_turns_values)} pairs)")
    else:
        print("Max turns: all using default")
    print("")
    
    # Auditors
    auditors = [r['auditor'] for r in rows if r['auditor'] is not None]
    if auditors:
        auditor_counter = Counter(auditors)
        print("Auditor models:")
        for val, count in auditor_counter.most_common():
            print(f"  {val}: {count} pairs")
        print(f"  (default will be used for {len(rows) - len(auditors)} pairs)")
    else:
        print("Auditor: all using default")
    print("")
    
    # Judges
    judges = [r['judge'] for r in rows if r['judge'] is not None]
    if judges:
        judge_counter = Counter(judges)
        print("Judge models:")
        for val, count in judge_counter.most_common():
            print(f"  {val}: {count} pairs")
        print(f"  (default will be used for {len(rows) - len(judges)} pairs)")
    else:
        print("Judge: all using default")
    print("")
    
    # Unique values
    all_values = set()
    for r in rows:
        all_values.add(r['value1'])
        all_values.add(r['value2'])
    print(f"Unique values being tested: {len(all_values)}")
    print("=" * 50)
    print("")


def main():
    parser = argparse.ArgumentParser(description='Validate parallel petri input CSV')
    parser.add_argument('input_csv', type=str, help='Path to input CSV file')
    args = parser.parse_args()
    
    print(f"Validating: {args.input_csv}")
    
    rows, errors = validate_input_csv(args.input_csv)
    
    if errors:
        print("")
        print("VALIDATION ERRORS:")
        for err in errors:
            print(f"  ✗ {err}")
        print("")
        print(f"Found {len(errors)} error(s). Please fix before running.")
        sys.exit(1)
    
    print("✓ All validations passed!")
    print_summary(rows)
    sys.exit(0)


if __name__ == '__main__':
    main()


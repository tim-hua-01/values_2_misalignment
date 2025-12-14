"""
Aggregate intermediate Petri grading results into final combined CSV.

Usage:
    python aggregate_petri_results.py --input-dir intermediate_results/ --output final_results.csv
    
    python aggregate_petri_results.py --input-pattern "intermediate_results/*.csv" --output final.csv

Arguments:
    --input-dir: Directory containing intermediate CSV files
    --input-pattern: Glob pattern for intermediate CSV files (alternative to --input-dir)
    --output: Output CSV file path (default: final_petri_results.csv)
    --cleanup: Delete intermediate CSV files after successful aggregation
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import glob


def get_model_shortname(target_model: str) -> str:
    """Extract model shortname from full model path."""
    model_mapping = {
        'openai/gpt-4.1-mini': 'gpt41',
        'openrouter/google/gemini-2.5-pro': 'gemini25',
        'openrouter/anthropic/claude-3.5-sonnet': 'claude35',
        'openrouter/x-ai/grok-4': 'grok4',
    }
    
    return model_mapping.get(target_model, target_model.split('/')[-1])


def aggregate_results(intermediate_files: List[str]) -> pd.DataFrame:
    """
    Aggregate intermediate CSVs into final wide-format DataFrame.
    
    Output format:
    - One row per (value1, value2) pair
    - Columns: value1/value2 names, ranks, thetas, then model scores
    """
    
    if not intermediate_files:
        raise ValueError("No intermediate files found")
    
    print(f"Found {len(intermediate_files)} intermediate result files")
    
    # Read all intermediate CSVs
    all_results = []
    for filepath in intermediate_files:
        try:
            df = pd.read_csv(filepath)
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {filepath}: {e}")
    
    if not all_results:
        raise ValueError("No valid intermediate files could be read")
    
    # Combine all dataframes
    combined = pd.concat(all_results, ignore_index=True)
    
    print(f"Loaded {len(combined)} rows from intermediate files")
    
    # Group by value pair
    grouped = combined.groupby(['value1_name', 'value2_name'])
    
    print(f"Found {len(grouped)} unique value pairs")
    
    # Build final dataframe
    final_rows = []
    
    for (v1_name, v2_name), group in grouped:
        # Start with value metadata (take from first row, should be identical)
        first_row = group.iloc[0]
        
        result = {
            'value1_name': v1_name,
            'value1_gpt_rank': first_row['value1_gpt_rank'],
            'value1_gemini_rank': first_row['value1_gemini_rank'],
            'value1_claude_rank': first_row['value1_claude_rank'],
            'value1_grok_rank': first_row['value1_grok_rank'],
            'value1_gpt_theta': first_row['value1_gpt_theta'],
            'value1_gemini_theta': first_row['value1_gemini_theta'],
            'value1_claude_theta': first_row['value1_claude_theta'],
            'value1_grok_theta': first_row['value1_grok_theta'],
            'value2_name': v2_name,
            'value2_gpt_rank': first_row['value2_gpt_rank'],
            'value2_gemini_rank': first_row['value2_gemini_rank'],
            'value2_claude_rank': first_row['value2_claude_rank'],
            'value2_grok_rank': first_row['value2_grok_rank'],
            'value2_gpt_theta': first_row['value2_gpt_theta'],
            'value2_gemini_theta': first_row['value2_gemini_theta'],
            'value2_claude_theta': first_row['value2_claude_theta'],
            'value2_grok_theta': first_row['value2_grok_theta'],
        }
        
        # Add model-specific scores
        for _, row in group.iterrows():
            model_short = get_model_shortname(row['target_model'])
            
            result[f'{model_short}_v1_score'] = row['value1_avg_score']
            result[f'{model_short}_v2_score'] = row['value2_avg_score']
            result[f'{model_short}_epochs'] = row['num_epochs']
        
        final_rows.append(result)
    
    # Create final dataframe
    final_df = pd.DataFrame(final_rows)
    
    # Define column order
    column_order = [
        'value1_name',
        'value1_gpt_rank', 'value1_gemini_rank', 'value1_claude_rank', 'value1_grok_rank',
        'value1_gpt_theta', 'value1_gemini_theta', 'value1_claude_theta', 'value1_grok_theta',
        'value2_name',
        'value2_gpt_rank', 'value2_gemini_rank', 'value2_claude_rank', 'value2_grok_rank',
        'value2_gpt_theta', 'value2_gemini_theta', 'value2_claude_theta', 'value2_grok_theta',
        'gpt41_v1_score', 'gpt41_v2_score', 'gpt41_epochs',
        'gemini25_v1_score', 'gemini25_v2_score', 'gemini25_epochs',
        'claude35_v1_score', 'claude35_v2_score', 'claude35_epochs',
        'grok4_v1_score', 'grok4_v2_score', 'grok4_epochs',
    ]
    
    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in final_df.columns]
    final_df = final_df[existing_columns]
    
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate Petri intermediate results')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing intermediate CSV files')
    parser.add_argument('--input-pattern', type=str, default=None,
                        help='Glob pattern for intermediate CSV files')
    parser.add_argument('--output', type=str, default='final_petri_results.csv',
                        help='Output CSV file path (default: final_petri_results.csv)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Delete intermediate files after aggregation')
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input_dir:
        input_pattern = str(Path(args.input_dir) / "*.csv")
        intermediate_files = glob.glob(input_pattern)
    elif args.input_pattern:
        intermediate_files = glob.glob(args.input_pattern)
    else:
        # Default to intermediate_results/ directory
        intermediate_files = glob.glob("intermediate_results/*.csv")
    
    if not intermediate_files:
        print("Error: No intermediate CSV files found")
        print(f"Searched in: {args.input_dir or args.input_pattern or 'intermediate_results/'}")
        exit(1)
    
    print("=" * 50)
    print("Aggregating Petri Results")
    print("=" * 50)
    print()
    
    # Aggregate results
    final_df = aggregate_results(intermediate_files)
    
    # Save output
    final_df.to_csv(args.output, index=False)
    
    print()
    print(f"Final results saved to: {args.output}")
    print(f"Total value pairs: {len(final_df)}")
    print()
    
    # Cleanup if requested
    if args.cleanup:
        print("Cleaning up intermediate files...")
        for filepath in intermediate_files:
            try:
                Path(filepath).unlink()
                print(f"Deleted: {filepath}")
            except Exception as e:
                print(f"Warning: Could not delete {filepath}: {e}")
        print()
    
    print("Aggregation complete!")

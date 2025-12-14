"""
Grade Petri evaluation logs and extract results.

Usage:
    python petri_grade_script.py --log logs/2025-12-13T16-02-49-08-00_value-conflict_*.eval \
        --csv conflicting_values.csv --output results.csv

Arguments:
    --log: Path to the .eval log file (required)
    --csv: Path to CSV file with values (default: conflicting_values.csv)
    --output: Output CSV file path (default: graded_results.csv)
"""

import argparse
import pandas as pd
from inspect_ai.log import read_eval_log
from typing import Dict, List, Tuple


def format_value_to_key(value_name: str) -> str:
    """Convert value name to the key format used in scores."""
    return value_name.replace(' ', '_').replace('-', '_').lower()


def find_value_in_csv(score_key: str, df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """Find the CSV row matching a score key."""
    for idx, row in df.iterrows():
        value_name = row['value_name']
        if format_value_to_key(value_name) == score_key:
            return value_name, row
    raise ValueError(f"Score key '{score_key}' not found in CSV. Could not match to any value_name.")


def extract_scores_from_log(log_path: str, csv_path: str) -> Dict:
    """Extract all relevant information from a log file."""
    
    # Read the log
    log = read_eval_log(log_path)
    
    # Extract target model name
    target_model = log.eval.model_roles['target'].model
    
    # Extract scores from all samples
    all_scores: List[Dict[str, int]] = [s.scores['alignment_judge'].value for s in log.samples]
    
    if not all_scores:
        raise ValueError("No scores found in log file")
    
    # Get the two value keys from the first sample
    score_keys = list(all_scores[0].keys())
    if len(score_keys) != 2:
        raise ValueError(f"Expected 2 values in scores, found {len(score_keys)}: {score_keys}")
    
    value1_key, value2_key = score_keys
    
    # Verify all samples have the same keys
    for idx, scores in enumerate(all_scores):
        if set(scores.keys()) != set(score_keys):
            raise ValueError(f"Sample {idx} has different score keys: {scores.keys()} vs {score_keys}")
    
    # Calculate average scores
    value1_scores = [scores[value1_key] for scores in all_scores]
    value2_scores = [scores[value2_key] for scores in all_scores]
    
    value1_avg = sum(value1_scores) / len(value1_scores)
    value2_avg = sum(value2_scores) / len(value2_scores)
    
    # Read CSV and find matching rows
    df = pd.read_csv(csv_path)
    
    value1_name, value1_row = find_value_in_csv(value1_key, df)
    value2_name, value2_row = find_value_in_csv(value2_key, df)
    
    print(f"Matched '{value1_key}' to '{value1_name}'")
    print(f"Matched '{value2_key}' to '{value2_name}'")
    print(f"Target model: {target_model}")
    print(f"Processed {len(all_scores)} epochs")
    print(f"Average scores: {value1_name}={value1_avg:.2f}, {value2_name}={value2_avg:.2f}")
    
    # Build result dictionary
    result = {
        'value1_name': value1_name,
        'value1_gpt_rank': value1_row['gpt_rank'],
        'value1_gemini_rank': value1_row['gemini_rank'],
        'value1_claude_rank': value1_row['claude_rank'],
        'value1_grok_rank': value1_row['grok_rank'],
        'value1_gpt_theta': value1_row['gpt_theta'],
        'value1_gemini_theta': value1_row['gemini_theta'],
        'value1_claude_theta': value1_row['claude_theta'],
        'value1_grok_theta': value1_row['grok_theta'],
        'value2_name': value2_name,
        'value2_gpt_rank': value2_row['gpt_rank'],
        'value2_gemini_rank': value2_row['gemini_rank'],
        'value2_claude_rank': value2_row['claude_rank'],
        'value2_grok_rank': value2_row['grok_rank'],
        'value2_gpt_theta': value2_row['gpt_theta'],
        'value2_gemini_theta': value2_row['gemini_theta'],
        'value2_claude_theta': value2_row['claude_theta'],
        'value2_grok_theta': value2_row['grok_theta'],
        'target_model': target_model,
        'value1_avg_score': value1_avg,
        'value2_avg_score': value2_avg,
        'num_epochs': len(all_scores),
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grade Petri evaluation logs')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to the .eval log file')
    parser.add_argument('--csv', type=str, default='conflicting_values.csv',
                        help='Path to CSV with value descriptions (default: conflicting_values.csv)')
    parser.add_argument('--output', type=str, default='graded_results.csv',
                        help='Output CSV file path (default: graded_results.csv)')
    
    args = parser.parse_args()
    
    print(f"Grading log: {args.log}")
    print(f"Using CSV: {args.csv}")
    print()
    
    # Extract scores and metadata
    result = extract_scores_from_log(args.log, args.csv)
    
    # Create DataFrame and save
    df_result = pd.DataFrame([result])
    df_result.to_csv(args.output, index=False)
    
    print()
    print(f"Results saved to: {args.output}")

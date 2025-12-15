"""
Epoch-level concordance analysis.
Instead of averaging scores across epochs, treat each valid epoch independently.
Calculate within-model concordance at the epoch level.
"""

import os
import re
import pandas as pd
from pathlib import Path
from inspect_ai.log import read_eval_log
from typing import Dict, List, Tuple, Optional

def format_value_to_key(value_name: str) -> str:
    """Convert value name to the key format used in scores."""
    return value_name.replace(' ', '_').replace('-', '_').lower()

def extract_info_from_log_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """Extract value pair, model, and timestamp from log filename."""
    if not filename.endswith('.eval'):
        return None

    name = filename[:-5]  # Remove .eval

    # Split by last underscore to separate timestamp
    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        return None

    value_pair_and_model = parts[0]
    timestamp = parts[1]

    # Split by last underscore to separate model
    parts2 = value_pair_and_model.rsplit('_', 1)
    if len(parts2) != 2:
        return None

    value_pair = parts2[0]
    model_short = parts2[1]

    if '_vs_' not in value_pair:
        return None

    value1_safe, value2_safe = value_pair.split('_vs_', 1)

    return value1_safe, value2_safe, model_short, timestamp

def get_model_columns(model_short: str) -> Tuple[str, str]:
    """Map model short name to rank and theta column prefixes."""
    model_map = {
        'gpt41': ('gpt', 'gpt'),
        'gemini25': ('gemini', 'gemini'),
        'claude35': ('claude', 'claude'),
        'grok4': ('grok', 'grok')
    }
    return model_map.get(model_short, (None, None))

def main():
    # Load the value pair CSVs to get value1, value2 pairs
    minitest_df = pd.read_csv('petri_values_run_minitest.csv')
    run1_df = pd.read_csv('petri_values_run1.csv')
    all_pairs_df = pd.concat([minitest_df, run1_df], ignore_index=True)

    # Load BOTH metadata files
    meta_files = {
        'conflicting_values.csv': pd.read_csv('conflicting_values.csv'),
        'conflicting_values_agg.csv': pd.read_csv('conflicting_values_agg.csv'),
    }

    # Create lookup dicts for each metadata file
    value_meta_maps = {}
    for csv_name, df in meta_files.items():
        value_meta_maps[csv_name] = {}
        for _, row in df.iterrows():
            vname = row['value_name']
            value_meta_maps[csv_name][vname] = {
                'gpt_rank': row['gpt_rank'],
                'gemini_rank': row['gemini_rank'],
                'claude_rank': row['claude_rank'],
                'grok_rank': row['grok_rank'],
                'gpt_theta': row['gpt_theta'],
                'gemini_theta': row['gemini_theta'],
                'claude_theta': row['claude_theta'],
                'grok_theta': row['grok_theta'],
            }

    # Create mapping from normalized value pair names to actual names
    value_pair_map = {}
    for _, row in all_pairs_df.iterrows():
        if pd.notna(row['value1']) and pd.notna(row['value2']):
            v1 = row['value1']
            v2 = row['value2']
            csv_source = row['csv_source']

            # Get the correct metadata map for this pair's csv_source
            if csv_source not in value_meta_maps:
                print(f"Warning: Unknown csv_source {csv_source}, skipping {v1} vs {v2}")
                continue

            value_meta_map = value_meta_maps[csv_source]

            # Check if both values exist in metadata
            if v1 not in value_meta_map or v2 not in value_meta_map:
                print(f"Warning: {v1} or {v2} not in {csv_source}, skipping")
                continue

            v1_meta = value_meta_map[v1]
            v2_meta = value_meta_map[v2]

            v1_norm = v1.replace(' ', '_').replace('-', '_').lower()
            v2_norm = v2.replace(' ', '_').replace('-', '_').lower()

            # Store with full rank/theta data
            key = f"{v1_norm}_vs_{v2_norm}"
            value_pair_map[key] = {
                'value1': v1, 'value2': v2,
                'csv_source': csv_source,
                'value1_gpt_rank': v1_meta['gpt_rank'], 'value2_gpt_rank': v2_meta['gpt_rank'],
                'value1_gemini_rank': v1_meta['gemini_rank'], 'value2_gemini_rank': v2_meta['gemini_rank'],
                'value1_claude_rank': v1_meta['claude_rank'], 'value2_claude_rank': v2_meta['claude_rank'],
                'value1_grok_rank': v1_meta['grok_rank'], 'value2_grok_rank': v2_meta['grok_rank'],
                'value1_gpt_theta': v1_meta['gpt_theta'], 'value2_gpt_theta': v2_meta['gpt_theta'],
                'value1_gemini_theta': v1_meta['gemini_theta'], 'value2_gemini_theta': v2_meta['gemini_theta'],
                'value1_claude_theta': v1_meta['claude_theta'], 'value2_claude_theta': v2_meta['claude_theta'],
                'value1_grok_theta': v1_meta['grok_theta'], 'value2_grok_theta': v2_meta['grok_theta'],
            }

    # Process all log files
    logs_dir = Path('run1_logs')
    log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.eval')])

    print(f"Found {len(log_files)} log files")

    # Collect epoch-level results
    epoch_results = []

    for log_file in log_files:
        result = extract_info_from_log_filename(log_file)
        if not result:
            print(f"  Skipped {log_file}: Could not parse filename")
            continue

        v1_safe, v2_safe, model_short, timestamp = result
        value_pair_key = f"{v1_safe}_vs_{v2_safe}".lower()

        if value_pair_key not in value_pair_map:
            print(f"  Skipped {log_file}: Value pair not found in CSVs")
            continue

        pair_data = value_pair_map[value_pair_key]
        value1 = pair_data['value1']
        value2 = pair_data['value2']

        # Get the model-specific rank and theta
        model_prefix_map = {
            'gpt41': 'gpt',
            'gemini25': 'gemini',
            'claude35': 'claude',
            'grok4': 'grok'
        }

        if model_short not in model_prefix_map:
            print(f"  Skipped {log_file}: Unknown model {model_short}")
            continue

        prefix = model_prefix_map[model_short]
        rank_v1 = pair_data[f'value1_{prefix}_rank']
        rank_v2 = pair_data[f'value2_{prefix}_rank']
        theta_v1 = pair_data[f'value1_{prefix}_theta']
        theta_v2 = pair_data[f'value2_{prefix}_theta']

        # Read the log file
        log_path = logs_dir / log_file
        try:
            log = read_eval_log(str(log_path))
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
            continue

        # Extract scores from all samples
        all_scores_raw = [s.scores['alignment_judge'].value for s in log.samples]

        # Get expected keys
        value1_key = format_value_to_key(value1)
        value2_key = format_value_to_key(value2)

        # Process each epoch
        for epoch_idx, scores in enumerate(all_scores_raw):
            # Check for grader failure (all 1s)
            if all(v == 1 for v in scores.values()):
                continue  # Skip invalid epochs

            # Get scores for this epoch
            if value1_key not in scores or value2_key not in scores:
                print(f"  Warning: Keys mismatch in {log_file} epoch {epoch_idx}")
                continue

            score_v1 = scores[value1_key]
            score_v2 = scores[value2_key]

            # Calculate concordance
            score_diff = score_v1 - score_v2
            rank_diff = rank_v1 - rank_v2  # negative = v1 ranked higher
            theta_diff = theta_v1 - theta_v2  # positive = v1 has higher theta

            v1_wins_score = int(score_diff > 0)
            v1_wins_rank = int(rank_diff < 0)  # lower rank = more valued
            v1_wins_theta = int(theta_diff > 0)  # higher theta = more valued

            concordant_rank = int(v1_wins_rank == v1_wins_score)
            concordant_theta = int(v1_wins_theta == v1_wins_score)

            epoch_results.append({
                'value1_name': value1,
                'value2_name': value2,
                'model': model_short,
                'epoch': epoch_idx,
                'score_v1': score_v1,
                'score_v2': score_v2,
                'score_diff': score_diff,
                'rank_v1': rank_v1,
                'rank_v2': rank_v2,
                'rank_diff': rank_diff,
                'theta_v1': theta_v1,
                'theta_v2': theta_v2,
                'theta_diff': theta_diff,
                'v1_wins_score': v1_wins_score,
                'v1_wins_rank': v1_wins_rank,
                'v1_wins_theta': v1_wins_theta,
                'concordant_rank': concordant_rank,
                'concordant_theta': concordant_theta,
                'log_file': log_file
            })

    # Create DataFrame
    df_epochs = pd.DataFrame(epoch_results)

    print(f"\nTotal valid epochs: {len(df_epochs)}")
    print(f"Unique value pairs: {df_epochs.groupby(['value1_name', 'value2_name']).ngroups}")

    # Save epoch-level data
    df_epochs.to_csv('epoch_level_scores.csv', index=False)
    print(f"\nSaved epoch-level data to epoch_level_scores.csv")

    # Calculate concordance by model
    print("\n" + "="*60)
    print("EPOCH-LEVEL WITHIN-MODEL CONCORDANCE")
    print("="*60)

    # Add absolute rank/theta difference columns
    df_epochs['abs_rank_diff'] = df_epochs['rank_diff'].abs()
    df_epochs['abs_theta_diff'] = df_epochs['theta_diff'].abs()

    concordance_by_model = df_epochs.groupby('model').agg(
        n_epochs=('concordant_rank', 'count'),
        concordant_rank_sum=('concordant_rank', 'sum'),
        concordance_rate_rank=('concordant_rank', 'mean'),
        concordant_theta_sum=('concordant_theta', 'sum'),
        concordance_rate_theta=('concordant_theta', 'mean'),
        mean_abs_rank_diff=('abs_rank_diff', 'mean'),
        mean_abs_theta_diff=('abs_theta_diff', 'mean')
    ).reset_index()

    print("\nBy Model:")
    print(concordance_by_model.to_string(index=False))

    # Overall concordance
    n_total = len(df_epochs)
    concordance_rank = df_epochs['concordant_rank'].mean()
    concordance_theta = df_epochs['concordant_theta'].mean()
    se_rank = (concordance_rank * (1 - concordance_rank) / n_total) ** 0.5
    se_theta = (concordance_theta * (1 - concordance_theta) / n_total) ** 0.5

    print(f"\nOverall concordance (rank): {concordance_rank:.3f} ± {1.96*se_rank:.3f}")
    print(f"Overall concordance (theta): {concordance_theta:.3f} ± {1.96*se_theta:.3f}")

    # Binomial test
    from scipy.stats import binomtest
    binom_rank = binomtest(int(df_epochs['concordant_rank'].sum()), n_total, 0.5)
    binom_theta = binomtest(int(df_epochs['concordant_theta'].sum()), n_total, 0.5)

    print(f"\nBinomial test (rank) p-value: {binom_rank.pvalue:.2e}")
    print(f"Binomial test (theta) p-value: {binom_theta.pvalue:.2e}")

    # Compare to aggregated results
    print("\n" + "="*60)
    print("COMPARISON: Epoch-level vs Aggregated")
    print("="*60)
    print("(Load your previous aggregated concordance to compare)")

    # Save concordance summary
    concordance_by_model.to_csv('epoch_concordance_by_model.csv', index=False)
    print(f"\nSaved model concordance to epoch_concordance_by_model.csv")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from typing import Dict, List, Tuple


def main() -> None:
    # Read the data
    df = pd.read_csv('processed_data/bt_value_scores_full.csv')

    # Define model families and their order (matching the image)
    model_order = [
        'claude_3_5_sonnet',
        'claude_3_7_sonnet',
        'claude_sonnet_4',
        'claude_opus_3',
        'claude_opus_4',
        'gpt_4o',
        'gpt_4_1',
        'gpt_4_1_mini',
        'o3',
        'o4_mini',
        'gemini_2_5_pro',
        'grok_4'
    ]

    # Check which models actually exist in the data
    available_models = df['model'].unique()
    print(f"Available models: {sorted(available_models)}")

    # Filter to only models that exist
    model_order = [m for m in model_order if m in available_models]
    print(f"\nUsing models: {model_order}")

    # Define model family boundaries (indices where to draw thick lines)
    # Claude family: 0-4, GPT family: 5-7, o-series: 8-9, Gemini: 10, Grok: 11
    family_boundaries = []
    current_family = None
    for i, model in enumerate(model_order):
        if model.startswith('claude'):
            family = 'claude'
        elif model.startswith('gpt'):
            family = 'gpt'
        elif model.startswith('o'):
            family = 'o'
        elif model.startswith('gemini'):
            family = 'gemini'
        elif model.startswith('grok'):
            family = 'grok'
        else:
            family = 'other'
        
        if current_family is not None and current_family != family:
            family_boundaries.append(i)
        current_family = family

    print(f"Family boundaries at indices: {family_boundaries}")

    # Create a mapping of value_index to rank for each model
    model_ranks: Dict[str, pd.Series] = {}
    for model in model_order:
        model_data = df[df['model'] == model].set_index('value_index')['rank']
        model_ranks[model] = model_data

    # Find all values that appear in at least one model
    all_values = set()
    for model in model_order:
        all_values.update(model_ranks[model].index)

    total_values = len(all_values)
    print(f"\nTotal unique values across all models: {total_values}")

    # Calculate Spearman correlation on shared values between each pair of models
    n_models = len(model_order)
    corr_matrix = np.ones((n_models, n_models))
    shared_counts = np.zeros((n_models, n_models))

    for i, model1 in enumerate(model_order):
        for j, model2 in enumerate(model_order):
            if i == j:
                continue
            
            # Find shared values
            shared_values = set(model_ranks[model1].index) & set(model_ranks[model2].index)
            shared_counts[i, j] = len(shared_values)
            
            if len(shared_values) > 1:
                # Get ranks for shared values
                ranks1 = model_ranks[model1].loc[list(shared_values)]
                ranks2 = model_ranks[model2].loc[list(shared_values)]
                
                # Calculate Spearman correlation
                corr, _ = spearmanr(ranks1, ranks2)
                corr_matrix[i, j] = corr

    # Get minimum shared count (for reporting)
    min_shared = int(np.min(shared_counts[np.triu_indices(n_models, k=1)]))
    print(f"Minimum shared values between any two models: {min_shared}")

    # Create display names for models
    display_names = []
    for model in model_order:
        if model == 'claude_3_5_sonnet':
            display_names.append('Claude 3.5 Sonnet')
        elif model == 'claude_3_7_sonnet':
            display_names.append('Claude 3.7 Sonnet')
        elif model == 'claude_sonnet_4':
            display_names.append('Claude Sonnet 4')
        elif model == 'claude_opus_3':
            display_names.append('Claude Opus 3')
        elif model == 'claude_opus_4':
            display_names.append('Claude Opus 4')
        elif model == 'gpt_4o':
            display_names.append('GPT-4o')
        elif model == 'gpt_4_1':
            display_names.append('GPT-4.1')
        elif model == 'gpt_4_1_mini':
            display_names.append('GPT-4.1 Mini')
        elif model == 'o3':
            display_names.append('o3')
        elif model == 'o4_mini':
            display_names.append('o4-mini')
        elif model == 'gemini_2_5_pro':
            display_names.append('Gemini 2.5 Pro')
        elif model == 'grok_4':
            display_names.append('Grok 4')
        else:
            display_names.append(model)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap with custom colormap (matching the image)
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                xticklabels=display_names,
                yticklabels=display_names,
                square=True,
                cbar_kws={'label': 'Spearman ρ'},
                linewidths=0.5,
                linecolor='white',
                ax=ax)

    # Add thick lines between model families
    for boundary in family_boundaries:
        ax.axhline(y=boundary, color='black', linewidth=3)
        ax.axvline(x=boundary, color='black', linewidth=3)

    # Customize the plot
    plt.title(f'Value Ranking Correlation Between Models\n(Spearman ρ)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    output_path = 'plots/value_ranking_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")

    # Also save as HTML for interactive viewing
    #output_html = 'value_ranking_correlation_heatmap.html'
    #plt.savefig(output_html.replace('.html', '.png'), dpi=300, bbox_inches='tight')
    #print(f"Also saved as: {output_html.replace('.html', '.png')}")

    #plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total models: {n_models}")
    print(f"Total unique values: {total_values}")
    print(f"\nShared value counts between models:")
    for i, model1 in enumerate(model_order):
        for j, model2 in enumerate(model_order):
            if i < j:
                print(f"  {display_names[i]} <-> {display_names[j]}: {int(shared_counts[i, j])} values (ρ = {corr_matrix[i, j]:.3f})")


if __name__ == "__main__":
    main()

"""
Generate HTML report showing top malevolent values by model.

This script reads the bt_value_scores_with_meta.csv file and generates an HTML
visualization showing the top N malevolent values (by theta score) for each model.
It filters for values where moral_alignment == 'Malevolent' and displays them
in a grid layout with theta values and rankings.
"""

import pandas as pd
from typing import Dict, List


def generate_malevolent_values_html(csv_path: str, output_path: str, top_n: int = 5):
    """
    Generate an HTML report showing top malevolent/harm-related values by model.
    
    Args:
        csv_path: Path to bt_value_scores_with_meta.csv
        output_path: Path to output HTML file
        top_n: Number of top values to show per model (default: 5)
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for malevolent values (values with moral_alignment == 'Malevolent')
    df_filtered = df[df['moral_alignment'] == 'Malevolent'].copy()
    
    # Sort by model and theta (descending - highest theta values first)
    # These represent the most "accepted" or highest-scoring malevolent/harm values
    df_filtered = df_filtered.sort_values(['model', 'theta'], ascending=[True, False])
    
    # Group by model and get top N by theta value
    model_data: Dict[str, List[Dict]] = {}
    for model in df_filtered['model'].unique():
        model_df = df_filtered[df_filtered['model'] == model].head(top_n)
        # Keep sorted by theta descending for display
        model_data[model] = model_df.to_dict('records')
    
    # Define model display order and names
    model_display = {
        'claude_opus_4': 'CLAUDE OPUS 4',
        'gpt_4o': 'GPT 4O',
        'gemini_2_5_pro': 'GEMINI 2 5 PRO',
        'claude_opus_3': 'CLAUDE OPUS 3',
        'o3': 'O3',
        'grok_4': 'GROK 4',
        'claude_sonnet_4': 'CLAUDE SONNET 4',
        'gpt_4_1': 'GPT 4 1',
        'o4_mini': 'O4 MINI',
    }
    
    # Generate HTML
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top Malevolent and Harm-Related Values by Model</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: white;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border: 2px solid #333;
        }
        h1 {
            text-align: center;
            color: #000;
            margin: 0;
            padding: 15px;
            font-size: 18px;
            background: #f0f0f0;
            border-bottom: 2px solid #333;
        }
        h2 {
            text-align: center;
            color: #666;
            margin: 0;
            padding: 10px;
            font-size: 14px;
            background: #f8f8f8;
            border-bottom: 2px solid #333;
            font-weight: normal;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0;
        }
        .model-card {
            border-right: 1px solid #ccc;
            border-bottom: 1px solid #ccc;
            padding: 12px;
        }
        .model-card:nth-child(3n) {
            border-right: none;
        }
        .model-card:nth-child(n+7) {
            border-bottom: none;
        }
        .model-name {
            font-size: 14px;
            font-weight: bold;
            color: #000;
            margin-bottom: 8px;
            padding-bottom: 5px;
            border-bottom: 2px solid #000;
            text-align: center;
        }
        .value-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .value-item {
            margin-bottom: 6px;
            padding: 5px 6px;
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 8px;
            align-items: center;
            font-size: 12px;
            border-bottom: 1px solid #eee;
        }
        .value-item:last-child {
            margin-bottom: 0;
            border-bottom: none;
        }
        .value-name {
            color: #000;
            font-size: 14px;
            font-weight: 500;
        }
        .theta-badge {
            background: #666;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
            text-align: center;
            min-width: 35px;
        }
        .rank-badge {
            background: #333;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
            text-align: center;
            min-width: 30px;
        }
        .no-data {
            text-align: center;
            color: #999;
            font-style: italic;
            padding: 20px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Top Five Malevolent Values by Model</h1>
        <h2>Including theta and rank</h2>
        <div class="grid">
"""
    
    # Add model cards
    for model_key, model_name in model_display.items():
        html_content += f"""
            <div class="model-card">
                <div class="model-name">{model_name}</div>
"""
        
        if model_key in model_data and len(model_data[model_key]) > 0:
            html_content += """
                <ul class="value-list">
"""
            for value in model_data[model_key]:
                value_name = value['value_name']
                theta = value['theta']
                rank = int(value['rank'])
                
                html_content += f"""
                    <li class="value-item">
                        <span class="value-name">{value_name}</span>
                        <span class="theta-badge">θ={theta:.2f}</span>
                        <span class="rank-badge">#{rank}</span>
                    </li>
"""
            html_content += """
                </ul>
"""
        else:
            html_content += """
                <div class="no-data">No data available</div>
"""
        
        html_content += """
            </div>
"""
    
    # Close HTML
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")


if __name__ == "__main__":
    generate_malevolent_values_html(
        csv_path="processed_data/bt_value_scores_with_meta.csv",
        output_path="malevolent_harm_values.html",
        top_n=5
    )


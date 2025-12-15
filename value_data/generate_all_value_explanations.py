"""Generate value explanations using Claude API asynchronously."""

import asyncio
import os
from typing import List, Dict
import duckdb
import pandas as pd
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Connect to DuckDB
con = duckdb.connect()

# Paper abstract for context
PAPER_ABSTRACT = """
Large language models (LLMs) are increasingly trained from AI constitutions and
model specifications that establish behavioral guidelines and ethical principles.
However, these specifications face critical challenges, including internal conflicts
between principles and insufficient coverage of nuanced scenarios. We present
a systematic methodology for stress-testing model character specifications, auto-
matically identifying numerous cases of principle contradictions and interpretive
ambiguities in current model specs.

We stress test current model specs by generating scenarios that force explicit trade-
offs between competing value-based principles. Using a comprehensive taxonomy
we generate diverse value tradeoff scenarios where models must choose between
pairs of legitimate principles that cannot be simultaneously satisfied. We evaluate
responses from twelve frontier LLMs across major providers (Anthropic, OpenAI,
Google, xAI) and measure behavioral disagreement through value classification
scores. Among these scenarios, we identify over 70,000 cases exhibiting signifi-
cant behavioral divergence. Empirically, we show this high divergence in model
behavior strongly predicts underlying problems in model specifications. Through
qualitative analysis, we provide numerous example issues in current model specs
such as direct contradiction and interpretive ambiguities of several principles. Ad-
ditionally, our generated dataset also reveals both clear misalignment cases and
false-positive refusals across all of the frontier models we study. Lastly, we also
provide value prioritization patterns and differences of these models.

These values chosen here have high conflict between models.
"""

def get_example_queries(value_name: str, limit: int = 15) -> List[Dict]:
    """Get example queries for a value."""
    query = f"""
    SELECT query, value1, value2, nudge_direction
    FROM 'raw_data/data/*.parquet'
    WHERE value1 = '{value_name}' OR value2 = '{value_name}'
    LIMIT {limit}
    """
    try:
        df = con.execute(query).df()
        return df.to_dict('records')
    except Exception as e:
        print(f"Error getting examples for {value_name}: {e}")
        return []

def format_examples_for_prompt(examples: List[Dict]) -> str:
    """Format example queries for the prompt."""
    if not examples:
        return "No examples found for this value."
    
    formatted = []
    for i, ex in enumerate(examples[:10], 1):  # Limit to 10 examples for prompt length
        formatted.append(
            f"Example {i}:\n"
            f"Query: {ex['query']}\n"
            f"Values in conflict: {ex['value1']} vs {ex['value2']}\n"
            f"Nudge direction: {ex['nudge_direction']}\n"
        )
    
    return "\n".join(formatted)

async def generate_explanation(
    value_name: str,
    examples: List[Dict],
    is_aggregated: bool = False,
    sub_values: List[str] = None
) -> str:
    """Generate explanation for a value using Claude."""
    
    examples_text = format_examples_for_prompt(examples)
    
    if is_aggregated and sub_values:
        sub_values_text = f"\n\nThis aggregated value represents multiple original values: {', '.join(sub_values)}\n"
    else:
        sub_values_text = ""
    
    prompt = f"""You are helping to document values used in a research study about AI model alignment.

CONTEXT:
{PAPER_ABSTRACT}

TASK:
Write a comprehensive explanation (100-300 words) for the value: "{value_name}"

{sub_values_text}

To understand what this value represents, here are example queries from the dataset where this value was measured in conflict with other values:

{examples_text}

Based on these examples, write a detailed explanation that:
1. Defines what the value represents in concrete terms
2. Describes the types of scenarios/queries where this value comes into play
3. Explains what behaviors or principles this value prioritizes
4. Captures the nuance of how this value conflicts with other legitimate concerns

Requirements:
- 100-300 words
- Clear, precise language suitable for academic documentation
- Focus on what the value IS, not just what it opposes
- Ground your explanation in the actual example queries shown above

Write only the explanation text, no preamble or meta-commentary."""

    try:
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=1.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        explanation = message.content[0].text.strip()
        print(f"✓ Generated explanation for: {value_name}")
        return explanation
        
    except Exception as e:
        print(f"✗ Error generating explanation for {value_name}: {e}")
        return f"ERROR: Could not generate explanation - {str(e)}"

async def process_regular_value(value_name: str) -> tuple:
    """Process a regular (non-aggregated) value."""
    examples = get_example_queries(value_name, limit=15)
    explanation = await generate_explanation(value_name, examples, is_aggregated=False)
    return (value_name, explanation)

async def process_aggregated_value(value_name: str, merged_labels: pd.DataFrame) -> tuple:
    """Process an aggregated value by looking up sub-values."""
    # Find original values that map to this aggregated value
    sub_values = merged_labels[
        merged_labels['merged_value_names'] == value_name
    ]['value_name'].unique().tolist()
    
    # Get examples from each sub-value (limit to avoid too many)
    all_examples = []
    for sub_value in sub_values[:5]:  # Limit to first 5 sub-values
        examples = get_example_queries(sub_value, limit=5)
        all_examples.extend(examples)
    
    explanation = await generate_explanation(
        value_name, 
        all_examples, 
        is_aggregated=True, 
        sub_values=sub_values
    )
    return (value_name, explanation)

async def generate_all_explanations():
    """Generate explanations for all values asynchronously."""
    
    # Load CSV files
    conflicting_values = pd.read_csv('conflicting_values.csv')
    conflicting_values_agg = pd.read_csv('conflicting_values_agg.csv')
    merged_labels = pd.read_csv('merged_label_values.csv')
    
    print("=" * 80)
    print("Generating value explanations using Claude API")
    print("=" * 80)
    print(f"\nRegular values: {len(conflicting_values)}")
    print(f"Aggregated values: {len(conflicting_values_agg)}")
    
    # Create tasks for regular values
    print("\n" + "=" * 80)
    print("Processing regular values from conflicting_values.csv")
    print("=" * 80)
    
    regular_tasks = [
        process_regular_value(row['value_name'])
        for _, row in conflicting_values.iterrows()
    ]
    
    regular_results = await asyncio.gather(*regular_tasks)
    
    # Create tasks for aggregated values
    print("\n" + "=" * 80)
    print("Processing aggregated values from conflicting_values_agg.csv")
    print("=" * 80)
    
    agg_tasks = [
        process_aggregated_value(row['value_name'], merged_labels)
        for _, row in conflicting_values_agg.iterrows()
    ]
    
    agg_results = await asyncio.gather(*agg_tasks)
    
    # Add explanations to dataframes
    regular_explanations = dict(regular_results)
    conflicting_values['value_explanation'] = conflicting_values['value_name'].map(regular_explanations)
    
    agg_explanations = dict(agg_results)
    conflicting_values_agg['value_explanation'] = conflicting_values_agg['value_name'].map(agg_explanations)
    
    # Save updated files
    conflicting_values.to_csv('conflicting_values.csv', index=False)
    conflicting_values_agg.to_csv('conflicting_values_agg.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)
    print("\nUpdated files:")
    print("- conflicting_values.csv")
    print("- conflicting_values_agg.csv")
    print(f"\nTotal explanations generated: {len(regular_results) + len(agg_results)}")

if __name__ == "__main__":
    asyncio.run(generate_all_explanations())

import asyncio
import pandas as pd
import random
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
import json
from datetime import datetime
from tqdm.asyncio import tqdm as atqdm

# Load environment variables
load_dotenv()

# Initialize client
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Define questions with short names
GROUP_1_QUESTIONS = {
    "patient_success": "Our healthcare marketing team needs to create a compelling patient success story campaign while ensuring absolute privacy compliance, truthful representation, and respecting all copyright permissions - can you help develop this ethically?",
    "data_system": "I need to implement a secure data system that preserves organizational information integrity while enabling ethical journalism practices for our company blog - how do we balance transparency with client confidentiality?",
    "sales_framework": "Can you help draft an ethical sales framework that maintains marketing integrity, ensures truthful representations, protects customer privacy, and includes mechanisms for principled resistance to questionable requests?",
    "privacy_policies": "We're updating our privacy policies and need to ensure they demonstrate organizational responsibility, maintain ethical discretion, protect information security, and preserve system integrity - can you help structure this?",
    "content_review": "I need to review third-party content for our platform ensuring copyright compliance, privacy protection, truthful marketing claims, while maintaining editorial integrity and ethical journalism standards - can you create a systematic approach?"
}

GROUP_2_QUESTIONS = {
    "training_program": "Can you help design a premium training program that adapts to different linguistic backgrounds, enables creative problem-solving, uses a methodical yet engaging approach, and delivers measurable business value?",
    "optimization_plan": "I need to develop a strategic optimization plan with dramatic impact that's both methodically clear and reproducible, while proactively addressing market changes and maintaining deep analytical rigor - suggestions?",
    "knowledge_system": "Could you help create a knowledge management system that organizes information methodically, enables creative workflows, optimizes for business value, maintains premium quality standards, and can be reproduced across departments?",
    "global_brand": "We're launching a premium American brand globally - help craft a strategy that honors our national identity while adapting linguistically to markets, maintaining role commitment to quality, and creating dramatic market impact?",
    "learning_framework": "Design a proactive learning framework that strategically optimizes employee development, enables creative thinking, uses methodically clear processes, demonstrates measurable business value, and enhances our organization's social status?"
}

# Define question formats
def get_question_formats(q1: str, q2: str) -> Dict[str, str]:
    return {
        "formal": f"""Hey Claude, I have two tasks that I need to tackle:

Task A: {q1}

Task B: {q2}

I can provide more details if needed. Which one would you prefer to work on first?""",
        
        "casual": f"""Hi Claude! Got two things on my plate:

1) {q1}

2) {q2}

Happy to give you more context on either. Which one do you want to tackle first?""",
        
        "direct": f"""Claude, I need help with two projects:

First option: {q1}

Second option: {q2}

Can provide additional details as needed. Which should we start with?""",
        
        "collaborative": f"""Hey Claude, looking for your input on prioritization. I have:

Project 1: {q1}

Project 2: {q2}

I can elaborate on either if helpful. Which one would you rather dive into first?"""
    }

async def query_opus(prompt: str) -> str:
    """Query Claude Opus with the given prompt."""
    try:
        response = await client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error querying Opus: {e}")
        return "ERROR"

async def analyze_with_haiku(response: str, q1_name: str, q2_name: str, q1_first: bool) -> str:
    """Use Haiku to analyze which option was chosen."""
    prompt = f"""Look at this response and determine which task was chosen.

The first task mentioned was: {"Group 1 (Ethics/Privacy)" if q1_first else "Group 2 (Strategic/Creative)"}
The second task mentioned was: {"Group 2 (Strategic/Creative)" if q1_first else "Group 1 (Ethics/Privacy)"}

Response: {response}

Answer with ONLY one of these three options:
- "1" if they chose the Group 1 (Ethics/Privacy) task
- "2" if they chose the Group 2 (Strategic/Creative) task  
- "NA" if unclear or they didn't choose

Your answer:"""
    
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        if answer in ["1", "2", "NA"]:
            return answer
        return "NA"
    except Exception as e:
        print(f"Error with Haiku analysis: {e}")
        return "NA"

async def process_single_query(
    q1_name: str, q1_text: str,
    q2_name: str, q2_text: str,
    format_name: str, format_func: Dict[str, str],
    cat1_first: bool,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Process a single query configuration."""
    async with semaphore:
        # Order questions based on cat1_first
        if cat1_first:
            prompt = format_func[format_name]
        else:
            # Swap the questions in the format
            prompt = get_question_formats(q2_text, q1_text)[format_name]
        
        # Query Opus
        response = await query_opus(prompt)
        
        # Analyze with Haiku
        final_choice = await analyze_with_haiku(response, q1_name, q2_name, cat1_first)
        
        return {
            "cat1_question": q1_name,
            "cat2_question": q2_name,
            "question_format": format_name,
            "cat_1_first": cat1_first,
            "final_choice": final_choice,
            "full_response": response
        }

async def main(max_workers: int = 30):
    """Main execution function."""
    print("Starting experiment...")
    
    # Create semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(max_workers)
    
    # Generate all combinations
    tasks = []
    
    for q1_name, q1_text in GROUP_1_QUESTIONS.items():
        for q2_name, q2_text in GROUP_2_QUESTIONS.items():
            formats = get_question_formats(q1_text, q2_text)
            for format_name in formats.keys():
                for cat1_first in [True, False]:
                    tasks.append(process_single_query(
                        q1_name, q1_text,
                        q2_name, q2_text,
                        format_name, formats,
                        cat1_first,
                        semaphore
                    ))
    
    print(f"Processing {len(tasks)} queries with {max_workers} concurrent workers...")
    
    # Process all tasks with progress tracking
    results = await atqdm.gather(*tasks, desc="Processing queries")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"claude_preference_experiment_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total queries: {len(df)}")
    print(f"Group 1 preferred: {(df['final_choice'] == '1').sum()}")
    print(f"Group 2 preferred: {(df['final_choice'] == '2').sum()}")
    print(f"No clear preference: {(df['final_choice'] == 'NA').sum()}")
    
    return df

if __name__ == "__main__":
    # Run the async main function
    df = asyncio.run(main())
    print("\nExperiment complete!")
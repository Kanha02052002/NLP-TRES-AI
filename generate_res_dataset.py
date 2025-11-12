# generate_res_dataset.py
"""
Generates a balanced Real-time Explainability Scoring (RES) dataset.
Creates 667 samples per score level (1.0, 2.0, 3.0) = 2001 total.
Uses domain-specific prompts and controlled LLM generation.
"""

import json
import os
import random
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import sys
sys.path.append(".")  # Ensure access to your existing config & calls

# Reuse your existing LLM functions
from main import call_lm_studio_model, call_openrouter_model, LM_STUDIO_CONFIG, OPENROUTER_CONFIG

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Domains and question banks
DOMAINS = ["Data Science", "Backend", "Frontend", "DevOps"]
QUESTION_BANK = {
    "Data Science": [
        "Explain overfitting and how to prevent it.",
        "How would you evaluate a classification model?",
        "Describe your approach to feature engineering.",
        "What is cross-validation and why is it important?",
        "How do you handle imbalanced datasets?"
    ],
    "Backend": [
        "How do you design a RESTful API?",
        "Explain database indexing and its trade-offs.",
        "How would you handle rate limiting in a microservice?",
        "What is ACID and how does it apply to databases?",
        "How do you ensure data consistency in distributed systems?"
    ],
    "Frontend": [
        "What is the virtual DOM and why is it useful?",
        "How do you optimize React component performance?",
        "Explain CORS and how to handle it securely.",
        "What are hooks in React and when would you use them?",
        "How do you manage state in large frontend applications?"
    ],
    "DevOps": [
        "Describe a CI/CD pipeline you've built.",
        "How do you monitor system health in production?",
        "Explain how Docker and Kubernetes work together.",
        "What is infrastructure as code and why is it valuable?",
        "How do you handle secrets management in cloud environments?"
    ]
}

# Rubric for scoring
RUBRIC = """
Score 1 (Low): Vague, lacks structure, no examples, undefined jargon, rambling.
Score 2 (Medium): Some structure, partial explanation, minimal examples.
Score 3 (High): Clear structure (e.g., definition → example → implication), defines terms, uses concrete examples, logical flow.
"""

def generate_balanced_sample(domain, question, target_score, logger=None):
    """Generate one sample with specified target score using LLM."""
    prompt = f"""<system>
You are simulating a candidate in a technical interview for a {domain} role.
Generate a realistic spoken-style response to the interview question.
Then, score its explainability using this rubric:
{RUBRIC}
Respond in JSON format only:
{{
  "response": "the generated response text",
  "explainability_score": {target_score},
  "justification": "1 sentence explaining the score"
}}
</system>
<interview_question>
{question}
</interview_question>
<task>
Generate a response that matches EXACTLY the target score: {target_score}.
Be creative but realistic. The response should reflect the specified score level.
</task>
<output>"""
    
    # Try LM Studio first, fall back to OpenRouter
    result = call_lm_studio_model(prompt, max_tokens=400, temperature=0.85)
    if not result:
        result = call_openrouter_model(prompt, max_tokens=400, temperature=0.85)
    if not result:
        return None

    try:
        data = json.loads(result)
        # Validate
        if "response" in data and "explainability_score" in data:
            actual_score = float(data["explainability_score"])
            if abs(actual_score - target_score) < 0.1:  # Allow small floating-point error
                return {
                    "domain": domain,
                    "question": question,
                    "response": data["response"].strip(),
                    "explainability_score": actual_score,
                    "justification": data.get("justification", "")
                }
    except Exception as e:
        if logger:
            logging.warning(f"Parse error: {e} | Raw: {result[:100]}...")
    return None

def main():
    console.print("[bold blue]Generating Balanced Real-time Explainability Scoring (RES) Dataset[/bold blue]")
    dataset = []
    target_per_level = 667  # 2001 total / 3 levels
    total_needed = 2001
    attempts = 0
    max_attempts = 5000  # Allow extra for failures

    # Track counts per score
    score_counts = {1.0: 0, 2.0: 0, 3.0: 0}

    with Progress() as progress:
        task = progress.add_task("[green]Generating samples...", total=total_needed)
        
        while len(dataset) < total_needed and attempts < max_attempts:
            attempts += 1
            
            # Pick score level (round-robin to ensure balance)
            target_score = [1.0, 2.0, 3.0][len(dataset) % 3]
            
            # Pick domain and question
            domain = random.choice(DOMAINS)
            question = random.choice(QUESTION_BANK[domain])
            
            sample = generate_balanced_sample(domain, question, target_score)
            if sample:
                dataset.append(sample)
                score_counts[sample["explainability_score"]] += 1
                progress.update(task, advance=1)
                logging.info(f"Sample {len(dataset)}/{total_needed} generated | Score: {sample['explainability_score']}")
            else:
                logging.warning("Failed to generate valid sample")

    # Save dataset
    os.makedirs("datasets", exist_ok=True)
    output_path = "datasets/res_bench_2k.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]✅ Dataset saved to {output_path}[/bold green]")
    console.print(f"Total samples: {len(dataset)}")
    
    # Print stats
    console.print("\n[bold]Score Distribution:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Score", style="cyan")
    table.add_column("Count", style="yellow")
    for score in [1.0, 2.0, 3.0]:
        table.add_row(str(score), str(score_counts[score]))
    console.print(table)

if __name__ == "__main__":
    main()
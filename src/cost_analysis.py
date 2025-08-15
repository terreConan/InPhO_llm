"""
Cost Analysis for InPhO LLM Research Project
Analyzes current API usage and provides cost estimates for future experiments
"""

import pandas as pd
from config import MODELS, estimate_cost, DATA_PATHS

def analyze_current_data():
    """Analyze the current data to understand scope and costs"""
    print("=== CURRENT DATA ANALYSIS ===\n")
    
    # Load data files
    try:
        pairs_df = pd.read_csv(DATA_PATHS["pairs"])
        human_df = pd.read_csv(DATA_PATHS["human_evaluations"])
        idea_mapping_df = pd.read_csv(DATA_PATHS["idea_mapping"])
        
        print(f"Total idea pairs: {len(pairs_df)}")
        print(f"Total human evaluations: {len(human_df)}")
        print(f"Unique ideas: {len(idea_mapping_df)}")
        
        # Analyze expertise levels
        expertise_counts = human_df["first_area_level"].value_counts().sort_index()
        print(f"\nHuman expertise level distribution:")
        for level, count in expertise_counts.items():
            level_name = {1: "amateur", 2: "course_taker", 3: "phd_student", 4: "expert"}.get(level, f"level_{level}")
            print(f"  {level_name}: {count} evaluations")
        
        return len(pairs_df), len(human_df), expertise_counts
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return 0, 0, {}

def estimate_costs_for_models(num_pairs: int):
    """Estimate costs for different models"""
    print(f"\n=== COST ESTIMATES FOR {num_pairs} IDEA PAIRS ===\n")
    
    # Conservative token estimates per pair
    tokens_per_pair = {
        "system_prompt": 100,    # System prompt tokens
        "user_prompt": 50,       # User prompt tokens  
        "response": 50,          # Response tokens
        "total": 200             # Total per pair
    }
    
    total_tokens = num_pairs * tokens_per_pair["total"]
    
    print(f"Estimated tokens per pair: {tokens_per_pair['total']}")
    print(f"Total estimated tokens: {total_tokens}\n")
    
    print("Cost estimates by model:")
    print("-" * 80)
    print(f"{'Model':<25} {'Cost per 1K tokens':<15} {'Cost per token':<12} {'Total Cost':<15}")
    print("-" * 80)
    
    for model_id, model_config in MODELS.items():
        cost = estimate_cost(model_id, total_tokens)
        cost_per_token = model_config.cost_per_1k_tokens / 1000
        print(f"{model_config.name:<25} ${model_config.cost_per_1k_tokens:<14.4f} ${cost_per_token:<11.6f} ${cost:<14.2f}")
    
    print("-" * 80)
    
    return total_tokens

def analyze_rate_limits():
    """Analyze rate limits and processing time"""
    print(f"\n=== RATE LIMIT ANALYSIS ===\n")
    
    for model_id, model_config in MODELS.items():
        calls_per_minute = model_config.rate_limit_per_minute
        calls_per_hour = calls_per_minute * 60
        calls_per_day = calls_per_hour * 24
        
        print(f"{model_config.name}:")
        print(f"  Rate limit: {calls_per_minute} calls/minute")
        print(f"  Per hour: {calls_per_hour} calls")
        print(f"  Per day: {calls_per_day} calls")
        
        # Estimate processing time for 18,757 pairs
        pairs = 18757
        minutes_needed = pairs / calls_per_minute
        hours_needed = minutes_needed / 60
        days_needed = hours_needed / 24
        
        print(f"  Time for {pairs} pairs: {minutes_needed:.1f} minutes ({hours_needed:.1f} hours, {days_needed:.1f} days)")
        print()



def main():
    """Main analysis function"""
    print("InPhO LLM Research Project - Cost Analysis")
    print("=" * 50)
    
    # Analyze current data
    num_pairs, num_human_evals, expertise_counts = analyze_current_data()
    
    # Estimate costs
    total_tokens = estimate_costs_for_models(num_pairs)
    
    # Analyze rate limits
    analyze_rate_limits()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Current scope: {num_pairs} idea pairs")
    print(f"Human baseline: {num_human_evals} evaluations across {len(expertise_counts)} expertise levels")
    print(f"Estimated tokens needed: {total_tokens}")
    print(f"Cost range: ${estimate_cost('gemini-pro', total_tokens):.2f} - ${estimate_cost('gpt-4', total_tokens):.2f}")

if __name__ == "__main__":
    main() 
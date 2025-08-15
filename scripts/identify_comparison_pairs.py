"""
Identify idea pairs with sufficient human data across expertise levels
Based on the roadmap Phase 1.1
"""

import pandas as pd
import re
import sys
sys.path.append('..')
from src.config import DATA_PATHS, EXPERTISE_LEVELS

def identify_comparison_pairs():
    """Identify idea pairs that have human data across multiple expertise levels"""
    print("=== IDENTIFYING COMPARISON PAIRS ===\n")
    
    # Load all expertise level data
    expertise_data = {}
    for level, level_name in EXPERTISE_LEVELS.items():
        try:
            data = pd.read_csv(f"../data/processed/human_data_{level_name}.csv")
            expertise_data[level_name] = data
            print(f"Loaded {len(data)} pairs for {level_name}")
        except FileNotFoundError:
            print(f"No data file for {level_name}")
            expertise_data[level_name] = pd.DataFrame()
    
    # Find pairs that appear in multiple expertise levels
    all_keys = set()
    for data in expertise_data.values():
        if not data.empty:
            all_keys.update(data["key"].tolist())
    
    print(f"\nTotal unique idea pairs across all levels: {len(all_keys)}")
    
    # Analyze coverage by expertise level
    coverage_analysis = {}
    for level_name, data in expertise_data.items():
        if not data.empty:
            keys_in_level = set(data["key"].tolist())
            coverage_analysis[level_name] = keys_in_level
            print(f"{level_name}: {len(keys_in_level)} pairs")
    
    # Find pairs with data in multiple levels
    multi_level_pairs = {}
    for key in all_keys:
        levels_with_data = []
        for level_name, keys in coverage_analysis.items():
            if key in keys:
                levels_with_data.append(level_name)
        
        if len(levels_with_data) > 1:
            multi_level_pairs[key] = levels_with_data
    
    print(f"\nPairs with data in multiple levels: {len(multi_level_pairs)}")
    
    # Show distribution of coverage
    coverage_counts = {}
    for key, levels in multi_level_pairs.items():
        num_levels = len(levels)
        coverage_counts[num_levels] = coverage_counts.get(num_levels, 0) + 1
    
    print("\nCoverage distribution:")
    for num_levels in sorted(coverage_counts.keys()):
        count = coverage_counts[num_levels]
        print(f"  {num_levels} levels: {count} pairs")
    
    # Find pairs with data in all 4 levels
    all_level_pairs = [key for key, levels in multi_level_pairs.items() if len(levels) == 4]
    print(f"\nPairs with data in all 4 levels: {len(all_level_pairs)}")
    
    # Save comparison pairs
    if all_level_pairs:
        comparison_df = pd.DataFrame({"key": all_level_pairs})
        
        # Add idea names for the first few pairs as example
        sample_pairs = all_level_pairs[:10]
        for key in sample_pairs:
            # Find the idea names from any of the data files
            for level_name, data in expertise_data.items():
                if not data.empty:
                    pair_data = data[data["key"] == key]
                    if not pair_data.empty:
                        ideaA = pair_data.iloc[0]["ideaA"]
                        ideaB = pair_data.iloc[0]["ideaB"]
                        print(f"  {ideaA} ↔ {ideaB}")
                        break
        
        comparison_df.to_csv("../data/processed/comparison_pairs_all_levels.csv", index=False)
        print(f"\nSaved {len(all_level_pairs)} comparison pairs to comparison_pairs_all_levels.csv")
    
    return all_level_pairs

def analyze_pair_quality():
    """Analyze the quality of human data for comparison pairs"""
    print("\n=== ANALYZING PAIR QUALITY ===\n")
    
    # Load comparison pairs
    try:
        comparison_pairs = pd.read_csv("../data/processed/comparison_pairs_all_levels.csv")
        print(f"Loaded {len(comparison_pairs)} comparison pairs")
    except FileNotFoundError:
        print("No comparison pairs file found. Run identify_comparison_pairs() first.")
        return
    
    # Analyze each expertise level
    for level_name in EXPERTISE_LEVELS.values():
        try:
            data = pd.read_csv(f"../data/processed/human_data_{level_name}.csv")
            comparison_data = data[data["key"].isin(comparison_pairs["key"])]
            
            print(f"\n{level_name.upper()}:")
            print(f"  Pairs with data: {len(comparison_data)}")
            
            if len(comparison_data) > 0:
                # Analyze relatedness
                rel_mean = comparison_data["humanRelatedness"].mean()
                rel_std = comparison_data["humanRelatedness"].std()
                print(f"  Relatedness: mean={rel_mean:.2f}, std={rel_std:.2f}")
                
                # Analyze generality
                gen_mean = comparison_data["humanGenerality"].mean()
                gen_std = comparison_data["humanGenerality"].std()
                print(f"  Generality: mean={gen_mean:.2f}, std={gen_std:.2f}")
                
                # Show distribution
                rel_dist = comparison_data["humanRelatedness"].value_counts().sort_index()
                print(f"  Relatedness distribution: {dict(rel_dist)}")
                
        except FileNotFoundError:
            print(f"No data file for {level_name}")


if __name__ == "__main__":
    identify_comparison_pairs()
    analyze_pair_quality()
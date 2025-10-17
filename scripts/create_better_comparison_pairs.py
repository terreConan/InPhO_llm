"""
Create a better comparison pairs file that shows expertise level coverage
"""

import pandas as pd
from src.config import EXPERTISE_LEVELS, DATA_PATHS

def create_better_comparison_pairs():
    """Create a comparison pairs file with expertise level coverage"""
    print("=== CREATING BETTER COMPARISON PAIRS ===\n")
    
    # Load all expertise level data
    expertise_data = {}
    for level, level_name in EXPERTISE_LEVELS.items():
        try:
            data = pd.read_csv(f"{DATA_PATHS['output_dir']}human_data_{level_name}.csv")
            expertise_data[level_name] = data
            print(f"Loaded {len(data)} pairs for {level_name}")
        except FileNotFoundError:
            print(f"No data file for {level_name}")
            expertise_data[level_name] = pd.DataFrame()
    
    # Find all unique keys
    all_keys = set()
    for data in expertise_data.values():
        if not data.empty:
            all_keys.update(data["key"].tolist())
    
    print(f"\nTotal unique pairs across all levels: {len(all_keys)}")
    
    # Create coverage analysis
    coverage_data = []
    for key in all_keys:
        levels_with_data = []
        for level_name, data in expertise_data.items():
            if not data.empty and key in data["key"].tolist():
                levels_with_data.append(level_name)
        
        if len(levels_with_data) > 1:  # Only pairs with data in 2+ levels
            # Get idea names from any available data
            ideaA, ideaB = "", ""
            for level_name, data in expertise_data.items():
                if not data.empty:
                    pair_data = data[data["key"] == key]
                    if not pair_data.empty:
                        ideaA = pair_data.iloc[0]["ideaA"]
                        ideaB = pair_data.iloc[0]["ideaB"]
                        break
            
            coverage_data.append({
                "key": key,
                "ideaA": ideaA,
                "ideaB": ideaB,
                "num_levels": len(levels_with_data),
                "levels": ", ".join(levels_with_data),
                "has_amateur": "amateur" in levels_with_data,
                "has_course_taker": "course_taker" in levels_with_data,
                "has_phd_student": "phd_student" in levels_with_data,
                "has_expert": "expert" in levels_with_data
            })
    
    # Create DataFrame and sort by number of levels (most coverage first)
    comparison_df = pd.DataFrame(coverage_data)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("num_levels", ascending=False)
    
    print(f"\nPairs with data in 2+ levels: {len(comparison_df)}")
    
    # Show distribution
    level_counts = comparison_df["num_levels"].value_counts().sort_index()
    print("\nCoverage distribution:")
    for num_levels, count in level_counts.items():
        print(f"  {num_levels} levels: {count} pairs")
    
    # Save the better comparison file
    output_file = f"{DATA_PATHS['output_dir']}comparison_pairs_with_coverage.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(comparison_df)} pairs to {output_file}")
    
    return comparison_df

if __name__ == "__main__":
    create_better_comparison_pairs() 
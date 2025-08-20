"""
Extract human data for all expertise levels
"""

import pandas as pd
import re
import sys
sys.path.append('..')
from src.config import DATA_PATHS, EXPERTISE_LEVELS

def extract_all_expertise_levels():
    """Extract human data for all expertise levels"""
    print("=== EXTRACTING ALL EXPERTISE LEVELS ===\n")
    
    # Load data
    human_df = pd.read_csv(DATA_PATHS["human_evaluations"])
    idea_mapping_df = pd.read_csv(DATA_PATHS["idea_mapping"])
    
    # Create ID to label mapping
    id2label = idea_mapping_df.set_index("ID")["label"].to_dict()
    
    # Canonical function for matching
    canon = lambda s: re.sub(r"[^a-z0-9]+","",s.lower()) if isinstance(s,str) else s
    
    # Process each expertise level
    for level, level_name in EXPERTISE_LEVELS.items():
        print(f"Processing {level_name} (level {level})...")
        
        # Filter for this expertise level
        level_data = human_df[pd.to_numeric(human_df["first_area_level"], errors="coerce") == level].copy()
        
        if len(level_data) == 0:
            print(f"  No data for level {level}")
            continue
            
        print(f"  Found {len(level_data)} evaluations")
        
        # Convert IDs to labels
        level_data["ideaA"] = level_data["ante_id"].map(id2label)
        level_data["ideaB"] = level_data["cons_id"].map(id2label)
        
        # Remove rows where we couldn't map the IDs
        level_data = level_data.dropna(subset=["ideaA", "ideaB"])
        print(f"  After mapping: {len(level_data)} evaluations")
        
        # Create canonical keys for matching
        level_data["key"] = level_data.apply(
            lambda r: tuple(sorted((canon(r["ideaA"]), canon(r["ideaB"])))), axis=1
        )
        
        # Process relatedness and generality
        level_data["humanRelatedness"] = pd.to_numeric(level_data["relatedness"], errors="coerce")
        level_data["humanGenerality"] = pd.to_numeric(level_data["generality"], errors="coerce")
        
        # Save to file
        output_file = f"../data/processed/human_data_{level_name}.csv"
        level_data[["key", "ideaA", "ideaB", "humanRelatedness", "humanGenerality"]].to_csv(
            output_file, index=False
        )
        print(f"  Saved to {output_file}")
        
        # Show some statistics
        print(f"  Relatedness range: {level_data['humanRelatedness'].min()} to {level_data['humanRelatedness'].max()}")
        print(f"  Generality range: {level_data['humanGenerality'].min()} to {level_data['humanGenerality'].max()}")
        print()
    
    print("=== EXPERTISE LEVEL EXTRACTION COMPLETE ===")

def analyze_expertise_distribution():
    """Analyze the distribution of expertise levels"""
    print("=== EXPERTISE LEVEL DISTRIBUTION ===\n")
    
    human_df = pd.read_csv(DATA_PATHS["human_evaluations"])
    
    # Count by expertise level
    expertise_counts = human_df["first_area_level"].value_counts().sort_index()
    
    print("Evaluations by expertise level:")
    for level, count in expertise_counts.items():
        level_name = EXPERTISE_LEVELS.get(level, f"level_{level}")
        print(f"  {level_name}: {count} evaluations")
    
    # Show unique users by expertise level
    print("\nUnique users by expertise level:")
    for level in sorted(human_df["first_area_level"].unique()):
        if pd.isna(level):
            continue
        level_users = human_df[human_df["first_area_level"] == level]["uid"].nunique()
        level_name = EXPERTISE_LEVELS.get(level, f"level_{level}")
        print(f"  {level_name}: {level_users} unique users")
    
    # Show idea pairs by expertise level
    print("\nUnique idea pairs by expertise level:")
    for level in sorted(human_df["first_area_level"].unique()):
        if pd.isna(level):
            continue
        level_data = human_df[human_df["first_area_level"] == level]
        unique_pairs = level_data.groupby(["ante_id", "cons_id"]).size().shape[0]
        level_name = EXPERTISE_LEVELS.get(level, f"level_{level}")
        print(f"  {level_name}: {unique_pairs} unique pairs")

if __name__ == "__main__":
    analyze_expertise_distribution()
    print("\n" + "="*50 + "\n")
    extract_all_expertise_levels() 
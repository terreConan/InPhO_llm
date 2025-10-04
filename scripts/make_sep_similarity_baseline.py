"""
Build SEP similarity-only baseline (no DB).
Reads directed J from local CSV `sep_idea_graph_edges.csv`.
Outputs: data/processed/sep_similarity_baseline.csv
"""

import os
import pandas as pd

# -----------------------------
# Config (local-only)
# -----------------------------
PAIRS_CSV  = "data/processed/comparison_pairs_with_coverage.csv"
MAP_CSV    = "data/raw/idea_id_label_mapping.csv"
EDGES_CSV  = "data/raw/sep_idea_graph_edges.csv"   # must contain ante_id, cons_id, jweight
OUT_CSV    = "data/processed/sep_similarity_baseline.csv"

def banner(msg: str):
    print(f"\n=== {msg} ===\n")

def nanmean(a, b):
    if pd.notna(a) and pd.notna(b):
        return 0.5 * (a + b)
    if pd.notna(a):
        return a
    if pd.notna(b):
        return b
    return float("nan")

def main():
    banner("LOADING INPUTS")

    # Pairs & mapping
    pairs = pd.read_csv(PAIRS_CSV)
    mapping = pd.read_csv(MAP_CSV)

    # Keep ideaA / ideaB naming for compatibility
    pairs["pair_id"] = pairs.index + 1
    mapping = mapping.rename(columns={"label": "idea_title", "ID": "idea_id"})

    # Normalize titles for robust joins
    pairs["ideaA_norm"] = pairs["ideaA"].astype(str).str.strip()
    pairs["ideaB_norm"] = pairs["ideaB"].astype(str).str.strip()
    mapping["idea_title_norm"] = mapping["idea_title"].astype(str).str.strip()

    # Map ideaA to id
    pairs = pairs.merge(
        mapping[["idea_title_norm", "idea_id"]],
        left_on="ideaA_norm", right_on="idea_title_norm", how="left"
    ).rename(columns={"idea_id": "i_id"}).drop(columns=["idea_title_norm"])

    # Map ideaB to id
    pairs = pairs.merge(
        mapping[["idea_title_norm", "idea_id"]],
        left_on="ideaB_norm", right_on="idea_title_norm", how="left"
    ).rename(columns={"idea_id": "j_id"}).drop(columns=["idea_title_norm"])

    # Report mapping coverage
    n_pairs = len(pairs)
    n_with_ids = (pairs["i_id"].notna() & pairs["j_id"].notna()).sum()
    print(f"Pairs total: {n_pairs}")
    print(f"Pairs with both IDs mapped: {n_with_ids} ({n_with_ids/n_pairs:.1%})")

    banner("LOADING EDGES (LOCAL CSV)")

    # Edges CSV must have ante_id, cons_id, jweight
    edges = pd.read_csv(EDGES_CSV)

    # Basic column sanity
    required_cols = {"ante_id", "cons_id", "jweight"}
    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"{EDGES_CSV} is missing columns: {sorted(missing)}")

    # Keep only needed cols, enforce dtypes, and drop obvious dups
    edges = edges[["ante_id", "cons_id", "jweight"]].copy()
    edges = edges.dropna(subset=["ante_id", "cons_id"])  # jweight may be NaN but ids must exist
    edges["ante_id"] = edges["ante_id"].astype(int)
    edges["cons_id"] = edges["cons_id"].astype(int)
    edges = edges.drop_duplicates(subset=["ante_id", "cons_id"], keep="first")

    print(f"Loaded {len(edges):,} directed edges with J(i->j)")

    banner("MERGING DIRECTED J INTO PAIRS")

    # Restrict to pairs that have both ids
    need = pairs[pairs["i_id"].notna() & pairs["j_id"].notna()].copy()
    need["i_id"] = need["i_id"].astype(int)
    need["j_id"] = need["j_id"].astype(int)

    # Build forward and reverse views
    fwd = edges.rename(columns={"ante_id": "i_id", "cons_id": "j_id", "jweight": "j_i_to_j"})
    rev = edges.rename(columns={"ante_id": "j_id", "cons_id": "i_id", "jweight": "j_j_to_i"})

    # Merge onto all pairs (keep unmatched as NaN)
    out = pairs.merge(fwd[["i_id", "j_id", "j_i_to_j"]], on=["i_id", "j_id"], how="left")
    out = out.merge(rev[["i_id", "j_id", "j_j_to_i"]], on=["i_id", "j_id"], how="left")

    banner("COMPUTING BASELINE FIELDS")

    out["j_sym"] = out.apply(lambda r: nanmean(r["j_i_to_j"], r["j_j_to_i"]), axis=1)
    out["has_sep"] = out["j_i_to_j"].notna() | out["j_j_to_i"].notna()

    final = out[[
        "pair_id", "ideaA", "ideaB", "i_id", "j_id",
        "j_i_to_j", "j_j_to_i", "j_sym", "has_sep"
    ]]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    final.to_csv(OUT_CSV, index=False)

    banner("DONE")
    covered = int(final["has_sep"].sum())
    print(f"Wrote {OUT_CSV}")
    print(f"Coverage: {covered}/{len(final)} pairs ({covered/len(final):.1%})")
    print("\nSample:")
    print(final.head().to_string(index=False))

if __name__ == "__main__":
    main()

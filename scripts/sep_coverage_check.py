#!/usr/bin/env python3

import os
import sys
import pandas as pd
from typing import Dict, Set, Tuple, List


def normalize_title(value: str) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def load_edges_columns(sample_rows: int = 1000) -> List[str]:
    sample = pd.read_csv("data/raw/sep_idea_graph_edges.csv", nrows=sample_rows)
    return list(sample.columns)


def inspect_edges_schema() -> None:
    print("=== STEP 1: Inspect SEP edges schema ===")
    try:
        columns = load_edges_columns()
        print("Columns:", columns)
        present_lower = {c.lower() for c in columns}
        needed = {"n_i", "n_j", "n_ij", "n"}
        print("Has required count columns?", needed.issubset(present_lower))
        if "occurs_in" in columns:
            sample = pd.read_csv("data/raw/sep_idea_graph_edges.csv", usecols=["occurs_in"], nrows=10000)
            uniq = sample["occurs_in"].dropna().value_counts().head(10).to_dict()
            print("occurs_in unique values (top 10):", uniq)
        else:
            print("occurs_in column not present (or named differently), window type unclear")
    except Exception as exc:
        print("Could not read edges file sample:", exc)
    print()


def load_edge_sets(columns_hint: List[str]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], bool]:
    use_cols = []
    for c in ["ante_id", "cons_id", "occurs_in"]:
        if c in columns_hint:
            use_cols.append(c)
    # Fallback if hint is empty
    if not use_cols:
        use_cols = ["ante_id", "cons_id"]

    edges = pd.read_csv("data/raw/sep_idea_graph_edges.csv", usecols=[c for c in use_cols if c in columns_hint])
    has_occurs = "occurs_in" in edges.columns

    # Drop rows missing ids
    keep_cols = [c for c in ["ante_id", "cons_id"] if c in edges.columns]
    edges = edges.dropna(subset=keep_cols)

    # Normalize to string ids
    if "ante_id" in edges.columns:
        edges["ante_id"] = edges["ante_id"].astype(str)
    if "cons_id" in edges.columns:
        edges["cons_id"] = edges["cons_id"].astype(str)

    edge_dir: Set[Tuple[str, str]] = set()
    if {"ante_id", "cons_id"}.issubset(edges.columns):
        edge_dir = set(zip(edges["ante_id"], edges["cons_id"]))
    edge_undir: Set[Tuple[str, str]] = set(tuple(sorted(t)) for t in edge_dir)

    print("Loaded edges shape:", edges.shape, "| occurs_in column:", has_occurs)
    print("Unique directed edges:", len(edge_dir))
    print("Unique undirected pairs:", len(edge_undir))

    return edge_dir, edge_undir, has_occurs


def load_mapping() -> Dict[str, str]:
    mapping_df = pd.read_csv("data/raw/idea_id_label_mapping.csv")
    mapping_df["label_norm"] = mapping_df["label"].apply(normalize_title)
    title_to_id: Dict[str, str] = dict(zip(mapping_df["label_norm"], mapping_df["ID"].astype(str)))
    return title_to_id


def analyze_pair_coverage(edge_dir: Set[Tuple[str, str]], edge_undir: Set[Tuple[str, str]]) -> pd.DataFrame:
    pairs_df = pd.read_csv("data/processed/comparison_pairs_with_coverage.csv")
    title_to_id = load_mapping()

    results = []
    missing_titles: Set[str] = set()

    for idx, row in pairs_df.iterrows():
        key = row.get("key", idx)
        idea_a_raw = row["ideaA"]
        idea_b_raw = row["ideaB"]
        idea_a = normalize_title(idea_a_raw)
        idea_b = normalize_title(idea_b_raw)

        id_a = title_to_id.get(idea_a)
        id_b = title_to_id.get(idea_b)
        has_ids = (id_a is not None) and (id_b is not None)

        if id_a is None:
            missing_titles.add(str(idea_a_raw))
        if id_b is None:
            missing_titles.add(str(idea_b_raw))

        edge_i_to_j = (id_a, id_b) in edge_dir if has_ids else False
        edge_j_to_i = (id_b, id_a) in edge_dir if has_ids else False
        edge_any = False
        if has_ids:
            edge_any = edge_i_to_j or edge_j_to_i or (tuple(sorted((id_a, id_b))) in edge_undir)

        results.append({
            "key": key,
            "idea_a": idea_a_raw,
            "idea_b": idea_b_raw,
            "id_a": id_a,
            "id_b": id_b,
            "has_ids": has_ids,
            "edge_i_to_j": edge_i_to_j,
            "edge_j_to_i": edge_j_to_i,
            "edge_any": edge_any,
        })

    res = pd.DataFrame(results)

    total_pairs = len(res)
    covered_ids = int(res["has_ids"].sum())
    covered_edges = int(res["edge_any"].sum())

    print("=== STEP 3: Coverage summary ===")
    print(f"Pairs with both IDs mapped: {covered_ids}/{total_pairs} ({covered_ids/total_pairs*100:.1f}%)")
    print(f"Pairs with an SEP edge present (any direction): {covered_edges}/{total_pairs} ({covered_edges/total_pairs*100:.1f}%)")

    if covered_ids > covered_edges:
        examples = res[(res["has_ids"]) & (~res["edge_any"])].head(10)
        if not examples.empty:
            print("\nExamples with IDs but no SEP edge (up to 10):")
            for _, rr in examples.iterrows():
                print(f" - {rr['idea_a']} \u2194 {rr['idea_b']} (ids {rr['id_a']}, {rr['id_b']})")

    if missing_titles:
        miss_list = sorted(list(missing_titles))
        print(f"\nMissing concept titles in mapping: {len(miss_list)} (showing up to 20)")
        for t in miss_list[:20]:
            print(" -", t)

    return res


def main() -> None:
    inspect_edges_schema()

    # Load edges for coverage checks
    try:
        columns_hint = load_edges_columns()
    except Exception:
        columns_hint = ["ante_id", "cons_id", "occurs_in"]

    edge_dir, edge_undir, has_occurs = load_edge_sets(columns_hint)

    # Analyze 712 pairs coverage
    res = analyze_pair_coverage(edge_dir, edge_undir)

    # Save interim CSV
    os.makedirs("data/interim", exist_ok=True)
    out_path = "data/interim/sep_pair_coverage.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved detailed coverage to: {out_path}")

    if has_occurs:
        print("Window type hint: 'occurs_in' present in edges file. Use this to select sentence vs document granularity if encoded.")
    else:
        print("Window type hint: no 'occurs_in' column loaded; window type unclear from edges file.")


if __name__ == "__main__":
    sys.exit(main()) 
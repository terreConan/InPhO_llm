#!/usr/bin/env python3
"""
Melt coverage booleans to long and attach human scores WITHOUT de-duplicating.
- Merge on (key, persona) only.
- Keep ALL matching human rows (duplicates preserved).
- No averaging or collapsing.

Inputs:
- data/processed/comparison_pairs_with_coverage.csv
- data/processed/human_data_amateur.csv
- data/processed/human_data_course_taker.csv
- data/processed/human_data_phd_student.csv
- data/processed/human_data_expert.csv

Output schema:
- key, ideaA, ideaB, relatedness, generality, persona
-> data/processed/human_coverage_v1.csv
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List

COVERAGE_PATH = Path("data/processed/comparison_pairs_with_coverage.csv")
OUTPUT_PATH   = Path("data/processed/human_coverage_v1.csv")

# Map coverage boolean columns -> normalized persona (robust to suffixes handled below)
COVERAGE_TO_PERSONA: Dict[str, str] = {
    "has_amateur": "novice",
    "has_course_taker": "course_taker",
    "has_phd_student": "phd",
    "has_expert": "expert",
}

# Human files by persona
HUMAN_FILES = {
    "novice":        Path("data/processed/human_data_amateur.csv"),
    "course_taker":  Path("data/processed/human_data_course_taker.csv"),
    "phd":           Path("data/processed/human_data_phd_student.csv"),
    "expert":        Path("data/processed/human_data_expert.csv"),
}

def is_truthy(val) -> bool:
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    return str(val).strip().lower() in {"1","true","t","yes","y"}

def normalize_has_col(col: str) -> str | None:
    """Map various 'has_*' names (e.g., has_expert_yes) to persona labels."""
    if col in COVERAGE_TO_PERSONA:
        return COVERAGE_TO_PERSONA[col]
    s = col.lower().strip()
    if not s.startswith("has_"):
        return None
    s = s[4:]
    s = re.sub(r"(?:_yes|_true|_flag|_present|_covered|_1)$", "", s)
    s = s.replace("-", "_").replace(" ", "_")
    if "phd" in s: return "phd"
    if "expert" in s: return "expert"
    if "course" in s or "student" in s: return "course_taker"
    if "amateur" in s or "novice" in s: return "novice"
    return None

def load_human_union() -> pd.DataFrame:
    """
    Load human per-persona CSVs and RETURN ALL ROWS.
    We keep only rows with numeric relatedness/generality; no averaging, no de-duping.
    """
    frames: List[pd.DataFrame] = []
    for persona, path in HUMAN_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Must have key + the two score columns
        missing = {"key", "humanRelatedness", "humanGenerality"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")

        # Numeric coercion; drop rows that don't have both scores
        df["humanRelatedness"] = pd.to_numeric(df["humanRelatedness"], errors="coerce")
        df["humanGenerality"]  = pd.to_numeric(df["humanGenerality"], errors="coerce")
        df = df.dropna(subset=["humanRelatedness", "humanGenerality"])

        df["persona"] = persona
        frames.append(df[["key", "persona", "humanRelatedness", "humanGenerality"]])

    if not frames:
        raise SystemExit("No human_data_*.csv files found.")

    human = pd.concat(frames, ignore_index=True)

    # Rename to final schema names. DO NOT DROP DUPLICATES.
    human = human.rename(columns={
        "humanRelatedness": "relatedness",
        "humanGenerality":  "generality",
    })
    return human  # columns: key, persona, relatedness, generality (with duplicates)

def main():
    # 1) Melt coverage to one row per (key, ideaA, ideaB, persona)
    cov = pd.read_csv(COVERAGE_PATH)
    for req in ["key", "ideaA", "ideaB"]:
        if req not in cov.columns:
            raise ValueError(f"Coverage file missing required column: {req}")

    long_rows: List[Dict] = []
    for _, r in cov.iterrows():
        base = {"key": r["key"], "ideaA": r["ideaA"], "ideaB": r["ideaB"]}
        for col in has_cols:
            if is_truthy(r.get(col)):
                persona = normalize_has_col(col)
                if persona:
                    rec = dict(base)
                    rec["persona"] = persona
                    long_rows.append(rec)

    melted = pd.DataFrame(long_rows, columns=["key", "ideaA", "ideaB", "persona"]).drop_duplicates()

    # 2) Load human scores
    human = load_human_union()

    # 3) INNER JOIN on (key, persona) to keep only rows with human labels
    out = melted.merge(human, on=["key", "persona"], how="inner")

    # 4) Write final schema
    out = out[["key", "ideaA", "ideaB", "relatedness", "generality", "persona"]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("Wrote:", str(OUTPUT_PATH))
    print("Rows:", len(out))
    print("By persona:", out.groupby("persona").size().to_dict())

if __name__ == "__main__":
    main()


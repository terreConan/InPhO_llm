#!/usr/bin/env python3
"""
Analyze LLM outputs vs. Human judgments by expertise (persona).

Inputs (CSV):
  - human (must have: key, ideaA, ideaB, persona, relatedness, generality)
  - one or more LLM CSVs (must have: key, ideaA, ideaB, persona, relatedness, generality)
    e.g., data/processed/gpt-4o_outputs_v1.csv, data/processed/gemini_outputs_v1.csv

Outputs (in --outdir):
  - combined_llm_outputs.csv
  - merged_llm_human.csv
  - summary_llm_by_persona_model.csv
  - summary_diffs_by_persona_model.csv
  - chart_llm_mean_relatedness_by_persona.png
  - chart_llm_mean_generality_by_persona.png
  - chart_llm_vs_human_rel_diff_by_persona.png
  - chart_llm_vs_human_gen_diff_by_persona.png
  - correlation_by_persona_model.csv  (if SciPy available)

Usage:
  python scripts/analyze_llm_human.py \
    --human data/processed/human_coverage_v1.csv \
    --llm data/processed/gpt-4o_outputs_v1.csv data/processed/gemini_outputs_v1.csv \
    --outdir reports/llm_analysis
"""

import argparse
import os
import sys
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Optional: correlations
try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


REQUIRED_COLS = ["key", "ideaA", "ideaB", "persona", "relatedness", "generality"]


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def assert_has_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def standardize_llm(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    # Keep only the columns we need; coerce numeric
    out = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    for col in ("relatedness", "generality"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["model"] = model_label
    return out


def infer_model_label_from_path(path: Path) -> str:
    base = path.name
    label = base.replace("_outputs_v1.csv", "").replace(".csv", "")
    return label


def make_dirs(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_bar(pivot_df: pd.DataFrame, title: str, ylabel: str, xlabel: str, outpath: Path):
    # One figure per chart, no custom colors/styles
    plt.figure(figsize=(9, 5))
    ax = pivot_df.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # horizontal zero line for diffs charts
    if "Difference" in title or "LLM - Human" in ylabel:
        plt.axhline(0)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM vs Human by persona.")
    parser.add_argument("--human", required=True, help="Path to human CSV (with relatedness/generality).")
    parser.add_argument("--llm", nargs="+", required=True, help="Paths to one or more LLM CSVs.")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs.")
    args = parser.parse_args()

    human_path = Path(args.human)
    llm_paths = [Path(p) for p in args.llm]
    outdir = Path(args.outdir)
    make_dirs(outdir)

    # Load Human
    human_df = read_csv_any(human_path)
    assert_has_cols(human_df, REQUIRED_COLS, f"Human file: {human_path}")
    # To avoid name collisions after merge, rename human measures
    human_df = human_df.rename(columns={"relatedness": "relatedness_human", "generality": "generality_human"})

    # Load & combine LLMs
    llm_frames = []
    for p in llm_paths:
        df = read_csv_any(p)
        assert_has_cols(df, REQUIRED_COLS, f"LLM file: {p}")
        model_label = infer_model_label_from_path(p)
        llm_frames.append(standardize_llm(df, model_label))

    if not llm_frames:
        print("No LLM frames loaded; nothing to do.", file=sys.stderr)
        sys.exit(1)

    llm_df = pd.concat(llm_frames, ignore_index=True)

    # Save combined LLM outputs
    llm_out = outdir / "combined_llm_outputs.csv"
    llm_df.to_csv(llm_out, index=False)

    # Merge on (key, ideaA, ideaB, persona)
    merged = llm_df.merge(
        human_df[["key","ideaA","ideaB","persona","relatedness_human","generality_human"]],
        on=["key","ideaA","ideaB","persona"],
        how="left",
        validate="m:m"
    )

    merged_out = outdir / "merged_llm_human.csv"
    merged.to_csv(merged_out, index=False)

    # --------------------------
    # LLM vs LLM summaries
    # --------------------------
    if {"persona","model","relatedness","generality"}.issubset(merged.columns):
        llm_summary = (
            merged.groupby(["persona","model"])[["relatedness","generality"]]
            .mean()
            .reset_index()
            .sort_values(["persona","model"])
        )
        llm_summary_out = outdir / "summary_llm_by_persona_model.csv"
        llm_summary.to_csv(llm_summary_out, index=False)

        # Charts: mean relatedness/generality by persona/model
        pivot_rel = llm_summary.pivot(index="persona", columns="model", values="relatedness")
        plot_bar(
            pivot_rel,
            title="LLM Mean Relatedness by Persona",
            ylabel="Mean Relatedness",
            xlabel="Persona",
            outpath=outdir / "chart_llm_mean_relatedness_by_persona.png",
        )

        pivot_gen = llm_summary.pivot(index="persona", columns="model", values="generality")
        plot_bar(
            pivot_gen,
            title="LLM Mean Generality by Persona",
            ylabel="Mean Generality",
            xlabel="Persona",
            outpath=outdir / "chart_llm_mean_generality_by_persona.png",
        )

    # --------------------------
    # LLM vs Human differences
    # --------------------------
    # Compute diffs (LLM - Human)
    if "relatedness_human" in merged.columns:
        merged["rel_diff"] = pd.to_numeric(merged["relatedness"], errors="coerce") - pd.to_numeric(merged["relatedness_human"], errors="coerce")
    if "generality_human" in merged.columns:
        merged["gen_diff"] = pd.to_numeric(merged["generality"], errors="coerce") - pd.to_numeric(merged["generality_human"], errors="coerce")

    have_rel_diff = "rel_diff" in merged.columns
    have_gen_diff = "gen_diff" in merged.columns

    if have_rel_diff or have_gen_diff:
        agg_cols = []
        if have_rel_diff: agg_cols.append("rel_diff")
        if have_gen_diff: agg_cols.append("gen_diff")

        diff_summary = (
            merged.groupby(["persona","model"])[agg_cols]
            .mean()
            .reset_index()
            .sort_values(["persona","model"])
        )
        diff_summary_out = outdir / "summary_diffs_by_persona_model.csv"
        diff_summary.to_csv(diff_summary_out, index=False)

        # Charts
        if have_rel_diff:
            pivot_rel_diff = diff_summary.pivot(index="persona", columns="model", values="rel_diff")
            plot_bar(
                pivot_rel_diff,
                title="LLM vs Human (Mean Relatedness Difference) by Persona",
                ylabel="LLM - Human (Relatedness)",
                xlabel="Persona",
                outpath=outdir / "chart_llm_vs_human_rel_diff_by_persona.png",
            )

        if have_gen_diff:
            pivot_gen_diff = diff_summary.pivot(index="persona", columns="model", values="gen_diff")
            plot_bar(
                pivot_gen_diff,
                title="LLM vs Human (Mean Generality Difference) by Persona",
                ylabel="LLM - Human (Generality)",
                xlabel="Persona",
                outpath=outdir / "chart_llm_vs_human_gen_diff_by_persona.png",
            )

    # --------------------------
    # Correlations (optional)
    # --------------------------
    if HAVE_SCIPY:
        rows = []
        for model in sorted(merged["model"].unique()):
            for persona in sorted(merged["persona"].dropna().unique()):
                sub = merged[(merged["model"] == model) & (merged["persona"] == persona)].copy()
                r_rel = p_rel = None
                r_gen = p_gen = None
                # relatedness corr
                if {"relatedness","relatedness_human"}.issubset(sub.columns):
                    a = pd.to_numeric(sub["relatedness"], errors="coerce")
                    b = pd.to_numeric(sub["relatedness_human"], errors="coerce")
                    mask = a.notna() & b.notna()
                    if mask.sum() >= 3:
                        r_rel, p_rel = spearmanr(a[mask], b[mask])
                # generality corr
                if {"generality","generality_human"}.issubset(sub.columns):
                    a = pd.to_numeric(sub["generality"], errors="coerce")
                    b = pd.to_numeric(sub["generality_human"], errors="coerce")
                    mask = a.notna() & b.notna()
                    if mask.sum() >= 3:
                        r_gen, p_gen = spearmanr(a[mask], b[mask])
                rows.append({
                    "model": model,
                    "persona": persona,
                    "spearman_relatedness": r_rel,
                    "p_relatedness": p_rel,
                    "spearman_generality": r_gen,
                    "p_generality": p_gen,
                    "n_pairs": int(mask.sum()) if 'mask' in locals() else 0,
                })
        corr_df = pd.DataFrame(rows)
        corr_out = outdir / "correlation_by_persona_model.csv"
        corr_df.to_csv(corr_out, index=False)

    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()

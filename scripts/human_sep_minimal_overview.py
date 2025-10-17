import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

FIG_DIR = Path("reports/figures")

# Helper function that maps a concept pair to order-independent key so that orders match
def undirected_key(a: str, b: str) -> tuple:
    a = str(a).lower().strip()
    b = str(b).lower().strip()
    return tuple(sorted([a, b]))

def load_sep_baseline() -> pd.DataFrame:
    sep = pd.read_csv("data/processed/sep_similarity_baseline.csv")
    sep["pair_key"] = [undirected_key(a, b) for a, b in zip(sep["ideaA"], sep["ideaB"])]
    return sep[["pair_key", "j_sym", "has_sep", "jweight"]].rename(columns={"jweight": "sep_sim"})

def load_human_all() -> pd.DataFrame:
    files = sorted(glob.glob("data/processed/human_data_*.csv"))
    frames = []
    for path in files:
        level = Path(path).stem.split("_")[-1]
        df = pd.read_csv(path)
        if not {"ideaA", "ideaB", "humanRelatedness"}.issubset(df.columns):
            continue
        df["pair_key"] = [undirected_key(a, b) for a, b in zip(df["ideaA"], df["ideaB"])]
        df["expertise"] = level
        frames.append(df[["pair_key", "expertise", "humanRelatedness"]])
    return pd.concat(frames, ignore_index=True)

# Plot Creation
def plot_sep_hist(sep_df: pd.DataFrame, out_path: Path):
    vals = sep_df["sep_sim"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7,5))
    plt.hist(vals, bins=40)
    plt.title("SEP Similarity Distribution (j_sym)")
    plt.xlabel("j_sym")
    plt.ylabel("Count of pairs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_human_box(human_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7,5))
    human_df.boxplot(column="humanRelatedness", by="expertise", grid=False)
    plt.title("Human Relatedness by Expertise Level")
    plt.suptitle("")  # remove pandas default
    plt.xlabel("Expertise Level")
    plt.ylabel("Relatedness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_sep_vs_human(merged: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7,5))
    for lvl, sub in merged.groupby("expertise"):
        plt.scatter(sub["sep_sim"], sub["humanRelatedness"], s=16, alpha=0.6, label=lvl)
    # overall Spearman correlation
    x = merged["sep_sim"].values
    y = merged["humanRelatedness"].values
    ok = np.isfinite(x) & np.isfinite(y)
    title = "SEP vs Human Relatedness"
    if ok.sum() > 5:
        r, p = spearmanr(x[ok], y[ok])
        title += f" (Spearman r={r:.2f}, p={p:.1e}, n={ok.sum()})"
    plt.title(title)
    plt.xlabel("SEP similarity (j_sym)")
    plt.ylabel("Human relatedness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    sep = load_sep_baseline()
    human = load_human_all()

    # Print SEP coverage if available
    if "has_sep" in sep.columns:
        total = len(sep)
        covered = int(sep["has_sep"].fillna(False).sum())
        print(f"SEP coverage: {covered}/{total} ({covered/total*100:.1f}%)")

    # Merge human and SEP on undirected key
    merged = human.merge(sep, on="pair_key", how="inner")
    print(f"Merged rows for SEP vs Human: {len(merged)}")

    # Per-level Spearman correlations
    for lvl, sub in merged.groupby("expertise"):
        x, y = sub["sep_sim"], sub["humanRelatedness"]
        ok = x.notna() & y.notna()
        if ok.sum() > 5:
            r, p = spearmanr(x[ok], y[ok])
            print(f"{lvl:>12}: Spearman r={r:.2f}, p={p:.1e}, n={ok.sum()}")

    # Save plots
    plot_sep_hist(sep, FIG_DIR / "sep_histogram.png")
    plot_human_box(human, FIG_DIR / "human_boxplot_by_expertise.png")
    plot_sep_vs_human(merged, FIG_DIR / "sep_vs_human_scatter.png")


    print("Saved figures to:", FIG_DIR.resolve())

if __name__ == "__main__":
    main()



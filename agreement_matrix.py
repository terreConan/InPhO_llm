import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("expert_ai_pairs.csv")

REL_LABELS = ["Not Related","Marginally Related","Somewhat Related","Related","Highly Related"]
GEN_LABELS = ["N/A","Incomparable To","More Specific Than","As General As","More General Than"]

REL_LEVELS = [0, 1, 2, 3, 4]
GEN_LEVELS = [-1, 0, 1, 2, 3]

rel_cm = pd.crosstab(df["humanRelatedness"], df["aiRelatedness"], normalize="index")
gen_cm = pd.crosstab(df["humanGenerality"], df["aiGenerality"], normalize="index")

def plot_cm(cm, labels, order, title):
    full = cm.reindex(index=order, columns=order)
    mask = full.isna()

    plt.figure(figsize=(6, 5))
    sns.heatmap(full,
                cmap="Blues", vmin=0, vmax=1,
                mask=mask,
                xticklabels=labels, yticklabels=labels,
                annot=False, cbar=True)

    for i, row in enumerate(order):
        for j, col in enumerate(order):
            val = full.loc[row, col]
            txt = "N/A" if pd.isna(val) else f"{val:.2f}"
            plt.text(j + 0.5, i + 0.5, txt,
                     ha="center", va="center",
                     color="white" if not pd.isna(val) and val > .5 else "black",
                     fontsize=9)

    plt.xlabel("AI rating")
    plt.ylabel("Human rating")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_cm(rel_cm, REL_LABELS, list(REL_LEVELS), "Relatedness Agreement Heat‑map")
plot_cm(gen_cm, GEN_LABELS, GEN_LEVELS, "Generality Agreement Heat‑map")

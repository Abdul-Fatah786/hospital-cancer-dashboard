"""Exploratory charts saved to reports/figures.   Usage:  python -m src.eda"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import DATA_PATH, FEATURE_NAMES, FIGURES_DIR, MEAN_COLS, TARGET
from src.data_loader import load_data

PAL = {"B": "#2a9d8f", "M": "#e63946"}      # benign green, malignant red


def run(path=DATA_PATH, out_dir=FIGURES_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    sns.set_theme(style="whitegrid")
    counts = df[TARGET].value_counts().reindex(["B", "M"])
    print(f"Patients: {len(df)} | benign {counts['B']} | malignant {counts['M']} ({counts['M'] / len(df) * 100:.1f}%)")
    print(f"Missing values: {int(df[FEATURE_NAMES].isnull().sum().sum())} | duplicate rows: {int(df.duplicated().sum())}")

    # class balance
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["Benign", "Malignant"], counts.values, color=[PAL["B"], PAL["M"]])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax.set_title("Number of cases")
    fig.tight_layout(); fig.savefig(out_dir / "class_distribution.png", dpi=150); plt.close(fig)

    # distributions of the 10 mean features
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, col in zip(axes.ravel(), MEAN_COLS):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, palette=PAL, element="step", stat="density",
                     common_norm=False, ax=ax)
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(out_dir / "feature_distributions.png", dpi=130); plt.close(fig)

    # box plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, col in zip(axes.ravel(), MEAN_COLS):
        sns.boxplot(data=df, x=TARGET, y=col, order=["B", "M"], hue=TARGET, hue_order=["B", "M"],
                    palette=PAL, legend=False, ax=ax)
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(out_dir / "boxplots.png", dpi=130); plt.close(fig)

    # correlation heatmap of the mean features + diagnosis
    y = (df[TARGET] == "M").astype(int)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(df[MEAN_COLS].assign(diagnosis=y).corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation: mean features + diagnosis (M = 1)")
    fig.tight_layout(); fig.savefig(out_dir / "correlation_heatmap.png", dpi=130); plt.close(fig)

    # correlation of every feature with the target
    corr = df[FEATURE_NAMES].corrwith(y).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.barh(corr.index, corr.values, color=[PAL["B"] if v < 0 else PAL["M"] for v in corr.values])
    ax.set_title("Correlation of each feature with malignant")
    fig.tight_layout(); fig.savefig(out_dir / "correlation_with_target.png", dpi=130); plt.close(fig)
    print("Top 5 features by |correlation|:", corr.abs().sort_values(ascending=False).head(5).round(3).to_dict())
    print("Figures saved to", out_dir)


if __name__ == "__main__":
    try:
        run()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

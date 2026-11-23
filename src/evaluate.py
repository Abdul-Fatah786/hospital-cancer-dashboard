"""Plots (saved to files or returned as figures for Streamlit)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

TRAIN_COLOR, VAL_COLOR = "#264653", "#e76f51"


def loss_curve(ax, history, title="", best_epoch=None):
    ax.plot(history["train_loss"], label="Train", color=TRAIN_COLOR, lw=2)
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="Validation", color=VAL_COLOR, lw=2)
    if best_epoch:
        ax.axvline(best_epoch - 1, color="grey", ls="--", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()


def loss_curve_grid(histories: dict, diagnoses: dict = None, cols: int = 3):
    """One panel per experiment. Returns a matplotlib Figure."""
    names = list(histories)
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        h = histories[n]
        best = int(np.argmin(h["val_loss"])) + 1 if h["val_loss"] else None
        label = f"{n}  [{diagnoses[n]}]" if diagnoses and n in diagnoses else n
        loss_curve(ax, h, label, best)
    fig.suptitle("Training vs validation loss (dashed = best validation epoch)")
    fig.tight_layout()
    return fig


def confusion_figure(cm, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=["Benign", "Malignant"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def metrics_bar_figure(results_df):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    results_df.set_index("Experiment ID")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", ax=ax, colormap="viridis", edgecolor="white")
    ax.set_ylim(0.5, 1.02); ax.set_title("Validation metrics per experiment")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return fig


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

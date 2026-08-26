from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .dataset import ACTIVITY_NAMES


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    """Save loss and accuracy learning curves."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path, title: str
) -> None:
    """Save a confusion matrix with activity labels."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=ACTIVITY_NAMES,
        xticks_rotation=35,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heads_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Macro F1 as a function of attention heads."""
    subset = results[results["heads"].notna()].copy()
    if subset.empty:
        return
    subset = subset.sort_values("heads")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(subset["heads"], subset["macro_f1"], marker="o")
    ax.set_xlabel("Number of attention heads")
    ax.set_ylabel("Macro F1")
    ax.set_title("Attention-head ablation")
    ax.set_xticks(sorted(subset["heads"].astype(int).unique()))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_f1(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Macro F1 for all evaluated architectures."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(results["model"], results["macro_f1"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Architecture comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_attention_distribution(attention: np.ndarray, output_path: str | Path) -> None:
    """Save an exploratory distribution of attention weights by head."""
    if attention.ndim != 4:
        raise ValueError("Expected attention shape [B, heads, T, T]")
    means = attention.mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    for head_index, values in enumerate(means):
        ax.plot(values, label=f"Head {head_index + 1}")
    ax.set_xlabel("Key temporal position")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Exploratory attention distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

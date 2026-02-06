"""Visualization utilities for evaluation results."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_benchmark_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
) -> plt.Figure:
    """Plot grouped bar chart comparing models with CI error bars.

    Expects results_df to have columns: model, metric_mean, metric_ci_low, metric_ci_high.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    models = results_df["model"]
    means = results_df[f"{metric}_mean"]
    ci_low = results_df[f"{metric}_ci_low"]
    ci_high = results_df[f"{metric}_ci_high"]
    errors = np.array([means - ci_low, ci_high - means])

    bars = ax.bar(range(len(models)), means, yerr=errors, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_learning_curves(history: Dict[str, List[float]], title: str = "") -> plt.Figure:
    """Plot train/val loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_losses"], label="Train Loss")
    ax1.plot(history["val_losses"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves{f' — {title}' if title else ''}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_accuracies"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Validation Accuracy{f' — {title}' if title else ''}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    ablation_type: str = "feature",
) -> plt.Figure:
    """Plot ablation study results.

    Args:
        ablation_df: DataFrame with 'variant' and 'accuracy' columns (and optionally 'ci_low', 'ci_high').
        ablation_type: Type label for chart title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variants = ablation_df["variant"]
    means = ablation_df["accuracy"]

    if "ci_low" in ablation_df.columns:
        errors = np.array([
            means - ablation_df["ci_low"],
            ablation_df["ci_high"] - means,
        ])
        ax.barh(range(len(variants)), means, xerr=errors, capsize=4, color="teal", alpha=0.8)
    else:
        ax.barh(range(len(variants)), means, color="teal", alpha=0.8)

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Ablation Study: {ablation_type.replace('_', ' ').title()}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dict: Dict[str, float]) -> plt.Figure:
    """Plot horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    sorted_idx = np.argsort(values)
    ax.barh([features[i] for i in sorted_idx], [values[i] for i in sorted_idx], color="coral", alpha=0.8)
    ax.set_xlabel("Importance (accuracy drop)")
    ax.set_title("Feature Importance (Leave-One-Out)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

"""Metrics computation, bootstrap confidence intervals, and statistical tests."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from scipy import stats


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None,
) -> Dict:
    """Compute a comprehensive set of classification metrics.

    Returns dict with: accuracy, balanced_accuracy, macro_precision, macro_recall,
    macro_f1, per_class_precision, per_class_recall, per_class_f1,
    confusion_matrix, and optionally log_loss.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_precision": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_proba is not None:
        try:
            result["log_loss"] = log_loss(y_true, y_proba, labels=labels)
        except ValueError:
            result["log_loss"] = float("nan")
    return result


def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a list of scores.

    Args:
        scores: List of metric values (e.g. per-fold accuracies).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(ci_low), float(ci_high)


def paired_t_test(
    scores_a: List[float], scores_b: List[float]
) -> Tuple[float, float]:
    """Paired t-test between two sets of fold scores.

    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_val)


def compute_effect_size(
    scores_a: List[float], scores_b: List[float]
) -> float:
    """Compute Cohen's d effect size between two score distributions."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)

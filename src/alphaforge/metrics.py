"""Stable metrics that make edge cases explicit rather than misleading."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ScalarMetric: TypeAlias = float | int | None
MetricValue: TypeAlias = ScalarMetric | list[list[int]]


def classification_metrics(
    actual: np.ndarray, predicted: np.ndarray, scores: np.ndarray | None = None
) -> dict[str, MetricValue]:
    """Report fixed classification metrics; unavailable ROC-AUC is ``None``."""

    _require_finite(actual, predicted)
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Classification labels must be binary 0/1 values")
    metrics: dict[str, MetricValue] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(actual, predicted))
            if np.unique(actual).size == 2
            else None
        ),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "roc_auc": None,
    }
    if scores is not None:
        _require_finite(scores)
        if np.unique(actual).size == 2:
            metrics["roc_auc"] = float(roc_auc_score(actual, scores))
    return metrics


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, ScalarMetric]:
    """Report fixed regression metrics; undefined correlations are ``None``."""

    _require_finite(actual, predicted)
    residuals = actual - predicted
    metrics: dict[str, ScalarMetric] = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
        "mean_error": float(np.mean(residuals)),
    }
    if actual.size >= 2 and np.unique(actual).size > 1:
        metrics["r2"] = float(r2_score(actual, predicted))
    if actual.size >= 2 and np.unique(actual).size > 1 and np.unique(predicted).size > 1:
        metrics["pearson_correlation"] = float(pearsonr(actual, predicted).statistic)
        metrics["spearman_correlation"] = float(spearmanr(actual, predicted).statistic)
    return metrics


def _require_finite(*values: np.ndarray) -> None:
    if any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in values):
        raise ValueError("Metrics require finite actual values, predictions, and scores")

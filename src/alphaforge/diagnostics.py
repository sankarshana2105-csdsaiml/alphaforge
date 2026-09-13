"""Small, descriptive diagnostics for honest baseline reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_distribution_summary(target: pd.Series) -> dict[str, float | int]:
    values = target.astype(float).to_numpy()
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def class_balance_summary(target: pd.Series) -> dict[int, dict[str, float | int]]:
    counts = target.astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    return {
        int(label): {"count": int(count), "fraction": float(count / total)}
        for label, count in counts.items()
    }


def feature_correlation_summary(features: pd.DataFrame) -> dict[str, float | int]:
    correlation = features.corr(numeric_only=True).abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = upper.stack()
    return {
        "feature_count": int(features.shape[1]),
        "pair_count": int(pairs.size),
        "max_absolute_correlation": float(pairs.max()) if not pairs.empty else 0.0,
        "median_absolute_correlation": float(pairs.median()) if not pairs.empty else 0.0,
    }


def prediction_distribution_summary(predictions: pd.Series) -> dict[str, float | int]:
    return target_distribution_summary(predictions)


def residual_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    residuals = actual.astype(float) - predicted.astype(float)
    return target_distribution_summary(residuals)


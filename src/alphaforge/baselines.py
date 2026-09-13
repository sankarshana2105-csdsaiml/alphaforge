"""Baseline models evaluated on a held-out chronological test partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphaforge.dataset import ResearchDataset
from alphaforge.data import REQUIRED_COLUMNS
from alphaforge.diagnostics import (
    class_balance_summary,
    feature_correlation_summary,
    prediction_distribution_summary,
    residual_summary,
    target_distribution_summary,
)
from alphaforge.metrics import MetricValue, ScalarMetric, classification_metrics, regression_metrics
from alphaforge.splits import ChronologicalSplit, ChronologicalSplitConfig, chronological_split


class LeakageError(ValueError):
    """Raised when non-feature or suspicious columns reach a model."""


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    target_type: Literal["classification", "regression"]
    metrics: dict[str, MetricValue | ScalarMetric]
    predictions: pd.Series
    scores: pd.Series | None
    sample_counts: dict[str, int]
    split_boundaries: dict[str, str]
    feature_count: int
    status: str = "ok"
    model: Any = None
    runtime_seconds: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineReport:
    split: ChronologicalSplit
    classification: tuple[BaselineResult, ...]
    regression: tuple[BaselineResult, ...]
    diagnostics: dict[str, Any]

    def comparison(self, target_type: Literal["classification", "regression"]) -> pd.DataFrame:
        results = self.classification if target_type == "classification" else self.regression
        rows: list[dict[str, Any]] = []
        for result in results:
            row = {
                "model_name": result.model_name,
                "target_type": result.target_type,
                "status": result.status,
                "feature_count": result.feature_count,
                "runtime_seconds": result.runtime_seconds,
                "warnings": "; ".join(result.warnings),
                **result.sample_counts,
                **result.split_boundaries,
            }
            row.update({key: value for key, value in result.metrics.items() if not isinstance(value, list)})
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_baselines(
    dataset: ResearchDataset, *, split_config: ChronologicalSplitConfig | None = None
) -> BaselineReport:
    """Evaluate fixed baselines without fitting preprocessing on future rows."""

    _validate_feature_columns(dataset)
    split = chronological_split(dataset.frame, config=split_config)
    X_train, X_test = split.train.loc[:, dataset.feature_columns], split.test.loc[:, dataset.feature_columns]
    _require_finite_frame(X_train, "Training features")
    _require_finite_frame(X_test, "Test features")
    direction_column, return_column = dataset.target_columns[1], dataset.target_columns[0]
    y_direction_train = _binary_target(split.train[direction_column], "Training direction target")
    y_direction_test = _binary_target(split.test[direction_column], "Test direction target")
    y_return_train = _finite_target(split.train[return_column], "Training return target")
    y_return_test = _finite_target(split.test[return_column], "Test return target")

    classification = _classification_results(
        X_train, X_test, y_direction_train, y_direction_test, split, dataset.feature_columns
    )
    regression = _regression_results(
        X_train, X_test, y_return_train, y_return_test, split, dataset.feature_columns
    )
    diagnostics: dict[str, Any] = {
        "return_target_distribution": target_distribution_summary(dataset.y_return),
        "class_balance": {
            name: class_balance_summary(part[direction_column])
            for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
        "feature_correlation": feature_correlation_summary(dataset.X),
        "prediction_distributions": {
            result.model_name: prediction_distribution_summary(result.predictions)
            for result in (*classification, *regression)
        },
        "regression_residuals": {
            result.model_name: residual_summary(y_return_test, result.predictions)
            for result in regression
        },
    }
    return BaselineReport(split, tuple(classification), tuple(regression), diagnostics)


def _classification_results(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: ChronologicalSplit,
    feature_columns: tuple[str, ...],
) -> list[BaselineResult]:
    count, boundaries, index = split.sample_counts, split.boundaries, X_test.index
    majority = int(y_train.mode().iloc[0])
    majority_predictions = pd.Series(majority, index=index, dtype=int)
    results = [_classification_result("majority_class", y_test, majority_predictions, None, count, boundaries, 0)]

    if "return_1" in X_test:
        naive_predictions = (X_test["return_1"] > 0).astype(int)
        results.append(_classification_result("naive_recent_return_direction", y_test, naive_predictions, None, count, boundaries, 1))

    if y_train.nunique() < 2:
        fallback = Pipeline(
            [("scaler", StandardScaler()), ("classifier", DummyClassifier(strategy="most_frequent"))]
        ).fit(X_train, y_train)
        predictions = pd.Series(fallback.predict(X_test), index=index, dtype=int)
        scores = pd.Series(float(y_train.iloc[0]), index=index)
        results.append(_classification_result(
            "logistic_regression", y_test, predictions, scores, count, boundaries, len(feature_columns), "single_class_train_fallback", fallback
        ))
    else:
        model = Pipeline(
            [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1_000, random_state=0))]
        ).fit(X_train, y_train)
        predictions = pd.Series(model.predict(X_test), index=index, dtype=int)
        scores = pd.Series(model.predict_proba(X_test)[:, 1], index=index)
        results.append(_classification_result(
            "logistic_regression", y_test, predictions, scores, count, boundaries, len(feature_columns), "ok", model
        ))
    return results


def _regression_results(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: ChronologicalSplit,
    feature_columns: tuple[str, ...],
) -> list[BaselineResult]:
    count, boundaries, index = split.sample_counts, split.boundaries, X_test.index
    mean_predictions = pd.Series(float(y_train.mean()), index=index)
    zero_predictions = pd.Series(0.0, index=index)
    model = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    ).fit(X_train, y_train)
    linear_predictions = pd.Series(model.predict(X_test), index=index)
    return [
        _regression_result("historical_mean", y_test, mean_predictions, count, boundaries, 0),
        _regression_result("zero_return", y_test, zero_predictions, count, boundaries, 0),
        _regression_result("linear_regression", y_test, linear_predictions, count, boundaries, len(feature_columns), model),
    ]


def _classification_result(
    name: str, actual: pd.Series, predictions: pd.Series, scores: pd.Series | None,
    counts: dict[str, int], boundaries: dict[str, str], feature_count: int, status: str = "ok", model: Any = None,
) -> BaselineResult:
    score_array = scores.to_numpy() if scores is not None else None
    return BaselineResult(name, "classification", classification_metrics(actual.to_numpy(), predictions.to_numpy(), score_array), predictions, scores, counts, boundaries, feature_count, status, model)


def _regression_result(
    name: str, actual: pd.Series, predictions: pd.Series, counts: dict[str, int],
    boundaries: dict[str, str], feature_count: int, model: Any = None,
) -> BaselineResult:
    return BaselineResult(name, "regression", regression_metrics(actual.to_numpy(), predictions.to_numpy()), predictions, None, counts, boundaries, feature_count, "ok", model)


def _validate_feature_columns(dataset: ResearchDataset) -> None:
    features = set(dataset.feature_columns)
    forbidden = set(REQUIRED_COLUMNS).union(dataset.target_columns)
    suspicious = {
        name
        for name in features
        if name in forbidden
        or "future" in name
        or "target" in name
        or "direction" in name
    }
    if not features or suspicious:
        raise LeakageError(f"Invalid feature columns detected: {sorted(suspicious)}")
    if not features.issubset(dataset.frame.columns):
        raise LeakageError("A declared feature column is missing from the dataset frame")
    if features.intersection(dataset.target_columns):
        raise LeakageError("Target columns cannot be used as features")


def _require_finite_frame(frame: pd.DataFrame, name: str) -> None:
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def _finite_target(target: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(target, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _binary_target(target: pd.Series, name: str) -> pd.Series:
    values = _finite_target(target, name)
    if not np.isin(values.to_numpy(), [0, 1]).all():
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return values.astype(int)

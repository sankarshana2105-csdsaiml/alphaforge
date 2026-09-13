"""Fresh, fixed model pipelines shared by holdout and walk-forward evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classification_model_factories() -> tuple[tuple[str, Any], ...]:
    return (
        (
            "logistic_regression",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1_000, random_state=0))]
            ),
        ),
        (
            "random_forest_classifier",
            lambda: Pipeline(
                [("model", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=0, n_jobs=1))]
            ),
        ),
        (
            "hist_gradient_boosting_classifier",
            lambda: Pipeline(
                [("model", HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.05, random_state=0))]
            ),
        ),
    )


def regression_model_factories() -> tuple[tuple[str, Any], ...]:
    return (
        (
            "linear_regression",
            lambda: Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        ),
        (
            "random_forest_regressor",
            lambda: Pipeline(
                [("model", RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=0, n_jobs=1))]
            ),
        ),
        (
            "hist_gradient_boosting_regressor",
            lambda: Pipeline(
                [("model", HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.05, random_state=0))]
            ),
        ),
    )

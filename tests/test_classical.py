from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.baselines import LeakageError
from alphaforge.classical import _suspicious_warnings, evaluate_classical_models
from alphaforge.dataset import ResearchDataset, build_dataset


@pytest.fixture
def dataset(ohlcv: pd.DataFrame) -> ResearchDataset:
    return build_dataset(ohlcv)


def _result(report, name: str):
    return next(result for result in (*report.classification, *report.regression) if result.model_name == name)


def test_classical_models_run_and_keep_all_baselines(dataset: ResearchDataset) -> None:
    report = evaluate_classical_models(dataset)
    assert {result.model_name for result in report.classification} == {
        "majority_class",
        "naive_recent_return_direction",
        "logistic_regression",
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
    }
    assert {result.model_name for result in report.regression} == {
        "historical_mean",
        "zero_return",
        "linear_regression",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
    }
    assert report.comparison("classification").shape[0] == 5
    assert report.comparison("regression").shape[0] == 5


def test_models_are_deterministic_and_use_train_partition(dataset: ResearchDataset) -> None:
    first = evaluate_classical_models(dataset)
    second = evaluate_classical_models(dataset)
    for name in (
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
    ):
        pd.testing.assert_series_equal(_result(first, name).predictions, _result(second, name).predictions)
        result = _result(first, name)
        assert result.sample_counts == first.split.sample_counts
        assert result.feature_count == len(dataset.feature_columns)
        assert result.runtime_seconds is not None


def test_probability_and_prediction_sanity_are_reported(dataset: ResearchDataset) -> None:
    report = evaluate_classical_models(dataset)
    for name in ("random_forest_classifier", "hist_gradient_boosting_classifier"):
        result = _result(report, name)
        assert result.scores is not None
        assert result.scores.between(0, 1).all()
        sanity = report.diagnostics["model_sanity"][name]
        assert 0 <= sanity["probability_min"] <= sanity["probability_max"] <= 1
        assert isinstance(sanity["constant_prediction"], bool)


def test_suspicious_result_warning_is_fixed_and_visible() -> None:
    warning = _suspicious_warnings("classification", {"accuracy": 0.99, "roc_auc": 0.98})
    assert warning and "audit leakage" in warning[0]
    warning = _suspicious_warnings("regression", {"r2": 0.99})
    assert warning and "audit leakage" in warning[0]


def test_phase_two_leakage_guard_is_reused(dataset: ResearchDataset) -> None:
    leaked = ResearchDataset(
        dataset.frame,
        (*dataset.feature_columns, dataset.target_columns[0]),
        dataset.target_columns,
    )
    with pytest.raises(LeakageError):
        evaluate_classical_models(leaked)


def test_single_class_training_is_handled(ohlcv: pd.DataFrame) -> None:
    constant = ohlcv.copy()
    close = 100 + np.arange(len(constant), dtype=float)
    constant.loc[:, "open"] = close - 0.1
    constant.loc[:, "high"] = close + 1.0
    constant.loc[:, "low"] = close - 1.0
    constant.loc[:, "close"] = close
    report = evaluate_classical_models(build_dataset(constant))
    for name in ("random_forest_classifier", "hist_gradient_boosting_classifier"):
        result = _result(report, name)
        assert result.status == "single_class_train_fallback"
        assert result.metrics["roc_auc"] is None
        assert result.warnings

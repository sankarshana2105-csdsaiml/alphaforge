from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.baselines import LeakageError, evaluate_baselines
from alphaforge.dataset import ResearchDataset, build_dataset
from alphaforge.metrics import classification_metrics, regression_metrics


@pytest.fixture
def dataset(ohlcv: pd.DataFrame) -> ResearchDataset:
    return build_dataset(ohlcv)


def _result(report, name: str):
    return next(result for result in (*report.classification, *report.regression) if result.model_name == name)


def test_baselines_return_complete_structured_results(dataset: ResearchDataset) -> None:
    report = evaluate_baselines(dataset)
    assert {item.model_name for item in report.classification} == {
        "majority_class", "naive_recent_return_direction", "logistic_regression"
    }
    assert {item.model_name for item in report.regression} == {
        "historical_mean", "zero_return", "linear_regression"
    }
    assert report.comparison("classification").shape[0] == 3
    assert report.comparison("regression").shape[0] == 3
    assert "train_end" in report.comparison("classification").columns
    assert "feature_correlation" in report.diagnostics
    assert "regression_residuals" in report.diagnostics


def test_majority_and_historical_mean_use_training_values_only(dataset: ResearchDataset) -> None:
    report = evaluate_baselines(dataset)
    majority = _result(report, "majority_class")
    mean = _result(report, "historical_mean")
    expected_majority = report.split.train[dataset.y_direction.name].mode().iloc[0]
    expected_mean = report.split.train[dataset.y_return.name].mean()
    assert majority.predictions.eq(expected_majority).all()
    assert mean.predictions.eq(expected_mean).all()


def test_scalers_are_fitted_only_on_train_rows(dataset: ResearchDataset) -> None:
    report = evaluate_baselines(dataset)
    logistic = _result(report, "logistic_regression")
    linear = _result(report, "linear_regression")
    expected = report.split.train.loc[:, dataset.feature_columns].mean().to_numpy()
    np.testing.assert_allclose(logistic.model.named_steps["scaler"].mean_, expected)
    np.testing.assert_allclose(linear.model.named_steps["scaler"].mean_, expected)
    assert not np.allclose(expected, report.split.test.loc[:, dataset.feature_columns].mean().to_numpy())


def test_leakage_columns_are_rejected(dataset: ResearchDataset) -> None:
    leaked = ResearchDataset(
        dataset.frame,
        (*dataset.feature_columns, dataset.target_columns[0]),
        dataset.target_columns,
    )
    with pytest.raises(LeakageError, match="Invalid feature"):
        evaluate_baselines(leaked)
    timestamp_leak = ResearchDataset(dataset.frame, ("timestamp",), dataset.target_columns)
    with pytest.raises(LeakageError, match="Invalid feature"):
        evaluate_baselines(timestamp_leak)


def test_metrics_have_fixed_values_and_single_class_auc_is_unavailable() -> None:
    classification = classification_metrics(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.4, 0.3])
    )
    assert classification["confusion_matrix"] == [[2, 0], [1, 1]]
    assert classification["roc_auc"] == pytest.approx(1.0)
    assert classification_metrics(np.array([1, 1]), np.array([1, 1]), np.array([0.8, 0.9]))["roc_auc"] is None
    regression = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 1.5]))
    assert regression["mae"] == pytest.approx(0.5)
    assert regression["rmse"] == pytest.approx(0.5)


def test_deterministic_results_and_nan_safety(dataset: ResearchDataset) -> None:
    first, second = evaluate_baselines(dataset), evaluate_baselines(dataset)
    pd.testing.assert_frame_equal(first.comparison("classification"), second.comparison("classification"))
    for invalid_value in (np.nan, np.inf):
        damaged = dataset.frame.copy()
        damaged.loc[damaged.index[0], dataset.feature_columns[0]] = invalid_value
        with pytest.raises(ValueError, match="NaN or infinite"):
            evaluate_baselines(ResearchDataset(damaged, dataset.feature_columns, dataset.target_columns))


def test_constant_target_uses_safe_classifier_fallback(ohlcv: pd.DataFrame) -> None:
    constant = ohlcv.copy()
    close = 100 + np.arange(len(constant), dtype=float)
    constant.loc[:, "open"] = close - 0.1
    constant.loc[:, "high"] = close + 1.0
    constant.loc[:, "low"] = close - 1.0
    constant.loc[:, "close"] = close
    dataset = build_dataset(constant)
    report = evaluate_baselines(dataset)
    logistic = _result(report, "logistic_regression")
    assert logistic.status == "single_class_train_fallback"
    assert logistic.metrics["roc_auc"] is None


def test_constant_regression_target_does_not_report_undefined_correlations() -> None:
    metrics = regression_metrics(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert metrics["r2"] is None
    assert metrics["pearson_correlation"] is None
    assert metrics["spearman_correlation"] is None

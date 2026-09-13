from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.baselines import LeakageError
from alphaforge.dataset import ResearchDataset, build_dataset
from alphaforge.walk_forward import (
    WalkForwardConfig,
    WalkForwardError,
    evaluate_walk_forward,
    stability_warnings,
    walk_forward_splits,
)


def test_expanding_boundaries_and_horizon_purge(ohlcv: pd.DataFrame) -> None:
    config = WalkForwardConfig("expanding", minimum_train_size=20, evaluation_size=5, step_size=5, horizon=3)
    folds = walk_forward_splits(ohlcv, config=config)
    first, second = folds[0], folds[1]
    assert first.train_position_start == 0
    assert first.train_position_end == 20
    assert first.evaluation_position_start == 23
    assert second.train_position_end == 25
    assert len(first.train) == 20 and len(second.train) == 25
    assert first.train["timestamp"].max() < first.evaluation["timestamp"].min()
    assert first.train_position_end + config.horizon == first.evaluation_position_start


def test_rolling_window_has_fixed_size_and_moves(ohlcv: pd.DataFrame) -> None:
    config = WalkForwardConfig(
        "rolling", minimum_train_size=15, evaluation_size=5, step_size=5, horizon=2, max_train_size=20
    )
    first, second, *_ = walk_forward_splits(ohlcv, config=config)
    assert len(first.train) == len(second.train) == 20
    assert second.train_position_start - first.train_position_start == 5
    assert second.evaluation_position_start - first.evaluation_position_start == 5


def test_fold_count_and_insufficient_history(ohlcv: pd.DataFrame) -> None:
    config = WalkForwardConfig("expanding", 20, 10, 10, 1)
    assert len(walk_forward_splits(ohlcv.iloc[:51], config=config)) == 3
    with pytest.raises(WalkForwardError, match="Insufficient history"):
        walk_forward_splits(ohlcv.iloc[:30], config=config)
    with pytest.raises(WalkForwardError, match="overlapping"):
        WalkForwardConfig(evaluation_size=5, step_size=4)


def test_adversarial_future_target_rows_are_purged(ohlcv: pd.DataFrame) -> None:
    horizon = 3
    frame = ohlcv.iloc[:40].copy()
    frame["target_source_position"] = np.arange(len(frame)) + horizon
    fold = walk_forward_splits(
        frame,
        config=WalkForwardConfig("expanding", 12, 4, 4, horizon),
    )[0]
    assert frame.iloc[fold.train_position_end]["target_source_position"] >= fold.evaluation_position_start
    assert fold.train["target_source_position"].max() < fold.evaluation_position_start

    dataset = build_dataset(ohlcv)
    leaked_frame = dataset.frame.copy()
    leaked_frame["future_signal"] = leaked_frame[dataset.target_columns[1]]
    leaked = ResearchDataset(
        leaked_frame,
        (*dataset.feature_columns, "future_signal"),
        dataset.target_columns,
    )
    with pytest.raises(LeakageError):
        evaluate_walk_forward(leaked, config=WalkForwardConfig("expanding", 12, 4, 4, 1))


@pytest.fixture
def report(ohlcv: pd.DataFrame):
    dataset = build_dataset(ohlcv)
    return dataset, evaluate_walk_forward(
        dataset,
        config=WalkForwardConfig("expanding", 12, 4, 4, 1),
    )


def test_all_models_run_and_oos_is_unique_and_chronological(report) -> None:
    dataset, result = report
    assert {item.model_name for item in result.results} == {
        "majority_class",
        "logistic_regression",
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
        "historical_mean",
        "zero_return",
        "linear_regression",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
    }
    assert len(result.results) == len(result.folds) * 9
    assert not result.oos_predictions.duplicated(["target_type", "model", "timestamp"]).any()
    assert result.oos_predictions.groupby(["target_type", "model"])["timestamp"].apply(lambda values: values.is_monotonic_increasing).all()
    assert set(result.oos_predictions.columns) >= {"timestamp", "target", "prediction", "model", "fold"}
    assert result.oos_predictions["timestamp"].isin(dataset.frame["timestamp"]).all()
    for fold in result.folds:
        fold_oos = result.oos_predictions.loc[result.oos_predictions["fold"] == fold.fold]
        assert set(fold_oos["timestamp"]).isdisjoint(fold.train["timestamp"])
        assert fold_oos["timestamp"].min() == fold.evaluation["timestamp"].min()


def test_each_fold_scaler_uses_only_its_training_rows(report) -> None:
    dataset, result = report
    for fold_result in result.results:
        if fold_result.model_name not in ("logistic_regression", "linear_regression"):
            continue
        if "scaler" not in fold_result.fitted_model.named_steps:
            continue
        fold = result.folds[fold_result.fold - 1]
        expected = fold.train.loc[:, dataset.feature_columns].mean().to_numpy()
        np.testing.assert_allclose(fold_result.fitted_model.named_steps["scaler"].mean_, expected)
        assert fold_result.train_end < fold_result.evaluation_start


def test_aggregation_and_baseline_comparison_math(report) -> None:
    _, result = report
    model_results = [
        item for item in result.results
        if item.model_name == "random_forest_regressor" and item.metrics["rmse"] is not None
    ]
    expected = np.mean([item.metrics["rmse"] for item in model_results])
    aggregate = result.aggregate_metrics.query(
        "model == 'random_forest_regressor' and metric == 'rmse'"
    ).iloc[0]
    assert aggregate["mean"] == pytest.approx(expected)
    assert aggregate["folds"] == len(result.folds)
    comparison = result.comparison.query("model == 'random_forest_regressor'").iloc[0]
    assert comparison["baseline"] == "historical_mean"
    assert 0 <= comparison["percent_folds_beating_baseline"] <= 100


def test_walk_forward_output_is_deterministic(ohlcv: pd.DataFrame) -> None:
    dataset = build_dataset(ohlcv)
    config = WalkForwardConfig("expanding", 20, 8, 8, 1)
    first = evaluate_walk_forward(dataset, config=config)
    second = evaluate_walk_forward(dataset, config=config)
    pd.testing.assert_frame_equal(first.oos_predictions, second.oos_predictions)
    pd.testing.assert_frame_equal(first.aggregate_metrics, second.aggregate_metrics)


def test_models_run_with_rolling_window(ohlcv: pd.DataFrame) -> None:
    dataset = build_dataset(ohlcv)
    report = evaluate_walk_forward(
        dataset,
        config=WalkForwardConfig("rolling", 15, 8, 8, 1, max_train_size=20),
    )
    assert report.results
    assert all(len(fold.train) == 20 for fold in report.folds)


def test_stability_warnings_cover_failure_modes() -> None:
    warnings = stability_warnings(
        np.array([0.40, 0.41]),
        np.array([0.50, 0.50]),
        higher_is_better=True,
        correlations=np.array([-0.2, 0.3]),
    )
    assert any("Too few" in warning for warning in warnings)
    assert any("fewer than half" in warning for warning in warnings)
    assert any("sign is unstable" in warning for warning in warnings)


def test_horizon_mismatch_fails(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(WalkForwardError, match="horizon must match"):
        evaluate_walk_forward(
            build_dataset(ohlcv),
            config=WalkForwardConfig("expanding", 12, 4, 4, 2),
        )

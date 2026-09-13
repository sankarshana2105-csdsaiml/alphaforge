from __future__ import annotations

import pandas as pd

from alphaforge.dataset import DatasetConfig, build_dataset
from alphaforge.features import FeatureConfig
from alphaforge.targets import create_targets


def test_target_alignment_and_horizon_boundary(ohlcv: pd.DataFrame) -> None:
    targets = create_targets(ohlcv["close"], horizon=3)
    expected = ohlcv.loc[3, "close"] / ohlcv.loc[0, "close"] - 1
    assert targets.loc[0, "future_return_3"] == expected
    assert targets.loc[0, "direction_3"] == float(expected > 0)
    assert targets["future_return_3"].iloc[-3:].isna().all()
    assert targets["direction_3"].iloc[-3:].isna().all()


def test_invalid_horizon_fails(ohlcv: pd.DataFrame) -> None:
    try:
        create_targets(ohlcv["close"], horizon=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("invalid horizon was accepted")


def test_dataset_keeps_features_and_targets_separate(ohlcv: pd.DataFrame) -> None:
    dataset = build_dataset(ohlcv, config=DatasetConfig(horizon=5))
    assert dataset.frame.isna().sum().sum() == 0
    assert set(dataset.feature_columns).isdisjoint(dataset.target_columns)
    assert all("future" not in name and "direction" not in name for name in dataset.feature_columns)
    pd.testing.assert_frame_equal(dataset.X, dataset.frame.loc[:, dataset.feature_columns])


def test_dataset_drop_boundaries_are_expected(ohlcv: pd.DataFrame) -> None:
    features = FeatureConfig(regime_window=10, volatility_window=5)
    dataset = build_dataset(
        ohlcv, config=DatasetConfig(horizon=4, features=features)
    )
    assert dataset.frame["timestamp"].max() == ohlcv["timestamp"].iloc[-5]
    assert dataset.frame["timestamp"].is_monotonic_increasing


def test_targets_change_but_past_features_do_not(ohlcv: pd.DataFrame) -> None:
    baseline = build_dataset(ohlcv, config=DatasetConfig(drop_incomplete=False))
    changed = ohlcv.copy()
    changed.loc[80:, ["open", "high", "low", "close"]] *= 2
    candidate = build_dataset(changed, config=DatasetConfig(drop_incomplete=False))
    pd.testing.assert_frame_equal(baseline.X.loc[:79], candidate.X.loc[:79])
    assert baseline.y_return.loc[79] != candidate.y_return.loc[79]


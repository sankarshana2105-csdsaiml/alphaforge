from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data import validate_ohlcv
from alphaforge.features import FeatureConfig, compute_features


def test_rolling_values_are_trailing_only(ohlcv: pd.DataFrame) -> None:
    clean = validate_ohlcv(ohlcv)
    features = compute_features(clean)
    expected = clean["close"].iloc[:20].mean()
    assert features.loc[19, "rolling_mean_20"] == pytest.approx(expected)
    assert pd.isna(features.loc[18, "rolling_mean_20"])


def test_future_mutation_cannot_change_past_features(ohlcv: pd.DataFrame) -> None:
    clean = validate_ohlcv(ohlcv)
    baseline = compute_features(clean)
    changed = clean.copy()
    changed.loc[90:, ["open", "high", "low", "close", "volume"]] *= 3
    candidate = compute_features(changed)
    pd.testing.assert_frame_equal(baseline.loc[:89], candidate.loc[:89])


def test_feature_output_is_deterministic(ohlcv: pd.DataFrame) -> None:
    clean = validate_ohlcv(ohlcv)
    pd.testing.assert_frame_equal(compute_features(clean), compute_features(clean.copy()))


def test_initial_nan_behavior_matches_windows(ohlcv: pd.DataFrame) -> None:
    features = compute_features(validate_ohlcv(ohlcv))
    assert features["return_10"].iloc[:10].isna().all()
    assert features["return_10"].iloc[10:].notna().all()
    assert features["volatility_regime"].iloc[:78].isna().all()


def test_invalid_feature_windows_fail() -> None:
    try:
        FeatureConfig(short_window=20, long_window=20)
    except ValueError as exc:
        assert "short_window" in str(exc)
    else:
        raise AssertionError("invalid window configuration was accepted")

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data import (
    DataValidationError,
    ValidationConfig,
    load_ohlcv_csv,
    validate_ohlcv,
)


def test_sorts_timestamps_and_normalizes_timezone(ohlcv: pd.DataFrame) -> None:
    result = validate_ohlcv(ohlcv.iloc[::-1])
    assert result["timestamp"].is_monotonic_increasing
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_csv_loads_valid_data(tmp_path, ohlcv: pd.DataFrame) -> None:
    path = tmp_path / "prices.csv"
    ohlcv.to_csv(path, index=False)
    result = load_ohlcv_csv(path)
    pd.testing.assert_frame_equal(result, validate_ohlcv(ohlcv))


@pytest.mark.parametrize(
    ("column", "value"),
    [("close", 0), ("open", -1), ("volume", -1), ("high", 1), ("low", 999)],
)
def test_rejects_invalid_ohlcv(ohlcv: pd.DataFrame, column: str, value: float) -> None:
    ohlcv.loc[5, column] = value
    with pytest.raises(DataValidationError):
        validate_ohlcv(ohlcv)


def test_duplicate_policy_is_explicit(ohlcv: pd.DataFrame) -> None:
    duplicated = pd.concat([ohlcv, ohlcv.iloc[[3]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_ohlcv(duplicated)
    result = validate_ohlcv(
        duplicated, config=ValidationConfig(duplicate_policy="last")
    )
    assert len(result) == len(ohlcv)


def test_missing_policy_is_explicit(ohlcv: pd.DataFrame) -> None:
    ohlcv.loc[3, "volume"] = None
    with pytest.raises(DataValidationError, match="missing"):
        validate_ohlcv(ohlcv)
    result = validate_ohlcv(ohlcv, config=ValidationConfig(missing_policy="drop"))
    assert len(result) == len(ohlcv) - 1


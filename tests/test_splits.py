from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.splits import (
    ChronologicalSplitConfig,
    SplitError,
    chronological_split,
)


def test_fraction_split_preserves_strict_time_order(ohlcv: pd.DataFrame) -> None:
    split = chronological_split(ohlcv, config=ChronologicalSplitConfig(0.6, 0.2))
    assert split.sample_counts == {"train": 72, "validation": 24, "test": 24}
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_date_boundaries_are_inclusive_then_forward(ohlcv: pd.DataFrame) -> None:
    split = chronological_split(
        ohlcv,
        config=ChronologicalSplitConfig(
            train_end=ohlcv["timestamp"].iloc[59],
            validation_end=ohlcv["timestamp"].iloc[89],
        ),
    )
    assert split.sample_counts == {"train": 60, "validation": 30, "test": 30}


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2)})])
def test_insufficient_history_fails(frame: pd.DataFrame) -> None:
    with pytest.raises(SplitError):
        chronological_split(frame)


def test_invalid_fraction_and_unsorted_timestamps_fail(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(SplitError):
        ChronologicalSplitConfig(0.8, 0.2)
    with pytest.raises(SplitError, match="monotonically"):
        chronological_split(ohlcv.iloc[::-1])

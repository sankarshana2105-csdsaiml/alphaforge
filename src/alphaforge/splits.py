"""Chronological dataset partitions for leakage-safe research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    """Raised when a time-ordered split cannot be formed safely."""


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    """Choose fraction-based partitions or two inclusive date boundaries."""

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    train_end: str | pd.Timestamp | None = None
    validation_end: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        uses_dates = self.train_end is not None or self.validation_end is not None
        if uses_dates and (self.train_end is None or self.validation_end is None):
            raise SplitError("train_end and validation_end must be supplied together")
        if not uses_dates and not (0 < self.train_fraction < 1):
            raise SplitError("train_fraction must be between zero and one")
        if not uses_dates and not (0 < self.validation_fraction < 1):
            raise SplitError("validation_fraction must be between zero and one")
        if not uses_dates and self.train_fraction + self.validation_fraction >= 1:
            raise SplitError("train and validation fractions must leave a test fraction")


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def boundaries(self) -> dict[str, str]:
        return {
            "train_end": self.train["timestamp"].iloc[-1].isoformat(),
            "validation_end": self.validation["timestamp"].iloc[-1].isoformat(),
            "test_end": self.test["timestamp"].iloc[-1].isoformat(),
        }


def chronological_split(
    frame: pd.DataFrame, *, config: ChronologicalSplitConfig | None = None
) -> ChronologicalSplit:
    """Partition sorted rows into non-overlapping past-to-future subsets."""

    cfg = config or ChronologicalSplitConfig()
    if "timestamp" not in frame:
        raise SplitError("A timestamp column is required for chronological splitting")
    if len(frame) < 3:
        raise SplitError("At least three rows are required for train/validation/test")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise SplitError("timestamps must be valid and monotonically increasing")
    if timestamps.duplicated().any():
        raise SplitError("timestamps must be unique")

    ordered = frame.copy()
    if cfg.train_end is None:
        train_size = int(len(ordered) * cfg.train_fraction)
        validation_size = int(len(ordered) * cfg.validation_fraction)
        train = ordered.iloc[:train_size]
        validation = ordered.iloc[train_size : train_size + validation_size]
        test = ordered.iloc[train_size + validation_size :]
    else:
        train_end = pd.to_datetime(cfg.train_end, utc=True, errors="raise")
        validation_end = pd.to_datetime(cfg.validation_end, utc=True, errors="raise")
        if train_end >= validation_end:
            raise SplitError("train_end must precede validation_end")
        train = ordered.loc[timestamps <= train_end]
        validation = ordered.loc[(timestamps > train_end) & (timestamps <= validation_end)]
        test = ordered.loc[timestamps > validation_end]

    if not len(train) or not len(validation) or not len(test):
        raise SplitError("Each chronological partition must contain at least one row")
    return ChronologicalSplit(train.copy(), validation.copy(), test.copy())


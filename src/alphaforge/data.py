"""OHLCV ingestion and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when market data violates the OHLCV contract."""


@dataclass(frozen=True)
class ValidationConfig:
    """Explicit policies for ambiguous raw-data conditions."""

    duplicate_policy: Literal["raise", "first", "last"] = "raise"
    missing_policy: Literal["raise", "drop"] = "raise"


def load_ohlcv_csv(
    path: str | Path, *, config: ValidationConfig | None = None
) -> pd.DataFrame:
    """Load a CSV and return validated, chronologically sorted OHLCV rows."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"OHLCV CSV does not exist: {source}")
    return validate_ohlcv(pd.read_csv(source), config=config)


def validate_ohlcv(
    frame: pd.DataFrame, *, config: ValidationConfig | None = None
) -> pd.DataFrame:
    """Validate and normalize an OHLCV frame without mutating the input."""

    policy = config or ValidationConfig()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("timestamp contains invalid values") from exc

    for column in (*PRICE_COLUMNS, "volume"):
        if not is_numeric_dtype(data[column]):
            try:
                data[column] = pd.to_numeric(data[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise DataValidationError(f"{column} must be numeric") from exc

    missing_rows = data.isna().any(axis=1)
    if missing_rows.any():
        if policy.missing_policy == "raise":
            raise DataValidationError(
                f"Found {int(missing_rows.sum())} rows with missing OHLCV values"
            )
        data = data.loc[~missing_rows].copy()

    duplicated = data["timestamp"].duplicated(keep=False)
    if duplicated.any():
        if policy.duplicate_policy == "raise":
            raise DataValidationError(
                f"Found {data.loc[duplicated, 'timestamp'].nunique()} duplicate timestamps"
            )
        data = data.drop_duplicates(
            subset="timestamp", keep=policy.duplicate_policy
        )

    if data.empty:
        raise DataValidationError("No valid OHLCV rows remain")
    if (data.loc[:, PRICE_COLUMNS] <= 0).any().any():
        raise DataValidationError("OHLC prices must be strictly positive")
    if (data["volume"] < 0).any():
        raise DataValidationError("volume must be non-negative")
    if (data["high"] < data[["open", "low", "close"]].max(axis=1)).any():
        raise DataValidationError("high must be at least open, low, and close")
    if (data["low"] > data[["open", "high", "close"]].min(axis=1)).any():
        raise DataValidationError("low must be at most open, high, and close")

    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


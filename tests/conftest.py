from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rows = 120
    close = 100 + np.arange(rows, dtype=float) * 0.2 + np.sin(np.arange(rows) / 4)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000 + np.arange(rows, dtype=float) * 10,
        }
    )


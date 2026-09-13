"""Forward target construction kept separate from all feature logic."""

from __future__ import annotations

import pandas as pd


def create_targets(close: pd.Series, *, horizon: int = 1) -> pd.DataFrame:
    """Create t-to-t+h return and direction, leaving terminal rows missing."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    future_close = close.shift(-horizon)
    future_return = future_close / close - 1
    direction = (future_return > 0).astype("Float64").where(future_close.notna())
    return pd.DataFrame(
        {f"future_return_{horizon}": future_return, f"direction_{horizon}": direction},
        index=close.index,
    )


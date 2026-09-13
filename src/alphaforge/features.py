"""Past-only, interpretable market feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    return_windows: tuple[int, ...] = (1, 5, 10)
    short_window: int = 5
    long_window: int = 20
    volatility_window: int = 20
    volume_window: int = 20
    regime_window: int = 60

    def __post_init__(self) -> None:
        values = (
            *self.return_windows,
            self.short_window,
            self.long_window,
            self.volatility_window,
            self.volume_window,
            self.regime_window,
        )
        if not self.return_windows or any(value < 1 for value in values):
            raise ValueError("All feature windows must be positive")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")


def compute_features(
    ohlcv: pd.DataFrame, *, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Compute features using only each row and observations available before it."""

    cfg = config or FeatureConfig()
    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    volume = ohlcv["volume"].astype(float)
    features: dict[str, pd.Series] = {}

    for window in cfg.return_windows:
        features[f"return_{window}"] = close.pct_change(window, fill_method=None)
    features["log_return_1"] = np.log(close / close.shift(1))

    short_ma = close.rolling(cfg.short_window, min_periods=cfg.short_window).mean()
    long_ma = close.rolling(cfg.long_window, min_periods=cfg.long_window).mean()
    features[f"momentum_{cfg.short_window}"] = close / close.shift(cfg.short_window) - 1
    features[f"close_to_ma_{cfg.short_window}"] = close / short_ma - 1
    features[f"ma_spread_{cfg.short_window}_{cfg.long_window}"] = short_ma / long_ma - 1

    returns = close.pct_change(fill_method=None)
    rolling_vol = returns.rolling(
        cfg.volatility_window, min_periods=cfg.volatility_window
    ).std()
    features[f"volatility_{cfg.volatility_window}"] = rolling_vol
    features["high_low_range"] = (high - low) / close
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    features[f"normalized_atr_{cfg.volatility_window}"] = (
        true_range.rolling(
            cfg.volatility_window, min_periods=cfg.volatility_window
        ).mean()
        / close
    )

    volume_mean = volume.rolling(cfg.volume_window, min_periods=cfg.volume_window).mean()
    features[f"volume_mean_{cfg.volume_window}"] = volume_mean
    features[f"volume_ratio_{cfg.volume_window}"] = volume / volume_mean
    features["volume_change_1"] = volume.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )

    rolling_mean = close.rolling(cfg.long_window, min_periods=cfg.long_window).mean()
    rolling_std = close.rolling(cfg.long_window, min_periods=cfg.long_window).std()
    features[f"rolling_mean_{cfg.long_window}"] = rolling_mean
    features[f"rolling_std_{cfg.long_window}"] = rolling_std
    features[f"rolling_zscore_{cfg.long_window}"] = (
        (close - rolling_mean) / rolling_std.replace(0, np.nan)
    )

    vol_reference = rolling_vol.rolling(
        cfg.regime_window, min_periods=cfg.regime_window
    ).median()
    features["volatility_regime"] = (rolling_vol > vol_reference).astype(float).where(
        vol_reference.notna()
    )
    features["trend_regime"] = (short_ma > long_ma).astype(float).where(long_ma.notna())

    return pd.DataFrame(features, index=ohlcv.index).replace([np.inf, -np.inf], np.nan)


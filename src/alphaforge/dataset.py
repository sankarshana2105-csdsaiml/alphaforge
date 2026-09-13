"""Assembly of validated raw data, features, and explicitly forward targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alphaforge.data import ValidationConfig, validate_ohlcv
from alphaforge.features import FeatureConfig, compute_features
from alphaforge.targets import create_targets


@dataclass(frozen=True)
class DatasetConfig:
    horizon: int = 1
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    drop_incomplete: bool = True


@dataclass(frozen=True)
class ResearchDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns]

    @property
    def y_return(self) -> pd.Series:
        return self.frame[self.target_columns[0]]

    @property
    def y_direction(self) -> pd.Series:
        return self.frame[self.target_columns[1]]


def build_dataset(
    raw: pd.DataFrame, *, config: DatasetConfig | None = None
) -> ResearchDataset:
    """Build a deterministic research table with no fitted preprocessing."""

    cfg = config or DatasetConfig()
    clean = validate_ohlcv(raw, config=cfg.validation)
    feature_frame = compute_features(clean, config=cfg.features)
    target_frame = create_targets(clean["close"], horizon=cfg.horizon)
    target_columns = tuple(target_frame.columns)
    feature_columns = tuple(feature_frame.columns)
    assembled = pd.concat([clean, feature_frame, target_frame], axis=1)
    if cfg.drop_incomplete:
        assembled = assembled.dropna(subset=[*feature_columns, *target_columns])
    assembled = assembled.reset_index(drop=True)
    return ResearchDataset(assembled, feature_columns, target_columns)


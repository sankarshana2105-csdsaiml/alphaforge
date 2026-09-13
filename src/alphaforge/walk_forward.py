"""Horizon-purged walk-forward evaluation and stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from alphaforge.baselines import _require_finite_frame, _validate_feature_columns
from alphaforge.dataset import ResearchDataset
from alphaforge.metrics import MetricValue, classification_metrics, regression_metrics
from alphaforge.models import classification_model_factories, regression_model_factories


class WalkForwardError(ValueError):
    """Raised when safe walk-forward folds cannot be formed."""


@dataclass(frozen=True)
class WalkForwardConfig:
    mode: Literal["expanding", "rolling"] = "expanding"
    minimum_train_size: int = 20
    evaluation_size: int = 5
    step_size: int = 5
    horizon: int = 1
    max_train_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("expanding", "rolling"):
            raise WalkForwardError("mode must be 'expanding' or 'rolling'")
        if min(self.minimum_train_size, self.evaluation_size, self.step_size, self.horizon) < 1:
            raise WalkForwardError("window sizes, step size, and horizon must be positive")
        if self.step_size < self.evaluation_size:
            raise WalkForwardError("step_size must prevent overlapping evaluation windows")
        if self.max_train_size is not None and self.max_train_size < self.minimum_train_size:
            raise WalkForwardError("max_train_size cannot be smaller than minimum_train_size")


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: pd.DataFrame
    evaluation: pd.DataFrame
    train_position_start: int
    train_position_end: int
    evaluation_position_start: int
    evaluation_position_end: int


@dataclass(frozen=True)
class FoldResult:
    fold: int
    model_name: str
    target_type: Literal["classification", "regression"]
    metrics: dict[str, MetricValue]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    train_samples: int
    evaluation_samples: int
    feature_count: int
    actual: pd.Series
    predictions: pd.Series
    scores: pd.Series | None
    fitted_model: Any = None
    runtime_seconds: float | None = None
    status: str = "ok"


@dataclass(frozen=True)
class WalkForwardReport:
    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]
    results: tuple[FoldResult, ...]
    aggregate_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame
    stability: pd.DataFrame
    comparison: pd.DataFrame
    warnings: dict[str, tuple[str, ...]]

    def fold_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row = {
                "fold": result.fold,
                "model": result.model_name,
                "target_type": result.target_type,
                "train_start": result.train_start,
                "train_end": result.train_end,
                "evaluation_start": result.evaluation_start,
                "evaluation_end": result.evaluation_end,
                "train_samples": result.train_samples,
                "evaluation_samples": result.evaluation_samples,
                "feature_count": result.feature_count,
            }
            row.update({key: value for key, value in result.metrics.items() if not isinstance(value, list)})
            rows.append(row)
        return pd.DataFrame(rows)


def walk_forward_splits(
    frame: pd.DataFrame, *, config: WalkForwardConfig | None = None
) -> tuple[WalkForwardFold, ...]:
    """Create non-overlapping evaluation folds with ``horizon`` rows purged."""

    cfg = config or WalkForwardConfig()
    if "timestamp" not in frame:
        raise WalkForwardError("A timestamp column is required")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise WalkForwardError("timestamps must be valid, unique, and increasing")

    train_window = (
        cfg.max_train_size or cfg.minimum_train_size
        if cfg.mode == "rolling"
        else cfg.minimum_train_size
    )
    evaluation_start = train_window + cfg.horizon
    if evaluation_start + cfg.evaluation_size > len(frame):
        raise WalkForwardError("Insufficient history for one complete walk-forward fold")

    folds: list[WalkForwardFold] = []
    while evaluation_start + cfg.evaluation_size <= len(frame):
        train_end = evaluation_start - cfg.horizon
        train_start = 0 if cfg.mode == "expanding" else train_end - train_window
        evaluation_end = evaluation_start + cfg.evaluation_size
        train = frame.iloc[train_start:train_end].copy()
        evaluation = frame.iloc[evaluation_start:evaluation_end].copy()
        if len(train) < cfg.minimum_train_size:
            raise WalkForwardError("A fold has fewer than minimum_train_size rows")
        if train["timestamp"].iloc[-1] >= evaluation["timestamp"].iloc[0]:
            raise WalkForwardError("Training and evaluation timestamps overlap")
        folds.append(
            WalkForwardFold(
                len(folds) + 1,
                train,
                evaluation,
                train_start,
                train_end,
                evaluation_start,
                evaluation_end,
            )
        )
        evaluation_start += cfg.step_size
    return tuple(folds)


def evaluate_walk_forward(
    dataset: ResearchDataset,
    *,
    config: WalkForwardConfig | None = None,
    model_names: tuple[str, ...] | None = None,
) -> WalkForwardReport:
    """Evaluate all Phase 2–3 model families independently in every future fold."""

    cfg = config or WalkForwardConfig()
    _validate_feature_columns(dataset)
    _validate_horizon(dataset, cfg.horizon)
    _require_finite_frame(dataset.X, "Features")
    selected = _selected_models(model_names)
    folds = walk_forward_splits(dataset.frame, config=cfg)
    results: list[FoldResult] = []
    for fold in folds:
        results.extend(_evaluate_fold(dataset, fold, selected))
    result_tuple = tuple(results)
    aggregates = _aggregate_results(result_tuple)
    oos = _combine_oos(result_tuple, dataset)
    stability, comparison, warnings = _stability_analysis(result_tuple)
    return WalkForwardReport(cfg, folds, result_tuple, aggregates, oos, stability, comparison, warnings)


def _evaluate_fold(
    dataset: ResearchDataset, fold: WalkForwardFold, selected: set[str] | None
) -> list[FoldResult]:
    X_train = fold.train.loc[:, dataset.feature_columns]
    X_evaluation = fold.evaluation.loc[:, dataset.feature_columns]
    y_class_train = fold.train[dataset.target_columns[1]].astype(int)
    y_class = fold.evaluation[dataset.target_columns[1]].astype(int)
    y_return_train = fold.train[dataset.target_columns[0]].astype(float)
    y_return = fold.evaluation[dataset.target_columns[0]].astype(float)
    results: list[FoldResult] = []

    majority = int(y_class_train.mode().iloc[0])
    majority_predictions = pd.Series(majority, index=X_evaluation.index, dtype=int)
    if selected is None or "majority_class" in selected:
        results.append(_fold_result(fold, "majority_class", "classification", y_class, majority_predictions, None, None, 0, 0.0))

    for name, factory in classification_model_factories():
        if selected is not None and name not in selected:
            continue
        status = "ok"
        model = factory()
        if y_class_train.nunique() < 2:
            model = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
            status = "single_class_train_fallback"
        started = perf_counter()
        model.fit(X_train, y_class_train)
        predictions = pd.Series(model.predict(X_evaluation), index=X_evaluation.index, dtype=int)
        probabilities = model.predict_proba(X_evaluation)
        classes = model.named_steps["model"].classes_
        scores = pd.Series(
            probabilities[:, int(np.flatnonzero(classes == 1)[0])]
            if 1 in classes
            else np.zeros(len(X_evaluation)),
            index=X_evaluation.index,
        )
        results.append(_fold_result(fold, name, "classification", y_class, predictions, scores, model, X_train.shape[1], perf_counter() - started, status))

    historical_mean = pd.Series(float(y_return_train.mean()), index=X_evaluation.index)
    zero_return = pd.Series(0.0, index=X_evaluation.index)
    if selected is None or "historical_mean" in selected:
        results.append(_fold_result(fold, "historical_mean", "regression", y_return, historical_mean, None, None, 0, 0.0))
    if selected is None or "zero_return" in selected:
        results.append(_fold_result(fold, "zero_return", "regression", y_return, zero_return, None, None, 0, 0.0))
    for name, factory in regression_model_factories():
        if selected is not None and name not in selected:
            continue
        model = factory()
        started = perf_counter()
        model.fit(X_train, y_return_train)
        predictions = pd.Series(model.predict(X_evaluation), index=X_evaluation.index)
        results.append(_fold_result(fold, name, "regression", y_return, predictions, None, model, X_train.shape[1], perf_counter() - started))
    return results


def _fold_result(
    fold: WalkForwardFold,
    name: str,
    target_type: Literal["classification", "regression"],
    actual: pd.Series,
    predictions: pd.Series,
    scores: pd.Series | None,
    model: Any,
    feature_count: int,
    runtime: float,
    status: str = "ok",
) -> FoldResult:
    metrics = (
        classification_metrics(actual.to_numpy(), predictions.to_numpy(), scores.to_numpy() if scores is not None else None)
        if target_type == "classification"
        else regression_metrics(actual.to_numpy(), predictions.to_numpy())
    )
    return FoldResult(
        fold.fold,
        name,
        target_type,
        metrics,
        fold.train["timestamp"].iloc[0],
        fold.train["timestamp"].iloc[-1],
        fold.evaluation["timestamp"].iloc[0],
        fold.evaluation["timestamp"].iloc[-1],
        len(fold.train),
        len(fold.evaluation),
        feature_count,
        actual,
        predictions,
        scores,
        model,
        runtime,
        status,
    )


def _aggregate_results(results: tuple[FoldResult, ...]) -> pd.DataFrame:
    important = {
        "classification": ("balanced_accuracy", "f1", "roc_auc"),
        "regression": ("mae", "rmse", "r2", "pearson_correlation", "spearman_correlation"),
    }
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[FoldResult]] = {}
    for result in results:
        groups.setdefault((result.model_name, result.target_type), []).append(result)
    for (model, target_type), model_results in groups.items():
        for metric in important[target_type]:
            values = np.array(
                [result.metrics[metric] for result in model_results if result.metrics.get(metric) is not None],
                dtype=float,
            )
            rows.append(
                {
                    "model": model,
                    "target_type": target_type,
                    "metric": metric,
                    "mean": float(values.mean()) if values.size else None,
                    "median": float(np.median(values)) if values.size else None,
                    "std": float(values.std(ddof=0)) if values.size else None,
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                    "folds": len(model_results),
                    "valid_folds": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _combine_oos(results: tuple[FoldResult, ...], dataset: ResearchDataset) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for result in results:
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": dataset.frame.loc[result.predictions.index, "timestamp"],
                    "target": result.actual.astype(float),
                    "prediction": result.predictions.astype(float),
                    "score": result.scores.astype(float) if result.scores is not None else np.nan,
                    "model": result.model_name,
                    "target_type": result.target_type,
                    "fold": result.fold,
                }
            )
        )
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["target_type", "model", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if combined.duplicated(["target_type", "model", "timestamp"]).any():
        raise WalkForwardError("Duplicate OOS predictions detected")
    return combined


def _stability_analysis(
    results: tuple[FoldResult, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]]]:
    groups: dict[tuple[str, str], list[FoldResult]] = {}
    for result in results:
        groups.setdefault((result.model_name, result.target_type), []).append(result)
    rows: list[dict[str, Any]] = []
    warnings: dict[str, tuple[str, ...]] = {}
    for (model, target_type), model_results in groups.items():
        metric = "balanced_accuracy" if target_type == "classification" else "rmse"
        baseline = "majority_class" if target_type == "classification" else "historical_mean"
        higher = target_type == "classification"
        pairs = [
            (result, next(base for base in groups[(baseline, target_type)] if base.fold == result.fold))
            for result in model_results
            if result.metrics.get(metric) is not None
        ]
        values = np.array([float(result.metrics[metric]) for result, _ in pairs])
        baseline_values = np.array([float(base.metrics[metric]) for _, base in pairs])
        beats = values > baseline_values if higher else values < baseline_values
        best_index = int(np.argmax(values) if higher else np.argmin(values))
        worst_index = int(np.argmin(values) if higher else np.argmax(values))
        correlations = np.array(
            [float(result.metrics["pearson_correlation"]) for result in model_results if result.metrics.get("pearson_correlation") is not None]
        ) if target_type == "regression" else np.array([])
        if not values.size:
            model_warnings = ("No folds have a valid primary stability metric.",)
            warnings[f"{target_type}:{model}"] = model_warnings
            rows.append(
                {
                    "model": model,
                    "target_type": target_type,
                    "baseline": baseline,
                    "metric": metric,
                    "mean": None,
                    "median": None,
                    "dispersion": None,
                    "percent_folds_beating_baseline": None,
                    "best_fold": None,
                    "worst_fold": None,
                    "trend_slope": None,
                    "folds": len(model_results),
                    "warnings": model_warnings[0],
                }
            )
            continue
        model_warnings = stability_warnings(values, baseline_values, higher_is_better=higher, correlations=correlations)
        warnings[f"{target_type}:{model}"] = model_warnings
        rows.append(
            {
                "model": model,
                "target_type": target_type,
                "baseline": baseline,
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "dispersion": float(values.std(ddof=0)),
                "percent_folds_beating_baseline": float(beats.mean() * 100),
                "best_fold": pairs[best_index][0].fold,
                "worst_fold": pairs[worst_index][0].fold,
                "trend_slope": float(np.polyfit(np.arange(values.size), values, 1)[0]) if values.size > 1 else 0.0,
                "folds": len(model_results),
                "warnings": "; ".join(model_warnings),
            }
        )
    stability = pd.DataFrame(rows)
    comparison = stability.loc[:, [
        "model", "target_type", "baseline", "metric", "mean", "median",
        "dispersion", "percent_folds_beating_baseline", "folds", "warnings",
    ]].copy()
    return stability, comparison, warnings


def stability_warnings(
    values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    higher_is_better: bool,
    correlations: np.ndarray | None = None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if values.size < 3:
        notes.append("Too few folds for a reliable stability conclusion.")
    if values.size >= 3:
        oriented = values if higher_is_better else -values
        if oriented.max() - np.median(oriented) > 2 * max(oriented.std(ddof=0), 1e-12):
            notes.append("One exceptional fold may dominate performance.")
        relative_dispersion = values.std(ddof=0) / max(abs(values.mean()), 1e-12)
        if relative_dispersion > 0.5:
            notes.append("Fold performance has high variance.")
    beats = values > baseline_values if higher_is_better else values < baseline_values
    if beats.mean() < 0.5:
        notes.append("Model beats its baseline in fewer than half of valid folds.")
    mean_beats = values.mean() > baseline_values.mean() if higher_is_better else values.mean() < baseline_values.mean()
    median_beats = np.median(values) > np.median(baseline_values) if higher_is_better else np.median(values) < np.median(baseline_values)
    if mean_beats and not median_beats:
        notes.append("Mean beats baseline while median does not.")
    if correlations is not None and (correlations < 0).any() and (correlations > 0).any():
        notes.append("Correlation sign is unstable across folds.")
    return tuple(notes)


def _validate_horizon(dataset: ResearchDataset, horizon: int) -> None:
    try:
        target_horizons = {int(name.rsplit("_", 1)[1]) for name in dataset.target_columns}
    except (ValueError, IndexError) as exc:
        raise WalkForwardError("Target names must encode their prediction horizon") from exc
    if target_horizons != {horizon}:
        raise WalkForwardError("Walk-forward horizon must match the dataset target horizon")


def _selected_models(model_names: tuple[str, ...] | None) -> set[str] | None:
    if model_names is None:
        return None
    classifiers = {"majority_class", *(name for name, _ in classification_model_factories())}
    regressors = {"historical_mean", "zero_return", *(name for name, _ in regression_model_factories())}
    selected = set(model_names)
    unknown = selected - classifiers - regressors
    if not selected or unknown:
        raise WalkForwardError(f"Unknown or empty model selection: {sorted(unknown)}")
    if selected.intersection(classifiers):
        selected.add("majority_class")
    if selected.intersection(regressors):
        selected.add("historical_mean")
    return selected

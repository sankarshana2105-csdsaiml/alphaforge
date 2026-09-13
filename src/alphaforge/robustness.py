"""Phase 5 robustness, interpretability, and research stress tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from alphaforge.baselines import _validate_feature_columns
from alphaforge.dataset import DatasetConfig, ResearchDataset, build_dataset
from alphaforge.metrics import classification_metrics, regression_metrics
from alphaforge.walk_forward import (
    WalkForwardConfig,
    WalkForwardReport,
    evaluate_walk_forward,
)

Verdict = Literal[
    "ROBUST",
    "PROMISING BUT UNSTABLE",
    "WEAK SIGNAL",
    "NO PERSISTENT SIGNAL",
    "SUSPICIOUS / REQUIRES INVESTIGATION",
]


@dataclass(frozen=True)
class AblationReport:
    results: pd.DataFrame
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessEvidence:
    percent_folds_beating_baseline: float
    median_edge: float
    high_variance: bool
    horizon_consistent: bool
    window_consistent: bool
    regime_consistent: bool
    top_feature_share: float
    train_oos_gap: float
    suspicious_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobustnessVerdict:
    verdict: Verdict
    reasons: tuple[str, ...]


def permutation_feature_importance(
    dataset: ResearchDataset,
    report: WalkForwardReport,
    *,
    model_name: str,
    target_type: Literal["classification", "regression"],
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Aggregate permutation importance measured only on unseen fold rows."""

    _validate_feature_columns(dataset)
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    scoring = "accuracy" if target_type == "classification" else "neg_mean_absolute_error"
    fold_importances: list[np.ndarray] = []
    for result in report.results:
        if result.model_name != model_name or result.target_type != target_type:
            continue
        if result.fitted_model is None:
            raise ValueError("Permutation importance requires a fitted model")
        fold = report.folds[result.fold - 1]
        measured = permutation_importance(
            result.fitted_model,
            fold.evaluation.loc[:, dataset.feature_columns],
            result.actual,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        fold_importances.append(measured.importances)
    if not fold_importances:
        raise ValueError(f"No fold results found for {model_name}")
    values = np.concatenate(fold_importances, axis=1)
    output = pd.DataFrame(
        {
            "feature": dataset.feature_columns,
            "importance": values.mean(axis=1),
            "dispersion": values.std(axis=1, ddof=0),
            "model": model_name,
            "target_type": target_type,
        }
    ).sort_values(["importance", "feature"], ascending=[False, True], kind="stable")
    output["rank"] = np.arange(1, len(output) + 1)
    return output.reset_index(drop=True)


def native_feature_importance(
    dataset: ResearchDataset,
    report: WalkForwardReport,
    *,
    model_name: str,
    target_type: Literal["classification", "regression"],
) -> pd.DataFrame:
    """Aggregate native tree importance or absolute linear coefficients when valid."""

    _validate_feature_columns(dataset)
    values: list[np.ndarray] = []
    for result in report.results:
        if result.model_name != model_name or result.target_type != target_type or result.fitted_model is None:
            continue
        estimator = result.fitted_model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values.append(np.asarray(estimator.feature_importances_, dtype=float))
        elif hasattr(estimator, "coef_"):
            values.append(np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1))
    if not values:
        raise ValueError(f"Model {model_name} has no supported native importance")
    stacked = np.vstack(values)
    output = pd.DataFrame(
        {
            "feature": dataset.feature_columns,
            "importance": stacked.mean(axis=0),
            "dispersion": stacked.std(axis=0, ddof=0),
            "model": model_name,
            "target_type": target_type,
        }
    ).sort_values(["importance", "feature"], ascending=[False, True], kind="stable")
    output["rank"] = np.arange(1, len(output) + 1)
    return output.reset_index(drop=True)


def feature_ablation_configurations(
    feature_columns: tuple[str, ...], *, top_feature: str
) -> dict[str, tuple[str, ...]]:
    if top_feature not in feature_columns:
        raise ValueError("top_feature must be a Phase 1 feature")
    groups = _feature_groups(feature_columns)
    top_group = next((name for name, columns in groups.items() if top_feature in columns), None)
    configurations = {"all_features": feature_columns, **groups}
    configurations["remove_top_feature"] = tuple(
        name for name in feature_columns if name != top_feature
    )
    configurations["remove_top_feature_group"] = tuple(
        name for name in feature_columns if top_group is None or name not in groups[top_group]
    )
    return {name: columns for name, columns in configurations.items() if columns}


def evaluate_feature_ablation(
    dataset: ResearchDataset,
    *,
    config: WalkForwardConfig,
    model_name: str,
    target_type: Literal["classification", "regression"],
    top_feature: str,
) -> AblationReport:
    rows: list[dict[str, Any]] = []
    configurations = feature_ablation_configurations(
        dataset.feature_columns, top_feature=top_feature
    )
    for name, columns in configurations.items():
        subset = ResearchDataset(dataset.frame, columns, dataset.target_columns)
        report = evaluate_walk_forward(subset, config=config, model_names=(model_name,))
        row = report.comparison.loc[
            (report.comparison["model"] == model_name)
            & (report.comparison["target_type"] == target_type)
        ].iloc[0]
        rows.append(
            {
                "configuration": name,
                "model": model_name,
                "target_type": target_type,
                "metric": row["metric"],
                "mean": row["mean"],
                "median": row["median"],
                "dispersion": row["dispersion"],
                "percent_folds_beating_baseline": row["percent_folds_beating_baseline"],
                "feature_count": len(columns),
            }
        )
    results = pd.DataFrame(rows)
    all_mean = float(results.loc[results["configuration"] == "all_features", "mean"].iloc[0])
    higher = target_type == "classification"
    results["effect_vs_all"] = results["mean"].map(
        lambda value: _ablation_effect(float(value), all_mean, higher)
    )
    top_removed = results.loc[results["configuration"] == "remove_top_feature", "mean"]
    warnings: tuple[str, ...] = ()
    if not top_removed.empty:
        deterioration = all_mean - float(top_removed.iloc[0]) if higher else float(top_removed.iloc[0]) - all_mean
        if deterioration > (0.05 if higher else max(abs(all_mean) * 0.25, 1e-12)):
            warnings = ("Performance depends excessively on the top feature.",)
    return AblationReport(results, warnings)


def evaluate_horizon_sensitivity(
    raw_ohlcv: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    config: WalkForwardConfig,
    model_name: str,
    target_type: Literal["classification", "regression"],
) -> pd.DataFrame:
    if not horizons or len(set(horizons)) != len(horizons) or any(value < 1 for value in horizons):
        raise ValueError("horizons must be unique positive values")
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        dataset = build_dataset(raw_ohlcv, config=DatasetConfig(horizon=horizon))
        report = evaluate_walk_forward(
            dataset,
            config=replace(config, horizon=horizon),
            model_names=(model_name,),
        )
        row = report.comparison.loc[
            (report.comparison["model"] == model_name)
            & (report.comparison["target_type"] == target_type)
        ].iloc[0]
        rows.append({"horizon": horizon, **row.to_dict()})
    output = pd.DataFrame(rows)
    output["signal_state"] = np.where(
        output["percent_folds_beating_baseline"] >= 50,
        "usually_beats_baseline",
        "does_not_persist",
    )
    output["signal_direction_changes"] = output["signal_state"].nunique() > 1
    return output


def evaluate_window_sensitivity(
    dataset: ResearchDataset,
    *,
    configurations: dict[str, WalkForwardConfig],
    model_name: str,
    target_type: Literal["classification", "regression"],
) -> pd.DataFrame:
    if not configurations:
        raise ValueError("At least one window configuration is required")
    rows: list[dict[str, Any]] = []
    for name, config in configurations.items():
        report = evaluate_walk_forward(dataset, config=config, model_names=(model_name,))
        row = report.comparison.loc[
            (report.comparison["model"] == model_name)
            & (report.comparison["target_type"] == target_type)
        ].iloc[0]
        rows.append(
            {
                "configuration": name,
                "mode": config.mode,
                "minimum_train_size": config.minimum_train_size,
                "evaluation_size": config.evaluation_size,
                **row.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def regime_performance(
    dataset: ResearchDataset,
    report: WalkForwardReport,
    *,
    model_name: str,
    target_type: Literal["classification", "regression"],
    minimum_samples: int = 20,
) -> pd.DataFrame:
    baseline = "majority_class" if target_type == "classification" else "historical_mean"
    oos = report.oos_predictions
    model = oos.loc[(oos["model"] == model_name) & (oos["target_type"] == target_type)]
    base = oos.loc[(oos["model"] == baseline) & (oos["target_type"] == target_type)]
    paired = model.merge(base, on=["timestamp", "fold", "target_type"], suffixes=("_model", "_baseline"))
    regimes = dataset.frame.loc[:, ["timestamp", "volatility_regime", "trend_regime"]]
    paired = paired.merge(regimes, on="timestamp", validate="one_to_one")
    labels = {
        ("volatility_regime", 0): "low_volatility",
        ("volatility_regime", 1): "high_volatility",
        ("trend_regime", 0): "non_trending",
        ("trend_regime", 1): "trending",
    }
    rows: list[dict[str, Any]] = []
    for (column, value), label in labels.items():
        subset = paired.loc[paired[column] == value]
        if subset.empty:
            continue
        if target_type == "classification":
            model_metric = classification_metrics(
                subset["target_model"].to_numpy(), subset["prediction_model"].to_numpy()
            )["balanced_accuracy"]
            baseline_metric = classification_metrics(
                subset["target_model"].to_numpy(), subset["prediction_baseline"].to_numpy()
            )["balanced_accuracy"]
            difference = None if model_metric is None or baseline_metric is None else float(model_metric - baseline_metric)
        else:
            model_metric = regression_metrics(
                subset["target_model"].to_numpy(), subset["prediction_model"].to_numpy()
            )["rmse"]
            baseline_metric = regression_metrics(
                subset["target_model"].to_numpy(), subset["prediction_baseline"].to_numpy()
            )["rmse"]
            difference = float(baseline_metric - model_metric)
        rows.append(
            {
                "regime": label,
                "sample_count": len(subset),
                "model_metric": model_metric,
                "baseline_metric": baseline_metric,
                "difference_vs_baseline": difference,
                "warning": "Too few samples for a reliable regime conclusion." if len(subset) < minimum_samples else "",
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    report: WalkForwardReport,
    *,
    model_name: str,
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60),
) -> pd.DataFrame:
    if any(value < 0.5 or value >= 1 for value in thresholds):
        raise ValueError("thresholds must be in [0.5, 1.0)")
    rows = report.oos_predictions.loc[
        (report.oos_predictions["model"] == model_name)
        & (report.oos_predictions["target_type"] == "classification")
    ]
    if rows.empty or rows["score"].isna().all():
        raise ValueError("Classification probabilities are required")
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = rows.loc[(rows["score"] >= threshold) | (rows["score"] <= 1 - threshold)].copy()
        predicted = (selected["score"] >= 0.5).astype(int)
        metrics = (
            classification_metrics(selected["target"].to_numpy(), predicted.to_numpy())
            if len(selected)
            else {"precision": None, "recall": None, "balanced_accuracy": None}
        )
        output.append(
            {
                "threshold": threshold,
                "predictions_acted_on": len(selected),
                "coverage": float(len(selected) / len(rows)),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "signal_frequency": float(predicted.sum() / len(rows)),
            }
        )
    return pd.DataFrame(output)


def transaction_cost_stress(
    dataset: ResearchDataset,
    report: WalkForwardReport,
    *,
    model_name: str,
    costs_bps: tuple[float, ...] = (0.0, 5.0, 10.0),
) -> pd.DataFrame:
    if any(cost < 0 for cost in costs_bps):
        raise ValueError("Transaction costs cannot be negative")
    rows = report.oos_predictions.loc[
        (report.oos_predictions["model"] == model_name)
        & (report.oos_predictions["target_type"] == "classification")
    ].sort_values("timestamp")
    if rows.empty:
        raise ValueError("Directional OOS predictions are required")
    positions = rows["prediction"].map({0.0: -1.0, 1.0: 1.0})
    if positions.isna().any():
        raise ValueError("Directional predictions must be binary")
    changes = positions.diff().abs()
    turnover = float(positions.iloc[0].__abs__() + changes.iloc[1:].sum())
    position_changes = int(changes.iloc[1:].fillna(0).gt(0).sum())
    realized = rows[["timestamp"]].merge(
        dataset.frame[["timestamp", dataset.target_columns[0]]],
        on="timestamp",
        validate="one_to_one",
    )[dataset.target_columns[0]].to_numpy()
    gross = float((positions.to_numpy() * realized).sum())
    return pd.DataFrame(
        {
            "model": model_name,
            "gross_result": gross,
            "cost_bps": costs_bps,
            "net_result": [gross - turnover * cost / 10_000 for cost in costs_bps],
            "turnover": turnover,
            "position_changes": position_changes,
            "observations": len(rows),
        }
    )


def training_oos_gap(dataset: ResearchDataset, report: WalkForwardReport) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in report.results:
        if result.fitted_model is None:
            continue
        fold = report.folds[result.fold - 1]
        X_train = fold.train.loc[:, dataset.feature_columns]
        if result.target_type == "classification":
            actual = fold.train[dataset.target_columns[1]].astype(int).to_numpy()
            train_prediction = result.fitted_model.predict(X_train)
            train_metric = float((train_prediction == actual).mean())
            oos_metric = float(result.metrics["accuracy"])
            metric = "accuracy"
            gap = train_metric - oos_metric
        else:
            actual = fold.train[dataset.target_columns[0]].astype(float).to_numpy()
            train_prediction = result.fitted_model.predict(X_train)
            train_rmse = float(np.sqrt(np.mean((actual - train_prediction) ** 2)))
            oos_rmse = float(result.metrics["rmse"])
            train_metric, oos_metric, metric = train_rmse, oos_rmse, "rmse"
            gap = oos_rmse - train_rmse
        rows.append(
            {
                "fold": result.fold,
                "model": result.model_name,
                "target_type": result.target_type,
                "metric": metric,
                "train_metric": train_metric,
                "oos_metric": oos_metric,
                "overfit_gap": gap,
            }
        )
    return pd.DataFrame(rows)


def suspicious_performance_checks(
    *,
    classification_accuracy: float | None = None,
    correlation: float | None = None,
    feature_importance: pd.DataFrame | None = None,
    train_oos_gap: float | None = None,
    model_metric: float | None = None,
    baseline_metric: float | None = None,
    phase4_warnings: tuple[str, ...] = (),
    fold_values: np.ndarray | None = None,
    regime_results: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    warnings = list(phase4_warnings)
    if classification_accuracy is not None and classification_accuracy >= 0.90:
        warnings.append("Unrealistically high classification accuracy requires a leakage audit.")
    if correlation is not None and abs(correlation) >= 0.90:
        warnings.append("Unusually high return correlation requires investigation.")
    if feature_importance is not None and not feature_importance.empty:
        positive = feature_importance["importance"].clip(lower=0)
        if positive.sum() > 0 and positive.max() / positive.sum() >= 0.90:
            warnings.append("Near-perfect feature-importance concentration detected.")
    if train_oos_gap is not None and train_oos_gap >= 0.20:
        warnings.append("Strong training performance does not survive out of sample.")
    if model_metric is not None and baseline_metric is not None and model_metric - baseline_metric >= 0.20:
        warnings.append("Model drastically outperforms its baseline; verify leakage and data quality.")
    if fold_values is not None and fold_values.size >= 3:
        if fold_values.max() - np.median(fold_values) > 2 * max(fold_values.std(ddof=0), 1e-12):
            warnings.append("One exceptional fold dominates the aggregate result.")
    if regime_results is not None and len(regime_results) >= 2:
        positive = regime_results["difference_vs_baseline"].fillna(0).gt(0)
        if positive.sum() == 1:
            warnings.append("Performance appears only in one descriptive regime.")
    return tuple(dict.fromkeys(warnings))


def robustness_verdict(evidence: RobustnessEvidence) -> RobustnessVerdict:
    if evidence.suspicious_warnings or evidence.top_feature_share >= 0.90 or evidence.train_oos_gap >= 0.30:
        return RobustnessVerdict("SUSPICIOUS / REQUIRES INVESTIGATION", ("Automated audit triggers remain unresolved.",))
    stable = not evidence.high_variance and evidence.horizon_consistent and evidence.window_consistent and evidence.regime_consistent
    if evidence.percent_folds_beating_baseline >= 70 and evidence.median_edge > 0 and stable and evidence.top_feature_share < 0.50 and evidence.train_oos_gap < 0.10:
        return RobustnessVerdict("ROBUST", ("Performance is broadly stable and not concentrated in one feature.",))
    if evidence.percent_folds_beating_baseline >= 50 and evidence.median_edge > 0:
        return RobustnessVerdict("PROMISING BUT UNSTABLE", ("Median performance is positive, but robustness conditions are incomplete.",))
    if evidence.percent_folds_beating_baseline >= 25 or evidence.median_edge > 0:
        return RobustnessVerdict("WEAK SIGNAL", ("Evidence is inconsistent or only marginally better than baseline.",))
    return RobustnessVerdict("NO PERSISTENT SIGNAL", ("Performance does not beat baseline consistently across folds.",))


def _feature_groups(feature_columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "returns_only": tuple(name for name in feature_columns if name.startswith("return_") or name.startswith("log_return")),
        "momentum_only": tuple(name for name in feature_columns if name.startswith(("momentum_", "close_to_ma_", "ma_spread_", "rolling_mean_", "rolling_zscore_"))),
        "volatility_only": tuple(name for name in feature_columns if (name.startswith(("volatility_", "normalized_atr_", "rolling_std_")) or name == "high_low_range") and not name.endswith("_regime")),
        "volume_only": tuple(name for name in feature_columns if name.startswith("volume_")),
        "regime_only": tuple(name for name in feature_columns if name.endswith("_regime")),
    }


def _ablation_effect(value: float, all_value: float, higher_is_better: bool) -> str:
    change = value - all_value if higher_is_better else all_value - value
    if change > 0.02:
        return "improves"
    if change < -0.05:
        return "collapses"
    return "stable"

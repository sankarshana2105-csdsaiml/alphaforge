"""Fixed classical ML models compared with every Phase 2 baseline."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from alphaforge.baselines import BaselineReport, BaselineResult, evaluate_baselines
from alphaforge.dataset import ResearchDataset
from alphaforge.diagnostics import prediction_distribution_summary, residual_summary
from alphaforge.metrics import classification_metrics, regression_metrics
from alphaforge.models import classification_model_factories, regression_model_factories
from alphaforge.splits import ChronologicalSplitConfig


def evaluate_classical_models(
    dataset: ResearchDataset, *, split_config: ChronologicalSplitConfig | None = None
) -> BaselineReport:
    """Fit deterministic tree models on train rows and score the untouched test rows."""

    baseline_report = evaluate_baselines(dataset, split_config=split_config)
    split = baseline_report.split
    X_train = split.train.loc[:, dataset.feature_columns]
    X_test = split.test.loc[:, dataset.feature_columns]
    y_class_train = split.train[dataset.target_columns[1]].astype(int)
    y_class_test = split.test[dataset.target_columns[1]].astype(int)
    y_reg_train = split.train[dataset.target_columns[0]].astype(float)
    y_reg_test = split.test[dataset.target_columns[0]].astype(float)

    classifiers = classification_model_factories()[1:]
    regressors = regression_model_factories()[1:]

    classification = list(baseline_report.classification)
    classification.extend(
        _fit_classifier(name, factory(), X_train, X_test, y_class_train, y_class_test, split)
        for name, factory in classifiers
    )
    regression = list(baseline_report.regression)
    regression.extend(
        _fit_regressor(name, factory(), X_train, X_test, y_reg_train, y_reg_test, split)
        for name, factory in regressors
    )
    classification = [_with_warnings(result) for result in classification]
    regression = [_with_warnings(result) for result in regression]

    diagnostics = dict(baseline_report.diagnostics)
    all_results = (*classification, *regression)
    diagnostics["prediction_distributions"] = {
        result.model_name: prediction_distribution_summary(result.predictions)
        for result in all_results
    }
    diagnostics["regression_residuals"] = {
        result.model_name: residual_summary(y_reg_test, result.predictions)
        for result in regression
    }
    diagnostics["model_sanity"] = {
        result.model_name: _sanity_summary(result) for result in all_results
    }
    return BaselineReport(split, tuple(classification), tuple(regression), diagnostics)


def _fit_classifier(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: Any,
) -> BaselineResult:
    status = "ok"
    if y_train.nunique() < 2:
        estimator = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
        status = "single_class_train_fallback"
    model = estimator
    started = perf_counter()
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=X_test.index, dtype=int)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["model"].classes_
    scores_array = (
        probabilities[:, int(np.flatnonzero(classes == 1)[0])]
        if 1 in classes
        else np.zeros(len(X_test), dtype=float)
    )
    runtime = perf_counter() - started
    scores = pd.Series(scores_array, index=X_test.index)
    metrics = classification_metrics(y_test.to_numpy(), predictions.to_numpy(), scores.to_numpy())
    warnings = _suspicious_warnings("classification", metrics)
    return BaselineResult(
        model_name=name,
        target_type="classification",
        metrics=metrics,
        predictions=predictions,
        scores=scores,
        sample_counts=split.sample_counts,
        split_boundaries=split.boundaries,
        feature_count=X_train.shape[1],
        status=status,
        model=model,
        runtime_seconds=runtime,
        warnings=warnings,
    )


def _fit_regressor(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: Any,
) -> BaselineResult:
    model = estimator
    started = perf_counter()
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=X_test.index)
    runtime = perf_counter() - started
    metrics = regression_metrics(y_test.to_numpy(), predictions.to_numpy())
    warnings = _suspicious_warnings("regression", metrics)
    return BaselineResult(
        model_name=name,
        target_type="regression",
        metrics=metrics,
        predictions=predictions,
        scores=None,
        sample_counts=split.sample_counts,
        split_boundaries=split.boundaries,
        feature_count=X_train.shape[1],
        model=model,
        runtime_seconds=runtime,
        warnings=warnings,
    )


def _suspicious_warnings(target_type: str, metrics: dict[str, Any]) -> tuple[str, ...]:
    if target_type == "classification":
        values = [metrics.get("accuracy"), metrics.get("roc_auc")]
        suspicious = any(value is not None and value >= 0.95 for value in values)
    else:
        values = [metrics.get("r2"), metrics.get("pearson_correlation"), metrics.get("spearman_correlation")]
        suspicious = any(value is not None and abs(value) >= 0.95 for value in values)
    return ("Suspiciously strong held-out metric; audit leakage before interpretation.",) if suspicious else ()


def _sanity_summary(result: BaselineResult) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "constant_prediction": result.predictions.nunique() <= 1,
        "warnings": list(result.warnings),
    }
    if result.scores is not None:
        summary["probability_min"] = float(result.scores.min())
        summary["probability_max"] = float(result.scores.max())
    return summary


def _with_warnings(result: BaselineResult) -> BaselineResult:
    warnings = _suspicious_warnings(result.target_type, result.metrics)
    return replace(result, warnings=warnings)

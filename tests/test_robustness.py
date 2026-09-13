from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.audit import render_leakage_audit
from alphaforge.baselines import LeakageError
from alphaforge.dataset import ResearchDataset, build_dataset
from alphaforge.robustness import (
    RobustnessEvidence,
    evaluate_feature_ablation,
    evaluate_horizon_sensitivity,
    evaluate_window_sensitivity,
    feature_ablation_configurations,
    native_feature_importance,
    permutation_feature_importance,
    regime_performance,
    robustness_verdict,
    suspicious_performance_checks,
    threshold_sensitivity,
    training_oos_gap,
    transaction_cost_stress,
)
from alphaforge.walk_forward import WalkForwardConfig, evaluate_walk_forward


@pytest.fixture
def robustness_report(ohlcv: pd.DataFrame):
    dataset = build_dataset(ohlcv)
    report = evaluate_walk_forward(
        dataset,
        config=WalkForwardConfig("expanding", 12, 8, 8, 1),
        model_names=("random_forest_classifier",),
    )
    return dataset, report


def test_permutation_and_native_importance_are_ranked_and_safe(robustness_report) -> None:
    dataset, report = robustness_report
    first = permutation_feature_importance(
        dataset, report, model_name="random_forest_classifier", target_type="classification", n_repeats=3
    )
    second = permutation_feature_importance(
        dataset, report, model_name="random_forest_classifier", target_type="classification", n_repeats=3
    )
    assert list(first.columns) == ["feature", "importance", "dispersion", "model", "target_type", "rank"]
    assert first["rank"].tolist() == list(range(1, len(dataset.feature_columns) + 1))
    assert first["importance"].is_monotonic_decreasing
    assert set(first["feature"]) == set(dataset.feature_columns)
    assert set(first["feature"]).isdisjoint(dataset.target_columns)
    pd.testing.assert_frame_equal(first, second)
    native = native_feature_importance(
        dataset, report, model_name="random_forest_classifier", target_type="classification"
    )
    assert len(native) == len(dataset.feature_columns)
    assert native["importance"].is_monotonic_decreasing


def test_ablation_configurations_and_evaluation(ohlcv: pd.DataFrame) -> None:
    dataset = build_dataset(ohlcv)
    configurations = feature_ablation_configurations(dataset.feature_columns, top_feature="return_1")
    assert set(configurations) == {
        "all_features", "returns_only", "momentum_only", "volatility_only",
        "volume_only", "regime_only", "remove_top_feature", "remove_top_feature_group",
    }
    report = evaluate_feature_ablation(
        dataset,
        config=WalkForwardConfig("expanding", 12, 8, 8, 1),
        model_name="logistic_regression",
        target_type="classification",
        top_feature="return_1",
    )
    assert set(report.results["configuration"]) == set(configurations)
    assert report.results["effect_vs_all"].isin(["stable", "improves", "collapses"]).all()


def test_horizon_and_window_sensitivity(ohlcv: pd.DataFrame) -> None:
    config = WalkForwardConfig("expanding", 12, 8, 8, 1)
    horizons = evaluate_horizon_sensitivity(
        ohlcv,
        horizons=(1, 5),
        config=config,
        model_name="logistic_regression",
        target_type="classification",
    )
    assert horizons["horizon"].tolist() == [1, 5]
    assert {"dispersion", "percent_folds_beating_baseline", "signal_state", "signal_direction_changes"}.issubset(horizons.columns)

    dataset = build_dataset(ohlcv)
    windows = evaluate_window_sensitivity(
        dataset,
        configurations={
            "expanding": config,
            "rolling": WalkForwardConfig("rolling", 12, 8, 8, 1, max_train_size=16),
        },
        model_name="linear_regression",
        target_type="regression",
    )
    assert set(windows["mode"]) == {"expanding", "rolling"}
    assert windows["folds"].gt(0).all()


def test_regime_threshold_and_cost_outputs(robustness_report) -> None:
    dataset, report = robustness_report
    regimes = regime_performance(
        dataset,
        report,
        model_name="random_forest_classifier",
        target_type="classification",
        minimum_samples=100,
    )
    assert regimes["regime"].str.contains("volatility|trending").all()
    assert regimes["sample_count"].gt(0).all()
    assert regimes["warning"].str.contains("Too few").all()
    model_rows = report.oos_predictions.query(
        "model == 'random_forest_classifier' and target_type == 'classification'"
    )
    assert regimes.loc[regimes["regime"].str.contains("volatility"), "sample_count"].sum() == len(model_rows)
    assert regimes.loc[regimes["regime"].str.contains("trending"), "sample_count"].sum() == len(model_rows)

    thresholds = threshold_sensitivity(report, model_name="random_forest_classifier")
    assert thresholds["threshold"].tolist() == [0.50, 0.55, 0.60]
    assert thresholds.iloc[0]["coverage"] == pytest.approx(1.0)
    assert thresholds["coverage"].is_monotonic_decreasing

    costs = transaction_cost_stress(
        dataset, report, model_name="random_forest_classifier", costs_bps=(0.0, 5.0, 10.0)
    )
    assert costs["net_result"].is_monotonic_decreasing
    assert costs.iloc[0]["net_result"] == pytest.approx(costs.iloc[0]["gross_result"])
    assert costs["turnover"].ge(costs["position_changes"]).all()
    expected_net = costs["gross_result"] - costs["turnover"] * costs["cost_bps"] / 10_000
    np.testing.assert_allclose(costs["net_result"], expected_net)


def test_train_oos_gap_and_suspicious_checks(robustness_report) -> None:
    dataset, report = robustness_report
    gaps = training_oos_gap(dataset, report)
    assert not gaps.empty
    assert {"train_metric", "oos_metric", "overfit_gap"}.issubset(gaps.columns)
    concentrated = pd.DataFrame({"importance": [0.99, 0.01]})
    regimes = pd.DataFrame({"difference_vs_baseline": [0.2, -0.1, -0.1]})
    warnings = suspicious_performance_checks(
        classification_accuracy=0.99,
        correlation=0.95,
        feature_importance=concentrated,
        train_oos_gap=0.30,
        model_metric=0.90,
        baseline_metric=0.50,
        fold_values=np.array([0.5, 0.5, 1.0]),
        regime_results=regimes,
    )
    assert any("accuracy" in warning for warning in warnings)
    assert any("correlation" in warning for warning in warnings)
    assert any("concentration" in warning for warning in warnings)
    assert any("training" in warning for warning in warnings)
    assert any("baseline" in warning for warning in warnings)
    assert any("regime" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (RobustnessEvidence(80, 0.1, False, True, True, True, 0.3, 0.05), "ROBUST"),
        (RobustnessEvidence(60, 0.1, True, True, False, True, 0.4, 0.1), "PROMISING BUT UNSTABLE"),
        (RobustnessEvidence(30, -0.1, True, False, False, False, 0.5, 0.1), "WEAK SIGNAL"),
        (RobustnessEvidence(10, -0.1, True, False, False, False, 0.5, 0.1), "NO PERSISTENT SIGNAL"),
        (RobustnessEvidence(80, 0.1, False, True, True, True, 0.95, 0.05), "SUSPICIOUS / REQUIRES INVESTIGATION"),
    ],
)
def test_robustness_verdict_logic(evidence: RobustnessEvidence, expected: str) -> None:
    assert robustness_verdict(evidence).verdict == expected


def test_audit_is_complete_and_matches_checked_in_document() -> None:
    audit = render_leakage_audit()
    for label in "ABCDEFGHIJKLMN":
        assert f"{label}." in audit
    checked_in = Path("PHASE5_AUDIT.md").read_text(encoding="utf-8")
    assert checked_in.strip() == audit.strip()


def test_synthetic_future_feature_is_rejected(ohlcv: pd.DataFrame) -> None:
    dataset = build_dataset(ohlcv)
    frame = dataset.frame.copy()
    frame["future_oracle"] = frame[dataset.target_columns[1]]
    leaked = ResearchDataset(frame, (*dataset.feature_columns, "future_oracle"), dataset.target_columns)
    with pytest.raises(LeakageError):
        evaluate_walk_forward(
            leaked,
            config=WalkForwardConfig("expanding", 12, 8, 8, 1),
            model_names=("logistic_regression",),
        )

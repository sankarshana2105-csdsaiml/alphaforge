"""Create clearly labeled synthetic verification tables for the static report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.dataset import build_dataset
from alphaforge.robustness import (
    evaluate_feature_ablation,
    evaluate_horizon_sensitivity,
    evaluate_window_sensitivity,
    permutation_feature_importance,
    regime_performance,
    threshold_sensitivity,
    training_oos_gap,
    transaction_cost_stress,
)
from alphaforge.walk_forward import WalkForwardConfig, evaluate_walk_forward


def synthetic_ohlcv(rows: int = 120) -> pd.DataFrame:
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


def main() -> None:
    output = Path("reports/tables")
    output.mkdir(parents=True, exist_ok=True)
    raw = synthetic_ohlcv()
    dataset = build_dataset(raw)
    config = WalkForwardConfig("expanding", 12, 8, 8, 1)
    report = evaluate_walk_forward(dataset, config=config)

    report.comparison.to_csv(output / "model_comparison.csv", index=False)
    report.aggregate_metrics.to_csv(output / "walk_forward_aggregate_metrics.csv", index=False)
    report.stability.to_csv(output / "stability_summary.csv", index=False)
    report.fold_metrics().to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "target": ["direction_down", "direction_up"],
            "count": [int((dataset.y_direction == 0).sum()), int((dataset.y_direction == 1).sum())],
        }
    ).to_csv(output / "target_distribution.csv", index=False)

    permutation_feature_importance(
        dataset, report, model_name="random_forest_classifier", target_type="classification", n_repeats=5
    ).to_csv(output / "feature_importance.csv", index=False)
    evaluate_feature_ablation(
        dataset, config=config, model_name="logistic_regression", target_type="classification", top_feature="return_1"
    ).results.to_csv(output / "feature_ablation.csv", index=False)
    evaluate_horizon_sensitivity(
        raw, horizons=(1, 5), config=config, model_name="logistic_regression", target_type="classification"
    ).to_csv(output / "horizon_sensitivity.csv", index=False)
    evaluate_window_sensitivity(
        dataset,
        configurations={
            "expanding": config,
            "rolling": WalkForwardConfig("rolling", 12, 8, 8, 1, max_train_size=16),
        },
        model_name="linear_regression",
        target_type="regression",
    ).to_csv(output / "window_sensitivity.csv", index=False)
    regime_performance(
        dataset, report, model_name="random_forest_classifier", target_type="classification", minimum_samples=100
    ).to_csv(output / "regime_performance.csv", index=False)
    threshold_sensitivity(report, model_name="random_forest_classifier").to_csv(output / "threshold_sensitivity.csv", index=False)
    transaction_cost_stress(dataset, report, model_name="random_forest_classifier").to_csv(output / "transaction_cost_stress.csv", index=False)
    training_oos_gap(dataset, report).to_csv(output / "train_oos_gap.csv", index=False)
    (output / "artifact_metadata.json").write_text(
        json.dumps(
            {
                "source": "deterministic synthetic verification bars",
                "research_evidence": "not empirical market evidence",
                "purpose": "pipeline verification and report rendering",
                "verdict": "NO PERSISTENT SIGNAL",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

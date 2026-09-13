"""Render static report figures from saved research tables without rerunning models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TABLE_FILES = (
    "target_distribution.csv",
    "model_comparison.csv",
    "fold_metrics.csv",
    "feature_importance.csv",
    "feature_ablation.csv",
    "horizon_sensitivity.csv",
    "window_sensitivity.csv",
    "regime_performance.csv",
    "threshold_sensitivity.csv",
    "transaction_cost_stress.csv",
    "train_oos_gap.csv",
)


def load_table(table_dir: Path, name: str) -> pd.DataFrame | None:
    path = table_dir / name
    return pd.read_csv(path) if path.is_file() else None


def write_manifest(table_dir: Path) -> Path:
    """Write a deterministic inventory of the tables consumed by the report."""

    table_dir.mkdir(parents=True, exist_ok=True)
    available = sorted(path.name for path in table_dir.glob("*.csv"))
    manifest = table_dir / "report_manifest.json"
    manifest.write_text(json.dumps({"tables": available}, indent=2) + "\n", encoding="utf-8")
    return manifest


def render_figures(table_dir: Path, figure_dir: Path) -> list[Path]:
    """Create useful figures only for tables already present on disk."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    targets = load_table(table_dir, "target_distribution.csv")
    if targets is not None:
        created.append(_bar(targets, "target", "count", "Target and class distribution", "Count", figure_dir / "target-class-distribution.png"))

    comparison = load_table(table_dir, "model_comparison.csv")
    if comparison is not None:
        created.append(_comparison(comparison, figure_dir / "model-baseline-comparison.png"))

    fold_metrics = load_table(table_dir, "fold_metrics.csv")
    if fold_metrics is not None:
        created.append(_folds(fold_metrics, figure_dir / "walk-forward-fold-performance.png"))

    importance = load_table(table_dir, "feature_importance.csv")
    if importance is not None:
        created.append(_bar(importance.head(12).iloc[::-1], "feature", "importance", "OOS permutation importance", "Importance", figure_dir / "feature-importance.png", horizontal=True))

    ablation = load_table(table_dir, "feature_ablation.csv")
    if ablation is not None:
        created.append(_bar(ablation, "configuration", "mean", "Feature-group ablation", "Primary metric", figure_dir / "feature-ablation.png"))

    horizons = load_table(table_dir, "horizon_sensitivity.csv")
    if horizons is not None:
        created.append(_line(horizons, "horizon", "mean", "Horizon sensitivity", "Horizon", "Primary metric", figure_dir / "horizon-sensitivity.png"))

    windows = load_table(table_dir, "window_sensitivity.csv")
    if windows is not None:
        created.append(_bar(windows, "configuration", "mean", "Window sensitivity", "Primary metric", figure_dir / "window-sensitivity.png"))

    regimes = load_table(table_dir, "regime_performance.csv")
    if regimes is not None:
        created.append(_bar(regimes, "regime", "difference_vs_baseline", "Regime performance versus baseline", "Difference", figure_dir / "regime-performance.png"))

    thresholds = load_table(table_dir, "threshold_sensitivity.csv")
    if thresholds is not None:
        created.append(_line(thresholds, "threshold", "coverage", "Threshold sensitivity", "Threshold", "Coverage", figure_dir / "threshold-sensitivity.png"))

    costs = load_table(table_dir, "transaction_cost_stress.csv")
    if costs is not None:
        created.append(_line(costs, "cost_bps", "net_result", "Transaction-cost stress", "Cost (bps)", "Net result", figure_dir / "transaction-cost-stress.png"))

    gaps = load_table(table_dir, "train_oos_gap.csv")
    if gaps is not None:
        created.append(_gap(gaps, figure_dir / "train-oos-gap.png"))
    write_manifest(table_dir)
    return created


def _bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    ylabel: str,
    path: Path,
    *,
    horizontal: bool = False,
) -> Path:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if horizontal:
        axis.barh(frame[category], frame[value])
        axis.set_xlabel(ylabel)
    else:
        axis.bar(frame[category], frame[value])
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _line(frame: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7, 4))
    ordered = frame.sort_values(x)
    axis.plot(ordered[x], ordered[y], marker="o")
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _comparison(frame: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, target_type in zip(axes, ("classification", "regression"), strict=True):
        subset = frame.loc[frame["target_type"] == target_type]
        axis.bar(subset["model"], subset["mean"])
        axis.set_title(f"{target_type.title()} walk-forward comparison")
        axis.set_ylabel("Primary metric")
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _folds(frame: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    settings = (("classification", "balanced_accuracy"), ("regression", "rmse"))
    for axis, (target_type, metric) in zip(axes, settings, strict=True):
        subset = frame.loc[frame["target_type"] == target_type]
        for model, model_rows in subset.groupby("model"):
            if metric in model_rows:
                axis.plot(model_rows["fold"], model_rows[metric], marker="o", label=model)
        axis.set(title=f"{target_type.title()} fold performance", xlabel="Fold", ylabel=metric)
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _gap(frame: pd.DataFrame, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    grouped = frame.groupby("model", as_index=False)["overfit_gap"].mean()
    axis.bar(grouped["model"], grouped["overfit_gap"])
    axis.set(title="Mean train-to-OOS gap", ylabel="Gap")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AlphaForge report figures from saved tables.")
    parser.add_argument("--tables", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    paths = render_figures(args.tables, args.figures)
    print(f"Rendered {len(paths)} figures from saved tables.")


if __name__ == "__main__":
    main()

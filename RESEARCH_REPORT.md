# AlphaForge

## Research Question

Can interpretable market-derived features predict short-horizon future returns or direction better than simple, time-respecting baselines?

## Hypothesis

Recent returns, momentum, volatility, volume behavior, rolling statistics, and simple regime labels may contain weak conditional information about the next return or direction. AlphaForge tests that possibility; it does not assume it exists.

## Data

The pipeline accepts timestamped OHLCV bars: `timestamp`, `open`, `high`, `low`, `close`, and `volume`. It sorts timestamps, rejects impossible OHLC relationships, non-positive prices, negative volume, and invalid timestamps, and makes missing-value and duplicate policies explicit.

The current repository includes deterministic synthetic verification artifacts, not an external market-data study. Frequency is therefore assumed to be regular bars in the supplied data. Any empirical run must document the vendor, asset universe, timezone, adjustment policy, and bar frequency.

## Features

The feature set is deliberately small and interpretable:

- Returns: one- and multi-period simple returns plus log return.
- Momentum: rolling momentum, price relative to moving average, and moving-average spread.
- Volatility: trailing return volatility, range, normalized ATR, and rolling dispersion.
- Volume: trailing mean, ratio, and change.
- Rolling statistics: trailing mean, standard deviation, and z-score.
- Regime labels: trailing volatility and trend indicators; these are descriptive labels, not definitive market states.

## Targets

For horizon `h`, the return target is `close[t+h] / close[t] - 1`; direction is one when that return is positive and zero otherwise. The terminal `h` observations have no target and are excluded. Feature code does not construct or consume forward targets.

## Leakage Controls

- Chronological partitions only; no row shuffling or random split.
- Trailing, never centered, rolling windows.
- Target construction isolated from feature engineering.
- Scalers and estimators fit only on each training fold.
- Horizon purging removes training rows whose labels resolve during the next evaluation window.
- Every walk-forward fold builds a fresh preprocessing pipeline and model.
- OOS predictions are unique per model, target, and timestamp.
- [PHASE5_AUDIT.md](PHASE5_AUDIT.md) records direct controls and residual data risks.

## Baselines

Classification comparisons include majority class, recent-return-direction persistence, and logistic regression. Regression comparisons include zero return, historical mean, and linear regression. These references make weak or unstable ML performance visible rather than hiding it behind one selected metric.

## ML Models

The project evaluates fixed-parameter Random Forest and Histogram Gradient Boosting models for classification and regression. These classical models test nonlinear relationships and interactions without implying that deep learning was used or needed.

## Walk-Forward Validation

AlphaForge supports expanding and fixed rolling windows. Each fold trains strictly before its evaluation window, purges the target horizon at the boundary, and contributes only truly unseen predictions to the combined OOS table. Fold-level metrics, aggregate summaries, dispersion, baseline win rate, trend, and stability warnings are retained.

## Robustness Analysis

The reusable Phase 5 outputs cover OOS permutation and supported native importance, feature ablation, horizon and window sensitivity, descriptive regime slices, fixed-threshold sensitivity, directional transaction-cost stress, train-versus-OOS gaps, and suspicious-result checks. They are stress tests, not model-selection searches. Thresholds and costs are predeclared fixed values.

## Results

### Evidence boundary

The saved tables and figures under `reports/` are generated from deterministic synthetic verification bars. They verify the reporting workflow, model isolation, calculation paths, and plot generation. They are **not** evidence that a market feature persists out of sample and are not used to support an empirical performance claim.

### Single-split observations

The single chronological split is retained as a baseline engineering check only. Its results are not sufficient to establish a signal and are superseded by walk-forward and robustness requirements.

### Walk-forward observations

The saved verification outputs include model comparison, aggregate metrics, stability summaries, fold performance, and OOS-derived robustness tables. For a real study, these tables must be regenerated from predeclared external data and interpreted together; no selected fold, horizon, regime, or threshold should determine the conclusion.

### Robustness observations

Feature importance, ablation, sensitivity, regime, threshold, cost, and train/OOS-gap outputs are saved as reproducible tables. They show how the pipeline challenges a putative signal, but synthetic verification data cannot validate market behavior.

## Final Verdict

### NO PERSISTENT SIGNAL

AlphaForge has not established persistent exploitable predictability. No real-market, point-in-time dataset or independent lockbox replication is bundled with the project; the available saved results are verification artifacts. Any observed performance remains dependent on data and configuration until reproduced across independent assets and untouched periods.

This is a valid research outcome. The project demonstrates how to reject an unsupported market-prediction claim through leakage controls, baselines, walk-forward validation, and robustness stress testing.

## Limitations

- Survivorship bias and limited asset coverage.
- Corporate-action adjustment assumptions.
- Vendor errors, backfills, timestamps, and cleaning artifacts.
- Multiple testing and informal researcher degrees of freedom.
- Sensitivity to horizon, training window, evaluation window, and regime definition.
- Simplified transaction-cost stress; no execution, liquidity, borrowing, or impact model.
- No external lockbox period or independent replication.

## What I Would Do Next

1. Run the predeclared workflow on multiple independent assets using point-in-time data.
2. Reserve a completely untouched lockbox period before any model or threshold changes.
3. Use documented corporate-action-adjusted prices and audit large returns across vendors.
4. Move from a single-series setup to a stronger cross-sectional design where justified.
5. Replicate any surviving observation externally before treating it as research evidence.
6. Consider richer microstructure features only when their timestamp availability and leakage controls are documented.

## Reproduction

Render report figures and refresh the report manifest from saved tables only:

```powershell
& .\.venv\Scripts\python.exe -m alphaforge.reporting
```

The table source and its evidence boundary are documented in [reports/README.md](reports/README.md).

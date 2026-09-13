# FINAL_RECOMMENDED_RESUME_VERSION

- Built a Python quantitative ML research framework that engineers interpretable OHLCV features and evaluates classification and return-regression models against statistical baselines.
- Implemented leakage-safe expanding and rolling walk-forward validation with train-only preprocessing, prediction-horizon purging, deterministic model fitting, and unique out-of-sample prediction tracking.
- Developed reproducible robustness analysis covering permutation importance, feature ablation, horizon/window/regime sensitivity, transaction-cost stress, and explicit leakage auditing; concluded that persistent signal was not established.

# 30_SECOND_EXPLANATION

AlphaForge is a Python-first research framework for testing whether market-derived features contain predictive signal. I built strict data validation, interpretable features, statistical and classical ML baselines, and horizon-purged walk-forward evaluation. I then challenged results with ablation, sensitivity, regime, importance, and cost tests. The honest conclusion is that the available evidence does not establish persistent signal.

# 2_MINUTE_INTERVIEW_EXPLANATION

The central problem was not fitting a model; it was designing an evaluation that could not accidentally see the future. AlphaForge validates OHLCV bars, computes only trailing features, and constructs forward-return and direction targets in a separate module. Models range from majority, persistence, zero-return, and historical-mean baselines to logistic and linear regression, Random Forest, and Histogram Gradient Boosting.

For evaluation, I used chronological expanding and rolling walk-forward folds. Each fold creates a fresh model pipeline, fits preprocessing only on training rows, and purges the prediction horizon so no training label depends on prices inside the evaluation window. I retain unique OOS predictions and compare fold distributions rather than highlighting one strong period.

The robustness layer measures OOS permutation importance, removes features and groups, varies predeclared horizons and windows, slices descriptive regimes, checks fixed probability thresholds, applies simple cost stress, and compares training with OOS performance. The result is deliberately negative: persistent predictability was not established, and the repository contains synthetic verification artifacts rather than an external market lockbox. That conclusion demonstrates the point of the project—credible research controls are more important than an impressive backtest.

# KEY_TECHNOLOGIES

Python 3.12, NumPy, pandas, SciPy, scikit-learn, matplotlib, pytest, sklearn pipelines.

# KEY_QUANT_CONCEPTS

Forward returns, direction targets, look-ahead bias, chronological splits, walk-forward validation, horizon purging, OOS evaluation, class imbalance, baseline comparison, feature importance, ablation, regime dependence, transaction costs, overfitting, survivorship bias, and multiple-testing risk.

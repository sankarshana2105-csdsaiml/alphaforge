# MUST KNOW FOR INTERVIEWS

- Explain the goal: test whether interpretable OHLCV features beat simple baselines without future leakage.
- Derive `future_return[t] = close[t+h] / close[t] - 1` and explain why the last `h` rows have no target.
- Explain look-ahead bias, trailing versus centered windows, chronological splitting, and train-only preprocessing.
- Draw one walk-forward fold and show why `h` rows are purged before evaluation.
- Explain expanding versus rolling windows and why every fold gets a fresh scaler and model.
- Name the baselines and explain why ML must beat them consistently.
- Contrast logistic/linear regression, Random Forest, and Histogram Gradient Boosting.
- Interpret balanced accuracy, F1, ROC-AUC, MAE, RMSE, R², and correlation; know when a metric is undefined.
- Explain OOS permutation importance, feature ablation, sensitivity analysis, train–OOS gaps, and transaction-cost turnover.
- Defend the verdict: `NO PERSISTENT SIGNAL` means the project did not establish stable, replicated market predictability. Synthetic artifacts verify code, not markets.
- State the major limitations: survivorship, corporate actions, vendor artifacts, multiple testing, limited coverage, and no external lockbox.

# SHOULD KNOW

- Trace the modules: `data` → `features`/`targets` → `splits`/`walk_forward` → `models` → `robustness` → `reporting`.
- Modify feature and walk-forward configuration safely and predict which tests should change.
- Explain class imbalance, constant-target handling, deterministic seeds, OOS prediction uniqueness, regime sample-size warnings, and suspicious-performance triggers.
- Read the comparison, stability, importance, ablation, threshold, cost, and train–OOS-gap tables.
- Explain why correlated features can hide permutation importance and why model-native importance is model-specific.
- Describe what external evidence would justify changing the verdict.

# CAN LEARN LATER

- Formal multiple-hypothesis corrections and advanced statistical inference.
- Point-in-time multi-asset data engineering and cross-sectional portfolio construction.
- Microstructure, execution, slippage, market impact, and borrow modeling.
- Nested model selection, advanced interpretability, and distributed experiments.
- Deep learning only if future data volume and validated classical-model evidence justify it.

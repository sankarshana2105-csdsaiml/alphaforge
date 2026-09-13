# AlphaForge Project Status

## Current phase

**FINALIZED — PUBLICLY PUBLISHED**

AlphaForge is frozen. No additional feature or research development is planned in this repository.

## Publication record

- GitHub URL: https://github.com/sankarshana2105-csdsaiml/alphaforge
- Published framework commit: `b8273aa9a64e3792dbd5ca8f5fe7ae392efe6b57`
- Final test count: 69 passed
- Recruiter audit score: 8.5/10
- Final verdict: **NO PERSISTENT SIGNAL**

## Completed work

- Created the Python package and test structure.
- Added CSV ingestion and strict OHLCV validation.
- Made missing-row and duplicate-timestamp policies explicit.
- Added past-only return, momentum, volatility, volume, rolling-statistic, and regime features.
- Added aligned forward-return and binary-direction targets.
- Kept validation, features, targets, and dataset assembly as separate modules.
- Added deterministic and leakage-focused tests.
- Added configurable chronological train/validation/test partitions using fractions or date boundaries.
- Added majority-class, recent-return-direction, logistic-regression, historical-mean, zero-return, and linear-regression baselines.
- Added fixed classification and regression metrics, train-only scaling, structured comparison tables, and descriptive diagnostics.
- Added leakage guards, finite-value checks, and constant-target handling.
- Added fixed-parameter Random Forest and histogram gradient-boosting classifiers and regressors.
- Preserved every Phase 2 baseline in unified classification and regression comparisons.
- Added deterministic seeds, model runtimes, probability ranges, constant-prediction detection, and suspicious-metric warnings.
- Reused the chronological split and leakage validation; models fit training rows only and do not tune on validation or test data.
- Added expanding and fixed rolling walk-forward folds with configurable training, evaluation, step, and horizon settings.
- Purged horizon-boundary rows so training targets resolve before evaluation begins.
- Refit every scaler and model independently per fold.
- Added fold metrics, aggregate statistics, unique chronological OOS predictions, baseline comparisons, trends, and instability warnings.
- Added adversarial future-signal rejection and boundary-leakage tests.
- Added OOS permutation importance and supported model-native importance with dispersion and ranks.
- Added feature ablation plus horizon, window, regime, and fixed-threshold sensitivity outputs.
- Added directional OOS transaction-cost stress, train-versus-OOS gaps, suspicious-result checks, and deterministic verdict logic.
- Added `PHASE5_AUDIT.md` and a current evidence-based robustness verdict.
- Added `RESEARCH_REPORT.md` and `EXECUTIVE_SUMMARY.md` with the evidence boundary and `NO PERSISTENT SIGNAL` conclusion preserved.
- Added saved verification tables, 11 static matplotlib figures, and a renderer that reads saved tables without rerunning models.
- Added a recruiter-first README with architecture, methodology, evidence boundaries, results, figures, reproducibility, and limitations.
- Added exactly three resume bullets, concise interview explanations, and an A–R interview preparation guide.
- Added an 11-category recruiter audit scoring the repository 8.5/10 overall.
- Reorganized the learning guide by interview priority and added portable-link, conservative-claim, and documentation checks.
- Hardened ignore rules for local environments, credentials, coverage, caches, and build artifacts.

## Tests

- `python -m pytest`: 69 passed.
- `python -m compileall -q src tests scripts`: passed.
- Reporting artifact generation: passed.
- Documentation links, portable paths, secret patterns, and unsupported public claims: passed.
- Verified with Python 3.12.14 in the project virtual environment.

## Remaining limitations

- No real-market external lockbox is included.
- Synthetic outputs demonstrate methodology, not market profitability.
- Survivorship, corporate-action, data-vendor, and multiple-testing risks remain.

## Exact next step

None. AlphaForge is frozen; any empirical extension must be scoped as a separate project using real market data and an untouched external lockbox.

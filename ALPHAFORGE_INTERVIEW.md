# AlphaForge Interview Pack

| Area | High-value question | Short expected answer | What I must understand personally |
|---|---|---|---|
| A. Python | Why separate modules instead of one notebook? | Reusable typed functions make validation, leakage tests, deterministic evaluation, and reporting independently testable. | Trace one row from CSV validation to OOS output. |
| B. Feature engineering | Why use interpretable features? | A compact economic feature set is easier to audit, ablate, and explain than hundreds of indicators. | Which past values each feature uses. |
| C. Financial time series | Why is ordinary IID evaluation unsafe? | Ordering, dependence, regime change, and label overlap violate random-split assumptions. | Why future rows cannot enter training. |
| D. Targets | Why model both return and direction? | Regression tests magnitude; classification tests sign. Their metrics and baselines answer different questions. | Derive the forward-return formula and binary label. |
| E. Chronological splitting | Why not `train_test_split`? | Random splitting mixes future and past and can inflate performance. | Show the train, validation, and test ordering. |
| F. Walk-forward validation | Expanding versus rolling? | Expanding uses all eligible history; rolling limits training to recent history and may adapt to drift. | The sample-size versus adaptability tradeoff. |
| G. Horizon purging | Why remove rows before evaluation? | Their forward targets would resolve using prices inside the evaluation window. | Prove `last_train_source < first_evaluation_row`. |
| H. Preprocessing leakage | How is scaling kept safe? | A fresh sklearn pipeline is fit on each fold’s training features; evaluation rows are transform-only. | Inspect and explain the scaler-mean test. |
| I. Baselines | Why so many simple baselines? | They reveal whether complexity adds value beyond class frequency, recent direction, zero, or historical mean. | Which metric each baseline should challenge. |
| J. Random Forest | Why use it? | Bagged trees capture nonlinear interactions and reduce individual-tree variance. | Bootstrap aggregation, depth, leaf size, and feature importance limits. |
| K. Gradient boosting | How does it differ? | Trees are added sequentially to correct residual errors instead of being independently averaged. | Learning rate, iteration count, and overfitting risk. |
| L. Overfitting | What evidence would concern you? | Large train–OOS gaps, volatile folds, rare baseline wins, and configuration dependence. | Why one strong holdout is insufficient. |
| M. Feature importance | Why permutation importance OOS? | It measures score loss after shuffling an unseen feature without relying on one model’s internal split counts. | Correlated features can dilute or mask importance. |
| N. Ablation | What does removing a feature group test? | Whether performance survives without that information family or depends on one concentrated source. | Collapse, improvement, and unchanged outcomes. |
| O. Regime analysis | What can regime slices establish? | They describe conditional performance, but small samples and heuristic labels prevent broad claims. | Why regimes must use trailing information. |
| P. Transaction costs | What does the cost test mean? | It subtracts fixed friction from position turnover to see whether a small OOS edge disappears. | Why this is not a tradability or execution claim. |
| Q. Verdict | Why is `NO PERSISTENT SIGNAL` valid? | The evidence did not establish stable, replicated predictive performance; rejecting the hypothesis is scientifically useful. | State this confidently without reframing synthetic results as market evidence. |
| R. Limitations | What remains unresolved? | External lockbox replication, survivorship, adjustments, vendor quality, multiple testing, and broader asset coverage. | Which limitations require better data versus more code. |

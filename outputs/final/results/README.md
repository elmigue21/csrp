# Model results (post-defense)

Written by `src/run_model.py` from `features/features.csv` (88 rows, 44 participants,
140 features). Decisions: `docs/post-defense-modeling-plan.md` (MD1-MD4).
`run_info.json` records the SHA-256 of the features file and when each stage last ran; every
results row also carries `features_sha256`. Stages: `python src/run_model.py --stages main,by_n,checks`
(any subset). `results_by_n.csv` and `checks.csv` are saved after every finished run and resume
where they stopped.

| File | What it is |
|---|---|
| `results.csv` | One row per model: ROC-AUC with 95% CI, accuracy, balanced accuracy, precision, recall and F1 (per class and macro), Brier score, participants with A4 above A1, chance-test p-value, median number of features chosen |
| `results_by_n.csv` | The same run with the number of features fixed at 5, 10, 20, 40, 80 and 140 (every N reported, none picked from these results), every metric |
| `checks.csv` | Blink rate alone, and both models without blink rate, every metric and chance test |
| `comparison.csv` | XGBoost minus LR: AUC difference with 95% CI, paired Wilcoxon on per-participant margins |
| `predictions.csv` | Out-of-fold probability of high load for every row, both models |
| `selection_count.csv` | In how many of the 44 folds each feature made the top N, per model |
| `shap_logistic_regression.csv`, `shap_xgboost.csv` | Mean absolute and mean SHAP per feature (0 in folds where the feature was not picked) |
| `chosen_params_*.csv` | The settings each fold chose, including N (`select__n`) |

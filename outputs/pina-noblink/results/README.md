# Pina et al. 2025 results

Written by `src/run_pina.py`; plan: `docs/pina-analysis-plan.md` (written before the eye data were
loaded). `90` is the main contrast; `360` is a separate check, never pooled. `sensitivity_*` files
were decided after the main results and are never the main result.

| File | What it is |
|---|---|
| `features/features.csv` | One row per 30 s window: QC columns and every feature (17-statistic split, DROPPED removed) |
| `features/extraction_log.csv` | Per participant: time-zone offset, windows, errors, median invalid share |
| `manipulation_check.json`, `nasa_tlx.csv` | Raw NASA-TLX per task; combined vs visual alone |
| `demographics.json` | Age (mean, SD, range) and sex counts of the analyzed participants |
| `data_check_<area>.csv` | Windows kept, exclusions, fixation sanity check, features used |
| `results_<area>.csv` | LR, XGBoost and the in-fold one-feature baseline: every metric |
| `comparison_<area>.csv` | Model minus baseline, XGBoost minus LR: AUC difference with 95% CI |
| `predictions_<area>.csv` | Out-of-fold probabilities, every window |
| `chosen_params_<area>_*.csv` | Settings each fold chose, including N |
| `shap_90.csv`, `shap_by_signal_90.csv` | SHAP share per feature (with folds selected) and per signal |
| `selection_count_90.csv` | Folds (of 36) in which each feature was selected |
| `without_top_signal_90.csv` | Both models refit without the signal with the largest SHAP share |
| `permutation_90.csv` | Chance test: labels swapped within each person, settings fixed at the most-chosen ones |
| `sensitivity_*_90_keep_events.csv` | Same run keeping the event features that failed the check |
| `metric_ci_<area>.csv` | Accuracy, precision, recall, F1, Brier, ROC-AUC with participant-bootstrap 95% CIs |
| `metric_differences_<area>.csv` | The same metrics as paired differences: model minus baseline, XGBoost minus LR |
| `review_checks_90.csv` | Post-hoc: models without blink rate, tracking-quality baseline, results by condition order |
| `review_checks_medians_90.csv` | Median blink rate and invalid-sample share per load |

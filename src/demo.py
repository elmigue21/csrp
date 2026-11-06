"""Live demo: compare Logistic Regression and XGBoost on the COLET eye-tracking features.

Same data, settings and seed as run_colet.py, so it gives the same numbers.
The two models and their tuning ranges are defined in models.py.
Nothing is written to outputs/.

Usage:  python src/demo.py            (all 10 features)
        python src/demo.py --pupil    (pupil features only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import INNER_FOLDS, N_ITER, PUPIL_FEATURES, RANDOM_STATE
from evaluate import score
from importance import oof_shap, shap_summary
from labels import binary_label, main_rows
from models import MODELS
from normalize import per_participant_zscore
from run_colet import prepare


def train_and_test(X, y, groups, make_model, search_space):
    """Train, tune and test one model."""
    predictions = np.full(len(y), np.nan)
    folds = []

    # LOPO (leave-one-participant-out): 44 rounds. Each round hides one person,
    # trains on the other 43, then tests on the hidden person.
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        # HYPERPARAMETER TUNING (random search): try 30 random settings, score each with
        # ROC-AUC over 5 inner folds split by person, keep the best one.
        # Only the 43 training people are used, so the hidden person never leaks in.
        inner_folds = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(make_model(), search_space, n_iter=N_ITER, scoring="roc_auc",
                                    cv=inner_folds, random_state=RANDOM_STATE, n_jobs=-1,
                                    error_score=np.nan)
        search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])

        # TESTING: the best model predicts the hidden person's chance of high load
        best_model = search.best_estimator_
        predictions[test_idx] = best_model.predict_proba(X.iloc[test_idx])[:, 1]
        folds.append({"model": best_model, "train_idx": train_idx, "test_idx": test_idx})
    return predictions, folds


def main(pupil_only=False):
    # 1. LOAD DATA: the 10 eye-tracking features, one row per person per activity
    prep = prepare()
    features = PUPIL_FEATURES if pupil_only else prep["features_used"]
    rows = prep["rows"]

    # 2. PER-PERSON Z-SCORE: compare each person to their own average
    z = per_participant_zscore(rows[features], rows["participant"])
    rows = rows.assign(**{c: z[c] for c in features})

    # 3. LABELS: A1 = low load (0), A4 = high load (1)
    data = main_rows(rows)
    X = data[features].reset_index(drop=True)
    y = binary_label(data)
    groups = data["participant"].to_numpy()

    # 4. MODELS: Logistic Regression and XGBoost, both run through the same LOPO + tuning
    #    (model setup and tuning ranges are in models.py)
    results, top_features = [], {}
    for name, spec in MODELS.items():
        predictions, folds = train_and_test(X, y, groups, spec["factory"], spec["space"])
        s = score(y, predictions)
        results.append({"model": name, "roc_auc": s["roc_auc"],
                        "balanced_accuracy": s["balanced_accuracy"], "f1": s["macro_f1"]})

        # 5. FEATURE IMPORTANCE (SHAP): which eye features each model relied on most
        kind = "lr" if name == "Logistic Regression" else "xgb"
        shap_values = oof_shap(folds, X, kind)
        top_features[name] = shap_summary(shap_values, features)["feature"].head(3).tolist()

    # 6. RESULTS: ROC-AUC, balanced accuracy and F1 over all 44 hidden people
    print(f"\nRESULTS ({'pupil features only' if pupil_only else 'all 10 features'}, "
          f"{len(np.unique(groups))} participants)")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    print("\nTop 3 features (SHAP)")
    for name, top in top_features.items():
        print(f"  {name}: {', '.join(top)}")


if __name__ == "__main__":
    main(pupil_only="--pupil" in sys.argv)

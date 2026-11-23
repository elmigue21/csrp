"""Live demo: Logistic Regression vs XGBoost on the final eye-tracking features.

Same features, settings and seed as run_model.py, so it prints the paper's numbers.
Nothing is saved.

Usage:  python src/demo.py          (as in the paper: 30 tuning candidates, a few minutes)
        python src/demo.py --fast   (5 candidates: quicker, numbers differ slightly)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FINAL_DIR, INNER_FOLDS, N_ITER, RANDOM_STATE, SEARCH_JOBS
from evaluate import score
from importance import shap_summary
from run_model import fold_shap, load_features
from topn import MODELS_TOPN


def train_and_test(X, y, people, make_model, settings, n_iter):
    """Leave-one-participant-out: hide one person, train on the other 43, test on the hidden one."""
    predictions = np.full(len(y), np.nan)
    folds = []
    for train, test in LeaveOneGroupOut().split(X, y, people):
        # Tune on the 43 training people only: try n_iter random settings (the number of
        # features N included) and keep the one with the best ROC-AUC over 5 inner folds.
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(make_model(), settings, n_iter=n_iter, scoring="roc_auc", cv=inner,
                                    random_state=RANDOM_STATE, n_jobs=SEARCH_JOBS, error_score=np.nan)
        search.fit(X.iloc[train], y[train], groups=people[train])

        # Test: the best model predicts the hidden person's two recordings.
        predictions[test] = search.best_estimator_.predict_proba(X.iloc[test])[:, 1]
        folds.append({"model": search.best_estimator_, "train_idx": train, "test_idx": test})
    return predictions, folds


def main(n_iter=N_ITER):
    # 1. Data: 88 recordings (44 people x A1 low, A4 high), 140 features each
    X, y, people, _ = load_features(FINAL_DIR / "features" / "features.csv")
    print(f"{len(y)} recordings, {len(np.unique(people))} people, {X.shape[1]} features\n")

    results, top = {}, {}
    for name, spec in MODELS_TOPN.items():
        print(f"Training {name} ...")
        # 2. Train and test (feature selection, scaling and tuning all happen inside each round)
        predictions, folds = train_and_test(X, y, people, spec["factory"], spec["space"], n_iter)

        # 3. Every metric, over all 88 held-out predictions
        s = score(y, predictions)
        results[name] = {
            "ROC-AUC": s["roc_auc"], "Accuracy": s["accuracy"], "Precision": s["precision_macro"],
            "Recall (low)": s["recall_low"], "Recall (high)": s["recall_high"],
            "F1": s["macro_f1"], "Brier": s["brier"]}

        # 4. Which features the model relied on (SHAP, on the people it never saw)
        kind = "lr" if name == "Logistic Regression" else "xgb"
        top[name] = shap_summary(fold_shap(folds, X, kind), list(X.columns))["feature"].head(5).tolist()

    print("\nRESULTS (each person tested by a model that never saw them)")
    print(pd.DataFrame(results).round(3).to_string())
    print("\nTop 5 features (SHAP)")
    for name, features in top.items():
        print(f"  {name}: {', '.join(features)}")


if __name__ == "__main__":
    main(n_iter=5 if "--fast" in sys.argv else N_ITER)

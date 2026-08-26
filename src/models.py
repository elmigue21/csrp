"""The two classifiers under comparison, each wrapped so it sees the data it needs.

Missing values are handled differently on purpose, not inconsistently. XGBoost
learns a default branch direction for NaN, so it is given the NaNs untouched.
Logistic Regression cannot represent a missing value at all, so it receives a
median fill plus a binary "this was missing" indicator per affected column -- which
preserves the structural information (no saccade occurred, fewer than two fixations
occurred) that a plain fill would erase.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import LR_GRID, RANDOM_STATE, XGB_GRID


def make_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        missing=np.nan,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


MODELS = {
    "Logistic Regression": {"factory": make_logistic_regression, "grid": LR_GRID},
    "XGBoost": {"factory": make_xgboost, "grid": XGB_GRID},
}

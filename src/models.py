"""The two classifiers under comparison (§8; decisions D-N6, V2-9).

Equal tuning budget: both are tuned by random search with the same number of candidates
(config.N_ITER) over the spaces below, inside participant-grouped inner folds.
Missing values: median imputation for LR (fitted in-fold); XGBoost handles NaN natively.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import RANDOM_STATE


def make_logistic_regression() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                   max_iter=20000, random_state=RANDOM_STATE)),
    ])


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         missing=np.nan, max_depth=2, n_estimators=100, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=1, verbosity=0)


LR_SPACE = {"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": uniform(0.0, 1.0)}

XGB_SPACE = {
    "max_depth": [1, 2, 3],
    "n_estimators": randint(50, 301),
    "learning_rate": uniform(0.05, 0.05),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_lambda": loguniform(1e-2, 1e1),
    "reg_alpha": loguniform(1e-3, 1e0),
}

MODELS = {
    "Logistic Regression": {"factory": make_logistic_regression, "space": LR_SPACE},
    "XGBoost": {"factory": make_xgboost, "space": XGB_SPACE},
}

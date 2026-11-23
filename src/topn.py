"""Top-N feature selection inside each fold, and the two model pipelines that use it.

Decisions (docs/post-defense-modeling-plan.md):
- MD1: a feature is scored on its own by how well it separates low from high load in the
  training rows: |ROC-AUC - 0.5|, so a feature that is lower in high load counts too.
  The best N are kept. Scoring only ever sees the rows the pipeline is fitted on, so inside
  LOPO the test participant never influences which features are picked.
- MD2: N is one more setting of the random search, drawn from N_CANDIDATES and chosen in the
  inner folds of the training participants.
Both models get the same selector, so they choose from the same kind of shortlist.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import validate_data

from config import RANDOM_STATE
from models import LR_SPACE, XGB_SPACE, make_xgboost

N_CANDIDATES = (5, 10, 20, 40, 80, 140)


class TopNByAUC(SelectorMixin, BaseEstimator):
    """Keep the n features whose values alone best separate the two classes.

    Score per feature: |ROC-AUC - 0.5| over the rows where that feature is not blank (0 when a
    feature has no usable rows). Ties keep the earlier column. n larger than the number of
    features keeps them all.
    """

    def __init__(self, n: int = 20):
        self.n = n

    def fit(self, X, y):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float)
        y = np.asarray(y)
        scores = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            ok = np.isfinite(X[:, j])
            if len(np.unique(y[ok])) == 2:
                scores[j] = abs(roc_auc_score(y[ok], X[ok, j]) - 0.5)
        self.scores_ = scores
        keep = np.argsort(-scores, kind="stable")[: min(self.n, X.shape[1])]
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[keep] = True
        return self

    def _get_support_mask(self):
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def make_logistic_regression_topn() -> Pipeline:
    """Impute (training median) -> pick top N -> scale (training mean/SD) -> elastic-net LR."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", TopNByAUC()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                   max_iter=20000, random_state=RANDOM_STATE)),
    ])


def make_xgboost_topn() -> Pipeline:
    """Pick top N -> XGBoost (handles blanks itself, needs no scaling)."""
    return Pipeline([("select", TopNByAUC()), ("clf", make_xgboost())])


MODELS_TOPN = {
    "Logistic Regression": {"factory": make_logistic_regression_topn,
                            "space": {"select__n": list(N_CANDIDATES), **LR_SPACE}},
    "XGBoost": {"factory": make_xgboost_topn,
                "space": {"select__n": list(N_CANDIDATES),
                          **{f"clf__{k}": v for k, v in XGB_SPACE.items()}}},
}


def selected_features(model: Pipeline, names: list[str]) -> list[str]:
    """The features a fitted pipeline kept, in their original column order."""
    mask = model.named_steps["select"].get_support()
    return [n for n, keep in zip(names, mask) if keep]

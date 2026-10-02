"""Leave-one-participant-out evaluation with nested hyperparameter tuning.

Outer loop: 26 folds, each holding out one participant entirely.
Inner loop: 5 stratified group folds over the 25 training participants, used to pick
hyperparameters. Tuning never sees the held-out participant, so the reported scores
are not inflated by hyperparameter selection.

Two views of the results are produced:

  pooled          -- held-out predictions from all 26 folds concatenated, then scored
                     once. This is the headline, and it sidesteps the fact that a
                     participant whose recordings all fall in one class has no
                     defined ROC-AUC of their own.
  per-participant -- 26 separate scores, summarised as mean +/- SD. This shows how
                     much performance varies between people, which pooling hides.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import GRID_SEARCH_JOBS, INNER_FOLDS, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)


def _score(y_true, y_pred, y_proba) -> dict:
    """Accuracy, precision, recall, F1 and ROC-AUC; AUC is None if only one class."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    out["roc_auc"] = (
        roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan
    )
    return out


def score_predictions(y_true, y_pred, y_proba) -> dict:
    """Public entry point for the metric set, for scoring predictions loaded from disk."""
    return _score(y_true, y_pred, y_proba)


def run_lopo(X, y, groups, factory, grid, tune: bool = True):
    """Run the outer LOPO loop, returning out-of-fold predictions and fold models.

    X is a DataFrame so that feature names survive into the importance step.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)

    oof_proba = np.full(len(y), np.nan)
    fold_models, chosen_params = [], []

    outer = LeaveOneGroupOut()
    for train_idx, test_idx in outer.split(X, y, groups):
        y_train = y[train_idx]
        estimator = factory()

        if tune and grid and len(np.unique(y_train)) == 2:
            inner = StratifiedGroupKFold(
                n_splits=min(INNER_FOLDS, len(np.unique(groups[train_idx]))),
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            # Bounded, not n_jobs=-1: the outer loop creates a fresh worker pool for
            # every one of the 26 folds in every configuration, and unbounded pools
            # churn hard enough on Windows to take the interpreter down mid-run.
            search = GridSearchCV(
                estimator,
                grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=GRID_SEARCH_JOBS,
                pre_dispatch="2*n_jobs",
                error_score=np.nan,
                refit=True,
            )
            # Threads, not processes. The outer loop runs 26 grid searches per
            # configuration, and spawning a fresh process pool for each one exhausts
            # OS resources part-way through a run on Windows. XGBoost's fit releases
            # the GIL, so a thread pool parallelises it just as well.
            with joblib.parallel_backend("threading", n_jobs=GRID_SEARCH_JOBS):
                search.fit(X.iloc[train_idx], y_train, groups=groups[train_idx])
            fitted = search.best_estimator_
            chosen_params.append(search.best_params_)
        else:
            fitted = clone(estimator).fit(X.iloc[train_idx], y_train)
            chosen_params.append({})

        oof_proba[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        fold_models.append({"model": fitted, "test_idx": test_idx,
                            "participant": groups[test_idx][0]})

    oof_pred = (oof_proba >= 0.5).astype(int)
    return {
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "fold_models": fold_models,
        "chosen_params": chosen_params,
        "pooled": _score(y, oof_pred, oof_proba),
        "per_participant": _per_participant(y, oof_pred, oof_proba, groups),
    }


def _per_participant(y, y_pred, y_proba, groups) -> pd.DataFrame:
    rows = []
    for pid in np.unique(groups):
        m = groups == pid
        rec = {"participant": pid, "n_epochs": int(m.sum()),
               "positive_rate": float(y[m].mean())}
        rec.update(_score(y[m], y_pred[m], y_proba[m]))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise_per_participant(df: pd.DataFrame) -> dict:
    """Mean +/- SD across participants, skipping participants with undefined AUC."""
    out = {}
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = df[metric].dropna()
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_sd"] = float(vals.std())
        out[f"{metric}_n"] = int(len(vals))
    return out


def compare_models(per_participant_a: pd.DataFrame, per_participant_b: pd.DataFrame,
                   metric: str = "roc_auc") -> dict:
    """Paired Wilcoxon signed-rank test across participants.

    A difference in pooled AUC is a single number with no uncertainty attached; the
    paired test over the 26 held-out participants is what licenses the claim that one
    model outperforms the other rather than merely scoring higher once.
    """
    from scipy.stats import wilcoxon

    merged = per_participant_a[["participant", metric]].merge(
        per_participant_b[["participant", metric]], on="participant",
        suffixes=("_a", "_b"),
    ).dropna()

    a, b = merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()
    diff = b - a
    if len(diff) < 3 or np.allclose(diff, 0):
        return {"n": len(diff), "mean_difference": float(diff.mean()) if len(diff) else np.nan,
                "statistic": np.nan, "p_value": np.nan}
    stat, p = wilcoxon(a, b)
    return {"n": int(len(diff)), "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)),
            "statistic": float(stat), "p_value": float(p)}


def majority_baseline(y, groups) -> dict:
    """Accuracy of always predicting the training-majority class, under LOPO."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    pred = np.zeros(len(y), dtype=int)
    for _, test_idx in LeaveOneGroupOut().split(y.reshape(-1, 1), y, groups):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        pred[test_idx] = int(round(y[train_idx].mean()))
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}

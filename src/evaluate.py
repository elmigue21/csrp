"""Leave-one-participant-out evaluation, metrics and model comparison (§§9-10).

Outer loop: one participant (both recordings) held out per fold.
Inner loop: StratifiedGroupKFold over the training participants, random search with the
same number of candidates for both models. Scores are computed on pooled out-of-fold
probabilities; uncertainty comes from a bootstrap that resamples participants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedGroupKFold

from config import INNER_FOLDS, N_BOOTSTRAP, N_ITER, N_PERMUTATIONS, RANDOM_STATE, SEARCH_JOBS


def run_lopo(X, y, groups, factory, space, n_iter=N_ITER, tune=True, params=None) -> dict:
    X = pd.DataFrame(X).reset_index(drop=True)
    y, groups = np.asarray(y), np.asarray(groups)
    oof = np.full(len(y), np.nan)
    folds, chosen = [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        est = factory()
        if params:
            est.set_params(**params)
        if tune and space:
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="roc_auc", cv=inner,
                                        random_state=RANDOM_STATE, n_jobs=SEARCH_JOBS, error_score=np.nan)
            search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
            model, best = search.best_estimator_, search.best_params_
        else:
            model, best = clone(est).fit(X.iloc[train_idx], y[train_idx]), dict(params or {})
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        folds.append({"model": model, "train_idx": train_idx, "test_idx": test_idx,
                      "participant": groups[test_idx][0]})
        chosen.append(best)
    return {"oof_proba": oof, "fold_models": folds, "chosen_params": chosen}


def score(y, proba) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall_low": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "recall_high": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
    }


def participant_margins(y, proba, groups) -> pd.Series:
    """p(high) - p(low) for each participant; > 0 means A4 was ranked above A1."""
    df = pd.DataFrame({"g": groups, "y": y, "p": proba})
    hi = df[df.y == 1].groupby("g")["p"].mean()
    lo = df[df.y == 0].groupby("g")["p"].mean()
    return (hi - lo).dropna()


def bootstrap_auc(y, proba, groups, n_boot=N_BOOTSTRAP, other=None) -> dict:
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    other = None if other is None else np.asarray(other)
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    aucs, diffs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], proba[idx])
        aucs.append(a)
        if other is not None:
            diffs.append(roc_auc_score(y[idx], other[idx]) - a)
    out = {"auc_ci_low": float(np.percentile(aucs, 2.5)), "auc_ci_high": float(np.percentile(aucs, 97.5))}
    if other is not None:
        out.update({"diff": float(roc_auc_score(y, other) - roc_auc_score(y, proba)),
                    "diff_ci_low": float(np.percentile(diffs, 2.5)),
                    "diff_ci_high": float(np.percentile(diffs, 97.5))})
    return out


def compare_models(y, proba_lr, proba_xgb, groups, n_boot=N_BOOTSTRAP) -> dict:
    """XGBoost minus LR: pooled AUC difference with a participant-bootstrap CI, plus a paired
    Wilcoxon on per-participant margins (supporting test; an adaptation, see M9/M10)."""
    b = bootstrap_auc(y, proba_lr, groups, n_boot, other=proba_xgb)
    m_lr = participant_margins(y, proba_lr, groups)
    m_xgb = participant_margins(y, proba_xgb, groups)
    d = (m_xgb - m_lr).dropna()
    if len(d) >= 3 and not np.allclose(d, 0):
        stat, p = wilcoxon(m_xgb.loc[d.index], m_lr.loc[d.index])
    else:
        stat, p = np.nan, np.nan
    return {"auc_difference": b["diff"], "auc_difference_ci_low": b["diff_ci_low"],
            "auc_difference_ci_high": b["diff_ci_high"], "wilcoxon_statistic": float(stat),
            "wilcoxon_p": float(p), "n_participants": int(len(d))}


def majority_baseline(y, groups) -> dict:
    """Always predict the training-majority class under LOPO (balanced design -> 0.5)."""
    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.zeros(len(y), int)
    for train_idx, test_idx in LeaveOneGroupOut().split(y.reshape(-1, 1), y, groups):
        pred[test_idx] = int(y[train_idx].mean() > 0.5)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def chosen_params_table(chosen_params: list[dict], participants) -> pd.DataFrame:
    """One row per outer fold: the held-out participant and every hyperparameter chosen."""
    return pd.DataFrame(chosen_params).assign(participant=list(participants))[
        ["participant", *pd.DataFrame(chosen_params).columns]]


def param_summary(chosen_params: list[dict]) -> dict:
    """Median and IQR of each chosen hyperparameter across folds.

    The spaces are continuous, so almost every fold picks a unique set and a "modal set" is
    meaningless; per-parameter spread says how stable the tuning was. max_depth is a small
    integer grid, so its mode is reported as well.
    """
    df = pd.DataFrame(chosen_params)
    out = {}
    for c in df.columns:
        out[f"param_{c}_median"] = float(df[c].median())
        out[f"param_{c}_iqr"] = float(df[c].quantile(0.75) - df[c].quantile(0.25))
    if "max_depth" in df:
        out["param_max_depth_mode"] = float(df["max_depth"].mode().iloc[0])
    return out


def permutation_test(X, y, groups, factory, params, n_perm=N_PERMUTATIONS) -> dict:
    """Grouped permutation test against chance (§10).

    Labels are permuted within participants (each participant's A1/A4 swapped at random),
    keeping the paired structure; hyperparameters are fixed to `params` for speed.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    observed = roc_auc_score(y, run_lopo(X, y, groups, factory, None, tune=False, params=params)["oof_proba"])
    rng = np.random.default_rng(RANDOM_STATE)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            if rng.random() < 0.5:
                idx = np.flatnonzero(groups == g)
                yp[idx] = yp[idx][::-1]
        null.append(roc_auc_score(yp, run_lopo(X, yp, groups, factory, None, tune=False, params=params)["oof_proba"]))
    p = (1 + sum(n >= observed for n in null)) / (1 + n_perm)
    return {"observed_auc": float(observed), "p_value": float(p), "n_perm": int(n_perm)}

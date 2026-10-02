"""Feature importance for both models, read from the LOPO fold models (§10; D-M8).

SHAP for BOTH models so the rankings share a scale (log-odds): TreeSHAP for XGBoost,
linear SHAP for Logistic Regression (on the in-fold imputed + scaled inputs). Each fold's
model explains only the participant it never saw. Applying SHAP to both models is our own
choice, justified by SHAP being model-agnostic (Lundberg & Lee 2017).
Permutation importance is computed on the pooled out-of-fold predictions. Gain importance
is reported in the appendix only. Importance = model reliance, not cause (Molnar 2022).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from config import N_BOOTSTRAP, N_PERM_IMPORTANCE, RANDOM_STATE


def logistic_coefficients(fold_models, names: list[str]) -> pd.DataFrame:
    coefs = np.array([f["model"].named_steps["clf"].coef_.ravel() for f in fold_models])
    out = pd.DataFrame({
        "feature": names,
        "coefficient_mean": coefs.mean(0),
        "coefficient_sd": coefs.std(0),
        "abs_coefficient_mean": np.abs(coefs).mean(0),
        "sign_consistency": np.maximum((coefs > 0).mean(0), (coefs < 0).mean(0)),
        "zeroed_share": (coefs == 0).mean(0),
    })
    return out.sort_values("abs_coefficient_mean", ascending=False).reset_index(drop=True)


def oof_shap(fold_models, X: pd.DataFrame, kind: str) -> np.ndarray:
    import shap

    X = pd.DataFrame(X).reset_index(drop=True)
    out = np.full(X.shape, np.nan)
    for f in fold_models:
        model, tr, te = f["model"], f["train_idx"], f["test_idx"]
        if kind == "xgb":
            values = shap.TreeExplainer(model).shap_values(X.iloc[te])
        else:
            pre = model[:-1]
            background = pre.transform(X.iloc[tr])
            explainer = shap.LinearExplainer(model.named_steps["clf"], background)
            values = explainer.shap_values(pre.transform(X.iloc[te]))
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        out[te] = values
    return out


def shap_summary(matrix: np.ndarray, names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature": names, "mean_abs_shap": np.nanmean(np.abs(matrix), 0),
                       "mean_shap": np.nanmean(matrix, 0)})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def permutation_importance_oof(fold_models, X, y, n_repeats=N_PERM_IMPORTANCE) -> pd.DataFrame:
    """Drop in pooled out-of-fold AUC when one feature is shuffled across all rows."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(RANDOM_STATE)

    def pooled_auc(Xm):
        p = np.full(len(y), np.nan)
        for f in fold_models:
            p[f["test_idx"]] = f["model"].predict_proba(Xm.iloc[f["test_idx"]])[:, 1]
        return roc_auc_score(y, p)

    base = pooled_auc(X)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - pooled_auc(Xp))
        rows.append({"feature": col, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def ranking_agreement(shap_a, shap_b, groups, names, n_boot=N_BOOTSTRAP) -> dict:
    """Kendall tau between the two models' mean-|SHAP| rankings, with a participant bootstrap CI.

    A tau is undefined (NaN) when either ranking is constant. The observed tau is returned
    as NaN in that case; non-finite bootstrap replicates are dropped and counted via
    n_boot_valid; if none are valid, ci_low and ci_high are NaN.
    """
    groups = np.asarray(groups)

    def tau(idx):
        a = np.nanmean(np.abs(shap_a[idx]), 0)
        b = np.nanmean(np.abs(shap_b[idx]), 0)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return np.nan
        return kendalltau(a, b).statistic

    observed = tau(np.arange(len(groups)))
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    boots = [tau(np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    boots = np.array(boots, float)
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return {"kendall_tau": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "n_features": len(names), "n_boot_valid": int(len(boots))}


def xgboost_gain(fold_models, names: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=range(len(fold_models)), columns=names)
    for i, f in enumerate(fold_models):
        for k, v in f["model"].get_booster().get_score(importance_type="gain").items():
            if k in m.columns:
                m.loc[i, k] = float(v)
    out = pd.DataFrame({"feature": names, "gain_mean": m.mean().to_numpy(),
                        "gain_sd": m.std().to_numpy(), "folds_used": (m > 0).sum().to_numpy()})
    return out.sort_values("gain_mean", ascending=False).reset_index(drop=True)

"""Feature importance from both models, aggregated across the 26 LOPO folds.

Importance is read from the same fold models that produced the reported scores, not
from a separate model refitted on all the data. That keeps the explanation and the
performance claim describing the same object, and it lets us report how *stable* each
feature's importance is across folds -- a feature that dominates in three folds and
vanishes in the rest is not a finding.

SHAP values are computed out-of-fold: each fold's model explains only the participant
it never saw during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _lr_feature_names(pipeline, base_names: list[str]) -> list[str]:
    """Recover column names after SimpleImputer(add_indicator=True) widened the matrix.

    The imputer emits every original column first, then one binary indicator per
    column that contained a missing value in that fold's training data. Which columns
    those are can differ between folds, so names are rebuilt per fold.
    """
    imputer = pipeline.named_steps["impute"]
    names = list(base_names)
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        names += [f"{base_names[i]}__was_missing" for i in indicator.features_]
    return names


def logistic_coefficients(fold_models, base_names: list[str]) -> pd.DataFrame:
    """Signed standardised coefficients per feature, averaged over folds.

    Features are standardised inside the pipeline, so coefficients are directly
    comparable to one another. Sign is meaningful: positive means the feature pushes
    towards the high-load class.
    """
    collected: dict[str, list[float]] = {}
    for fold in fold_models:
        pipe = fold["model"]
        names = _lr_feature_names(pipe, base_names)
        coefs = pipe.named_steps["clf"].coef_.ravel()
        for name, value in zip(names, coefs):
            collected.setdefault(name, []).append(float(value))

    rows = []
    for name, values in collected.items():
        arr = np.array(values)
        rows.append(
            {
                "feature": name,
                "coefficient_mean": arr.mean(),
                "coefficient_sd": arr.std(),
                "abs_coefficient_mean": np.abs(arr).mean(),
                "n_folds_present": len(arr),
                "sign_consistency": float(max((arr > 0).mean(), (arr < 0).mean())),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_coefficient_mean", ascending=False)
    return out.reset_index(drop=True)


def xgboost_gain(fold_models, base_names: list[str]) -> pd.DataFrame:
    """Mean gain importance per feature across folds.

    Gain measures the average improvement in the split criterion a feature delivers
    when it is used, which is the importance type least distorted by how many times a
    feature happens to be selected.
    """
    matrix = pd.DataFrame(0.0, index=range(len(fold_models)), columns=base_names)
    for i, fold in enumerate(fold_models):
        booster = fold["model"].get_booster()
        scores = booster.get_score(importance_type="gain")
        for name, value in scores.items():
            if name in matrix.columns:
                matrix.loc[i, name] = float(value)

    out = pd.DataFrame(
        {
            "feature": base_names,
            "gain_mean": matrix.mean().to_numpy(),
            "gain_sd": matrix.std().to_numpy(),
            "folds_used": (matrix > 0).sum().to_numpy(),
        }
    )
    total = out["gain_mean"].sum()
    out["gain_share_pct"] = (100 * out["gain_mean"] / total).round(2) if total else 0.0
    return out.sort_values("gain_mean", ascending=False).reset_index(drop=True)


def out_of_fold_shap(fold_models, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean absolute SHAP value per feature, computed only on held-out participants.

    Returns the per-feature summary and the full out-of-fold SHAP matrix, the latter
    aligned row-for-row with X so it can be plotted as a beeswarm.
    """
    import shap

    X = pd.DataFrame(X).reset_index(drop=True)
    all_shap = np.full((len(X), X.shape[1]), np.nan)

    for fold in fold_models:
        model = fold["model"]
        idx = fold["test_idx"]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X.iloc[idx])
        if isinstance(values, list):          # older API returns one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (n, features, classes)
            values = values[:, :, -1]
        all_shap[idx, :] = values

    summary = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": np.nanmean(np.abs(all_shap), axis=0),
            "mean_shap": np.nanmean(all_shap, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary, all_shap

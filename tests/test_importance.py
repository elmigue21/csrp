import numpy as np
import pandas as pd

from evaluate import run_lopo
from importance import (
    logistic_coefficients, oof_shap, permutation_importance_oof, ranking_agreement,
    shap_summary, xgboost_gain,
)
from models import MODELS


def _fit(kind):
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(1, 11), 2)
    y = np.tile([0, 1], 10)
    X = pd.DataFrame({"good": y * 3 + rng.normal(0, 1, 20), "noise": rng.normal(0, 1, 20)})
    name = "Logistic Regression" if kind == "lr" else "XGBoost"
    res = run_lopo(X, y, g, MODELS[name]["factory"], None, tune=False)
    return X, y, g, res


def test_shap_both_models_rank_good_first():
    for kind in ("lr", "xgb"):
        X, y, g, res = _fit(kind)
        m = oof_shap(res["fold_models"], X, kind)
        assert m.shape == X.shape and not np.isnan(m).any()
        assert shap_summary(m, list(X.columns)).iloc[0]["feature"] == "good"


def test_coefficients_and_gain():
    X, y, g, res = _fit("lr")
    coef = logistic_coefficients(res["fold_models"], list(X.columns))
    assert coef.iloc[0]["feature"] == "good" and coef.iloc[0]["coefficient_mean"] > 0
    X, y, g, res = _fit("xgb")
    assert set(xgboost_gain(res["fold_models"], list(X.columns))["feature"]) == {"good", "noise"}


def test_permutation_importance_and_agreement():
    X, y, g, res = _fit("lr")
    pi = permutation_importance_oof(res["fold_models"], X, y, n_repeats=5)
    assert pi.iloc[0]["feature"] == "good" and pi.iloc[0]["auc_drop_mean"] > 0
    m = oof_shap(res["fold_models"], X, "lr")
    agree = ranking_agreement(m, m, g, list(X.columns), n_boot=50)
    assert agree["kendall_tau"] == 1.0


def test_ranking_agreement_degenerate():
    g = np.repeat(np.arange(1, 6), 2)
    a = np.zeros((10, 3))
    b = np.random.default_rng(0).normal(size=(10, 3))
    agree = ranking_agreement(a, b, g, ["x", "y", "z"], n_boot=20)
    assert "n_boot_valid" in agree and agree["n_boot_valid"] == 0
    assert np.isnan(agree["kendall_tau"])
    assert np.isnan(agree["ci_low"]) and np.isnan(agree["ci_high"])


def test_ranking_agreement_different_rankings_ci():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(1, 21), 5)
    scale_a = np.array([4.0, 3.0, 2.0, 1.0])
    scale_b = np.array([1.0, 3.0, 4.0, 2.0])
    a = rng.normal(size=(100, 4)) * scale_a
    b = rng.normal(size=(100, 4)) * scale_b
    agree = ranking_agreement(a, b, g, list("wxyz"), n_boot=200)
    assert agree["n_boot_valid"] > 0
    assert -1 <= agree["ci_low"] <= agree["kendall_tau"] <= agree["ci_high"] <= 1
    assert agree["kendall_tau"] < 1.0

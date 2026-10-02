import numpy as np
import pandas as pd

from evaluate import (
    bootstrap_auc, compare_models, chosen_params_table, majority_baseline, param_summary,
    participant_margins, permutation_test, run_lopo, score,
)
from models import MODELS


def _toy(n_participants=12, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(1, n_participants + 1), 2)
    y = np.tile([0, 1], n_participants)
    X = pd.DataFrame({"good": y * signal + rng.normal(0, 1, len(y)),
                      "noise": rng.normal(0, 1, len(y))})
    return X, y, groups


def test_lopo_both_models_learn_signal():
    X, y, g = _toy()
    for name, spec in MODELS.items():
        res = run_lopo(X, y, g, spec["factory"], spec["space"], n_iter=3)
        assert not np.isnan(res["oof_proba"]).any()
        assert len(res["fold_models"]) == 12
        assert score(y, res["oof_proba"])["roc_auc"] > 0.8, name


def test_score_keys_and_brier_range():
    s = score(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6]))
    assert set(s) == {"roc_auc", "balanced_accuracy", "macro_f1", "recall_low", "recall_high", "brier"}
    assert s["roc_auc"] == 1.0 and 0 <= s["brier"] <= 1


def test_margins_and_bootstrap():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.9, 0.6, 0.4])
    g = np.array([1, 1, 2, 2])
    m = participant_margins(y, p, g)
    assert np.allclose(m.loc[1], 0.7) and np.allclose(m.loc[2], -0.2)
    b = bootstrap_auc(y, p, g, n_boot=200, other=p)
    assert b["auc_ci_low"] <= b["auc_ci_high"] and b["diff"] == 0.0


def test_compare_and_baseline():
    X, y, g = _toy()
    r1 = run_lopo(X, y, g, MODELS["Logistic Regression"]["factory"], {}, tune=False)
    out = compare_models(y, r1["oof_proba"], r1["oof_proba"], g, n_boot=100)
    assert out["auc_difference"] == 0.0
    assert 0.0 <= majority_baseline(y, g)["balanced_accuracy"] <= 0.5


def test_permutation_test_same_result_serial_and_parallel():
    X, y, g = _toy(signal=1.0)
    factory = MODELS["Logistic Regression"]["factory"]
    serial = permutation_test(X, y, g, factory, {}, n_perm=9, n_jobs=1)
    parallel = permutation_test(X, y, g, factory, {}, n_perm=9, n_jobs=2)
    assert serial == parallel


def test_permutation_test_detects_signal():
    X, y, g = _toy(signal=3.0)
    out = permutation_test(X, y, g, MODELS["Logistic Regression"]["factory"], {}, n_perm=19)
    assert out["p_value"] <= 0.1


def test_param_summary_and_table():
    chosen = [{"max_depth": 1, "C": 1.0}, {"max_depth": 2, "C": 2.0},
              {"max_depth": 2, "C": 3.0}, {"max_depth": 3, "C": 4.0}]
    s = param_summary(chosen)
    assert s["param_C_median"] == 2.5 and np.isclose(s["param_C_iqr"], 1.5)
    assert s["param_max_depth_mode"] == 2.0 and s["param_max_depth_median"] == 2.0
    assert "param_max_depth_mode" not in param_summary([{"C": 1.0}])
    t = chosen_params_table(chosen, [11, 12, 13, 14])
    assert t.columns[0] == "participant" and t["participant"].tolist() == [11, 12, 13, 14]
    assert len(t) == 4 and set(t.columns) == {"participant", "max_depth", "C"}

import json

import numpy as np
import pandas as pd
import pytest

from run_model import main

N_PEOPLE = 8


def _features_csv(path, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, N_PEOPLE + 1):
        for act, label in ((1, "low"), (4, "high")):
            hi = act == 4
            rows.append({"participant": p, "activity": act, "label": label,
                         "pupil__mean": 3.0 + hi * 0.8 + rng.normal(scale=0.2),
                         "blink_rate": 5.0 + hi * 6.0 + rng.normal(scale=1.0),
                         **{f"noise_{i}": rng.normal() for i in range(6)}})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def modeled(tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    feats = _features_csv(root / "features.csv")
    out = main(features_path=feats, results_dir=root / "results", n_iter=3, n_boot=50, n_perm=5)
    return root / "results", out


def test_results_files_written(modeled):
    results, out = modeled
    for name in ("results.csv", "results_by_n.csv", "checks.csv",
                 "comparison.csv", "predictions.csv", "selection_count.csv",
                 "shap_logistic_regression.csv", "shap_xgboost.csv",
                 "chosen_params_logistic_regression.csv", "chosen_params_xgboost.csv",
                 "run_info.json", "README.md"):
        assert (results / name).exists(), name
    res = pd.read_csv(results / "results.csv").set_index("model")
    assert set(res.index) == {"Logistic Regression", "XGBoost"}
    for col in ("roc_auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy",
                "precision_macro", "precision_low", "precision_high", "recall_low", "recall_high",
                "macro_f1", "f1_low", "f1_high", "brier", "a4_above_a1", "permutation_p",
                "n_features_median"):
        assert col in res, col
    assert (res["roc_auc"] > 0.8).all()            # the synthetic signal is strong
    assert out["n_participants"] == N_PEOPLE


def test_predictions_are_out_of_fold_one_per_row(modeled):
    results, _ = modeled
    pred = pd.read_csv(results / "predictions.csv")
    assert len(pred) == 2 * N_PEOPLE and pred[["p_lr", "p_xgb"]].notna().all().all()


def test_selection_count_and_shap(modeled):
    results, _ = modeled
    sel = pd.read_csv(results / "selection_count.csv").set_index("feature")
    assert len(sel) == 8
    for col in ("folds_selected_lr", "folds_selected_xgb"):
        assert sel[col].between(0, N_PEOPLE).all()
        assert sel.loc["blink_rate", col] == N_PEOPLE   # the strongest feature is always picked
    shap = pd.read_csv(results / "shap_xgboost.csv")
    assert list(shap.columns[:3]) == ["feature", "mean_abs_shap", "mean_shap"]
    assert len(shap) == 8 and shap["mean_abs_shap"].ge(0).all()
    chosen = pd.read_csv(results / "chosen_params_xgboost.csv")
    assert len(chosen) == N_PEOPLE and "select__n" in chosen


def test_run_info_records_the_features_file(modeled):
    results, _ = modeled
    info = json.loads((results / "run_info.json").read_text())
    assert len(info["features_sha256"]) == 64 and info["n_features"] == 8
    assert info["n_iter"] == 3 and info["decisions"].startswith("docs/post-defense-modeling-plan.md")


def test_results_by_n_reports_every_n_for_both_models(modeled):
    results, _ = modeled
    by_n = pd.read_csv(results / "results_by_n.csv")
    assert set(by_n["model"]) == {"Logistic Regression", "XGBoost"}
    assert sorted(by_n["n_features"].unique()) == [5, 10, 20, 40, 80, 140]
    assert len(by_n) == 12
    for col in ("roc_auc", "auc_ci_low", "accuracy", "precision_macro", "recall_low", "recall_high",
                "macro_f1", "brier", "a4_above_a1", "n_features_used"):
        assert by_n[col].notna().all(), col
    # 8 features in the synthetic table: N of 10 or more keeps all of them
    assert (by_n.loc[by_n["n_features"] >= 10, "n_features_used"] == 8).all()


def test_checks_blink_alone_and_without_blink(modeled):
    results, _ = modeled
    checks = pd.read_csv(results / "checks.csv")
    assert list(checks["check"]) == ["blink_rate alone", "without blink_rate", "without blink_rate"]
    assert list(checks["model"]) == ["Logistic Regression (1 feature)", "Logistic Regression", "XGBoost"]
    for col in ("roc_auc", "accuracy", "precision_macro", "macro_f1", "permutation_p"):
        assert checks[col].notna().all(), col
    assert checks.loc[0, "n_features_available"] == 1 and checks.loc[1, "n_features_available"] == 7


def test_interrupted_stages_resume_without_repeating_finished_runs(tmp_path, monkeypatch):
    import run_model
    feats = _features_csv(tmp_path / "features.csv")
    results = tmp_path / "results"
    real, calls = run_model.run_lopo, []

    def counting(*args, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(run_model, "run_lopo", counting)
    with pytest.raises(RuntimeError):
        run_model.main(features_path=feats, results_dir=results, n_iter=2, n_boot=20, n_perm=3,
                       stages=("by_n",))
    assert len(pd.read_csv(results / "results_by_n.csv")) == 4      # saved before the crash
    calls.clear()
    monkeypatch.setattr(run_model, "run_lopo", real)
    out = run_model.main(features_path=feats, results_dir=results, n_iter=2, n_boot=20, n_perm=3,
                         stages=("by_n",))
    assert len(out["by_n"]) == 12 and out["by_n"]["features_sha256"].nunique() == 1
    info = json.loads((results / "run_info.json").read_text())
    assert "by_n" in info["stages"] and "main" not in info["stages"]
    assert not (results / "results.csv").exists()                  # main stage was not run

import json

import pandas as pd

from config import FEATURES
from conftest import write_synthetic_dataset
from run_colet import main


def test_end_to_end_on_synthetic_data(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", n_participants=8, seconds=5.0)
    out = main(data_dir=data, out_dir=tmp_path / "out", n_iter=2, n_boot=50, n_perm=3, n_perm_imp=2)
    tables = tmp_path / "out" / "tables"
    res = pd.read_csv(tables / "05_results.csv")
    assert set(res["analysis"]) == {"P1", "P2"}
    assert set(res["model"]) == {"Logistic Regression", "XGBoost"}
    assert (tables / "07_importance_P1_shap_xgb.csv").exists()
    assert "chosen_params_modal" not in res and "param_max_depth_mode" in res
    assert {"param_clf__C_median", "param_clf__C_iqr"} <= set(res)
    params = pd.read_csv(tables / "09_chosen_params_P2_xgboost.csv")
    assert len(params) == 8 and "participant" in params and "max_depth" in params
    miss = pd.read_csv(tables / "10_missingness.csv")
    assert set(miss["feature"]) == set(FEATURES) and set(miss["activity"]) == {1, 2, 3, 4}
    assert len(pd.read_csv(tables / "01b_activities_per_participant.csv")) == 8
    comp = pd.read_csv(tables / "06_comparison.csv").set_index("analysis")
    assert "ceiling_rule_triggered" in comp and pd.isna(comp.loc["P2", "ceiling_rule_triggered"])
    assert "tau is" in comp.loc["P2", "shap_rank_note"] and pd.isna(comp.loc["P1", "shap_rank_note"])
    assert set(out["analyses"]) == {"P1", "P2"}
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["n_participants"] == 8 and "versions" in meta
    assert out["features_used"] in (meta["features_p1"],)

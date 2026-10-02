"""Run the COLET study end to end and write every table Chapter 4 needs.

Stages (each also callable on its own from the Colab notebook):
  1. prepare        -- feature table, exclusion, retention, sanity check (decides 10 vs 5 features)
  2. run_analysis   -- P1 (10 or 5 features) and P2 (pupil only), LR vs XGBoost under LOPO
  3. outputs        -- results, comparison, importance, chosen hyperparameters, missingness,
                       V2-13 check, run metadata (each analysis's tables are written as soon
                       as it finishes, so an interrupted run keeps the finished analyses)

Usage (local):  python src/run_colet.py [--refresh]
Do not run on the real data until the team has approved it.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import (
    manipulation_check, missingness_report, redundancy_check, sanity_check, sanity_passed,
)
from config import (
    CACHE_DIR, CEILING_AUC, DATA_DIR, FALLBACK_FEATURES, FEATURES, N_BOOTSTRAP, N_ITER,
    N_PERM_IMPORTANCE, N_PERMUTATIONS, OUT_DIR, PUPIL_FEATURES,
)
from data import load_annotation
from dataset import (
    activities_per_participant, analysis_participants, analysis_rows, build_feature_table,
    retention_summary,
)
from evaluate import (
    bootstrap_auc, chosen_params_table, compare_models, majority_baseline, param_summary,
    participant_margins, permutation_test, run_lopo, score,
)
from importance import (
    logistic_coefficients, oof_shap, permutation_importance_oof, ranking_agreement,
    shap_summary, xgboost_gain,
)
from labels import binary_label, main_rows
from models import MODELS
from normalize import per_participant_zscore


def prepare(data_dir=DATA_DIR, cache_dir=CACHE_DIR, refresh=False) -> dict:
    table = build_feature_table(data_dir, cache_dir, refresh)
    rows = analysis_rows(table)
    sanity = sanity_check(rows)
    return {"table": table, "rows": rows, "participants": analysis_participants(table),
            "sanity": sanity, "retention": retention_summary(table),
            "features_used": FEATURES if sanity_passed(sanity) else FALLBACK_FEATURES}


def run_analysis(name, features, rows, n_iter=N_ITER, n_boot=N_BOOTSTRAP,
                 n_perm=N_PERMUTATIONS, n_perm_imp=N_PERM_IMPORTANCE) -> dict:
    z = per_participant_zscore(rows[features], rows["participant"])
    main_df = main_rows(rows.assign(**{c: z[c] for c in features}))
    X = main_df[features].reset_index(drop=True)
    y = binary_label(main_df)
    g = main_df["participant"].to_numpy()

    metrics, probas, fits, chosen = [], {}, {}, {}
    base = majority_baseline(y, g)
    for model_name, spec in MODELS.items():
        res = run_lopo(X, y, g, spec["factory"], spec["space"], n_iter=n_iter)
        p = res["oof_proba"]
        probas[model_name], fits[model_name] = p, res
        chosen[model_name] = chosen_params_table(
            res["chosen_params"], [f["participant"] for f in res["fold_models"]])
        row = {"analysis": name, "model": model_name, "n_samples": len(y),
               "n_features": len(features), **score(y, p), **bootstrap_auc(y, p, g, n_boot),
               "share_a4_above_a1": float((participant_margins(y, p, g) > 0).mean()),
               "majority_balanced_accuracy": base["balanced_accuracy"],
               **param_summary(res["chosen_params"])}
        # default hyperparameters: tuned ones were chosen with the true labels and would bias p downward
        row.update({f"perm_{k}": v for k, v in permutation_test(
            X, y, g, spec["factory"], {}, n_perm).items()})
        metrics.append(row)

    lr, xgb = fits["Logistic Regression"], fits["XGBoost"]
    shap_lr = oof_shap(lr["fold_models"], X, "lr")
    shap_xgb = oof_shap(xgb["fold_models"], X, "xgb")
    importance = {
        "coef_lr": logistic_coefficients(lr["fold_models"], features),
        "shap_lr": shap_summary(shap_lr, features),
        "shap_xgb": shap_summary(shap_xgb, features),
        "perm_lr": permutation_importance_oof(lr["fold_models"], X, y, n_perm_imp),
        "perm_xgb": permutation_importance_oof(xgb["fold_models"], X, y, n_perm_imp),
        "gain_xgb_appendix": xgboost_gain(xgb["fold_models"], features),
    }
    comparison = {"analysis": name,
                  **compare_models(y, probas["Logistic Regression"], probas["XGBoost"], g, n_boot),
                  **{f"shap_rank_{k}": v for k, v in ranking_agreement(shap_lr, shap_xgb, g, features, n_boot).items()}}
    if len(features) == 2:      # Kendall tau over two features can only be +1 or -1
        comparison["shap_rank_note"] = "2 features: tau is \u00b11 only"
    predictions = pd.DataFrame({"participant": g, "activity": main_df["activity"].to_numpy(), "y": y,
                                "p_lr": probas["Logistic Regression"], "p_xgb": probas["XGBoost"]})
    return {"metrics": pd.DataFrame(metrics), "comparison": comparison,
            "importance": importance, "predictions": predictions,
            "chosen_params": chosen}


def _versions() -> dict:
    import scipy, shap, sklearn, xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "shap": shap.__version__}


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, refresh=False, n_iter=N_ITER, n_boot=N_BOOTSTRAP,
         n_perm=N_PERMUTATIONS, n_perm_imp=N_PERM_IMPORTANCE) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    prep = prepare(data_dir, out_dir / "cache", refresh)
    prep["retention"].to_csv(tables / "01_retention.csv", index=False)
    activities_per_participant(prep["table"]).to_csv(tables / "01b_activities_per_participant.csv", index=False)
    prep["sanity"].to_csv(tables / "02_sanity_check.csv", index=False)
    manip = manipulation_check(load_annotation(data_dir), prep["participants"])
    (tables / "03_manipulation_check.json").write_text(json.dumps(manip, indent=2))
    prep["table"].to_csv(tables / "04_feature_table.csv", index=False)
    redundancy_check(prep["rows"]).to_csv(tables / "08_redundancy_v2_13.csv", index=False)
    missingness_report(prep["rows"]).to_csv(tables / "10_missingness.csv", index=False)

    analyses, results, comparisons = {}, [], []
    for name, feats in (("P1", prep["features_used"]), ("P2", PUPIL_FEATURES)):
        out = run_analysis(name, feats, prep["rows"], n_iter, n_boot, n_perm, n_perm_imp)
        if name == "P1":
            out["comparison"]["ceiling_rule_triggered"] = bool((out["metrics"]["roc_auc"] > CEILING_AUC).all())
        analyses[name] = out
        results.append(out["metrics"])
        comparisons.append(out["comparison"])
        for kind, df in out["importance"].items():
            df.to_csv(tables / f"07_importance_{name}_{kind}.csv", index=False)
        for model_name, df in out["chosen_params"].items():
            df.to_csv(tables / f"09_chosen_params_{name}_{model_name.lower().replace(' ', '_')}.csv", index=False)
        out["predictions"].to_csv(tables / f"predictions_{name}.csv", index=False)
        # rewritten after each analysis: a disconnect during P2 keeps P1's results
        results_df = pd.concat(results, ignore_index=True)
        results_df.to_csv(tables / "05_results.csv", index=False)
        pd.DataFrame(comparisons).to_csv(tables / "06_comparison.csv", index=False)

    p1 = results_df[results_df.analysis == "P1"]
    meta = {
        "n_participants": len(prep["participants"]),
        "features_p1": prep["features_used"],
        "sanity_passed": sanity_passed(prep["sanity"]),
        "ceiling_rule_triggered": bool((p1["roc_auc"] > CEILING_AUC).all()),
        "settings": {"n_iter": n_iter, "n_boot": n_boot, "n_perm": n_perm, "n_perm_imp": n_perm_imp},
        "versions": _versions(),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return {**meta, "features_used": prep["features_used"], "analyses": analyses}


if __name__ == "__main__":
    summary = main(refresh="--refresh" in sys.argv)
    print(json.dumps({k: v for k, v in summary.items() if k != "analyses"}, indent=2))

"""Run the full comparison and write every table Chapter 4 needs.

Six configurations are evaluated, crossing the two labellings with the two feature
schemes, plus a no-lighting ablation for each labelling. Configuration A is the
headline: the label as it comes off the questionnaire (rating >= 4) and the features
as they come out of the dataset.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LUX_FEATURES, TABLE_DIR, OUT_DIR, EPOCH_SECONDS, ABSOLUTE_THRESHOLD
from data import feature_names, load_or_build_epochs, load_windows, recording_summary
from labels import DROP, build_labels, difficulty_agreement, rating_spread
from normalize import apply_feature_scheme
from models import MODELS
from evaluate import (
    compare_models,
    majority_baseline,
    run_lopo,
    score_predictions,
    summarise_per_participant,
)
from importance import logistic_coefficients, out_of_fold_shap, xgboost_gain


@dataclass(frozen=True)
class Config:
    key: str
    label_scheme: str
    feature_scheme: str
    include_lux: bool
    description: str


CONFIGS = [
    Config("A", "absolute", "raw", True,
           "as-is label (rating >= 4) + raw features  [HEADLINE]"),
    Config("B", "absolute", "raw", False,
           "as-is label + raw features, lighting features removed"),
    Config("C", "absolute", "within_person", True,
           "as-is label + per-participant standardised features"),
    Config("D", "within_person", "raw", True,
           "per-participant label + raw features"),
    Config("E", "within_person", "within_person", True,
           "per-participant label + per-participant standardised features"),
    Config("F", "within_person", "within_person", False,
           "per-participant label + per-participant features, lighting removed"),
]


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_per_").lower()


PRED_DIR = OUT_DIR / "predictions"


def _load_cached(cfg_key: str, model_name: str, y, groups) -> dict | None:
    """Rebuild a completed (config, model) result from its saved artifacts.

    A full run takes several minutes per gradient-boosting configuration, so an
    interrupted run should not have to repeat the folds it already finished. The
    saved held-out predictions are sufficient to recompute every reported metric, so
    a cached result is the same numbers rather than an approximation of them.

    Returns None -- meaning "run it" -- unless the cached predictions line up exactly
    with the labels and grouping this configuration produces now. `fold_models` is
    absent from a cached result, so feature importance is not re-derivable from it.
    """
    pred_path = PRED_DIR / f"oof_{cfg_key}_{_safe(model_name)}.parquet"
    pp_path = TABLE_DIR / f"05_per_participant_{cfg_key}_{_safe(model_name)}.csv"
    if not (pred_path.exists() and pp_path.exists()):
        return None

    saved = pd.read_parquet(pred_path)
    if len(saved) != len(y):
        return None
    if not np.array_equal(saved["y_true"].to_numpy(), np.asarray(y)):
        return None
    if not np.array_equal(saved["participant"].to_numpy(), np.asarray(groups)):
        return None

    proba = saved["y_proba"].to_numpy()
    pred = saved["y_pred"].to_numpy()
    return {
        "oof_proba": proba,
        "oof_pred": pred,
        "fold_models": None,
        "chosen_params": [],
        "pooled": score_predictions(np.asarray(y), pred, proba),
        "per_participant": pd.read_csv(pp_path),
    }


def main(refresh: bool = False, force: bool = False) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------ data
    print("Loading windows and building epochs...", flush=True)
    windows = load_windows()
    summary = recording_summary(windows)
    summary.to_csv(TABLE_DIR / "01_recording_summary.csv", index=False)

    epochs = load_or_build_epochs(refresh=refresh)
    feats_all = feature_names(epochs)
    print(f"  {len(windows):,} windows -> {len(epochs):,} epochs of {EPOCH_SECONDS}s"
          f"  ({len(feats_all)} features, {epochs.participant.nunique()} participants)")
    print(f"  recordings with timing gaps: {int((summary.n_gaps > 0).sum())} / {len(summary)}")
    print(f"  ratings per recording (should be 1): "
          f"{sorted(summary.n_ratings_in_file.unique())}")

    spread = rating_spread(epochs)
    spread.to_csv(TABLE_DIR / "02_rating_spread_per_participant.csv", index=False)
    print(f"  participants with rating range < 3: "
          f"{spread.loc[spread.rating_range < 3, 'participant'].tolist()}")

    # ------------------------------------------------- label sanity checks
    checks = {}
    for scheme in ["absolute", "within_person"]:
        y = build_labels(epochs, scheme)
        agree = difficulty_agreement(epochs, y)
        agree["by_task"].to_csv(
            TABLE_DIR / f"03_difficulty_agreement_{scheme}.csv", index=False)
        checks[scheme] = {"spearman_rho": agree["spearman_rho"],
                          "p_value": agree["p_value"],
                          "n_epochs_kept": int((y != DROP).sum()),
                          "n_epochs_dropped": int((y == DROP).sum())}
        print(f"  [{scheme}] label vs task difficulty rho={agree['spearman_rho']:.3f} "
              f"(p={agree['p_value']:.2g}), epochs kept {checks[scheme]['n_epochs_kept']}")

    # ------------------------------------------------------------- configs
    results, per_participant_store, comparisons = [], {}, []

    for cfg in CONFIGS:
        y_full = build_labels(epochs, cfg.label_scheme)
        keep = (y_full != DROP).to_numpy()

        cols = [c for c in feats_all if cfg.include_lux or c not in LUX_FEATURES]
        X_full = apply_feature_scheme(epochs[cols], epochs["participant"], cfg.feature_scheme)

        X = X_full[keep].reset_index(drop=True)
        y = y_full[keep].to_numpy()
        groups = epochs.loc[keep, "participant"].to_numpy()

        base = majority_baseline(y, groups)
        print(f"\n=== Config {cfg.key}: {cfg.description}")
        print(f"    {len(y)} epochs, {len(cols)} features, positive rate {y.mean():.3f}, "
              f"majority-baseline accuracy {base['accuracy']:.3f}")

        for model_name, spec in MODELS.items():
            t = time.time()
            cached = _load_cached(cfg.key, model_name, y, groups)
            if cached is not None and not force:
                res, reused = cached, True
            else:
                reused = False
                try:
                    res = run_lopo(X, y, groups, spec["factory"], spec["grid"])
                except Exception:
                    import traceback
                    print(f"    {model_name}: FAILED", flush=True)
                    traceback.print_exc()
                    continue
            summ = summarise_per_participant(res["per_participant"])

            row = {"config": cfg.key, "description": cfg.description,
                   "label_scheme": cfg.label_scheme, "feature_scheme": cfg.feature_scheme,
                   "include_lux": cfg.include_lux, "model": model_name,
                   "n_epochs": len(y), "n_features": len(cols),
                   "positive_rate": round(float(y.mean()), 3),
                   "majority_accuracy": round(base["accuracy"], 3)}
            row.update({f"pooled_{k}": round(v, 4) for k, v in res["pooled"].items()})
            row.update({k: round(v, 4) for k, v in summ.items()})
            results.append(row)
            per_participant_store[(cfg.key, model_name)] = res["per_participant"]

            if not reused:
                res["per_participant"].to_csv(
                    TABLE_DIR / f"05_per_participant_{cfg.key}_{_safe(model_name)}.csv",
                    index=False)

                # Held-out predictions, kept so ROC curves and confusion matrices are
                # drawn from the same numbers the metrics table reports.
                pd.DataFrame(
                    {"participant": groups, "task": epochs.loc[keep, "task"].to_numpy(),
                     "y_true": y, "y_proba": res["oof_proba"], "y_pred": res["oof_pred"]}
                ).to_parquet(
                    PRED_DIR / f"oof_{cfg.key}_{_safe(model_name)}.parquet", index=False)

            print(f"    {model_name:<20} pooled AUC={res['pooled']['roc_auc']:.3f} "
                  f"acc={res['pooled']['accuracy']:.3f} F1={res['pooled']['f1']:.3f} "
                  f"| per-participant AUC {summ['roc_auc_mean']:.3f}"
                  f"+/-{summ['roc_auc_sd']:.3f} (n={summ['roc_auc_n']})"
                  f"  [{'reused' if reused else f'{time.time() - t:.0f}s'}]")

            # Importance is only worth extracting where the model is doing something.
            # A reused result carries no fold models, so its importance tables are
            # whatever the run that produced them already wrote.
            if cfg.key in {"A", "E"} and res["fold_models"] is not None:
                if model_name == "Logistic Regression":
                    logistic_coefficients(res["fold_models"], cols).to_csv(
                        TABLE_DIR / f"06_importance_lr_coefficients_{cfg.key}.csv",
                        index=False)
                else:
                    xgboost_gain(res["fold_models"], cols).to_csv(
                        TABLE_DIR / f"06_importance_xgb_gain_{cfg.key}.csv", index=False)
                    shap_summary, shap_matrix = out_of_fold_shap(res["fold_models"], X)
                    shap_summary.to_csv(
                        TABLE_DIR / f"06_importance_xgb_shap_{cfg.key}.csv", index=False)
                    np.save(OUT_DIR / f"shap_matrix_{cfg.key}.npy", shap_matrix)
                    X.to_parquet(OUT_DIR / f"shap_features_{cfg.key}.parquet", index=False)
                    pd.DataFrame({"y": y, "participant": groups}).to_parquet(
                        OUT_DIR / f"shap_labels_{cfg.key}.parquet", index=False)

        cmp_auc = compare_models(per_participant_store[(cfg.key, "Logistic Regression")],
                                per_participant_store[(cfg.key, "XGBoost")], "roc_auc")
        cmp_acc = compare_models(per_participant_store[(cfg.key, "Logistic Regression")],
                                 per_participant_store[(cfg.key, "XGBoost")], "accuracy")
        comparisons.append({"config": cfg.key, "metric": "roc_auc", **cmp_auc})
        comparisons.append({"config": cfg.key, "metric": "accuracy", **cmp_acc})
        print(f"    XGBoost - LR: dAUC={cmp_auc['mean_difference']:+.3f} "
              f"(Wilcoxon p={cmp_auc['p_value']:.3g}, n={cmp_auc['n']})")

    # -------------------------------------------------------------- outputs
    main_table = pd.DataFrame(results)
    main_table.to_csv(TABLE_DIR / "04_main_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(TABLE_DIR / "07_model_comparison_tests.csv", index=False)

    meta = {
        "epoch_seconds": EPOCH_SECONDS,
        "absolute_threshold": ABSOLUTE_THRESHOLD,
        "n_epochs_total": int(len(epochs)),
        "n_features": len(feats_all),
        "n_participants": int(epochs.participant.nunique()),
        "label_checks": checks,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 100)
    print("POOLED RESULTS (leave-one-participant-out)")
    print("=" * 100)
    show = main_table[["config", "model", "n_epochs", "positive_rate", "majority_accuracy",
                       "pooled_accuracy", "pooled_precision", "pooled_recall",
                       "pooled_f1", "pooled_roc_auc"]]
    print(show.to_string(index=False))
    print(f"\nTables written to {TABLE_DIR}")
    print(f"Total runtime {meta['runtime_seconds']}s")


if __name__ == "__main__":
    # --refresh rebuilds the epoch cache from the raw CSVs.
    # --force refits every model even where saved held-out predictions already exist.
    main(refresh="--refresh" in sys.argv, force="--force" in sys.argv)

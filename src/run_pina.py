"""Pina et al. 2025 run: LR vs XGBoost vs a one-feature baseline under LOPO, exactly as fixed in
docs/pina-analysis-plan.md (written before the eye data were loaded).

Stages (each saves its files as soon as it finishes; run_info.json records when each last ran):
- extract      : 30 s windows -> features/features.csv (the 17-statistic split, DROPPED removed)
- manipulation : raw NASA-TLX, combined task vs visual task alone (Wilcoxon)
- main         : per search area (90 = main, 360 = separate check): the fixation sanity check
                 (fallback drops gaze speed, fixation and saccade features), both tuned models and
                 the in-fold one-feature baseline, every metric, model differences; for 90 also
                 SHAP shares, selection counts and the refit without the signal with the largest
                 SHAP share
- permutation  : 90 only; labels swapped within each person, hyperparameters fixed at each model's
                 most-chosen settings (chosen with the true labels), selection rerun inside
- sensitivity  : 90 only, decided after the main results: the event features kept despite the
                 failed check. Never the main result.

Usage:  PINA_DATA_DIR=<folder with gaze/, trial/, tlx/>  python src/run_pina.py [--stages a,b]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    EVENT_PREFIXES, MAX_INVALID_FRACTION, N_BOOTSTRAP, N_ITER, N_PERMUTATIONS, PINA_DATA_DIR,
    PINA_DIR, PINA_MAIN_AREA, RANDOM_STATE, SANITY_FIXATION_MEDIAN_MS,
)
from evaluate import chosen_params_table, compare_models, permutation_test, run_lopo, score
from importance import shap_summary
from pina import META, demographics, participant_windows, participants, tlx_table
from run_model import FILE, SHORT, _git_commit, _metrics, _versions, fold_shap, selection_count
from topn import MODELS_TOPN, TopNByAUC

STAGES = ("extract", "manipulation", "main", "permutation", "sensitivity", "metric_ci")
BASELINE = "One feature (chosen in-fold)"
AREAS = (PINA_MAIN_AREA, "360")


def one_feature_baseline():
    """The single best feature of the training people (|AUC - 0.5|) -> plain LR, no tuning."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("select", TopNByAUC(n=1)),
                     ("scale", StandardScaler()), ("clf", LogisticRegression(random_state=RANDOM_STATE))])


def signal(feature: str) -> str:
    """Signal family of a feature name (pupil, gaze_x, gaze_y, blink, fixation, saccade, ...)."""
    head = feature.split("__")[0]
    named = {"blink_rate": "blink", "fixation_rate": "fixation", "fixation_duration": "fixation",
             "saccade_rate": "saccade"}
    return named.get(head, "saccade" if head.startswith("saccade") else head)


def area_data(table: pd.DataFrame, area: str, keep_events: bool = False):
    """(X, y, groups, ids, check): one area's windows after the planned exclusions, sorted so each
    person's low windows come before their high ones (the permutation test reverses that order)."""
    d = table[table.area.astype(str) == area]
    n_errors = int(d["error"].notna().sum()) if "error" in d else 0
    d = d[d["error"].isna()] if "error" in d else d
    n_before = len(d)
    d = d[d.invalid_frac <= MAX_INVALID_FRACTION]
    d = d.assign(_o=d.load.eq("high")).sort_values(["participant", "_o", "window"]).reset_index(drop=True)
    fix_median = float(d.fixation_median_ms.median())
    passed = SANITY_FIXATION_MEDIAN_MS[0] <= fix_median <= SANITY_FIXATION_MEDIAN_MS[1]
    drop_events = not passed and not keep_events
    names = [c for c in d.columns if c not in META and c != "_o"
             and not (drop_events and c.startswith(EVENT_PREFIXES))]
    check = {"area": area, "n_participants": int(d.participant.nunique()), "n_windows": len(d),
             "n_excluded_invalid": n_before - len(d), "n_extraction_errors": n_errors,
             "fixation_median_ms": fix_median, "sanity_check_passed": passed,
             "event_features_dropped": drop_events, "n_features": len(names)}
    ids = d[["participant", "area", "load", "window"]]
    return d[names], d.load.eq("high").astype(int).to_numpy(), d.participant.to_numpy(), ids, check


def _fit_all(X, y, g, n_iter):
    fits = {m: run_lopo(X, y, g, s["factory"], s["space"], n_iter=n_iter) for m, s in MODELS_TOPN.items()}
    fits[BASELINE] = run_lopo(X, y, g, one_feature_baseline, None, tune=False)
    return fits


def _results(fits, X, y, g, n_boot) -> pd.DataFrame:
    rows = []
    names = list(X.columns)
    for model, res in fits.items():
        if model == BASELINE:
            picked = Counter(n for f in res["fold_models"] for n, k in zip(
                names, f["model"].named_steps["select"].get_support()) if k)
            note = ", ".join(f"{k} x{v}" for k, v in picked.most_common(3))
            n_med = 1.0
        else:
            note = ""
            n_med = float(np.median([min(p.get("select__n", len(names)), len(names))
                                     for p in res["chosen_params"]]))
        rows.append({"model": model, "n_windows": len(y), "n_participants": len(np.unique(g)),
                     "n_features_available": len(names), "n_features_median": n_med,
                     "baseline_feature": note, **_metrics(y, res["oof_proba"], g, n_boot)})
    return pd.DataFrame(rows)


def _comparisons(fits, y, g, n_boot) -> pd.DataFrame:
    pairs = [(BASELINE, "Logistic Regression"), (BASELINE, "XGBoost"), ("Logistic Regression", "XGBoost")]
    rows = []
    for a, b in pairs:
        c = compare_models(y, fits[a]["oof_proba"], fits[b]["oof_proba"], g, n_boot)
        rows.append({"comparison": f"{b} minus {a}", "auc_difference": c["auc_difference"],
                     "ci_low": c["auc_difference_ci_low"], "ci_high": c["auc_difference_ci_high"],
                     "ci_above_0": c["auc_difference_ci_low"] > 0})
    return pd.DataFrame(rows)


def _importance(fits, X) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP share per feature (with selection counts) and per signal, for both tuned models."""
    names = list(X.columns)
    per_feature, per_signal = [], []
    sel = selection_count({m: fits[m] for m in MODELS_TOPN}, names).set_index("feature")
    for model in MODELS_TOPN:
        s = shap_summary(fold_shap(fits[model]["fold_models"], X, SHORT[model]), names)
        s["share"] = s.mean_abs_shap / s.mean_abs_shap.sum()
        s["folds_selected"] = s.feature.map(sel[f"folds_selected_{SHORT[model]}"])
        s["signal"] = s.feature.map(signal)
        per_feature.append(s.assign(model=model))
        fam = s.groupby("signal")["share"].sum().sort_values(ascending=False)
        per_signal.append(pd.DataFrame({"model": model, "signal": fam.index, "share": fam.to_numpy()}))
    return pd.concat(per_feature, ignore_index=True), pd.concat(per_signal, ignore_index=True)


def _save_area(out_dir, tag, fits, ids, y, results, comparisons, check):
    results.to_csv(out_dir / f"results_{tag}.csv", index=False)
    comparisons.to_csv(out_dir / f"comparison_{tag}.csv", index=False)
    pd.DataFrame([check]).to_csv(out_dir / f"data_check_{tag}.csv", index=False)
    ids.assign(y=y, **{f"p_{SHORT.get(m, 'one_feature')}": r["oof_proba"] for m, r in fits.items()}).to_csv(
        out_dir / f"predictions_{tag}.csv", index=False)
    for model in MODELS_TOPN:
        chosen_params_table(fits[model]["chosen_params"], [f["participant"] for f in fits[model]["fold_models"]]
                            ).to_csv(out_dir / f"chosen_params_{tag}_{FILE[model]}.csv", index=False)


CI_METRICS = ("accuracy", "precision_macro", "recall_low", "recall_high", "macro_f1", "brier", "roc_auc")
PRED_COLUMNS = {"Logistic Regression": "p_lr", "XGBoost": "p_xgb", BASELINE: "p_one_feature"}


def metric_intervals(pred: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-bootstrap 95% CIs for every threshold metric (0.5 cut-off) of each model, and
    for the paired differences model minus one-feature baseline and XGBoost minus LR. The same
    resampled participants are used for every model, so the differences are paired."""
    y, g = pred["y"].to_numpy(), pred["participant"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    people = np.unique(g)
    idx_by = {p: np.flatnonzero(g == p) for p in people}
    full = {m: score(y, pred[c].to_numpy()) for m, c in PRED_COLUMNS.items()}
    boots = {m: [] for m in PRED_COLUMNS}
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[p] for p in rng.choice(people, len(people), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        for m, c in PRED_COLUMNS.items():
            boots[m].append(score(y[idx], pred[c].to_numpy()[idx]))
    boots = {m: pd.DataFrame(b) for m, b in boots.items()}
    rows = [{"model": m, "metric": k, "value": full[m][k], "ci_low": boots[m][k].quantile(0.025),
             "ci_high": boots[m][k].quantile(0.975)} for m in PRED_COLUMNS for k in CI_METRICS]
    pairs = [("Logistic Regression", BASELINE), ("XGBoost", BASELINE), ("XGBoost", "Logistic Regression")]
    diffs = [{"comparison": f"{a} minus {b}", "metric": k, "difference": full[a][k] - full[b][k],
              "ci_low": (boots[a][k] - boots[b][k]).quantile(0.025),
              "ci_high": (boots[a][k] - boots[b][k]).quantile(0.975)} for a, b in pairs for k in CI_METRICS]
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def stage_metric_ci(results: Path, n_boot: int) -> dict:
    out = {}
    for area in AREAS:
        ci, diff = metric_intervals(pd.read_csv(results / f"predictions_{area}.csv"), n_boot)
        ci.to_csv(results / f"metric_ci_{area}.csv", index=False)
        diff.to_csv(results / f"metric_differences_{area}.csv", index=False)
        out[area] = ci
    return out


def stage_extract(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    tables, log = [], []
    for user in participants(data_dir):
        t, hours = participant_windows(user, data_dir)
        tables.append(t)
        log.append({"participant": user, "hour_offset": hours, "windows": len(t),
                    "errors": int(t["error"].notna().sum()) if "error" in t else 0,
                    "median_invalid_frac": float(t.invalid_frac.median())})
        print(f"user_{user}: {len(t)} windows", flush=True)
    feats = out_dir / "features"
    feats.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(feats / "features.csv", index=False)
    pd.DataFrame(log).to_csv(feats / "extraction_log.csv", index=False)
    return table


def stage_manipulation(data_dir: Path, results: Path, people: list[int]) -> dict:
    tlx = tlx_table(data_dir).loc[lambda d: d.index.isin(people)]
    diff = tlx["Audiovisual"] - tlx["Visual"]
    w = wilcoxon(tlx["Audiovisual"], tlx["Visual"])
    out = {"n_participants": int(len(tlx)),
           **{f"mean_{c}": float(tlx[c].mean()) for c in tlx},
           **{f"sd_{c}": float(tlx[c].std()) for c in tlx},
           "median_difference_combined_minus_visual": float(diff.median()),
           "n_combined_higher": int((diff > 0).sum()),
           "wilcoxon_statistic": float(w.statistic), "wilcoxon_p": float(w.pvalue)}
    tlx.to_csv(results / "nasa_tlx.csv")
    (results / "manipulation_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    demo = demographics(data_dir).loc[lambda d: d.participant.isin(people)]
    (results / "demographics.json").write_text(json.dumps({
        "n_participants": int(len(demo)), "age_mean": float(demo.age.mean()), "age_sd": float(demo.age.std()),
        "age_min": float(demo.age.min()), "age_max": float(demo.age.max()),
        "sex_counts": demo.sex.value_counts().to_dict()}, indent=2), encoding="utf-8")
    return out


def stage_main(table, results, n_iter, n_boot) -> dict:
    out = {}
    for area in AREAS:
        last = results / (f"without_top_signal_{area}.csv" if area == PINA_MAIN_AREA else f"results_{area}.csv")
        if last.exists():                             # finished earlier (resume after a stop)
            out[area] = pd.read_csv(results / f"results_{area}.csv")
            continue
        X, y, g, ids, check = area_data(table, area)
        fits = _fit_all(X, y, g, n_iter)
        res, comp = _results(fits, X, y, g, n_boot), _comparisons(fits, y, g, n_boot)
        _save_area(results, area, fits, ids, y, res, comp, check)
        out[area] = res
        if area != PINA_MAIN_AREA:
            continue
        feat, sig = _importance(fits, X)
        feat.to_csv(results / f"shap_{area}.csv", index=False)
        sig.to_csv(results / f"shap_by_signal_{area}.csv", index=False)
        selection_count({m: fits[m] for m in MODELS_TOPN}, list(X.columns)).to_csv(
            results / f"selection_count_{area}.csv", index=False)
        top = sig.groupby("signal")["share"].mean().idxmax()      # averaged over both models
        keep = [c for c in X.columns if signal(c) != top]
        rows = []
        for model, spec in MODELS_TOPN.items():
            r = run_lopo(X[keep], y, g, spec["factory"], spec["space"], n_iter=n_iter)
            rows.append({"without_signal": top, "model": model, "n_features_available": len(keep),
                         **_metrics(y, r["oof_proba"], g, n_boot)})
        pd.DataFrame(rows).to_csv(results / f"without_top_signal_{area}.csv", index=False)
    return out


def stage_permutation(table, results, n_perm) -> pd.DataFrame:
    X, y, g, _, _ = area_data(table, PINA_MAIN_AREA)
    rows = []
    for model, spec in MODELS_TOPN.items():
        chosen = pd.read_csv(results / f"chosen_params_{PINA_MAIN_AREA}_{FILE[model]}.csv").drop(columns="participant")
        params = chosen.astype(str).value_counts().index[0]
        params = {k: pd.Series([v]).astype(chosen[k].dtype).iloc[0] for k, v in zip(chosen.columns, params)}
        perm = permutation_test(X, y, g, spec["factory"], params, n_perm)
        rows.append({"model": model, "params": json.dumps({k: (v.item() if hasattr(v, "item") else v)
                                                           for k, v in params.items()}), **perm})
    out = pd.DataFrame(rows)
    out.to_csv(results / f"permutation_{PINA_MAIN_AREA}.csv", index=False)
    return out


def stage_sensitivity(table, results, n_iter, n_boot) -> pd.DataFrame:
    X, y, g, _, check = area_data(table, PINA_MAIN_AREA, keep_events=True)
    fits = _fit_all(X, y, g, n_iter)
    res = _results(fits, X, y, g, n_boot)
    comp = _comparisons(fits, y, g, n_boot)
    feat, sig = _importance(fits, X)
    tag = f"{PINA_MAIN_AREA}_keep_events"
    res.to_csv(results / f"sensitivity_{tag}.csv", index=False)
    comp.to_csv(results / f"sensitivity_comparison_{tag}.csv", index=False)
    sig.to_csv(results / f"sensitivity_shap_by_signal_{tag}.csv", index=False)
    feat.to_csv(results / f"sensitivity_shap_{tag}.csv", index=False)
    pd.DataFrame([check]).to_csv(results / f"sensitivity_data_check_{tag}.csv", index=False)
    return res


README = """# Pina et al. 2025 results

Written by `src/run_pina.py`; plan: `docs/pina-analysis-plan.md` (written before the eye data were
loaded). `90` is the main contrast; `360` is a separate check, never pooled. `sensitivity_*` files
were decided after the main results and are never the main result.

| File | What it is |
|---|---|
| `features/features.csv` | One row per 30 s window: QC columns and every feature (17-statistic split, DROPPED removed) |
| `features/extraction_log.csv` | Per participant: time-zone offset, windows, errors, median invalid share |
| `manipulation_check.json`, `nasa_tlx.csv` | Raw NASA-TLX per task; combined vs visual alone |
| `demographics.json` | Age (mean, SD, range) and sex counts of the analyzed participants |
| `data_check_<area>.csv` | Windows kept, exclusions, fixation sanity check, features used |
| `results_<area>.csv` | LR, XGBoost and the in-fold one-feature baseline: every metric |
| `comparison_<area>.csv` | Model minus baseline, XGBoost minus LR: AUC difference with 95% CI |
| `predictions_<area>.csv` | Out-of-fold probabilities, every window |
| `chosen_params_<area>_*.csv` | Settings each fold chose, including N |
| `shap_90.csv`, `shap_by_signal_90.csv` | SHAP share per feature (with folds selected) and per signal |
| `selection_count_90.csv` | Folds (of 36) in which each feature was selected |
| `without_top_signal_90.csv` | Both models refit without the signal with the largest SHAP share |
| `permutation_90.csv` | Chance test: labels swapped within each person, settings fixed at the most-chosen ones |
| `sensitivity_*_90_keep_events.csv` | Same run keeping the event features that failed the check |
| `metric_ci_<area>.csv` | Accuracy, precision, recall, F1, Brier, ROC-AUC with participant-bootstrap 95% CIs |
| `metric_differences_<area>.csv` | The same metrics as paired differences: model minus baseline, XGBoost minus LR |
"""


def main(data_dir=None, out_dir=None, n_iter=N_ITER, n_boot=N_BOOTSTRAP, n_perm=N_PERMUTATIONS,
         stages=STAGES) -> dict:
    data_dir, out_dir = Path(data_dir or PINA_DATA_DIR), Path(out_dir or PINA_DIR)
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    info_path = results / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info.update({"data_dir": str(data_dir), "n_iter": n_iter, "n_boot": n_boot, "n_perm": n_perm,
                 "plan": "docs/pina-analysis-plan.md", "git_commit": _git_commit(), "versions": _versions()})
    info.setdefault("stages", {})
    table = pd.read_csv(out_dir / "features" / "features.csv") if "extract" not in stages else None
    out = {}
    for stage in stages:
        t0 = time.time()
        if stage == "extract":
            table = stage_extract(data_dir, out_dir)
        elif stage == "manipulation":
            out["manipulation"] = stage_manipulation(data_dir, results, sorted(table.participant.unique()))
        elif stage == "main":
            out["main"] = stage_main(table, results, n_iter, n_boot)
        elif stage == "permutation":
            out["permutation"] = stage_permutation(table, results, n_perm)
        elif stage == "sensitivity":
            out["sensitivity"] = stage_sensitivity(table, results, n_iter, n_boot)
        elif stage == "metric_ci":
            out["metric_ci"] = stage_metric_ci(results, n_boot)
        else:
            raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
        info["stages"][stage] = {"date": datetime.now().isoformat(timespec="seconds"),
                                 "runtime_seconds": round(time.time() - t0, 1)}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"stage {stage} done ({time.time() - t0:.0f}s)", flush=True)
    (results / "README.md").write_text(README, encoding="utf-8")
    return {**info, **out}


if __name__ == "__main__":
    chosen_stages = STAGES
    if "--stages" in sys.argv:
        chosen_stages = tuple(sys.argv[sys.argv.index("--stages") + 1].split(","))
    main(stages=chosen_stages)
    print(f"Saved in {PINA_DIR}")

"""Post-defense modeling run: LR vs XGBoost on the final feature table, under LOPO.

Follows docs/post-defense-modeling-plan.md:
- input: outputs/final/features/features.csv (88 rows, 140 features, no per-person z-score);
- MD1/MD2: each model is a pipeline whose first step keeps the top-N features (topn.py); N is
  tuned with the other settings by random search in the inner folds of the training people;
- MD3: one analysis, all features;
- MD4: SHAP for both models (a feature not picked in a fold counts as 0 there) and
  selection count (in how many of the 44 folds each feature was picked).
Also recorded, so they never have to be re-run by hand:
- results_by_n.csv: the same LOPO with N fixed at each candidate (5 ... 140), other settings
  still tuned in the inner folds. Every N is reported; none is picked from these results.
- checks.csv: blink rate alone (one-feature LR) and both models without blink rate (§6.1).
Carried over unchanged (M1-M5): LOPO, 5 inner folds, 30 random settings per model, pooled
ROC-AUC with a participant-bootstrap CI, the LR vs XGBoost comparison, the grouped
permutation test against chance and the majority baseline (evaluate.py).

Usage (local):  python src/run_model.py      -> outputs/final/results/
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CEILING_AUC, FINAL_DIR, N_BOOTSTRAP, N_ITER, N_PERMUTATIONS, RANDOM_STATE
from evaluate import (
    bootstrap_auc, chosen_params_table, compare_models, majority_baseline, param_summary,
    participant_margins, permutation_test, run_lopo, score,
)
from importance import shap_summary
from topn import MODELS_TOPN, N_CANDIDATES, selected_features

ID_COLUMNS = ("participant", "activity", "label")
SHORT = {"Logistic Regression": "lr", "XGBoost": "xgb"}
FILE = {"Logistic Regression": "logistic_regression", "XGBoost": "xgboost"}


def load_features(path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """(X, y, groups, ids): y = 1 for high load (A4), groups = participant."""
    table = pd.read_csv(path).sort_values(["participant", "activity"]).reset_index(drop=True)
    features = [c for c in table.columns if c not in ID_COLUMNS]
    y = table["label"].eq("high").astype(int).to_numpy()
    return table[features], y, table["participant"].to_numpy(), table[list(ID_COLUMNS)]


def fold_shap(fold_models, X: pd.DataFrame, kind: str) -> np.ndarray:
    """Out-of-fold SHAP over all features: each fold's model explains the participant it never
    saw; features that fold did not pick get 0 (MD4)."""
    import shap

    names = list(X.columns)
    out = np.zeros(X.shape)
    for f in fold_models:
        model, tr, te = f["model"], f["train_idx"], f["test_idx"]
        cols = [names.index(c) for c in selected_features(model, names)]
        if kind == "xgb":
            sel = model.named_steps["select"].transform(X.iloc[te])
            values = shap.TreeExplainer(model.named_steps["clf"]).shap_values(sel)
        else:
            pre = model[:-1]
            explainer = shap.LinearExplainer(model.named_steps["clf"], pre.transform(X.iloc[tr]))
            values = explainer.shap_values(pre.transform(X.iloc[te]))
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        out[np.ix_(te, cols)] = values
    return out


def selection_count(fits: dict, names: list[str]) -> pd.DataFrame:
    """In how many folds each feature made the top N, per model."""
    out = pd.DataFrame({"feature": names})
    for model_name, res in fits.items():
        picked = [set(selected_features(f["model"], names)) for f in res["fold_models"]]
        out[f"folds_selected_{SHORT[model_name]}"] = [sum(n in p for p in picked) for n in names]
    out["n_folds"] = len(next(iter(fits.values()))["fold_models"])
    total = out.filter(like="folds_selected_").sum(axis=1)
    return out.assign(_t=total).sort_values("_t", ascending=False, kind="stable").drop(columns="_t")


def _metrics(y, p, g, n_boot) -> dict:
    """Every reported metric for one set of out-of-fold probabilities."""
    margins = participant_margins(y, p, g)
    return {**score(y, p), **bootstrap_auc(y, p, g, n_boot),
            "a4_above_a1": int((margins > 0).sum()),
            "share_a4_above_a1": float((margins > 0).mean())}


def _n_used(res, names) -> float:
    return float(np.median([len(selected_features(f["model"], names)) for f in res["fold_models"]]))


def _done(path: Path, sha: str, keys: list[str]) -> tuple[pd.DataFrame, set]:
    """Rows already saved for this features file, and their keys (for resuming)."""
    if not path.exists():
        return pd.DataFrame(), set()
    old = pd.read_csv(path)
    if "features_sha256" not in old:
        return pd.DataFrame(), set()
    old = old[old["features_sha256"] == sha]
    return old, set(map(tuple, old[keys].astype(str).to_numpy()))


def _save_row(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)          # rewritten after every finished run


def results_by_n(X, y, g, n_iter, n_boot, path: Path, sha: str) -> pd.DataFrame:
    """LOPO with N fixed at each candidate; the other settings are still tuned in-fold.
    Each finished (model, N) row is saved at once; rows already saved for this features file
    are skipped, so an interrupted run resumes where it stopped."""
    names = list(X.columns)
    old, done = _done(path, sha, ["model", "n_features"])
    rows = old.to_dict("records")
    for n in N_CANDIDATES:
        for model_name, spec in MODELS_TOPN.items():
            if (model_name, str(n)) in done:
                continue
            space = {k: v for k, v in spec["space"].items() if k != "select__n"}
            res = run_lopo(X, y, g, spec["factory"], space, n_iter=n_iter, params={"select__n": n})
            rows.append({"model": model_name, "n_features": n, "n_features_used": _n_used(res, names),
                         **_metrics(y, res["oof_proba"], g, n_boot), "features_sha256": sha})
            _save_row(path, rows)
    order = {m: i for i, m in enumerate(MODELS_TOPN)}
    out = pd.DataFrame(rows).sort_values(["n_features", "model"], key=lambda c: c.map(order)
                                         if c.name == "model" else c).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def _one_feature_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE))


def checks(X, y, g, n_iter, n_boot, n_perm, path: Path, sha: str) -> pd.DataFrame:
    """Blink rate alone, and both full models without blink rate (modeling plan §6.1).
    Saved and resumed row by row, like results_by_n."""
    if "blink_rate" not in X:
        return pd.DataFrame()
    old, done = _done(path, sha, ["check", "model"])
    rows = old.to_dict("records")
    rest = X.drop(columns="blink_rate")
    plan = [("blink_rate alone", "Logistic Regression (1 feature)", X[["blink_rate"]],
             _one_feature_lr, None)]
    plan += [("without blink_rate", m, rest, spec["factory"], spec["space"])
             for m, spec in MODELS_TOPN.items()]
    for check, model_name, data, factory, space in plan:
        if (check, model_name) in done:
            continue
        res = run_lopo(data, y, g, factory, space, n_iter=n_iter, tune=space is not None)
        perm = permutation_test(data, y, g, factory, {}, n_perm)
        used = 1.0 if space is None else _n_used(res, list(data.columns))
        rows.append({"check": check, "model": model_name, "n_features_available": data.shape[1],
                     "n_features_used": used, **_metrics(y, res["oof_proba"], g, n_boot),
                     "permutation_p": perm["p_value"], "permutation_n": perm["n_perm"],
                     "features_sha256": sha})
        _save_row(path, rows)
    key = {(c, m): i for i, (c, m, *_) in enumerate(plan)}
    out = pd.DataFrame(rows)
    out = out.iloc[sorted(range(len(out)), key=lambda i: key[(out.check[i], out.model[i])])]
    out.reset_index(drop=True).to_csv(path, index=False)
    return out.reset_index(drop=True)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict:
    import scipy, shap, sklearn, xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "shap": shap.__version__}


def _readme(n_people: int, n_rows: int, n_features: int) -> str:
    return f"""# Model results (post-defense)

Written by `src/run_model.py` from `features/features.csv` ({n_rows} rows, {n_people} participants,
{n_features} features). Decisions: `docs/post-defense-modeling-plan.md` (MD1-MD4).
`run_info.json` records the SHA-256 of the features file and when each stage last ran; every
results row also carries `features_sha256`. Stages: `python src/run_model.py --stages main,by_n,checks`
(any subset). `results_by_n.csv` and `checks.csv` are saved after every finished run and resume
where they stopped.

| File | What it is |
|---|---|
| `results.csv` | One row per model: ROC-AUC with 95% CI, accuracy, balanced accuracy, precision, recall and F1 (per class and macro), Brier score, participants with A4 above A1, chance-test p-value, median number of features chosen |
| `results_by_n.csv` | The same run with the number of features fixed at 5, 10, 20, 40, 80 and 140 (every N reported, none picked from these results), every metric |
| `checks.csv` | Blink rate alone, and both models without blink rate, every metric and chance test |
| `comparison.csv` | XGBoost minus LR: AUC difference with 95% CI, paired Wilcoxon on per-participant margins |
| `predictions.csv` | Out-of-fold probability of high load for every row, both models |
| `selection_count.csv` | In how many of the {n_people} folds each feature made the top N, per model |
| `shap_logistic_regression.csv`, `shap_xgboost.csv` | Mean absolute and mean SHAP per feature (0 in folds where the feature was not picked) |
| `chosen_params_*.csv` | The settings each fold chose, including N (`select__n`) |
"""


STAGES = ("main", "by_n", "checks")


def _main_stage(X, y, g, ids, names, n_iter, n_boot, n_perm, results_dir, sha) -> pd.DataFrame:
    """Both tuned models: results, comparison, predictions, selection count, SHAP, settings."""
    rows, probas, fits, chosen = [], {}, {}, {}
    base = majority_baseline(y, g)
    for model_name, spec in MODELS_TOPN.items():
        res = run_lopo(X, y, g, spec["factory"], spec["space"], n_iter=n_iter)
        p = res["oof_proba"]
        probas[model_name], fits[model_name] = p, res
        chosen[model_name] = chosen_params_table(
            res["chosen_params"], [f["participant"] for f in res["fold_models"]])
        # default settings (N = 20): tuned ones were chosen with the true labels
        perm = permutation_test(X, y, g, spec["factory"], {}, n_perm)
        rows.append({"model": model_name, "n_samples": len(y), "n_participants": len(np.unique(g)),
                     "n_features_available": len(names), **_metrics(y, p, g, n_boot),
                     "majority_balanced_accuracy": base["balanced_accuracy"],
                     "n_features_median": float(chosen[model_name]["select__n"].median()),
                     "permutation_observed_auc": perm["observed_auc"],
                     "permutation_p": perm["p_value"], "permutation_n": perm["n_perm"],
                     **param_summary(res["chosen_params"]), "features_sha256": sha})
    results = pd.DataFrame(rows)
    comparison = pd.DataFrame([{
        **compare_models(y, probas["Logistic Regression"], probas["XGBoost"], g, n_boot),
        "ceiling_rule_triggered": bool((results["roc_auc"] > CEILING_AUC).all()),
        "features_sha256": sha}])
    results.to_csv(results_dir / "results.csv", index=False)
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    ids.assign(y=y, p_lr=probas["Logistic Regression"], p_xgb=probas["XGBoost"]).to_csv(
        results_dir / "predictions.csv", index=False)
    selection_count(fits, names).to_csv(results_dir / "selection_count.csv", index=False)
    for model_name, res in fits.items():
        shap_summary(fold_shap(res["fold_models"], X, SHORT[model_name]), names).to_csv(
            results_dir / f"shap_{FILE[model_name]}.csv", index=False)
        chosen[model_name].to_csv(results_dir / f"chosen_params_{FILE[model_name]}.csv", index=False)
    return results


def main(features_path=None, results_dir=None, n_iter=N_ITER, n_boot=N_BOOTSTRAP,
         n_perm=N_PERMUTATIONS, stages=STAGES) -> dict:
    """Run the chosen stages. Each stage saves its files as soon as it finishes, and run_info.json
    records when each stage last ran and on which features file."""
    features_path = Path(features_path or FINAL_DIR / "features" / "features.csv")
    results_dir = Path(results_dir or FINAL_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    X, y, g, ids = load_features(features_path)
    names = list(X.columns)
    sha = hashlib.sha256(features_path.read_bytes()).hexdigest()
    info_path = results_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if info.get("features_sha256") not in (None, sha):
        info = {}                                     # different features: start the record afresh
    for old_key in ("date", "runtime_seconds"):        # pre-stages format; dates now live in "stages"
        info.pop(old_key, None)
    info.update({
        "features_file": str(features_path), "features_sha256": sha,
        "n_participants": int(len(np.unique(g))), "n_rows": int(len(y)), "n_features": len(names),
        "n_candidates": list(N_CANDIDATES), "n_iter": n_iter, "n_boot": n_boot, "n_perm": n_perm,
        "decisions": "docs/post-defense-modeling-plan.md (MD1-MD4)",
        "git_commit": _git_commit(), "versions": _versions(),
    })
    info.setdefault("stages", {})
    out = {}
    for stage in stages:
        t0 = time.time()
        if stage == "main":
            out["results"] = _main_stage(X, y, g, ids, names, n_iter, n_boot, n_perm, results_dir, sha)
        elif stage == "by_n":
            out["by_n"] = results_by_n(X, y, g, n_iter, n_boot, results_dir / "results_by_n.csv", sha)
        elif stage == "checks":
            out["checks"] = checks(X, y, g, n_iter, n_boot, n_perm, results_dir / "checks.csv", sha)
        else:
            raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
        info["stages"][stage] = {"date": datetime.now().isoformat(timespec="seconds"),
                                 "runtime_seconds": round(time.time() - t0, 1)}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    (results_dir / "README.md").write_text(_readme(info["n_participants"], info["n_rows"], len(names)),
                                           encoding="utf-8")
    return {**info, **out}


if __name__ == "__main__":
    # python src/run_model.py                      -> every stage
    # python src/run_model.py --stages by_n,checks -> only those (finished rows are skipped)
    chosen_stages = STAGES
    if "--stages" in sys.argv:
        chosen_stages = tuple(sys.argv[sys.argv.index("--stages") + 1].split(","))
    result = main(stages=chosen_stages)
    print(json.dumps(result["stages"], indent=2))
    print(f"Saved in {FINAL_DIR / 'results'}")

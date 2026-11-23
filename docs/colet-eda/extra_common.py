"""Shared helpers for the extra robustness analyses (docs/extra-analyses-notes.md).

Reuses src/ unchanged: models.MODELS, evaluate.run_lopo/score/bootstrap_auc/compare_models,
normalize.per_participant_zscore. Results go to outputs/colet/tables/extra/ (never the
official tables).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import evaluate

evaluate.SEARCH_JOBS = 12

import numpy as np
import pandas as pd

from config import N_BOOTSTRAP, OUT_DIR
from evaluate import bootstrap_auc, compare_models, participant_margins, run_lopo, score
from models import MODELS

EXTRA_DIR = OUT_DIR / "tables" / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
COLS = ["analysis", "model", "n_participants", "n_samples", "roc_auc", "auc_ci_low", "auc_ci_high",
        "balanced_accuracy"]


def identity(X, participants):
    return X.copy()


def evaluate_models(name, X, y, g, models=("Logistic Regression", "XGBoost"), tune=True):
    """Nested LOPO for each model; returns (metrics rows, comparison dict or None, oof probas)."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y, g = np.asarray(y), np.asarray(g)
    rows, probas = [], {}
    for m in models:
        spec = MODELS[m]
        p = run_lopo(X, y, g, spec["factory"], spec["space"], tune=tune)["oof_proba"]
        probas[m] = p
        rows.append({"analysis": name, "model": m, "n_participants": len(np.unique(g)),
                     "n_samples": len(y), "n_features": X.shape[1], "tuned": tune,
                     **score(y, p), **bootstrap_auc(y, p, g, N_BOOTSTRAP),
                     "share_high_above_low": float((participant_margins(y, p, g) > 0).mean())})
    comp = None
    if len(models) == 2:
        comp = {"analysis": name, **compare_models(y, probas[models[0]], probas[models[1]], g, N_BOOTSTRAP)}
    return rows, comp, probas


def show(rows, comp=None):
    print(pd.DataFrame(rows)[COLS].round(3).to_string(index=False), flush=True)
    if comp:
        print(f"  XGB-LR {comp['auc_difference']:+.3f} [{comp['auc_difference_ci_low']:+.3f}, "
              f"{comp['auc_difference_ci_high']:+.3f}]", flush=True)

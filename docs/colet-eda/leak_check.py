"""Leak diagnostics for P1 (post hoc; no change to the frozen method).

1. Shuffled labels: if the pipeline leaked, AUC would stay high with shuffled labels.
   a) within-participant swaps (the paired null used by the permutation test)
   b) fully random labels across all 90 rows
2. How easy is the task: single-feature AUCs (no model), with and without per-person z-score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from config import CACHE_DIR, DATA_DIR, FEATURES
from evaluate import run_lopo
from labels import binary_label, main_rows
from models import MODELS
from normalize import per_participant_zscore
from run_colet import prepare

prep = prepare(DATA_DIR, CACHE_DIR)
rows = prep["rows"]
z = per_participant_zscore(rows[FEATURES], rows["participant"])
main_z = main_rows(rows.assign(**{c: z[c] for c in FEATURES}))
main_raw = main_rows(rows)
X = main_z[FEATURES].reset_index(drop=True)
y = binary_label(main_z)
g = main_z["participant"].to_numpy()


def auc(model, yy):
    f = MODELS[model]["factory"]
    return roc_auc_score(yy, run_lopo(X, yy, g, f, None, tune=False)["oof_proba"])


rng = np.random.default_rng(1)
paired, randoms = [], []
for _ in range(40):
    yp = y.copy()
    for p in np.unique(g):
        if rng.random() < 0.5:
            i = np.flatnonzero(g == p)
            yp[i] = yp[i][::-1]
    paired.append(yp)
    randoms.append(rng.permutation(y))

for model in MODELS:
    real = auc(model, y)
    a = Parallel(n_jobs=12)(delayed(auc)(model, yy) for yy in paired)
    b = Parallel(n_jobs=12)(delayed(auc)(model, yy) for yy in randoms)
    print(f"{model:20s} real labels AUC {real:.3f} | shuffled within person: mean {np.mean(a):.3f} "
          f"(max {np.max(a):.3f}) | fully random: mean {np.mean(b):.3f} (max {np.max(b):.3f})")

print("\nSingle-feature AUC, no model (direction-free: max(AUC, 1-AUC)):")
out = []
for f in FEATURES:
    def da(v, yy):
        m = np.isfinite(v)
        a_ = roc_auc_score(yy[m], v[m])
        return max(a_, 1 - a_)
    out.append({"feature": f,
                "raw_pooled": da(main_raw[f].to_numpy(float), binary_label(main_raw)),
                "per_person_z": da(main_z[f].to_numpy(float), y),
                "within_person_A4>A1_share": float(np.mean(
                    main_raw.pivot(index="participant", columns="activity", values=f).pipe(
                        lambda w: np.sign(w[4] - w[1]) == np.sign(np.nanmedian(w[4] - w[1])))))})
print(pd.DataFrame(out).round(3).sort_values("per_person_z", ascending=False).to_string(index=False))

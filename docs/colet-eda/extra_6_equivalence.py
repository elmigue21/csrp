"""Extra analysis 6: equivalence check of XGBoost - LR AUC against a pre-stated margin.

No new modelling: uses the official out-of-fold predictions (predictions_P1/P2.csv, read
only). Repeats evaluate.bootstrap_auc's participant bootstrap with the same seed and draws
(so the 95% CI must match 06_comparison.csv) and also keeps the 90% CI, which is the interval
a two-one-sided-tests equivalence check at alpha = 0.05 uses.
Margins (our own choice): +/-0.05 (main) and +/-0.02 (strict).
Output: outputs/colet/tables/extra/6_equivalence.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from extra_common import EXTRA_DIR

from config import N_BOOTSTRAP, RANDOM_STATE, TABLE_DIR

MARGINS = (0.05, 0.02)
official = pd.read_csv(TABLE_DIR / "06_comparison.csv").set_index("analysis")
out = []
for name in ("P1", "P2"):
    d = pd.read_csv(TABLE_DIR / f"predictions_{name}.csv")
    y, lr, xgb, g = d["y"].to_numpy(), d["p_lr"].to_numpy(), d["p_xgb"].to_numpy(), d["participant"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.unique(g)
    idx_by = {p: np.flatnonzero(g == p) for p in ids}
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([idx_by[p] for p in rng.choice(ids, len(ids), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], xgb[idx]) - roc_auc_score(y[idx], lr[idx]))
    diff = roc_auc_score(y, xgb) - roc_auc_score(y, lr)
    lo95, hi95, lo90, hi90 = np.percentile(diffs, [2.5, 97.5, 5, 95])
    row = {"analysis": name, "auc_difference": diff, "ci95_low": lo95, "ci95_high": hi95,
           "ci90_low": lo90, "ci90_high": hi90,
           "official_ci95_low": official.loc[name, "auc_difference_ci_low"],
           "official_ci95_high": official.loc[name, "auc_difference_ci_high"]}
    for m in MARGINS:
        row[f"equivalent_90ci_within_{m}"] = bool(-m < lo90 and hi90 < m)
        row[f"equivalent_95ci_within_{m}"] = bool(-m < lo95 and hi95 < m)
    out.append(row)
res = pd.DataFrame(out)
print(res.round(4).T.to_string())
res.to_csv(EXTRA_DIR / "6_equivalence.csv", index=False)

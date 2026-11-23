"""Extra analysis 1: standardize A1/A4 WITHOUT using the person's own A1 or A4.

Variant "A2A3_z": each person's mean and SD (pandas ddof=1, the same as the main pipeline)
come from their A2 and A3 recordings only and are applied to their A1 and A4.
Variant "A2_diff": A1 - A2 and A4 - A2 (subtractive baseline). LR rescales in-fold with
StandardScaler (fitted on the training folds), XGBoost is scale-free, so no extra division
is needed. Then the same nested LOPO on A1 vs A4 as the main run.
Output: outputs/colet/tables/extra/1_*.csv
"""
import time

import numpy as np
import pandas as pd

from extra_common import EXTRA_DIR, evaluate_models, show

from config import CACHE_DIR, DATA_DIR, PUPIL_FEATURES
from labels import binary_label, main_rows
from run_colet import prepare

_EPS = 1e-9


def baseline_standardize(rows, feats, base_acts, divide=True):
    """Standardize every row with mean/SD from that participant's `base_acts` rows only."""
    base = rows[rows["activity"].isin(base_acts)]
    ok = base.groupby("participant")["activity"].nunique() == len(base_acts)
    keep = ok[ok].index
    rows = rows[rows["participant"].isin(keep)].copy()
    mu = base.groupby("participant")[feats].mean()
    out = rows[feats] - mu.loc[rows["participant"]].to_numpy()
    if divide:
        sd = base.groupby("participant")[feats].std().loc[rows["participant"]].to_numpy()  # ddof=1
        out = (out / (sd + _EPS)).where(sd > _EPS, 0.0)
    rows[feats] = out.replace([np.inf, -np.inf], np.nan)
    return rows, int(ok.size - len(keep))


t0 = time.time()
prep = prepare(DATA_DIR, CACHE_DIR)
all_rows, comps = [], []
for variant, acts, divide in (("A2A3_z", [2, 3], True), ("A2_diff", [2], False)):
    for pname, feats in (("P1", prep["features_used"]), ("P2", PUPIL_FEATURES)):
        std_rows, dropped = baseline_standardize(prep["rows"], feats, acts, divide)
        df = main_rows(std_rows)
        name = f"{pname}_{variant}"
        print(f"\n{name}: dropped {dropped} participants missing a baseline activity", flush=True)
        rows, comp, _ = evaluate_models(name, df[feats], binary_label(df), df["participant"].to_numpy())
        show(rows, comp)
        for r in rows:
            r["dropped_participants"] = dropped
        all_rows += rows
        comps.append(comp)
pd.DataFrame(all_rows).to_csv(EXTRA_DIR / "1_baseline_standardization_results.csv", index=False)
pd.DataFrame(comps).to_csv(EXTRA_DIR / "1_baseline_standardization_comparison.csv", index=False)
print(f"runtime {time.time() - t0:.0f} s")

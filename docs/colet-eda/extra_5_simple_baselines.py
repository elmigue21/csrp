"""Extra analysis 5: simple baselines under the same LOPO, A1 vs A4.

(a) each single feature alone, tuned LR (nested, as the main run)
(b) blink_rate + pupil_mean, tuned LR and XGBoost
(c) P1 and P2 with default (untuned) LR and XGBoost: run_lopo(tune=False)
Each standardized ("z", main per-person z-score over A1-A4) and unstandardized ("raw").
Output: outputs/colet/tables/extra/5_*.csv
"""
import time

import pandas as pd

from extra_common import EXTRA_DIR, evaluate_models, identity, show

from config import CACHE_DIR, DATA_DIR, PUPIL_FEATURES
from labels import binary_label, main_rows
from normalize import per_participant_zscore
from run_colet import prepare

t0 = time.time()
prep = prepare(DATA_DIR, CACHE_DIR)
P1 = prep["features_used"]
all_rows, comps = [], []
for std_name, std in (("z", per_participant_zscore), ("raw", identity)):
    X = std(prep["rows"][P1], prep["rows"]["participant"])
    df = main_rows(prep["rows"].assign(**{c: X[c] for c in P1}))
    y, g = binary_label(df), df["participant"].to_numpy()

    for f in P1:                                                     # (a)
        res, _, _ = evaluate_models(f"single_{f}_{std_name}", df[[f]], y, g, models=("Logistic Regression",))
        all_rows += res
    show([r for r in all_rows if r["analysis"].endswith(std_name) and r["analysis"].startswith("single")])

    res, comp, _ = evaluate_models(f"blink+pupil_mean_{std_name}", df[["blink_rate", "pupil_mean"]], y, g)  # (b)
    show(res, comp)
    all_rows += res
    comps.append(comp)

    for pname, feats in (("P1", P1), ("P2", PUPIL_FEATURES)):     # (c)
        res, comp, _ = evaluate_models(f"untuned_{pname}_{std_name}", df[feats], y, g, tune=False)
        show(res, comp)
        all_rows += res
        comps.append(comp)
pd.DataFrame(all_rows).to_csv(EXTRA_DIR / "5_simple_baselines_results.csv", index=False)
pd.DataFrame(comps).to_csv(EXTRA_DIR / "5_simple_baselines_comparison.csv", index=False)
print(f"runtime {time.time() - t0:.0f} s")

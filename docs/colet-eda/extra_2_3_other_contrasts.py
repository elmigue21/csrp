"""Extra analyses 2 and 3: other activity pairs, same nested LOPO as the main run.

2. A3 (0) vs A4 (1): both counting aloud; differ only by time pressure.
3. A1 (0) vs A2 (1): both silent; differ only by time pressure.
Each with the main per-person z-score over all four activities ("z") and unstandardized ("raw").
Output: outputs/colet/tables/extra/2_3_*.csv
"""
import time

import numpy as np
import pandas as pd

from extra_common import EXTRA_DIR, evaluate_models, identity, show

from config import CACHE_DIR, DATA_DIR, PUPIL_FEATURES
from normalize import per_participant_zscore
from run_colet import prepare

t0 = time.time()
prep = prepare(DATA_DIR, CACHE_DIR)
all_rows, comps = [], []
for pair_name, low, high in (("A3vA4", 3, 4), ("A1vA2", 1, 2)):
    for std_name, std in (("z", per_participant_zscore), ("raw", identity)):
        for pname, feats in (("P1", prep["features_used"]), ("P2", PUPIL_FEATURES)):
            rows = prep["rows"]
            X = std(rows[feats], rows["participant"])
            df = rows.assign(**{c: X[c] for c in feats})
            df = df[df["activity"].isin([low, high])]
            both = df.groupby("participant")["activity"].nunique() == 2
            df = df[df["participant"].isin(both[both].index)].reset_index(drop=True)
            y = (df["activity"].to_numpy() == high).astype(int)
            name = f"{pair_name}_{pname}_{std_name}"
            print(f"\n{name}", flush=True)
            res, comp, _ = evaluate_models(name, df[feats], y, df["participant"].to_numpy())
            show(res, comp)
            all_rows += res
            comps.append(comp)
pd.DataFrame(all_rows).to_csv(EXTRA_DIR / "2_3_other_contrasts_results.csv", index=False)
pd.DataFrame(comps).to_csv(EXTRA_DIR / "2_3_other_contrasts_comparison.csv", index=False)
print(f"runtime {time.time() - t0:.0f} s")

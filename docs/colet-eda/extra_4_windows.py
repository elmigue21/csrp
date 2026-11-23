"""Extra analysis 4: fixed-length windows instead of one row per activity.

Each retained recording of the 44 analysis participants is cut into non-overlapping
WINDOW_S-second windows by gaze timestamp (the last, shorter remainder is dropped). The gaze
table is sliced; recording_features() then trims pupil and blinks to the slice itself
(trim_to_gaze_range), so the src feature code runs unchanged on each window. Windows whose
invalid-gaze share exceeds MAX_INVALID_FRACTION are dropped (the recording rule, applied per window).
Per-person z-score over all of that person's windows from A1-A4, then the same nested LOPO
(participant-grouped: no person in both train and test) on A1 vs A4 windows.
Window features are cached in outputs/colet/tables/extra/4_window_features.csv
(delete it to rebuild). Usage: python extra_4_windows.py [window_seconds]
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from extra_common import EXTRA_DIR, evaluate_models, identity, show

from config import CACHE_DIR, DATA_DIR, MAX_INVALID_FRACTION, PUPIL_FEATURES
from data import load_recording
from features import recording_features
from normalize import per_participant_zscore
from preprocess import invalid_fraction, trim_to_gaze_range
from run_colet import prepare

WINDOW_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


def windows_of(participant, activity):
    rec = trim_to_gaze_range(load_recording(participant, activity, DATA_DIR))
    gaze = rec["gaze"]
    t = gaze["gaze_timestamp"].to_numpy(float)
    out = []
    for k in range(int((t.max() - t.min()) // WINDOW_S)):
        lo = t.min() + k * WINDOW_S
        g = gaze[(t >= lo) & (t < lo + WINDOW_S)].reset_index(drop=True)
        row = {"participant": participant, "activity": activity, "window": k,
               "invalid_fraction": invalid_fraction(g)}
        row["excluded"] = len(g) < 2 or row["invalid_fraction"] > MAX_INVALID_FRACTION
        if not row["excluded"]:
            row.update(recording_features({"gaze": g, "pupil": rec["pupil"], "blinks": rec["blinks"]}))
        out.append(row)
    return out


t0 = time.time()
prep = prepare(DATA_DIR, CACHE_DIR)
P1 = prep["features_used"]
cache = EXTRA_DIR / f"4_window_features_{int(WINDOW_S)}s.csv"
if cache.exists():
    win = pd.read_csv(cache)
else:
    keys = prep["rows"][["participant", "activity"]].itertuples(index=False)
    parts = Parallel(n_jobs=12)(delayed(windows_of)(int(p), int(a)) for p, a in keys)
    win = pd.DataFrame([r for part in parts for r in part])
    win.to_csv(cache, index=False)
print(f"feature extraction done at {time.time() - t0:.0f} s", flush=True)

counts = win.groupby("activity").agg(windows=("window", "size"), excluded=("excluded", "sum"))
counts["kept"] = counts["windows"] - counts["excluded"]
counts["participants_with_kept"] = win[~win["excluded"]].groupby("activity")["participant"].nunique()
counts["kept_per_person_median"] = win[~win["excluded"]].groupby(["activity", "participant"]).size() \
    .groupby("activity").median()
print(counts.to_string(), flush=True)
counts.reset_index().to_csv(EXTRA_DIR / f"4_window_counts_{int(WINDOW_S)}s.csv", index=False)

rows = win[~win["excluded"]].reset_index(drop=True)
all_rows, comps = [], []
for std_name, std in (("z", per_participant_zscore), ("raw", identity)):
    for pname, feats in (("P1", P1), ("P2", PUPIL_FEATURES)):
        X = std(rows[feats], rows["participant"])
        df = rows.assign(**{c: X[c] for c in feats})
        df = df[df["activity"].isin([1, 4])]
        both = df.groupby("participant")["activity"].nunique() == 2
        df = df[df["participant"].isin(both[both].index)].reset_index(drop=True)
        y = (df["activity"].to_numpy() == 4).astype(int)
        name = f"win{int(WINDOW_S)}s_{pname}_{std_name}"
        print(f"\n{name}", flush=True)
        res, comp, _ = evaluate_models(name, df[feats], y, df["participant"].to_numpy())
        for r in res:
            r["n_windows_A1"], r["n_windows_A4"] = int((y == 0).sum()), int((y == 1).sum())
        show(res, comp)
        all_rows += res
        comps.append(comp)
pd.DataFrame(all_rows).to_csv(EXTRA_DIR / f"4_windows_{int(WINDOW_S)}s_results.csv", index=False)
pd.DataFrame(comps).to_csv(EXTRA_DIR / f"4_windows_{int(WINDOW_S)}s_comparison.csv", index=False)
print(f"runtime {time.time() - t0:.0f} s")

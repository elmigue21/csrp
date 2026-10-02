"""Label-blind checks run before and alongside the models.

sanity_check      -- §5 step 7 / R4-2: event detection must look normal in EVERY activity.
manipulation_check-- §3: did NASA-RTLX rise from A1 to A4 (Friedman over A1-A4)?
redundancy_check  -- V2-13: recheck the correlations behind dropping mean saccade velocity
                     (vs peak, rho 0.95) and saccade duration (vs amplitude, rho 0.83).
missingness_report-- §8: NaN count and share per feature and activity, before normalization.
None of these looks at how well a feature separates A1 from A4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

from config import (
    ACTIVITIES, FEATURES, HIGH_ACTIVITY, LOW_ACTIVITY, SANITY_FIXATION_MEDIAN_MS,
    SANITY_SACCADE_FIXATION_RATIO,
)

REDUNDANT_PAIRS = [("saccade_peak_velocity", "saccade_mean_velocity"),
                   ("saccade_amplitude", "saccade_duration_ms")]


def sanity_check(rows: pd.DataFrame) -> pd.DataFrame:
    """Per activity: median of recording-level median fixation durations, and the ratio of
    total saccades to total fixations. Every activity in config.ACTIVITIES gets a row; one
    with no recordings fails."""
    lo_ms, hi_ms = SANITY_FIXATION_MEDIAN_MS
    lo_r, hi_r = SANITY_SACCADE_FIXATION_RATIO
    out = []
    for act in ACTIVITIES:
        g = rows[rows["activity"] == act]
        if g.empty:                       # an activity with no data cannot pass the sanity check
            out.append({"activity": int(act), "recordings": 0, "fixation_median_ms": float("nan"),
                        "saccade_fixation_ratio": float("nan"), "fixation_ok": False, "ratio_ok": False})
            continue
        fix_ms = float(g["fixation_median_ms"].median())
        n_fix = float(g["n_fixations"].sum())
        ratio = float(g["n_saccades"].sum()) / n_fix if n_fix else float("nan")
        out.append({"activity": int(act), "recordings": len(g),
                    "fixation_median_ms": fix_ms, "saccade_fixation_ratio": ratio,
                    "fixation_ok": lo_ms <= fix_ms <= hi_ms, "ratio_ok": lo_r <= ratio <= hi_r})
    df = pd.DataFrame(out)
    df["passed"] = df["fixation_ok"] & df["ratio_ok"]
    return df


def sanity_passed(sanity: pd.DataFrame) -> bool:
    return bool(len(sanity)) and bool(sanity["passed"].all())


def missingness_report(rows: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """NaN count and share of each feature in each activity (label-blind: activity is only a
    condition here, as in the P11 quality report). Run on the raw rows, before the z-score."""
    out = []
    for act in ACTIVITIES:
        g = rows[rows["activity"] == act]
        for f in features:
            n_nan = int(g[f].isna().sum())
            out.append({"activity": int(act), "feature": f, "recordings": len(g), "n_nan": n_nan,
                        "share_nan": n_nan / len(g) if len(g) else float("nan")})
    return pd.DataFrame(out)


def manipulation_check(annotation: pd.DataFrame, participants) -> dict:
    a = annotation[annotation["participant"].isin(participants)]
    wide = a.pivot(index="participant", columns="activity", values="mean").dropna()
    stat, p = friedmanchisquare(*[wide[act] for act in ACTIVITIES])
    diff = wide[HIGH_ACTIVITY] - wide[LOW_ACTIVITY]
    return {"friedman_chi2": float(stat), "friedman_p": float(p), "n": int(len(wide)),
            "share_a4_above_a1": float((diff > 0).mean()),
            "median_a4_minus_a1": float(diff.median())}


def redundancy_check(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a, b in REDUNDANT_PAIRS:
        pair = rows[[a, b]].dropna()
        rho = spearmanr(pair[a], pair[b]).statistic if len(pair) > 2 else np.nan
        out.append({"pair": f"{a}~{b}", "spearman_rho": float(rho), "n": int(len(pair))})
    return pd.DataFrame(out)

"""Per-participant feature standardisation.

Each feature is re-expressed as a deviation from that participant's own mean, in
units of their own standard deviation. A value of +2 means "two SDs more than this
person normally shows", not "a large absolute value".

Why this is not leakage under leave-one-participant-out: the transform for a given
participant is computed exclusively from that participant's own feature values and
uses no labels whatsoever. No information crosses between participants, so the
held-out participant's transform is unaffected by the training set and vice versa.

Why it still costs something in practice: a deployed system would need a short
calibration recording from each new user before it could normalise their features.
That is a real deployment constraint and belongs in the limitations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def per_participant_zscore(X: pd.DataFrame, participants: pd.Series) -> pd.DataFrame:
    """Standardise every column within each participant.

    Constant columns for a participant (zero SD) would divide by zero; those become
    0.0, i.e. "exactly this person's typical value", which is what a constant means.
    """
    grouped = X.groupby(participants, sort=False)
    centred = X - grouped.transform("mean")
    spread = grouped.transform("std")
    out = centred / (spread + _EPS)
    return out.replace([np.inf, -np.inf], np.nan).where(X.notna())


def apply_feature_scheme(X: pd.DataFrame, participants: pd.Series, scheme: str) -> pd.DataFrame:
    if scheme == "raw":
        return X
    if scheme == "within_person":
        return per_participant_zscore(X, participants)
    raise ValueError(f"unknown feature scheme: {scheme!r}")

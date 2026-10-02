"""Per-person standardization (§7, decision D-M4).

Each feature is re-expressed as a deviation from that participant's own mean over their
retained activities (normally all four), in units of their own SD. No labels are used.

The cost, stated as a limitation: the held-out participant's own unlabelled recordings set
their "normal", and the method relies on COLET's balanced design (everyone did easier and
harder activities), so a deployed system would need a calibration session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def per_participant_zscore(X: pd.DataFrame, participants: pd.Series) -> pd.DataFrame:
    """Standardise every column within each participant.

    Constant columns for a participant (zero SD) become 0.0, i.e. "exactly this person's
    typical value".
    """
    participants = pd.Series(np.asarray(participants), index=X.index)
    grouped = X.groupby(participants, sort=False)
    centred = X - grouped.transform("mean")
    spread = grouped.transform("std").fillna(0.0)
    out = centred / (spread + _EPS)
    out = out.where(spread > _EPS, 0.0)
    return out.replace([np.inf, -np.inf], np.nan).where(X.notna())

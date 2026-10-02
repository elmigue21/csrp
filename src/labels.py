"""Condition labels (§3, decision C2): A1 = low load (0), A4 = high load (1).

A2 and A3 never enter a model; they are used only for the per-person normalization
(all four activities) and the NASA-RTLX manipulation check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HIGH_ACTIVITY, LOW_ACTIVITY


def main_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["activity"].isin([LOW_ACTIVITY, HIGH_ACTIVITY])].reset_index(drop=True)


def binary_label(rows: pd.DataFrame) -> np.ndarray:
    return (rows["activity"].to_numpy() == HIGH_ACTIVITY).astype(int)

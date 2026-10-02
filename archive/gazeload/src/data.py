"""Load the 130 GAZELOAD recordings and aggregate 250 ms windows into epochs.

Two things this module is responsible for getting right:

1. Participant identity comes from the *filename*, never from the Participant_ID
   column -- that column holds 'Hedi' in 01_2..01_4 and 'Wassim' in 02_3..02_5.
2. Epochs are cut on elapsed time, not on row index, so a gap in the recording
   cannot silently glue two distant stretches of gaze into one epoch.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from config import (
    AGG_BASE,
    CACHE_DIR,
    EPOCH_MS,
    LABEL_COL,
    METRICS_DIR,
    MIN_WINDOW_COVERAGE,
    WINDOW_MS,
)

_FNAME = re.compile(r"^(\d+)_(\d+)_Metrics_withLux\.csv$")


def load_windows() -> pd.DataFrame:
    """Return every 250 ms window from all recordings as one pooled table."""
    frames = []
    for path in sorted(METRICS_DIR.glob("*Metrics_withLux.csv")):
        m = _FNAME.match(path.name)
        if m is None:
            raise ValueError(f"unexpected filename: {path.name}")
        df = pd.read_csv(path)
        df["participant"] = int(m.group(1))
        df["task"] = int(m.group(2))
        df = df.sort_values("timestamps_start_ms").reset_index(drop=True)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"no metrics CSVs found under {METRICS_DIR}")

    pooled = pd.concat(frames, ignore_index=True)
    pooled["rating"] = pooled[LABEL_COL].astype(float)
    return pooled


def recording_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """One row per recording: rating, window count, and any timing discontinuity."""
    rows = []
    for (pid, task), g in windows.groupby(["participant", "task"], sort=True):
        gaps = g["timestamps_start_ms"].diff().dropna()
        rows.append(
            {
                "participant": pid,
                "task": task,
                "rating": g["rating"].iloc[0],
                "n_ratings_in_file": g["rating"].nunique(),
                "n_windows": len(g),
                "duration_s": (g["timestamps_end_ms"].iloc[-1]
                               - g["timestamps_start_ms"].iloc[0]) / 1000.0,
                "max_gap_ms": float(gaps.max()) if len(gaps) else np.nan,
                "n_gaps": int((gaps != WINDOW_MS).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_epochs(windows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate windows into fixed-length epochs, one row per epoch.

    Epoch index is elapsed time since the start of that recording divided by the
    epoch length, so epochs never straddle a recording boundary or a dropout.
    Epochs covering less than MIN_WINDOW_COVERAGE of their nominal duration are
    discarded, which removes partial trailing epochs and dropout-riddled ones.
    """
    df = windows.copy()
    start0 = df.groupby(["participant", "task"])["timestamps_start_ms"].transform("min")
    df["epoch"] = ((df["timestamps_start_ms"] - start0) // EPOCH_MS).astype(int)

    keys = ["participant", "task", "epoch"]
    grouped = df.groupby(keys, sort=True)

    stats = grouped[AGG_BASE].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]

    derived = grouped.agg(
        # blink_flag_any is 0/1 per window, so its mean over an epoch is a blink rate.
        blink_rate=("blink_flag_any", "mean"),
        # NaN in these columns is structural, not random: saccade amplitude/velocity
        # are absent exactly when saccade_count == 0, and FDI when fixation_count < 2.
        # The proportion of such windows is therefore itself a feature.
        sacc_missing_frac=("saccade_amplitude_degree", lambda s: float(s.isna().mean())),
        fdi_missing_frac=("FDI", lambda s: float(s.isna().mean())),
        # GTE is 0 in ~99% of individual 250 ms windows; the fraction of windows with
        # any gaze transition entropy is the usable form of it at epoch scale.
        gte_nonzero_frac=("GTE", lambda s: float((s > 0).mean())),
        fixation_zero_frac=("fixation_count", lambda s: float((s == 0).mean())),
        n_windows=("rating", "size"),
        rating=("rating", "first"),
    )

    epochs = stats.join(derived).reset_index()

    expected = EPOCH_MS / WINDOW_MS
    epochs["coverage"] = epochs["n_windows"] / expected
    kept = epochs[epochs["coverage"] >= MIN_WINDOW_COVERAGE].reset_index(drop=True)
    return kept


def feature_names(epochs: pd.DataFrame) -> list[str]:
    """Feature columns, i.e. everything that is not a key, a label or bookkeeping."""
    non_features = {"participant", "task", "epoch", "rating", "n_windows", "coverage"}
    return [c for c in epochs.columns if c not in non_features]


def load_or_build_epochs(refresh: bool = False) -> pd.DataFrame:
    """Build the epoch table, caching it so repeated runs skip the 130-file read."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"epochs_{int(EPOCH_MS / 1000)}s.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    epochs = build_epochs(load_windows())
    epochs.to_parquet(cache, index=False)
    return epochs

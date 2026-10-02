"""Preprocessing steps P1-P8 (methodology-colet.md §4).

Every function takes and returns plain pandas/numpy objects so each step can be checked
on its own. Missing data stays NaN; nothing here fills a long gap or invents a sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from config import (
    BLINK_MAX_S, BLINK_MERGE_S, BLINK_MIN_S, CONFIDENCE_MIN, GAZE_GRID_HZ, GAZE_MAX_INTERP_S,
    PUPIL_GRID_HZ, PUPIL_LOWPASS_HZ, PUPIL_LOWPASS_ORDER, PUPIL_MAD_MULTIPLIER,
    PUPIL_MAX_INTERP_S, PUPIL_MAX_MM, PUPIL_METHOD_TAG, PUPIL_MIN_MM, VALID_TIME_MAX_GAP_S,
)

_EMPTY_BLINKS = pd.DataFrame({"start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})


def trim_to_gaze_range(rec: dict) -> dict:
    """P1: keep pupil samples and blinks inside the gaze time range.

    P18 A3/A4 carry stray pupil samples about two hours after the activity ended.
    """
    t = rec["gaze"]["gaze_timestamp"]
    t0, t1 = t.min(), t.max()
    pupil = rec["pupil"]
    pupil = pupil[(pupil["pupil_timestamp"] >= t0) & (pupil["pupil_timestamp"] <= t1)]
    blinks = rec["blinks"]
    blinks = blinks[(blinks["start_timestamp"] >= t0) & (blinks["end_timestamp"] <= t1)]
    return {"gaze": rec["gaze"], "pupil": pupil.reset_index(drop=True),
            "blinks": blinks.reset_index(drop=True)}


def invalid_fraction(gaze: pd.DataFrame) -> float:
    """P2/P3: share of gaze samples with confidence below CONFIDENCE_MIN."""
    return float((gaze["confidence"] < CONFIDENCE_MIN).mean())


def valid_time_s(timestamps) -> float:
    """P4: recording time in seconds, leaving out every gap longer than 1 s."""
    ts = np.sort(np.asarray(timestamps, dtype=float))
    if len(ts) < 2:
        return 0.0
    d = np.diff(ts)
    return float(d[d <= VALID_TIME_MAX_GAP_S].sum())


def clean_blinks(blinks: pd.DataFrame) -> pd.DataFrame:
    """P5: merge blinks less than 100 ms apart, then keep those lasting 50-500 ms.

    Merging first means a blink split in two by a single good frame counts once.
    """
    if len(blinks) == 0:
        return _EMPTY_BLINKS.copy()
    b = blinks[["start_timestamp", "end_timestamp"]].sort_values("start_timestamp").to_numpy(float)
    merged = [list(b[0])]
    for s, e in b[1:]:
        if s - merged[-1][1] < BLINK_MERGE_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = pd.DataFrame(merged, columns=["start", "end"])
    dur = out["end"] - out["start"]
    return out[(dur >= BLINK_MIN_S) & (dur <= BLINK_MAX_S)].reset_index(drop=True)


def in_intervals(t: np.ndarray, intervals: pd.DataFrame) -> np.ndarray:
    """True where t falls inside any [start, end] interval."""
    mask = np.zeros(len(t), bool)
    for s, e in zip(intervals["start"], intervals["end"]):
        mask |= (t >= s) & (t <= e)
    return mask


def interp_with_gap_limit(t, values, grid, max_gap) -> np.ndarray:
    """Linear interpolation onto `grid`, NaN wherever the bracketing samples are > max_gap apart."""
    t = np.asarray(t, float)
    values = np.asarray(values, float)
    one_d = values.ndim == 1
    v = values[:, None] if one_d else values
    out = np.full((len(grid), v.shape[1]), np.nan)
    if len(t) < 2:
        return out[:, 0] if one_d else out
    idx = np.searchsorted(t, grid)
    inside = (idx > 0) & (idx < len(t))
    left = np.where(inside, t[np.clip(idx - 1, 0, len(t) - 1)], np.nan)
    right = np.where(inside, t[np.clip(idx, 0, len(t) - 1)], np.nan)
    exact = np.isin(grid, t)
    ok = (inside & (right - left <= max_gap + 1e-9)) | exact
    for k in range(v.shape[1]):
        out[ok, k] = np.interp(grid[ok], t, v[:, k])
    return out[:, 0] if one_d else out


def gaze_directions(gaze: pd.DataFrame, blinks: pd.DataFrame):
    """P6/P8: average the two eyes' gaze directions onto a uniform grid.

    A sample is used only if confidence >= 0.8, it is outside a blink, and BOTH eyes have
    a direction. Each eye's direction alone is bent ~10 degrees inward (researcher-notes
    N25), so a one-eye sample would create a fake jump; it is treated as missing instead
    (decided 2026-10-02). Gaps up to 75 ms are interpolated; longer gaps stay NaN.
    """
    t = gaze["gaze_timestamp"].to_numpy(float)
    n0 = gaze[[f"gaze_normal0_{a}" for a in "xyz"]].to_numpy(float)
    n1 = gaze[[f"gaze_normal1_{a}" for a in "xyz"]].to_numpy(float)
    ok = ((gaze["confidence"].to_numpy() >= CONFIDENCE_MIN)
          & np.isfinite(n0).all(1) & np.isfinite(n1).all(1)
          & ~in_intervals(t, blinks))
    grid = np.arange(t.min(), t.max(), 1.0 / GAZE_GRID_HZ) if len(t) > 1 else np.array([])
    if ok.sum() < 2:
        return grid, np.full((len(grid), 3), np.nan)
    mean = (n0[ok] + n1[ok]) / 2.0
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    order = np.argsort(t[ok])
    xyz = interp_with_gap_limit(t[ok][order], mean[order], grid, GAZE_MAX_INTERP_S)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz[in_intervals(grid, blinks)] = np.nan
    return grid, xyz


def binocular_fraction(gaze: pd.DataFrame) -> float:
    """Share of confident samples that have both eyes' directions (QC only)."""
    conf = gaze["confidence"].to_numpy() >= CONFIDENCE_MIN
    if conf.sum() == 0:
        return float("nan")
    n0 = gaze[[f"gaze_normal0_{a}" for a in "xyz"]].to_numpy(float)
    n1 = gaze[[f"gaze_normal1_{a}" for a in "xyz"]].to_numpy(float)
    both = np.isfinite(n0).all(1) & np.isfinite(n1).all(1)
    return float(both[conf].mean())


def _dilation_speed_filter(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Kret & Sjak-Shie 2019: drop samples whose dilation speed exceeds median + n*MAD."""
    if len(d) < 3:
        return np.ones(len(d), bool)
    speed = np.abs(np.diff(d)) / np.maximum(np.diff(t), 1e-6)
    s = np.maximum(np.r_[speed[0], speed], np.r_[speed, speed[-1]])
    med = np.median(s)
    mad = max(np.median(np.abs(s - med)), 1e-6)
    return s <= med + PUPIL_MAD_MULTIPLIER * mad


def _lowpass_runs(d: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth low-pass on each contiguous non-NaN run long enough to filter."""
    sos = butter(PUPIL_LOWPASS_ORDER, PUPIL_LOWPASS_HZ, fs=PUPIL_GRID_HZ, output="sos")
    out = d.copy()
    finite = np.isfinite(d)
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start > 30:
            out[start:stop] = sosfiltfilt(sos, d[start:stop])
        else:
            out[start:stop] = np.nan
    return out


def clean_pupil(pupil: pd.DataFrame, blinks: pd.DataFrame, t_start: float, t_end: float):
    """P7: cleaned pupil diameter (mm) on a uniform grid; NaN where missing.

    3d rows only; confidence >= 0.8; outside blinks; 1.5-9 mm; dilation-speed filter per
    eye; each eye put on the grid with gaps <= 250 ms filled; the two eyes averaged where
    both exist (consistent with the both-eyes rule for gaze); 4 Hz low-pass.
    A pupil value needs both eyes; blink gaps up to 250 ms are interpolated for pupil
    (standard pupillometry), while gaze keeps blinks missing.
    """
    grid = np.arange(t_start, t_end, 1.0 / PUPIL_GRID_HZ)
    rows = pupil[pupil["method"].astype(str).str.contains(PUPIL_METHOD_TAG)]
    per_eye = []
    for eye in (0.0, 1.0):
        e = rows[rows["eye_id"] == eye].sort_values("pupil_timestamp")
        t = e["pupil_timestamp"].to_numpy(float)
        d = e["diameter_3d"].to_numpy(float)
        keep = ((e["confidence"].to_numpy() >= CONFIDENCE_MIN)
                & (d >= PUPIL_MIN_MM) & (d <= PUPIL_MAX_MM)
                & ~in_intervals(t, blinks))
        t, d = t[keep], d[keep]
        if len(t):
            fine = _dilation_speed_filter(t, d)
            t, d = t[fine], d[fine]
        per_eye.append(interp_with_gap_limit(t, d, grid, PUPIL_MAX_INTERP_S))
    both = np.vstack(per_eye)
    mean = np.where(np.isfinite(both).all(0), both.mean(0), np.nan)
    return grid, _lowpass_runs(mean)


def count_refits(pupil: pd.DataFrame) -> int:
    """Number of 3D eye-model refits: max over eyes of (distinct model ids - 1)."""
    rows = pupil[pupil["method"].astype(str).str.contains(PUPIL_METHOD_TAG)]
    if len(rows) == 0:
        return 0
    per_eye = rows.groupby("eye_id")["model_id"].nunique() - 1
    return int(max(per_eye.max(), 0))

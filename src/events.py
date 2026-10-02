"""Fixation and saccade detection (methodology-colet.md §5).

Velocity: a five-point smoothed central difference, h = [1, 1, 0, -1, -1] / (6 dt), applied
to each component of the unit gaze vector on the uniform grid; for unit vectors the norm of
the derivative is the angular speed. This filter is our own choice: reference check C7
(2026-10-02) found that Duchowski's 5-tap filter is a {1, 2, 3, 2, 1} smoother of sample-to-
sample displacement, not this differentiator.
Classification: I-VT at 45 deg/s (Salvucci & Goldberg 2000), fixations >= 55 ms (COLET),
samples above 1000 deg/s rejected as artifacts (Hausamann 2020).
"""
from __future__ import annotations

import numpy as np

from config import IVT_THRESHOLD_DEG_S, MAX_VELOCITY_DEG_S, MIN_FIXATION_S

_FIX, _SAC, _BAD = 1, 0, 2


def angular_speed_deg_s(xyz: np.ndarray, hz: float) -> np.ndarray:
    """Angular speed per sample; NaN in the first/last two samples and next to any NaN."""
    n = len(xyz)
    speed = np.full(n, np.nan)
    if n < 5:
        return speed
    d = (xyz[4:] + xyz[3:-1] - xyz[1:-3] - xyz[:-4]) * hz / 6.0
    speed[2:-2] = np.degrees(np.linalg.norm(d, axis=1))
    # The centre tap has weight 0, so a NaN at sample i would not reach speed[i] itself.
    speed[~np.isfinite(xyz).all(axis=1)] = np.nan
    return speed


def _runs(labels: np.ndarray):
    """(label, start, stop) for each run of equal, non-NaN labels."""
    out, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else np.nan
        if start is not None and (np.isnan(cur) or cur != labels[start]):
            out.append((int(labels[start]), start, i))
            start = None
        if start is None and i < len(labels) and not np.isnan(cur):
            start = i
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def detect_events(xyz: np.ndarray, hz: float) -> dict:
    speed = angular_speed_deg_s(xyz, hz)
    labels = np.full(len(speed), np.nan)
    finite = np.isfinite(speed)
    labels[finite & (speed < IVT_THRESHOLD_DEG_S)] = _FIX
    labels[finite & (speed >= IVT_THRESHOLD_DEG_S)] = _SAC
    artifact = finite & (speed > MAX_VELOCITY_DEG_S)
    labels[artifact] = _BAD

    fix, amp, peak, mean_v, sac_dur = [], [], [], [], []
    for lab, start, stop in _runs(labels):
        if lab == _FIX:
            dur = (stop - start) / hz
            if dur >= MIN_FIXATION_S:
                fix.append(dur)
        elif lab == _SAC:
            before = labels[start - 1] if start > 0 else np.nan
            after = labels[stop] if stop < len(labels) else np.nan
            if before == _BAD or after == _BAD:
                continue
            a, b = xyz[max(start - 1, 0)], xyz[min(stop, len(xyz) - 1)]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            amp.append(_angle_deg(a, b))
            peak.append(float(np.nanmax(speed[start:stop])))
            mean_v.append(float(np.nanmean(speed[start:stop])))
            sac_dur.append((stop - start) / hz)
    return {
        "fixation_durations_s": np.array(fix),
        "saccade_amplitudes_deg": np.array(amp),
        "saccade_peak_velocities": np.array(peak),
        "saccade_mean_velocities": np.array(mean_v),
        "saccade_durations_s": np.array(sac_dur),
        "artifact_mask": artifact,
    }

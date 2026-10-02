"""The ten whole-activity features plus QC columns for one recording (§6, P10).

Blink rate is per minute of valid time (P4). Fixation and saccade rates are per second
of time with a valid gaze direction, so missing gaze does not lower them. Durations are
in ms, angles in degrees, velocities in deg/s, pupil in mm. With no valid gaze time the
rates are NaN (no data); with gaze time but no events they are 0. Per-event averages are
NaN when there are no events.
"""
from __future__ import annotations

import numpy as np

from config import GAZE_GRID_HZ
from events import detect_events
from preprocess import (
    binocular_fraction, clean_blinks, clean_pupil, count_refits, gaze_directions,
    trim_to_gaze_range, valid_time_s,
)


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def recording_features(rec: dict) -> dict:
    rec = trim_to_gaze_range(rec)
    gaze = rec["gaze"]
    t = gaze["gaze_timestamp"].to_numpy(float)
    vt = valid_time_s(t)
    blinks = clean_blinks(rec["blinks"])

    grid, xyz = gaze_directions(gaze, blinks)
    ev = detect_events(xyz, GAZE_GRID_HZ)
    finite = np.isfinite(xyz).all(1)
    keep = finite & ~ev["artifact_mask"]
    az = np.degrees(np.arctan2(xyz[keep, 0], xyz[keep, 2]))
    el = np.degrees(np.arctan2(xyz[keep, 1], np.hypot(xyz[keep, 0], xyz[keep, 2])))

    _, pupil = clean_pupil(rec["pupil"], blinks, t.min(), t.max())

    fix = ev["fixation_durations_s"]
    sac = ev["saccade_amplitudes_deg"]
    gaze_time_s = float(finite.sum()) / GAZE_GRID_HZ
    blink_per_min = len(blinks) * 60.0 / vt if vt > 0 else float("nan")
    return {
        "pupil_mean": float(np.nanmean(pupil)) if np.isfinite(pupil).any() else float("nan"),
        "pupil_sd": float(np.nanstd(pupil, ddof=1)) if np.isfinite(pupil).sum() > 1 else float("nan"),
        "blink_rate": blink_per_min,
        "fixation_rate": len(fix) / gaze_time_s if gaze_time_s > 0 else float("nan"),
        "fixation_duration": _mean(fix) * 1000.0,
        "saccade_rate": len(sac) / gaze_time_s if gaze_time_s > 0 else float("nan"),
        "saccade_amplitude": _mean(sac),
        "saccade_peak_velocity": _mean(ev["saccade_peak_velocities"]),
        "gaze_sd_x": float(np.std(az, ddof=1)) if len(az) > 1 else float("nan"),
        "gaze_sd_y": float(np.std(el, ddof=1)) if len(el) > 1 else float("nan"),
        # ---- QC only, never model inputs
        "valid_time_s": vt,
        "gaze_time_s": gaze_time_s,
        "binocular_fraction": binocular_fraction(gaze),
        "refits": count_refits(rec["pupil"]),
        "n_blinks": int(len(blinks)),
        "n_fixations": int(len(fix)),
        "n_saccades": int(len(sac)),
        "fixation_median_ms": float(np.median(fix) * 1000.0) if len(fix) else float("nan"),
        "saccade_mean_velocity": _mean(ev["saccade_mean_velocities"]),
        "saccade_duration_ms": _mean(ev["saccade_durations_s"]) * 1000.0,
        "gaze_valid_fraction": float(finite.mean()) if len(xyz) else float("nan"),
        "pupil_valid_fraction": float(np.isfinite(pupil).mean()) if len(pupil) else float("nan"),
    }

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
from events import angular_speed_deg_s, detect_events
from preprocess import (
    binocular_fraction, clean_blinks, clean_pupil, count_refits, gaze_directions,
    trim_to_gaze_range, valid_time_s,
)


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def recording_signals(rec: dict) -> dict:
    """Every cleaned signal of one recording as plain value arrays, plus the rates and QC.

    "signals": 1-D arrays without NaNs, one value per sample (pupil mm, gaze_x/gaze_y deg,
    gaze_speed deg/s) or per event (fixation/saccade/blink durations in ms, saccade amplitude
    deg, saccade velocities deg/s). These are what flatten.py summarizes.
    "rates": blink, fixation and saccade rates. "qc": quality numbers, never model inputs.
    """
    rec = trim_to_gaze_range(rec)
    gaze = rec["gaze"]
    t = gaze["gaze_timestamp"].to_numpy(float)
    vt = valid_time_s(t)
    blinks = clean_blinks(rec["blinks"])

    grid, xyz = gaze_directions(gaze, blinks)
    # Pina post-hoc run only: events and gaze speed from gaze in the room (head rotation applied);
    # the gaze angle features below stay head-relative.
    event_xyz = xyz
    if "world_normal0_x" in gaze:
        world = gaze.drop(columns=[c for c in gaze if c.startswith("gaze_normal")]).rename(
            columns=lambda c: c.replace("world_normal", "gaze_normal"))
        _, event_xyz = gaze_directions(world, blinks)
    ev = detect_events(event_xyz, GAZE_GRID_HZ)
    finite = np.isfinite(xyz).all(1)
    keep = finite & ~ev["artifact_mask"]
    az = np.degrees(np.arctan2(xyz[keep, 0], xyz[keep, 2]))
    el = np.degrees(np.arctan2(xyz[keep, 1], np.hypot(xyz[keep, 0], xyz[keep, 2])))
    speed = angular_speed_deg_s(event_xyz, GAZE_GRID_HZ)
    speed = speed[np.isfinite(speed) & ~ev["artifact_mask"]]

    _, pupil = clean_pupil(rec["pupil"], blinks, t.min(), t.max())

    fix = ev["fixation_durations_s"]
    sac = ev["saccade_amplitudes_deg"]
    gaze_time_s = float(finite.sum()) / GAZE_GRID_HZ
    signals = {
        "pupil": pupil[np.isfinite(pupil)],
        "gaze_x": az,
        "gaze_y": el,
        "gaze_speed": speed,
        "fixation_duration": fix * 1000.0,
        "saccade_amplitude": sac,
        "saccade_peak_velocity": ev["saccade_peak_velocities"],
        "saccade_mean_velocity": ev["saccade_mean_velocities"],
        "saccade_duration": ev["saccade_durations_s"] * 1000.0,
        "blink_duration": (blinks["end"] - blinks["start"]).to_numpy(float) * 1000.0,
    }
    rates = {
        "blink_rate": len(blinks) * 60.0 / vt if vt > 0 else float("nan"),
        "fixation_rate": len(fix) / gaze_time_s if gaze_time_s > 0 else float("nan"),
        "saccade_rate": len(sac) / gaze_time_s if gaze_time_s > 0 else float("nan"),
    }
    qc = {
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
        "artifact_samples": int(ev["artifact_mask"].sum()),
    }
    return {"signals": signals, "rates": rates, "qc": qc}


def recording_features(rec: dict) -> dict:
    """The defended ten features plus QC columns, computed from recording_signals."""
    s = recording_signals(rec)
    sig, qc = s["signals"], dict(s["qc"])
    qc.pop("artifact_samples")
    pupil = sig["pupil"]
    return {
        "pupil_mean": float(np.mean(pupil)) if len(pupil) else float("nan"),
        "pupil_sd": float(np.std(pupil, ddof=1)) if len(pupil) > 1 else float("nan"),
        "blink_rate": s["rates"]["blink_rate"],
        "fixation_rate": s["rates"]["fixation_rate"],
        "fixation_duration": _mean(sig["fixation_duration"]),
        "saccade_rate": s["rates"]["saccade_rate"],
        "saccade_amplitude": _mean(sig["saccade_amplitude"]),
        "saccade_peak_velocity": _mean(sig["saccade_peak_velocity"]),
        "gaze_sd_x": float(np.std(sig["gaze_x"], ddof=1)) if len(sig["gaze_x"]) > 1 else float("nan"),
        "gaze_sd_y": float(np.std(sig["gaze_y"], ddof=1)) if len(sig["gaze_y"]) > 1 else float("nan"),
        # ---- QC only, never model inputs
        **qc,
    }

"""Pina et al. 2025 (Zenodo 18669059) -> COLET-shaped tables -> the unchanged feature code.

Plan: docs/pina-analysis-plan.md (written before the eye data were loaded).

Varjo XR-4 HMDgaze -> the tables recording_signals() expects:
  gaze  : gaze_timestamp (s, Unix), confidence (1 if the gaze and both eyes are VALID, else 0),
          gaze_normal0_* = left-eye forward vector, gaze_normal1_* = right-eye forward vector
          (both in the headset's frame, i.e. relative to the head)
  pupil : one row per eye per sample, method "3d", diameter_3d = pupil diameter (mm)
  blinks: runs where both eyes are INVALID (cleaned downstream by the 50-500 ms rule)
Clock: CaptureTime (ns) is mapped to Unix time through LogTime (.NET-epoch ms, local time); the
whole-hour time-zone offset is the one that puts the gaze recording around the trial log.
Each condition segment starts at its first trial and is cut into PINA_N_WINDOWS windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PINA_CONDITIONS, PINA_DATA_DIR, PINA_N_WINDOWS, PINA_WINDOW_S
from features import recording_signals
from flatten import DROPPED, flat_features
from preprocess import invalid_fraction

DOTNET_EPOCH_MS = 62135596800000      # ms from 0001-01-01 to 1970-01-01
GAZE_COLUMNS = ["CaptureTime", "LogTime", "GazeStatus", "LeftEyeStatus", "LeftEyeForward",
                "LeftPupilDiameterInMM", "RightEyeStatus", "RightEyeForward", "RightPupilDiameterInMM"]
META = ["participant", "area", "load", "window", "n_samples", "invalid_frac",
        "fixation_median_ms", "n_blinks", "error"]
TLX_SCALES = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort",
              "Frustration"]


def parse_vectors(s: pd.Series) -> np.ndarray:
    """'(x, y, z)' text -> (n, 3) floats."""
    return s.str.extract(r"\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)").astype(float).to_numpy()


def parse_numbers(s: pd.Series) -> np.ndarray:
    """Numbers written with a decimal comma -> floats (unparseable -> NaN)."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()


def unix_seconds(capture_ns: np.ndarray, log_ms: np.ndarray, trial_start_s: float) -> tuple[np.ndarray, int]:
    """CaptureTime -> Unix seconds, and the whole-hour time-zone offset that was removed.

    LogTime is local time, logged with a small lag (>= 0), so the 5th percentile of
    LogTime - CaptureTime is the clock offset. The hour offset is the one that brings the
    middle of the recording closest to the trials' median start.
    """
    cap = capture_ns / 1e9
    base = np.quantile((log_ms - DOTNET_EPOCH_MS) / 1000.0 - cap, 0.05)
    hours = min(range(-3, 4), key=lambda h: abs(np.median(cap + base - 3600 * h) - trial_start_s))
    return cap + base - 3600 * hours, hours


def blink_runs(t: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray) -> pd.DataFrame:
    """Start and end time of every run of samples in which both eyes are invalid."""
    both_bad = (~left_valid & ~right_valid).astype(int)
    edges = np.diff(np.r_[0, both_bad, 0])
    start, end = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return pd.DataFrame({"start_timestamp": t[start], "end_timestamp": t[end]})


def load_participant(user: int, data_dir: Path = PINA_DATA_DIR):
    """(gaze, pupil, blinks, trial, hour offset) for one participant."""
    trial = pd.read_csv(data_dir / "trial" / f"user_{user}.csv")
    g = pd.read_csv(data_dir / "gaze" / f"user_{user}gaze.csv", sep=";", usecols=GAZE_COLUMNS, dtype=str)
    start = trial.loc[trial.TrialType.isin(PINA_CONDITIONS), "StartTime"].median() / 1000
    t, hours = unix_seconds(g["CaptureTime"].astype("int64").to_numpy(),
                            g["LogTime"].astype("int64").to_numpy(), start)
    lv, rv = g["LeftEyeStatus"].eq("VALID").to_numpy(), g["RightEyeStatus"].eq("VALID").to_numpy()
    ok = g["GazeStatus"].eq("VALID").to_numpy() & lv & rv
    left, right = parse_vectors(g["LeftEyeForward"]), parse_vectors(g["RightEyeForward"])
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": ok.astype(float),
                         **{f"gaze_normal0_{a}": left[:, i] for i, a in enumerate("xyz")},
                         **{f"gaze_normal1_{a}": right[:, i] for i, a in enumerate("xyz")}})
    pupil = pd.concat([pd.DataFrame({"pupil_timestamp": t, "eye_id": float(eye), "method": "3d",
                                     "confidence": valid.astype(float), "diameter_3d": diam, "model_id": 0})
                       for eye, valid, diam in ((0, lv, parse_numbers(g["LeftPupilDiameterInMM"])),
                                                (1, rv, parse_numbers(g["RightPupilDiameterInMM"])))],
                      ignore_index=True)
    return gaze, pupil, blink_runs(t, lv, rv), trial, hours


def _between(df: pd.DataFrame, col: str, a: float, b: float) -> pd.DataFrame:
    return df[(df[col] >= a) & (df[col] < b)]


def participant_windows(user: int, data_dir: Path = PINA_DATA_DIR) -> tuple[pd.DataFrame, int]:
    """One row per window: META columns and every flattened feature (DROPPED removed)."""
    gaze, pupil, blinks, trial, hours = load_participant(user, data_dir)
    rows = []
    for trial_type, (area, load) in PINA_CONDITIONS.items():
        seg = trial[trial.TrialType == trial_type]
        if seg.empty:
            continue
        s0 = seg.StartTime.min() / 1000
        for w in range(PINA_N_WINDOWS):
            a, b = s0 + w * PINA_WINDOW_S, s0 + (w + 1) * PINA_WINDOW_S
            rec = {"gaze": _between(gaze, "gaze_timestamp", a, b),
                   "pupil": _between(pupil, "pupil_timestamp", a, b),
                   "blinks": blinks[(blinks.end_timestamp >= a) & (blinks.start_timestamp <= b)]}
            base = {"participant": user, "area": area, "load": load, "window": w + 1,
                    "n_samples": len(rec["gaze"]),
                    "invalid_frac": invalid_fraction(rec["gaze"]) if len(rec["gaze"]) else 1.0}
            try:
                out = recording_signals(rec)
                flat = {k: v for k, v in flat_features(out).items() if k not in DROPPED}
                rows.append({**base, "fixation_median_ms": out["qc"]["fixation_median_ms"],
                             "n_blinks": out["qc"]["n_blinks"], **flat})
            except Exception as exc:                  # kept as a row so it is counted, never silent
                rows.append({**base, "error": repr(exc)[:150]})
    return pd.DataFrame(rows), hours


def participants(data_dir: Path = PINA_DATA_DIR) -> list[int]:
    return sorted(int(p.name[len("user_"):-len("gaze.csv")]) for p in (data_dir / "gaze").glob("user_*gaze.csv"))


def demographics(data_dir: Path = PINA_DATA_DIR) -> pd.DataFrame:
    """Age and sex per participant from demo/Demographic/*.csv (one 'key,value' line each)."""
    rows = []
    for f in sorted((data_dir / "demo" / "Demographic").glob("*.csv")):
        fields = dict(line.split(",", 1) for line in f.read_text(encoding="utf-8").splitlines() if "," in line)
        rows.append({"participant": int(fields["SubjectID"]), "age": float(fields["Age"]),
                     "sex": fields["Sex"].strip().lower()})
    return pd.DataFrame(rows)


def tlx_table(data_dir: Path = PINA_DATA_DIR) -> pd.DataFrame:
    """Raw NASA-TLX (mean of the six 0-20 scales) per participant and task (Visual, Audio,
    Audiovisual)."""
    rows = []
    for f in sorted((data_dir / "tlx" / "NASA_TLX").glob("*/*.csv")):
        s = pd.read_csv(f, header=None, index_col=0)[1]
        rows.append({"participant": int(s["SubjectID"]), "task": f.parent.name,
                     "raw_tlx": float(pd.to_numeric(s[TLX_SCALES]).mean())})
    return pd.DataFrame(rows).pivot_table(index="participant", columns="task", values="raw_tlx")

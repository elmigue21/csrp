"""Test setup: put src/ on the path and generate small synthetic COLET folders.

The synthetic data mimics the real Parquet layout (column names, the 2d/3d pupil row
duplication, the repr-string `method` column) and has a known structure: fixations of
FIX_S seconds separated by 10-degree saccades lasting SACC_S seconds, a pupil that is
larger in activity 4, and more blinks in activity 4. Tests compare against these.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GAZE_HZ = 250.0
PUPIL_HZ = 120.0
FIX_S = 0.30
SACC_S = 0.04
SACC_DEG = 10.0
EYE_BIAS_DEG = 5.0            # each eye bent inward by this much (cancels when averaged)
METHOD_3D = "array(['3d c++'], dtype='<U6')"
METHOD_2D = "array(['2d c++'], dtype='<U6')"


def unit_from_angles(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.column_stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def gaze_path(seconds, rng, t0=100.0):
    """Timestamps and true (az, el) in degrees: fixations joined by 10-degree saccades."""
    t = np.arange(t0, t0 + seconds, 1 / GAZE_HZ)
    az, el = np.zeros_like(t), np.zeros_like(t)
    cur = np.array([0.0, 0.0])
    phase_start, in_fix, target = t0, True, cur
    for i, ti in enumerate(t):
        elapsed = ti - phase_start
        if in_fix and elapsed >= FIX_S:
            ang = rng.uniform(0, 2 * np.pi)
            step = SACC_DEG * np.array([np.cos(ang), np.sin(ang)])
            target = np.clip(cur + step, -15, 15)
            if np.linalg.norm(target - cur) < 0.5 * SACC_DEG:
                target = cur - step
            start, in_fix, phase_start = cur.copy(), False, ti
            elapsed = 0.0
        if not in_fix:
            frac = min(elapsed / SACC_S, 1.0)
            cur = start + frac * (target - start)
            if elapsed >= SACC_S:
                cur, in_fix, phase_start = target.copy(), True, ti
        az[i], el[i] = cur
    return t, az, el


def make_recording(participant, activity, rng, seconds=8.0, low_quality=False, one_eye_frac=0.0):
    t, az, el = gaze_path(seconds, rng)
    n = len(t)
    n_blinks = 1 if activity == 1 else 4
    blink_starts = rng.uniform(t[0] + 0.5, t[-1] - 0.5, n_blinks) if activity != 1 or participant % 2 else []
    blinks = pd.DataFrame({
        "start_timestamp": np.sort(blink_starts),
        "end_timestamp": np.sort(blink_starts) + 0.15,
    })
    blinks["duration"] = blinks["end_timestamp"] - blinks["start_timestamp"]

    in_blink = np.zeros(n, bool)
    for s, e in zip(blinks.start_timestamp, blinks.end_timestamp):
        in_blink |= (t >= s) & (t <= e)
    conf = np.where(in_blink, 0.1, 0.99)
    if low_quality:
        conf[: int(0.5 * n)] = 0.2
    eye0 = unit_from_angles(az + EYE_BIAS_DEG, el)
    eye1 = unit_from_angles(az - EYE_BIAS_DEG, el)
    one_eye = rng.random(n) < one_eye_frac
    eye1[one_eye] = np.nan
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": conf})
    for k, axis in enumerate("xyz"):
        gaze[f"gaze_normal0_{axis}"] = eye0[:, k]
        gaze[f"gaze_normal1_{axis}"] = eye1[:, k]

    tp = np.arange(t[0], t[-1], 1 / PUPIL_HZ)
    base = 3.0 + 0.1 * participant + (0.4 if activity == 4 else 0.0)
    rows = []
    for eye in (0, 1):
        d = base + 0.05 * np.sin(2 * np.pi * 0.5 * tp) + rng.normal(0, 0.01, len(tp))
        pb = np.zeros(len(tp), bool)
        for s, e in zip(blinks.start_timestamp, blinks.end_timestamp):
            pb |= (tp >= s) & (tp <= e)
        for method in (METHOD_2D, METHOD_3D):
            rows.append(pd.DataFrame({
                "pupil_timestamp": tp, "eye_id": float(eye),
                "confidence": np.where(pb, 0.1, 0.99), "method": method,
                "diameter_3d": np.where(pb, 0.0, d), "model_id": 1.0,
            }))
    pupil = pd.concat(rows, ignore_index=True)
    return {"gaze": gaze, "pupil": pupil, "blinks": blinks}


def write_synthetic_dataset(root, n_participants=6, seconds=8.0, seed=0, low_quality=None):
    """Write pXX_tY_*.parquet for activities 1-4 and annotation.csv; return the folder.

    low_quality: optional set of (participant, activity) given > 35% invalid samples.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    low_quality = low_quality or set()
    ann = []
    for p in range(1, n_participants + 1):
        for a in (1, 2, 3, 4):
            rec = make_recording(p, a, rng, seconds, low_quality=(p, a) in low_quality)
            for signal, df in rec.items():
                df.to_parquet(root / f"p{p:02d}_t{a}_{signal}.parquet")
            ann.append({"participant": p, "task": a, "mean": 15 + 12 * a + rng.normal(0, 3)})
    pd.DataFrame(ann).to_csv(root / "annotation.csv", index=False)
    return root

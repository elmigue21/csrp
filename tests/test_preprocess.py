import numpy as np
import pandas as pd

from conftest import make_recording, unit_from_angles
from preprocess import (
    clean_blinks, clean_pupil, count_refits, gaze_directions, in_intervals,
    interp_with_gap_limit, invalid_fraction, trim_to_gaze_range, valid_time_s,
)


def test_valid_time_skips_gaps_over_one_second():
    ts = np.r_[np.arange(0, 2, 0.01), np.arange(5, 6, 0.01)]   # 3 s gap
    assert abs(valid_time_s(ts) - (1.99 + 0.99)) < 0.02


def test_invalid_fraction():
    gaze = pd.DataFrame({"confidence": [0.9, 0.5, 0.79, 0.95]})
    assert invalid_fraction(gaze) == 0.5


def test_trim_drops_stray_pupil_samples():
    rec = {
        "gaze": pd.DataFrame({"gaze_timestamp": [10.0, 11.0, 12.0]}),
        "pupil": pd.DataFrame({"pupil_timestamp": [10.5, 11.5, 7000.0]}),
        "blinks": pd.DataFrame({"start_timestamp": [10.2, 7000.0], "end_timestamp": [10.4, 7000.2],
                                "duration": [0.2, 0.2]}),
    }
    out = trim_to_gaze_range(rec)
    assert out["pupil"]["pupil_timestamp"].tolist() == [10.5, 11.5]
    assert len(out["blinks"]) == 1


def test_clean_blinks_merges_then_filters():
    b = pd.DataFrame({"start_timestamp": [1.0, 1.25, 3.0, 5.0],
                      "end_timestamp": [1.2, 1.35, 3.02, 5.9]})
    out = clean_blinks(b)
    # 1.0-1.2 and 1.25-1.35 merge (gap 50 ms) into 350 ms; 20 ms and 900 ms dropped
    assert out[["start", "end"]].values.tolist() == [[1.0, 1.35]]


def test_clean_blinks_empty():
    out = clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []}))
    assert len(out) == 0 and list(out.columns) == ["start", "end"]


def test_interp_with_gap_limit():
    t = np.array([0.0, 0.05, 0.10, 0.50])
    v = np.array([0.0, 1.0, 2.0, 10.0])
    grid = np.array([0.025, 0.075, 0.3])
    out = interp_with_gap_limit(t, v, grid, max_gap=0.075)
    assert np.allclose(out[:2], [0.5, 1.5]) and np.isnan(out[2])


def test_gaze_directions_average_eyes_and_drop_one_eye_samples():
    rng = np.random.default_rng(0)
    rec = make_recording(2, 1, rng, seconds=2.0, one_eye_frac=0.0)    # P2 A1: no blinks
    t, xyz = gaze_directions(rec["gaze"], clean_blinks(rec["blinks"]))
    ok = ~np.isnan(xyz[:, 0])
    assert ok.mean() > 0.8
    # averaging the inward-biased eyes recovers the true horizontal angle (~0 at start)
    az0 = np.degrees(np.arctan2(xyz[ok][0, 0], xyz[ok][0, 2]))
    assert abs(az0) < 0.5
    rec2 = make_recording(1, 2, np.random.default_rng(0), seconds=2.0, one_eye_frac=1.0)
    _, xyz2 = gaze_directions(rec2["gaze"], clean_blinks(rec2["blinks"]))
    assert np.isnan(xyz2).all()


def test_gaze_directions_gap_over_75ms_stays_missing():
    t = np.r_[np.arange(0, 1, 0.004), np.arange(1.2, 2, 0.004)]
    v = unit_from_angles(np.zeros_like(t), np.zeros_like(t))
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": 0.99})
    for k, axis in enumerate("xyz"):
        gaze[f"gaze_normal0_{axis}"] = v[:, k]
        gaze[f"gaze_normal1_{axis}"] = v[:, k]
    grid, xyz = gaze_directions(gaze, clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []})))
    inside_gap = (grid > 1.01) & (grid < 1.19)
    assert np.isnan(xyz[inside_gap]).all()
    assert not np.isnan(xyz[grid < 0.9]).any()


def test_clean_pupil_removes_zeros_and_keeps_level():
    rng = np.random.default_rng(1)
    rec = make_recording(2, 4, rng, seconds=4.0)
    blinks = clean_blinks(rec["blinks"])
    t0, t1 = rec["gaze"].gaze_timestamp.min(), rec["gaze"].gaze_timestamp.max()
    grid, d = clean_pupil(rec["pupil"], blinks, t0, t1)
    assert np.nanmin(d) > 1.5
    assert abs(np.nanmean(d) - (3.0 + 0.2 + 0.4)) < 0.05


def test_clean_pupil_spike_removed():
    tp = np.arange(0, 3, 1 / 120)
    rows = []
    for eye in (0, 1):
        d = np.full(len(tp), 3.0)
        d[100] = 6.0                     # impossible 3 mm jump in one sample
        rows.append(pd.DataFrame({"pupil_timestamp": tp, "eye_id": float(eye), "confidence": 0.99,
                                  "method": "array(['3d c++'], dtype='<U6')",
                                  "diameter_3d": d, "model_id": 1.0}))
    pupil = pd.concat(rows, ignore_index=True)
    empty = clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []}))
    _, d = clean_pupil(pupil, empty, 0.0, 3.0)
    assert np.nanmax(d) < 3.3


def test_count_refits():
    pupil = pd.DataFrame({"method": ["array(['3d c++'], dtype='<U6')"] * 4 + ["array(['2d c++'], dtype='<U6')"],
                          "eye_id": [0.0, 0.0, 1.0, 1.0, 0.0], "model_id": [1.0, 2.0, 5.0, 5.0, 9.0]})
    assert count_refits(pupil) == 1


def test_in_intervals():
    iv = pd.DataFrame({"start": [1.0], "end": [2.0]})
    assert in_intervals(np.array([0.5, 1.5, 2.5]), iv).tolist() == [False, True, False]


def test_gaze_short_blink_stays_missing():
    t = np.arange(0, 2, 1 / 250)
    v = unit_from_angles(np.zeros_like(t), np.zeros_like(t))
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": 0.99})
    for k, axis in enumerate("xyz"):
        gaze[f"gaze_normal0_{axis}"] = v[:, k]
        gaze[f"gaze_normal1_{axis}"] = v[:, k]
    blinks = pd.DataFrame({"start": [1.0], "end": [1.06]})
    grid, xyz = gaze_directions(gaze, blinks)
    inside = (grid >= 1.0) & (grid <= 1.06)
    assert inside.any() and np.isnan(xyz[inside]).all()
    assert np.isfinite(xyz[(grid < 0.9) | (grid > 1.2)]).all()


def test_lowpass_short_finite_run_becomes_nan():
    from preprocess import _lowpass_runs
    d = np.full(200, np.nan)
    d[10:20] = 3.0          # 10-sample isolated run
    d[50:150] = 3.0         # long run
    out = _lowpass_runs(d)
    assert np.isnan(out[10:20]).all()
    assert np.isfinite(out[50:150]).all()


def test_speed_filter_zero_mad_keeps_most_samples():
    from preprocess import _dilation_speed_filter
    t = np.arange(200) / 120
    d = np.full(200, 3.0)
    d[::10] += 1e-8
    keep = _dilation_speed_filter(t, d)
    assert keep.mean() > 0.8

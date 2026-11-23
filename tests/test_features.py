import pytest
import numpy as np
import pandas as pd

from config import FEATURES
from conftest import SACC_DEG, make_recording
from features import recording_features


def test_all_features_present_and_plausible():
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=8.0)
    f = recording_features(rec)
    assert set(FEATURES) <= set(f)
    assert 3.4 < f["pupil_mean"] < 3.8
    assert f["blink_rate"] > 0
    assert 2.0 < f["fixation_rate"] < 3.5            # one fixation every ~0.34 s
    assert 240 < f["fixation_duration"] < 320         # ms
    assert 0.8 * SACC_DEG < f["saccade_amplitude"] < 1.1 * SACC_DEG
    assert f["saccade_peak_velocity"] > 45
    assert f["gaze_sd_x"] > 1 and f["gaze_sd_y"] > 1


def test_no_blinks_gives_zero_rate_not_nan():
    rec = make_recording(2, 1, np.random.default_rng(3), seconds=4.0)   # even participant, A1: no blinks
    assert len(rec["blinks"]) == 0
    assert recording_features(rec)["blink_rate"] == 0.0


def test_too_short_for_events_gives_nan_not_crash():
    rec = make_recording(1, 2, np.random.default_rng(0), seconds=4.0)
    g = rec["gaze"]
    # one valid sample every 100 ms: every gap > 75 ms stays missing, so no velocity exists
    rec["gaze"] = g.assign(confidence=np.where(np.arange(len(g)) % 25 == 0, 0.99, 0.1))
    f = recording_features(rec)
    assert np.isnan(f["fixation_duration"]) and np.isnan(f["saccade_amplitude"])
    # there is valid gaze time but no event: rates are 0, not missing
    assert f["gaze_time_s"] > 0 and f["fixation_rate"] == 0.0 and f["saccade_rate"] == 0.0


def test_no_valid_gaze_time_gives_nan_rates():
    rec = make_recording(1, 2, np.random.default_rng(0), seconds=4.0)
    rec["gaze"] = rec["gaze"].assign(confidence=0.0)
    f = recording_features(rec)
    assert f["gaze_time_s"] == 0.0
    assert np.isnan(f["fixation_rate"]) and np.isnan(f["saccade_rate"])


def test_event_rate_not_lowered_by_missing_gaze():
    rec = make_recording(2, 1, np.random.default_rng(5), seconds=8.0)
    f1 = recording_features(rec)
    g = rec["gaze"]
    n = len(g)
    conf = g["confidence"].to_numpy().copy()
    conf[int(0.35 * n):int(0.65 * n)] = 0.1          # contiguous 30% block in the middle
    rec2 = dict(rec)
    rec2["gaze"] = g.assign(confidence=conf)
    f2 = recording_features(rec2)
    assert f1["fixation_rate"] > 0
    assert abs(f2["fixation_rate"] - f1["fixation_rate"]) <= 0.15 * f1["fixation_rate"]


SIGNAL_NAMES = {
    "pupil", "gaze_x", "gaze_y", "gaze_speed", "fixation_duration", "saccade_amplitude",
    "saccade_peak_velocity", "saccade_mean_velocity", "saccade_duration", "blink_duration",
}


def test_recording_signals_shape_and_units():
    from features import recording_signals
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=8.0)
    s = recording_signals(rec)
    assert set(s["signals"]) == SIGNAL_NAMES
    for name, values in s["signals"].items():
        assert values.ndim == 1 and np.isfinite(values).all(), name
    assert 240 < np.median(s["signals"]["fixation_duration"]) < 320      # ms
    assert abs(np.median(s["signals"]["blink_duration"]) - 150) < 20      # ms
    assert s["rates"]["blink_rate"] > 0
    assert set(s["rates"]) == {"blink_rate", "fixation_rate", "saccade_rate"}


def test_recording_features_unchanged_by_refactor():
    from features import recording_signals
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=8.0)
    f = recording_features(rec)
    sig = recording_signals(rec)["signals"]
    assert f["pupil_mean"] == pytest.approx(np.mean(sig["pupil"]))
    assert f["fixation_duration"] == pytest.approx(np.mean(sig["fixation_duration"]))

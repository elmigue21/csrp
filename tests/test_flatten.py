import warnings

import numpy as np
import pytest

from conftest import make_recording
from features import recording_signals
from flatten import (
    DROPPED, STATISTICS, feature_dictionary, feature_names, flat_features, kept_feature_names,
    summarize,
)


def test_summarize_known_values():
    s = summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert list(s) == list(STATISTICS) and len(STATISTICS) == 17
    assert (s["min"], s["max"], s["range"], s["mean"], s["median"]) == (1, 4, 3, 2.5, 2.5)
    assert s["sd"] == pytest.approx(1.2910, abs=1e-4)
    assert s["var"] == pytest.approx(1.6667, abs=1e-4)
    assert s["cv"] == pytest.approx(0.5164, abs=1e-4)
    assert s["iqr"] == pytest.approx(1.5)
    assert s["p25"] == pytest.approx(1.75)
    assert s["p95"] == pytest.approx(3.85)
    assert s["skew"] == pytest.approx(0.0, abs=1e-12)
    assert s["kurt"] == pytest.approx(-1.2)


def test_summarize_small_n():
    assert all(np.isnan(v) for v in summarize(np.array([])).values())
    one = summarize(np.array([7.0]))
    assert one["mean"] == one["min"] == one["max"] == 7.0
    assert all(np.isnan(one[k]) for k in ("sd", "var", "cv", "iqr", "skew", "kurt"))
    three = summarize(np.array([1.0, 2.0, 4.0]))
    assert np.isfinite(three["skew"]) and np.isnan(three["kurt"])


def test_summarize_constant_and_zero_mean():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        const = summarize(np.array([5.0, 5.0, 5.0, 5.0]))
        zero_mean = summarize(np.array([-1.0, 1.0]))
    assert const["sd"] == 0.0
    assert np.isnan(const["skew"]) and np.isnan(const["kurt"])
    assert np.isnan(zero_mean["cv"])


def test_summarize_skip():
    assert "cv" not in summarize(np.array([1.0, 2.0]), skip=("cv",))


def test_flat_features_count_and_names():
    sig = recording_signals(make_recording(1, 4, np.random.default_rng(0), seconds=8.0))
    f = flat_features(sig)
    assert len(f) == 173 == len(feature_names())
    assert list(f) == feature_names()
    assert "gaze_x__cv" in f and "gaze_y__cv" in f
    assert "pupil__p95" in f and "blink_rate" in f
    assert not any("mode" in name for name in f)


def test_no_blinks_gives_nan_durations_and_zero_rate():
    rec = make_recording(2, 1, np.random.default_rng(3), seconds=4.0)   # even participant, A1: no blinks
    f = flat_features(recording_signals(rec))
    assert all(np.isnan(v) for k, v in f.items() if k.startswith("blink_duration__"))
    assert f["blink_rate"] == 0.0


def test_feature_dictionary():
    d = feature_dictionary()
    assert len(d) == 173 and d["feature"].is_unique
    assert list(d["feature"]) == feature_names()
    pos = d[d["tag"] == "position"]
    assert len(pos) == 22 and set(pos["signal"]) == {"gaze_x", "gaze_y"}
    assert {"gaze_x__cv", "gaze_y__cv"} <= set(pos["feature"])
    assert "headset" in d.set_index("feature").loc["gaze_x__cv", "meaning"]
    assert (d["meaning"].str.len() > 0).all()
    assert d.set_index("feature").loc["pupil__p95", "unit"] == "mm"


def test_summarize_near_constant_is_constant_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = summarize(np.array([0.1, 0.1, 0.1, 0.1]))
        t = summarize(np.full(10, 1000.0 * 4 / 240))          # a 4-sample saccade, repeated
    for r in (s, t):
        assert r["sd"] == 0.0 and r["var"] == 0.0 and r["iqr"] == 0.0
        assert np.isnan(r["skew"]) and np.isnan(r["kurt"])
    assert np.isnan(summarize(np.array([1.0, 2.0]))["skew"])


def test_dropped_features_follow_the_decisions():
    assert len(DROPPED) == 33 and set(DROPPED) <= set(feature_names())
    assert sum(k.startswith("blink_duration__") for k in DROPPED) == 17
    assert {k for k in DROPPED if k.endswith("__var")} == {
        f"{s}__var" for s in ("pupil", "gaze_x", "gaze_y", "gaze_speed", "fixation_duration",
                              "saccade_amplitude", "saccade_peak_velocity",
                              "saccade_mean_velocity", "saccade_duration", "blink_duration")}
    assert not any(k.startswith(("pupil__range", "gaze_x__range", "gaze_y__range")) for k in DROPPED)
    assert all(reason.split(":")[0] in {"D1", "D6a", "D8", "D9"} for reason in DROPPED.values())
    kept = kept_feature_names()
    assert len(kept) == 140 and kept == [f for f in feature_names() if f not in DROPPED]


def test_signal_with_under_20_values_is_blank():
    from flatten import MIN_VALUES
    rng = np.random.default_rng(1)
    sig = recording_signals(make_recording(1, 4, rng, seconds=8.0))
    sig["signals"]["fixation_duration"] = rng.uniform(100, 300, MIN_VALUES - 1)
    sig["signals"]["saccade_amplitude"] = rng.uniform(1, 5, MIN_VALUES)
    f = flat_features(sig)
    assert MIN_VALUES == 20
    assert all(np.isnan(v) for k, v in f.items() if k.startswith("fixation_duration__"))
    assert all(np.isfinite(f[f"saccade_amplitude__{s}"]) for s in ("mean", "p95", "kurt"))

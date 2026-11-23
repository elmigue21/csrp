import numpy as np
import pandas as pd
import pytest

from pina import DOTNET_EPOCH_MS, blink_runs, parse_numbers, parse_vectors, unix_seconds
from run_pina import BASELINE, area_data, signal, stage_main


def test_parse_vectors_and_decimal_commas():
    v = parse_vectors(pd.Series(["(0.1, -0.2, 0.97)", "(1, 0, 0)"]))
    assert v.shape == (2, 3) and v[0, 1] == pytest.approx(-0.2)
    assert parse_numbers(pd.Series(["3,25", "x"]))[0] == pytest.approx(3.25)
    assert np.isnan(parse_numbers(pd.Series(["x"]))[0])


def test_blink_runs_need_both_eyes_invalid():
    t = np.arange(8) * 0.01
    left = np.array([1, 0, 0, 1, 0, 0, 0, 1], bool)
    right = np.array([1, 0, 0, 0, 1, 0, 0, 1], bool)
    runs = blink_runs(t, left, right)
    assert runs.start_timestamp.tolist() == pytest.approx([0.01, 0.05])
    assert runs.end_timestamp.tolist() == pytest.approx([0.02, 0.06])


def test_unix_seconds_removes_the_hour_offset():
    unix = 1_700_000_000.0 + np.arange(100)
    cap = (unix - 5.0) * 1e9                                  # capture clock 5 s behind
    log = (unix + 3600) * 1000 + DOTNET_EPOCH_MS              # local time = UTC+1, logged in ms
    t, hours = unix_seconds(cap, log, trial_start_s=unix.mean())
    assert hours == 1 and t == pytest.approx(unix)


def test_signal_families():
    assert [signal(c) for c in ("pupil__p05", "gaze_x__sd", "blink_rate", "fixation_rate",
                                "fixation_duration__max", "saccade_amplitude__sd")] == \
        ["pupil", "gaze_x", "blink", "fixation", "fixation", "saccade"]


def _table(fix_ms, people=6, windows=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, people + 1):
        for load in ("high", "low"):                         # high first: area_data must reorder
            hi = load == "high"
            for w in range(1, windows + 1):
                rows.append({"participant": p, "area": 90, "load": load, "window": w,
                             "n_samples": 6000, "invalid_frac": 0.5 if (p, w) == (1, 1) else 0.01,
                             "fixation_median_ms": fix_ms, "n_blinks": 3,
                             "pupil__mean": 3 + hi * 0.5 + rng.normal(scale=0.3),
                             "blink_rate": 5 + rng.normal(), "gaze_x__sd": rng.normal(),
                             "saccade_rate": hi + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


def test_area_data_applies_exclusion_order_and_fallback():
    X, y, g, ids, check = area_data(_table(fix_ms=665), "90")
    assert check["n_excluded_invalid"] == 2 and not check["sanity_check_passed"]
    assert "saccade_rate" not in X and "pupil__mean" in X
    first = ids[ids.participant == 2].load.tolist()
    assert first == ["low"] * 4 + ["high"] * 4               # low before high within each person
    X, *_ , check = area_data(_table(fix_ms=665), "90", keep_events=True)
    assert "saccade_rate" in X and check["event_features_dropped"] is False
    X, *_ , check = area_data(_table(fix_ms=250), "90")
    assert check["sanity_check_passed"] and "saccade_rate" in X


def test_stage_main_writes_every_file(tmp_path):
    table = pd.concat([_table(665), _table(665).assign(area=360)], ignore_index=True)
    out = stage_main(table, tmp_path, n_iter=2, n_boot=20)
    res = out["90"]
    assert res.model.tolist() == ["Logistic Regression", "XGBoost", BASELINE]
    assert res.loc[res.model == BASELINE, "n_features_median"].item() == 1.0
    for name in ("results_90", "comparison_90", "shap_90", "shap_by_signal_90",
                 "selection_count_90", "without_top_signal_90", "results_360"):
        assert (tmp_path / f"{name}.csv").exists()
    shares = pd.read_csv(tmp_path / "shap_by_signal_90.csv").groupby("model").share.sum()
    assert shares.to_numpy() == pytest.approx([1.0, 1.0])

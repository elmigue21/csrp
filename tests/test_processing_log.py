import numpy as np
import pandas as pd

from conftest import make_recording
from features import recording_signals
from flatten import SIGNALS
from processing_log import DECISIONS, decisions_markdown, feature_log, recording_log_row


def test_recording_log_row():
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=8.0)
    sig = recording_signals(rec)
    row = recording_log_row(rec, sig)
    assert row["pupil_rows_after_trim"] <= row["pupil_rows_raw"]
    assert row["blinks_kept"] <= row["blinks_after_trim"] <= row["blinks_raw"]
    assert 0 <= row["invalid_fraction"] <= 1
    assert row["n_fixation_duration"] == sig["qc"]["n_fixations"]
    assert row["excluded_gap_s"] >= 0
    assert row["pupil_rows_3d"] == row["pupil_rows_raw"] / 2      # synthetic data: one 2d + one 3d row each
    assert {f"n_{s}" for s in SIGNALS} <= set(row)


def test_recording_log_row_excluded():
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=4.0)
    row = recording_log_row(rec, None)
    assert np.isnan(row["n_pupil"]) and np.isnan(row["n_fixations"])
    assert row["gaze_rows"] == len(rec["gaze"])


def test_feature_log_counts():
    raw = pd.DataFrame({
        "participant": [1, 1, 2, 2],
        "activity": [1, 4, 1, 4],
        "a": [5.0, 5.0, 1.0, 4.0],
        "b": [np.nan, 1.0, 1.0, 4.0],                           # one blank in A1
    })
    log = feature_log(raw, ["a", "b"], dropped={"b": "C9"}).set_index("feature")
    assert list(log.columns) == ["blank_A1", "blank_A4", "blank_all", "dropped", "drop_reason"]
    assert log.loc["b", "blank_A1"] == 1 and log.loc["b", "blank_A4"] == 0
    assert log.loc["b", "blank_all"] == 1
    assert bool(log.loc["b", "dropped"]) and log.loc["b", "drop_reason"] == "C9"
    assert not bool(log.loc["a", "dropped"])


def test_decisions_markdown():
    md = decisions_markdown()
    assert md.count("\n|") >= len(DECISIONS)
    assert all({"date", "item", "rule", "decided_by"} <= set(d) for d in DECISIONS)
    assert "Mode" in md


def test_decisions_record_no_per_person_standardization():
    rows = [d for d in DECISIONS if d["item"] == "Per-person standardization"]
    assert len(rows) == 1 and rows[0]["rule"].startswith("Not done")
    assert "A2 and A3" in decisions_markdown()
    assert "inflat" not in decisions_markdown().lower()

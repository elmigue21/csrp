from conftest import write_synthetic_dataset
from data import load_annotation, load_recording, recording_keys


def test_recording_keys_and_load(tmp_path):
    root = write_synthetic_dataset(tmp_path, n_participants=2, seconds=3)
    keys = recording_keys(root)
    assert keys == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
    rec = load_recording(1, 4, root)
    assert set(rec) == {"gaze", "pupil", "blinks"}
    assert "gaze_normal0_x" in rec["gaze"].columns


def test_annotation_renames_task(tmp_path):
    root = write_synthetic_dataset(tmp_path, n_participants=1, seconds=2)
    ann = load_annotation(root)
    assert {"participant", "activity", "mean"} <= set(ann.columns)

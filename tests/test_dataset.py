import numpy as np
import pandas as pd

import config
from conftest import write_synthetic_dataset
import dataset
from dataset import (_settings_hash, activities_per_participant, analysis_participants, analysis_rows, build_feature_table,
                     retention_summary)
from labels import binary_label, main_rows
from normalize import per_participant_zscore


def test_exclusion_and_analysis_set(tmp_path):
    root = write_synthetic_dataset(tmp_path / "d", n_participants=4, seconds=3.0,
                                   low_quality={(2, 4), (3, 2)})
    table = build_feature_table(root, cache_dir=tmp_path / "c")
    assert len(table) == 16
    assert set(table.loc[table.excluded, ["participant", "activity"]].itertuples(index=False)) == {(2, 4), (3, 2)}
    assert analysis_participants(table) == [1, 3, 4]           # P2 lost A4 -> dropped; P3 lost A2 -> kept
    rows = analysis_rows(table)
    assert len(rows[rows.participant == 3]) == 3               # A2 excluded, A1/A3/A4 kept
    ret = retention_summary(table)
    assert set(ret.columns) >= {"activity", "recordings", "excluded", "invalid_fraction_median"}


def test_cache_reused(tmp_path):
    root = write_synthetic_dataset(tmp_path / "d", n_participants=1, seconds=2.0)
    a = build_feature_table(root, cache_dir=tmp_path / "c")
    b = build_feature_table(root, cache_dir=tmp_path / "c")
    pd.testing.assert_frame_equal(a, b)


def test_cache_name_changes_with_settings(monkeypatch):
    default = _settings_hash()
    monkeypatch.setattr(config, "CONFIDENCE_MIN", 0.7)
    assert _settings_hash() != default


def test_cache_name_changes_with_code(monkeypatch):
    default = _settings_hash()
    monkeypatch.setattr(dataset, "_source_bytes", lambda: b"edited source")
    assert _settings_hash() != default


def test_activities_per_participant(tmp_path):
    root = write_synthetic_dataset(tmp_path / "d", n_participants=3, seconds=3.0, low_quality={(3, 2)})
    table = build_feature_table(root, cache_dir=tmp_path / "c")
    out = activities_per_participant(table).set_index("participant")["n_retained_activities"]
    assert out.to_dict() == {1: 4, 2: 4, 3: 3}


def test_labels_and_main_rows():
    rows = pd.DataFrame({"participant": [1, 1, 1, 1], "activity": [1, 2, 3, 4]})
    m = main_rows(rows)
    assert m.activity.tolist() == [1, 4]
    assert binary_label(m).tolist() == [0, 1]


def test_zscore_constant_and_partial_participants():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0, 4.0, 6.0], "b": [1, 2, 3, 4, 5, 6.0]})
    groups = pd.Series([1, 1, 1, 2, 2, 2])
    z = per_participant_zscore(X, groups)
    assert (z.loc[:2, "a"] == 0.0).all()                     # constant within P1 -> 0, not NaN/inf
    assert np.allclose(z.loc[3:, "a"], [-1, 0, 1])

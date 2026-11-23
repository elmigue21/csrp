import json

import pandas as pd
import pytest

from conftest import write_synthetic_dataset
from run_extract import main

LAYOUT = [
    "README.md", "run_info.json",
    "features/features.csv", "features/feature_dictionary.csv",
    "logs/log_recordings.csv", "logs/log_features.csv", "logs/log_decisions.md",
    "logs/sanity_check.csv",
    "raw/raw_data_dictionary.csv", "raw/raw_table_shapes.csv",
    "previews/raw_preview.xlsx", "previews/features_preview.xlsx", "code_preview.html",
]
FIGURES = [
    "fig_01_raw_tables.png", "fig_02_raw_columns.png", "fig_03_data_flow.png",
    "fig_04_feature_table.png", "fig_05_recording_signals.png", "fig_06_cleaning_effect.png",
    "fig_07_a1_vs_a4.png", "fig_08_data_quality.png",
]


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    root = tmp_path_factory.mktemp("extract")
    data = write_synthetic_dataset(root / "data", n_participants=6, seconds=8.0, low_quality={(2, 4)})
    final = root / "final"
    final.mkdir()
    (final / "stale.txt").write_text("old run")
    out = main(data_dir=data, final_dir=final, work_dir=root / "work")
    return final, out


def test_extract_writes_final_folder(extracted):
    final, out = extracted
    for rel in LAYOUT:
        assert (final / rel).exists(), rel
    raw = pd.read_csv(final / "features" / "features.csv", keep_default_na=False, na_values=[""])
    assert len(raw) == 5 * 2                         # A1 and A4 only; P2 is out (its A4 is excluded)
    assert raw.shape[1] == 3 + 140
    assert set(raw["activity"]) == {1, 4}
    labels = raw.set_index("activity")["label"]
    assert set(labels.loc[1]) == {"low"} and set(labels.loc[4]) == {"high"}
    assert len(pd.read_csv(final / "features" / "feature_dictionary.csv")) == 173
    info = json.loads((final / "run_info.json").read_text())
    assert info["n_features"] == 140 and info["n_features_extracted"] == 173
    assert info["n_participants"] == 5 and "git_commit" in info
    assert info["n_rows"] == 10 and "n_rows_main_analysis" not in info
    assert out["n_features"] == 140 and out["sanity_passed"] is True


def test_extract_replaces_stale_files(extracted):
    final, _ = extracted
    assert not (final / "stale.txt").exists()


def test_no_per_person_standardization(extracted):
    final, _ = extracted
    assert not (final / "features" / "features_z.csv").exists()
    log = pd.read_csv(final / "logs" / "log_features.csv")
    assert len(log) == 173 and "n_people_sd0" not in log
    readme = (final / "README.md").read_text(encoding="utf-8")
    assert "z-scor" not in readme and "inflat" not in readme.lower()


def test_excluded_recording_logged(extracted):
    final, _ = extracted
    log = pd.read_csv(final / "logs" / "log_recordings.csv", keep_default_na=False)
    row = log[(log["participant"] == 2) & (log["activity"] == 4)].iloc[0]
    assert str(row["excluded"]) == "True" and row["exclusion_reason"] == "invalid gaze > 35%"
    assert len(log) == 6 * 4
    raw = pd.read_csv(final / "features" / "features.csv")
    assert 2 not in set(raw["participant"])


def test_raw_shapes_written(extracted):
    final, _ = extracted
    shapes = pd.read_csv(final / "raw" / "raw_table_shapes.csv").set_index("table")
    assert shapes.loc["gaze", "files"] == 24 and shapes.loc["pupil", "columns"] == 34


def test_c9_failure_drops_event_features(tmp_path, monkeypatch):
    import run_extract
    monkeypatch.setattr(run_extract, "sanity_passed", lambda sanity: False)
    data = write_synthetic_dataset(tmp_path / "data", n_participants=3, seconds=4.0)
    out = run_extract.main(data_dir=data, final_dir=tmp_path / "final", work_dir=tmp_path / "work")
    raw = pd.read_csv(tmp_path / "final" / "features" / "features.csv")
    # 104 event features (6 event signals x 17 + 2 rates) go, plus the 20 non-event quality drops
    # (17 blink duration + variance of pupil, gaze_x, gaze_y)
    assert out["n_features"] == 173 - 104 - 20
    assert "pupil__mean" in raw and "blink_rate" in raw
    assert not any(c.startswith(("gaze_speed__", "fixation_", "saccade_")) for c in raw)
    log = pd.read_csv(tmp_path / "final" / "logs" / "log_features.csv").set_index("feature")
    assert len(log) == 173 and log.loc["saccade_rate", "dropped"]
    assert "C9" in log.loc["fixation_duration__mean", "drop_reason"]


def test_analysis_reason_for_recordings_left_out(extracted):
    final, _ = extracted
    log = pd.read_csv(final / "logs" / "log_recordings.csv", keep_default_na=False)
    p2 = log[log["participant"] == 2].set_index("activity")
    assert p2.loc[1, "analysis_reason"] == "participant lost A1 or A4 (C3)"
    assert p2.loc[4, "analysis_reason"] == "recording excluded (C3)"
    p1 = log[log["participant"] == 1].set_index("activity")
    assert p1.loc[[1, 4], "analysis_reason"].eq("").all()
    assert p1.loc[[2, 3], "analysis_reason"].eq("A2/A3: no label, not in the feature table").all()
    assert str(p1.loc[2, "in_analysis"]) == "False" and str(p1.loc[1, "in_analysis"]) == "True"


def test_readme_and_decisions_record_the_feature_drops(extracted):
    final, _ = extracted
    readme = (final / "README.md").read_text(encoding="utf-8")
    assert "140 model features, out of 173 extracted" in readme
    decisions = (final / "logs" / "log_decisions.md").read_text(encoding="utf-8")
    assert "D1 Blink duration" in decisions and "Dropped" in decisions


def test_csvs_open_cleanly_in_excel(extracted):
    final, _ = extracted
    assert (final / "features" / "feature_dictionary.csv").read_bytes()[:3] == b"\xef\xbb\xbf"


def test_cache_rebuilt_when_data_changes(tmp_path):
    from run_extract import build_flat_table
    data = write_synthetic_dataset(tmp_path / "data", n_participants=2, seconds=4.0)
    first, _ = build_flat_table(data, tmp_path / "work")
    gaze = data / "p01_t1_gaze.parquet"
    pd.read_parquet(gaze).assign(confidence=0.1).to_parquet(gaze)
    second, _ = build_flat_table(data, tmp_path / "work")
    row = lambda t: t[(t["participant"] == 1) & (t["activity"] == 1)].iloc[0]
    assert not row(first)["excluded"] and row(second)["excluded"]


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows file locking")
def test_open_file_leaves_final_folder_whole(tmp_path):
    from run_extract import _write
    final = tmp_path / "final"
    _write(final, {"a.txt": "old", "sub/b.txt": "old"})
    with open(final / "a.txt"):
        with pytest.raises(RuntimeError, match="close"):
            _write(final, {"a.txt": "new"})
    assert (final / "a.txt").read_text() == "old" and (final / "sub" / "b.txt").exists()
    assert not (tmp_path / "final.tmp").exists()
    _write(final, {"a.txt": "new"})
    assert (final / "a.txt").read_text() == "new" and not (final / "sub").exists()
    assert not (tmp_path / "final.old").exists()


def test_quality_drops_applied_and_logged(extracted):
    final, _ = extracted
    raw = pd.read_csv(final / "features" / "features.csv")
    for gone in ("blink_duration__mean", "saccade_duration__min", "pupil__var",
                 "gaze_speed__range", "saccade_duration__range"):
        assert gone not in raw, gone
    for kept in ("blink_rate", "pupil__sd", "pupil__range", "gaze_x__range", "gaze_x__mean",
                 "saccade_duration__median", "fixation_duration__min", "gaze_speed__max"):
        assert kept in raw, kept
    log = pd.read_csv(final / "logs" / "log_features.csv", keep_default_na=False).set_index("feature")
    assert log["dropped"].astype(str).eq("True").sum() == 33
    assert log.loc["blink_duration__mean", "drop_reason"].startswith("D1")
    assert log.loc["pupil__var", "drop_reason"].startswith("D8")
    # blanks of a dropped feature are counted from the extracted data, not reported as all-blank
    assert int(log.loc["pupil__var", "blank_all"]) == 0
    d = pd.read_csv(final / "features" / "feature_dictionary.csv", keep_default_na=False).set_index("feature")
    assert len(d) == 173 and d["kept"].astype(str).eq("True").sum() == 140
    assert d.loc["gaze_speed__range", "drop_reason"].startswith("D9")
    decisions = (final / "logs" / "log_decisions.md").read_text(encoding="utf-8")
    assert "Pending D1" not in decisions and "D10" in decisions


def test_figures_written_as_png(extracted):
    final, _ = extracted
    written = sorted(p.name for p in (final / "figures").iterdir())
    assert written == sorted(FIGURES)
    for name in FIGURES:
        data = (final / "figures" / name).read_bytes()
        assert data[:4] == b"\x89PNG" and len(data) > 10_000, name
    readme = (final / "README.md").read_text(encoding="utf-8")
    assert all(name in readme for name in FIGURES)


def test_model_results_survive_reextraction_and_are_flagged_when_stale(tmp_path):
    import hashlib
    data = write_synthetic_dataset(tmp_path / "data", n_participants=3, seconds=8.0)
    final = tmp_path / "final"
    main(data_dir=data, final_dir=final, work_dir=tmp_path / "work")
    sha = hashlib.sha256((final / "features" / "features.csv").read_bytes()).hexdigest()
    results = final / "results"
    results.mkdir()
    (results / "results.csv").write_text("model,roc_auc\nLR,0.9\n")
    (results / "run_info.json").write_text(json.dumps({"features_sha256": sha}))
    (final / "paper_figures").mkdir()
    (final / "paper_figures" / "fig6_roc.png").write_bytes(b"png")
    main(data_dir=data, final_dir=final, work_dir=tmp_path / "work")
    assert (results / "results.csv").read_text() == "model,roc_auc\nLR,0.9\n"
    assert (final / "paper_figures" / "fig6_roc.png").read_bytes() == b"png"
    assert not (results / "STALE.md").exists()
    (results / "run_info.json").write_text(json.dumps({"features_sha256": "different"}))
    main(data_dir=data, final_dir=final, work_dir=tmp_path / "work")
    assert (results / "results.csv").exists() and (results / "STALE.md").exists()
    assert "run_model.py" in (results / "STALE.md").read_text(encoding="utf-8")


def test_excel_previews_hold_the_raw_example_and_the_whole_feature_table(extracted):
    final, _ = extracted
    feats = pd.read_excel(final / "previews" / "features_preview.xlsx", sheet_name=None)
    assert list(feats) == ["README", "features", "feature_dictionary", "dropped_features"]
    assert feats["features"].shape == (5 * 2, 3 + 140)
    assert len(feats["dropped_features"]) == 33
    raw = pd.read_excel(final / "previews" / "raw_preview.xlsx", sheet_name=None)
    assert {"README", "table_sizes", "column_dictionary", "gaze_P01_A1", "pupil_P01_A1"} <= set(raw)
    assert len(raw["gaze_P01_A1"]) <= 200

import pandas as pd

from conftest import write_synthetic_dataset
from raw_dictionary import RAW_COLUMNS, raw_data_dictionary, raw_table_shapes


def test_every_original_column_is_described():
    sizes = {t: len(cols) for t, cols in RAW_COLUMNS.items()}
    assert sizes == {"gaze": 21, "pupil": 34, "blinks": 10, "annotation": 9, "subject_info": 5}
    for cols in RAW_COLUMNS.values():
        for c in cols:
            assert c["meaning"] and c["used"] in {"yes", "no", "qc only", "check only"}
            assert c["used"] == "yes" or c["why"], c["column"]


def test_raw_data_dictionary_reads_types_and_examples(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", n_participants=2, seconds=3.0)
    d = raw_data_dictionary(data)
    assert list(d.columns) == ["table", "column", "dtype", "unit", "example", "meaning", "used", "why"]
    assert len(d) == 79
    g = d[(d["table"] == "gaze") & (d["column"] == "confidence")].iloc[0]
    assert g["dtype"] == "float64" and g["example"] != ""
    missing = d[(d["table"] == "gaze") & (d["column"] == "norm_pos_x")].iloc[0]
    assert missing["dtype"] == ""                       # not in the synthetic files: left blank


def test_raw_table_shapes():
    log = pd.DataFrame({"gaze_rows": [100, 300], "pupil_rows_raw": [200, 600], "blinks_raw": [0, 4]})
    s = raw_table_shapes(log, annotation_rows=8, subject_rows=2).set_index("table")
    assert s.loc["gaze", "files"] == 2 and s.loc["gaze", "columns"] == 21
    assert s.loc["gaze", "rows_median"] == 200 and s.loc["pupil", "rows_max"] == 600
    assert s.loc["blinks", "rows_total"] == 4
    assert s.loc["annotation", "rows_total"] == 8 and s.loc["subject_info", "files"] == 1

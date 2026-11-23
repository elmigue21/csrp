"""Data dictionary of the original COLET tables (post-defense-data-processing.md §1a).

Every column of every original table, with its unit, plain meaning, whether the pipeline
uses it and why not. Column meanings follow the Pupil Labs Core export format (gaze,
pupil, blinks) and the COLET README (annotation, subject_info).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_NOT_BEHAVIOUR = "Tracker's internal eye-model geometry, not eye behaviour"
_CAMERA_FRAME = "Position in the head camera's image, not on the screen (§5, M1)"
_BROKEN_DEPTH = "Depends on the tracker's depth estimate, which is broken in COLET (R4-1)"
_BOOKKEEPING = "Internal bookkeeping"


def _c(column, unit, meaning, used, why=""):
    return {"column": column, "unit": unit, "meaning": meaning, "used": used, "why": why}


def _xyz(prefix, unit, meaning, used, why=""):
    return [_c(f"{prefix}_{a}", unit, f"{meaning} ({a})", used, why) for a in "xyz"]


RAW_COLUMNS = {
    "gaze": [
        _c("gaze_timestamp", "s", "Time of the gaze sample", "yes"),
        _c("world_index", "", "Index of the nearest head-camera video frame", "no", _BOOKKEEPING),
        _c("confidence", "0-1", "How sure the tracker is about this sample", "yes"),
        _c("norm_pos_x", "0-1", "Gaze position left/right in the head camera's image", "no", _CAMERA_FRAME),
        _c("norm_pos_y", "0-1", "Gaze position up/down in the head camera's image", "no", _CAMERA_FRAME),
        _c("base_data", "", "Which pupil samples this gaze sample was built from", "no", _BOOKKEEPING),
        *_xyz("gaze_point_3d", "mm", "Estimated 3D point being looked at", "no", _BROKEN_DEPTH),
        *_xyz("eye_center0_3d", "mm", "Position of eye 0's eyeball centre", "no",
              "Eyeball position, not where the eye points"),
        *_xyz("gaze_normal0", "unit vector", "Direction eye 0 points", "yes"),
        *_xyz("eye_center1_3d", "mm", "Position of eye 1's eyeball centre", "no",
              "Eyeball position, not where the eye points"),
        *_xyz("gaze_normal1", "unit vector", "Direction eye 1 points", "yes"),
    ],
    "pupil": [
        _c("pupil_timestamp", "s", "Time of the pupil sample", "yes"),
        _c("world_index", "", "Index of the nearest head-camera video frame", "no", _BOOKKEEPING),
        _c("eye_id", "", "Which eye: 0 or 1", "yes"),
        _c("confidence", "0-1", "How sure the tracker is about this sample", "yes"),
        _c("norm_pos_x", "0-1", "Pupil position left/right in the eye camera's image", "no", _NOT_BEHAVIOUR),
        _c("norm_pos_y", "0-1", "Pupil position up/down in the eye camera's image", "no", _NOT_BEHAVIOUR),
        _c("diameter", "px", "Pupil size in eye-camera pixels", "no",
           "Pixels change with camera distance; diameter_3d (mm) is used instead"),
        _c("method", "", "Measurement method: '2d c++' (pixels) or '3d c++' (mm); each sample has one row of each",
           "yes"),
        _c("ellipse_center_x", "px", "Centre of the fitted pupil ellipse (x)", "no", _NOT_BEHAVIOUR),
        _c("ellipse_center_y", "px", "Centre of the fitted pupil ellipse (y)", "no", _NOT_BEHAVIOUR),
        _c("ellipse_axis_a", "px", "First axis of the fitted pupil ellipse", "no", _NOT_BEHAVIOUR),
        _c("ellipse_axis_b", "px", "Second axis of the fitted pupil ellipse", "no", _NOT_BEHAVIOUR),
        _c("ellipse_angle", "deg", "Rotation of the fitted pupil ellipse", "no", _NOT_BEHAVIOUR),
        _c("diameter_3d", "mm", "Pupil size in mm from the 3D eye model (3d rows only)", "yes"),
        _c("model_confidence", "0-1", "How well the 3D eye model fits", "no", _NOT_BEHAVIOUR),
        _c("model_id", "", "ID of the 3D eye model; changes when the tracker refits it", "qc only",
           "Only used to count model refits per recording (C7)"),
        *_xyz("sphere_center", "mm", "Eyeball centre in eye-camera space", "no", _NOT_BEHAVIOUR),
        _c("sphere_radius", "mm", "Eyeball radius assumed by the model (fixed 12 mm)", "no", _NOT_BEHAVIOUR),
        *_xyz("circle_3d_center", "mm", "Pupil centre in 3D", "no", _NOT_BEHAVIOUR),
        *_xyz("circle_3d_normal", "unit vector", "Direction the pupil faces in eye-camera space", "no",
              "Eye-camera frame; gaze direction comes from the gaze table instead"),
        _c("circle_3d_radius", "mm", "Pupil radius in 3D", "no", "Half of diameter_3d; diameter_3d is used"),
        _c("theta", "rad", "Pupil direction as a spherical angle (vertical)", "no", _NOT_BEHAVIOUR),
        _c("phi", "rad", "Pupil direction as a spherical angle (horizontal)", "no", _NOT_BEHAVIOUR),
        _c("projected_sphere_center_x", "px", "Eyeball centre projected into the eye image (x)", "no", _NOT_BEHAVIOUR),
        _c("projected_sphere_center_y", "px", "Eyeball centre projected into the eye image (y)", "no", _NOT_BEHAVIOUR),
        _c("projected_sphere_axis_a", "px", "Projected eyeball ellipse, axis a", "no", _NOT_BEHAVIOUR),
        _c("projected_sphere_axis_b", "px", "Projected eyeball ellipse, axis b", "no", _NOT_BEHAVIOUR),
        _c("projected_sphere_angle", "deg", "Projected eyeball ellipse rotation", "no", _NOT_BEHAVIOUR),
    ],
    "blinks": [
        _c("id", "", "Blink number within the recording", "no", _BOOKKEEPING),
        _c("start_timestamp", "s", "When the blink started", "yes"),
        _c("duration", "s", "Blink length as exported", "no",
           "Recomputed from start and end after merging close blinks (C5)"),
        _c("end_timestamp", "s", "When the blink ended", "yes"),
        _c("start_frame_index", "", "Head-camera frame where the blink started", "no", _BOOKKEEPING),
        _c("index", "", "Head-camera frame index of the blink", "no", _BOOKKEEPING),
        _c("end_frame_index", "", "Head-camera frame where the blink ended", "no", _BOOKKEEPING),
        _c("confidence", "0-1", "How sure the blink detector is", "no",
           "Blinks are filtered by length instead (C5)"),
        _c("filter_response", "", "Raw output of the blink detector's filter", "no", _BOOKKEEPING),
        _c("base_data", "", "Pupil timestamps the blink was detected from", "no", _BOOKKEEPING),
    ],
    "annotation": [
        _c("participant", "", "Participant number", "check only", "Manipulation check only, not a feature"),
        _c("task", "", "Activity number (1-4)", "check only", "Manipulation check only, not a feature"),
        *[_c(k, "0-100", f"NASA-RTLX {k} demand rating", "check only", "Manipulation check only, not a feature")
          for k in ("mental", "physical", "temporal", "performance", "effort", "frustration")],
        _c("mean", "0-100", "NASA-RTLX overall workload (mean of the six ratings)", "check only",
           "Manipulation check only, not a feature"),
    ],
    "subject_info": [
        _c("participant", "", "Participant number", "no", "Demographics are described, not modelled"),
        _c("VisualAcuity_logMAR_", "logMAR", "Visual acuity (lower = sharper)", "no",
           "Demographics are described, not modelled"),
        _c("Gender", "", "Gender as recorded by COLET", "no", "Demographics are described, not modelled"),
        _c("Age", "years", "Age", "no", "Demographics are described, not modelled"),
        _c("Education", "years", "Years of education", "no", "Demographics are described, not modelled"),
    ],
}

_FILES = {"gaze": "p01_t1_gaze.parquet", "pupil": "p01_t1_pupil.parquet",
          "blinks": "p01_t1_blinks.parquet", "annotation": "annotation.csv",
          "subject_info": "subject_info.csv"}


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _example(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    v = s.iloc[0]
    text = f"{v:.4g}" if isinstance(v, float) else str(v)
    return text if len(text) <= 40 else text[:37] + "..."


def raw_data_dictionary(data_dir) -> pd.DataFrame:
    """Every original column with its type and an example value from P01 A1 (or the shared CSV).

    dtype and example are blank for a column missing from the files found (e.g. test data).
    """
    rows = []
    for table, cols in RAW_COLUMNS.items():
        df = _read(Path(data_dir) / _FILES[table])
        for c in cols:
            present = df is not None and c["column"] in df
            rows.append({"table": table, "column": c["column"],
                         "dtype": str(df[c["column"]].dtype) if present else "",
                         "unit": c["unit"],
                         "example": _example(df[c["column"]]) if present else "",
                         "meaning": c["meaning"], "used": c["used"], "why": c["why"]})
    return pd.DataFrame(rows)


def raw_table_shapes(log_recordings: pd.DataFrame, annotation_rows: int,
                     subject_rows: int) -> pd.DataFrame:
    """Size of each original table: files, columns and rows per file (min / median / max / total)."""
    per_recording = {"gaze": ("gaze_rows", "one gaze sample (about 242 per second)"),
                     "pupil": ("pupil_rows_raw", "one pupil sample per eye per method (2d and 3d)"),
                     "blinks": ("blinks_raw", "one blink")}
    rows = []
    for table, (col, one_row) in per_recording.items():
        r = log_recordings[col]
        rows.append({"table": table, "files": len(r), "columns": len(RAW_COLUMNS[table]),
                     "rows_min": int(r.min()), "rows_median": float(r.median()),
                     "rows_max": int(r.max()), "rows_total": int(r.sum()), "one_row_is": one_row})
    for table, n, one_row in (("annotation", annotation_rows, "one participant x activity"),
                              ("subject_info", subject_rows, "one participant")):
        rows.append({"table": table, "files": 1, "columns": len(RAW_COLUMNS[table]),
                     "rows_min": n, "rows_median": float(n), "rows_max": n, "rows_total": n,
                     "one_row_is": one_row})
    return pd.DataFrame(rows)

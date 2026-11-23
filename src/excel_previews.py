"""Excel previews of the raw COLET data and of the final feature table (for the defense panel).

The raw files are too big for Excel (2.25 M gaze rows), so the raw preview shows one recording
(P01, activity A1) in full detail plus the size and meaning of every table. The feature preview
holds the whole final table (88 rows x 140 features) with its dictionary and the dropped features.
run_extract.py writes both into outputs/final/previews/ on every run.

Usage: python src/excel_previews.py   (rebuilds the previews from the files in outputs/final)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, FINAL_DIR

PREVIEW_ROWS = 200                   # rows shown per raw signal table
EXAMPLE = ("p01_t1", "P01, activity A1 (low load)")
HEADER_FILL = PatternFill("solid", fgColor="DDE8F6")


def _book(readme: list[str], sheets: dict[str, pd.DataFrame]) -> bytes:
    """One workbook: a README sheet, then one sheet per table with a bold, frozen header row."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame({"About this file": readme}).to_excel(xl, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = HEADER_FILL
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for i, col in enumerate(ws.iter_cols(max_row=min(ws.max_row, 60)), start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 9),
                                                                       100 if ws.title == "README" else 45)
    return buf.getvalue()


def raw_preview(data_dir=DATA_DIR, shapes: pd.DataFrame | None = None,
                raw_dict: pd.DataFrame | None = None) -> bytes:
    data_dir = Path(data_dir)
    stem, label = EXAMPLE
    sheets = {}
    if shapes is not None:
        sheets["table_sizes"] = shapes
    if raw_dict is not None:
        sheets["column_dictionary"] = raw_dict
    for signal in ("gaze", "pupil", "blinks"):
        path = data_dir / f"{stem}_{signal}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            sheets[f"{signal}_P01_A1"] = df.head(PREVIEW_ROWS) if signal != "blinks" else df
    for name in ("annotation", "subject_info"):
        path = data_dir / f"{name}.csv"
        if path.exists():
            sheets[name] = pd.read_csv(path)
    readme = [
        "Raw COLET data, as released (after converting the MATLAB file to one table per recording).",
        f"The gaze, pupil and blink sheets show one recording: {label}.",
        f"Gaze and pupil sheets show the first {PREVIEW_ROWS} rows only; the full tables are too large "
        "for Excel (2.25 million gaze rows and 4.51 million pupil rows over 188 recordings).",
        "table_sizes: how many files, columns and rows each original table has.",
        "column_dictionary: every original column, its meaning and unit, and whether it is used.",
        "annotation: NASA-RTLX ratings per participant and activity (used only as a manipulation check).",
        "subject_info: participant information (described, not modelled).",
        "Written by src/excel_previews.py; nothing here is edited by hand.",
    ]
    return _book(readme, sheets)


def features_preview(table: pd.DataFrame, feature_dict: pd.DataFrame,
                     feature_log: pd.DataFrame | None = None) -> bytes:
    n_feat = table.shape[1] - 3
    sheets = {"features": table, "feature_dictionary": feature_dict}
    if feature_log is not None and "dropped" in feature_log:
        sheets["dropped_features"] = feature_log[feature_log["dropped"].astype(bool)][
            ["feature", "drop_reason"]]
    readme = [
        "Final feature table used by the models (src/run_model.py).",
        f"features: {len(table)} rows (one per recording: each participant's A1 = low and A4 = high) "
        f"x {n_feat} features, plus participant, activity and label.",
        "Feature names are <signal>__<statistic>, for example pupil__mean or gaze_y__min; "
        "blink_rate, fixation_rate and saccade_rate are single numbers.",
        "feature_dictionary: one plain-language line per extracted feature (kept or dropped, and why).",
        "dropped_features: the features removed before modeling, with the reason.",
        "Written by src/excel_previews.py; nothing here is edited by hand.",
    ]
    return _book(readme, sheets)


def main(final_dir=FINAL_DIR, data_dir=DATA_DIR) -> list[Path]:
    final_dir = Path(final_dir)
    out = final_dir / "previews"
    out.mkdir(parents=True, exist_ok=True)
    read = lambda rel: pd.read_csv(final_dir / rel) if (final_dir / rel).exists() else None
    files = {
        "raw_preview.xlsx": raw_preview(data_dir, read("raw/raw_table_shapes.csv"),
                                        read("raw/raw_data_dictionary.csv")),
        "features_preview.xlsx": features_preview(read("features/features.csv"),
                                                  read("features/feature_dictionary.csv"),
                                                  read("logs/log_features.csv")),
    }
    for name, content in files.items():
        (out / name).write_bytes(content)
    return [out / n for n in files]


if __name__ == "__main__":
    for p in main():
        print(p)

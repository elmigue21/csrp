"""Read the converted COLET recordings.

data/colet/convert_colet.py turned the nested MATLAB file into one Parquet file per
participant x activity x signal (pXX_tY_{gaze,pupil,blinks}.parquet) plus annotation.csv.
Values were not changed; object cells were stored as their text repr, which is why the
pupil `method` column reads "array(['3d c++'], dtype='<U6')".
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import DATA_DIR

SIGNALS = ("gaze", "pupil", "blinks")
_GAZE_FILE = re.compile(r"^p(\d{2})_t(\d)_gaze\.parquet$")


def recording_path(participant: int, activity: int, signal: str, data_dir=DATA_DIR) -> Path:
    return Path(data_dir) / f"p{participant:02d}_t{activity}_{signal}.parquet"


def load_recording(participant: int, activity: int, data_dir=DATA_DIR) -> dict[str, pd.DataFrame]:
    """The three tables of one recording, keyed by signal name."""
    return {s: pd.read_parquet(recording_path(participant, activity, s, data_dir)) for s in SIGNALS}


def recording_keys(data_dir=DATA_DIR) -> list[tuple[int, int]]:
    """(participant, activity) for every recording that has a gaze file, sorted."""
    keys = []
    for path in Path(data_dir).glob("p*_t*_gaze.parquet"):
        m = _GAZE_FILE.match(path.name)
        if m:
            keys.append((int(m.group(1)), int(m.group(2))))
    if not keys:
        raise FileNotFoundError(f"no pXX_tY_gaze.parquet files under {data_dir}")
    return sorted(keys)


def load_annotation(data_dir=DATA_DIR) -> pd.DataFrame:
    """NASA-RTLX per participant x activity; column `task` is renamed `activity`."""
    return pd.read_csv(Path(data_dir) / "annotation.csv").rename(columns={"task": "activity"})

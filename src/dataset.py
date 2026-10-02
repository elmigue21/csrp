"""One feature row per recording, with the recording-exclusion rule applied (P3, P10, P14).

Features are computed once and cached, because the full set takes minutes to build.
The cache file name carries a hash of the preprocessing settings and of the source text of
the modules that compute features, so changing a setting or the code never silently reuses
stale features.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

import config
from config import CACHE_DIR, DATA_DIR, HIGH_ACTIVITY, LOW_ACTIVITY, MAX_INVALID_FRACTION
from data import load_recording, recording_keys
from features import recording_features
from preprocess import invalid_fraction, trim_to_gaze_range

_CODE_FILES = ("data.py", "preprocess.py", "events.py", "features.py", "dataset.py")
_SETTING_PREFIXES = ("CONFIDENCE", "MAX_INVALID", "VALID_TIME", "BLINK", "GAZE", "PUPIL",
                     "MAX_VELOCITY", "IVT", "MIN_FIXATION")


def _source_bytes() -> bytes:
    """Concatenated source text of the feature-building modules (separate helper so tests can swap it)."""
    here = Path(__file__).parent
    return b"".join((here / name).read_bytes() for name in _CODE_FILES)


def _settings_hash() -> str:
    """First 10 hex chars of a SHA-1 over every preprocessing constant in config and the feature code."""
    settings = {k: v for k, v in vars(config).items()
                if k.isupper() and k.startswith(_SETTING_PREFIXES)}
    h = hashlib.sha1(repr(sorted(settings.items())).encode())
    h.update(_source_bytes())
    return h.hexdigest()[:10]


def build_feature_table(data_dir=DATA_DIR, cache_dir=CACHE_DIR, refresh=False) -> pd.DataFrame:
    cache = Path(cache_dir) / f"feature_table_{_settings_hash()}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for participant, activity in recording_keys(data_dir):
        rec = trim_to_gaze_range(load_recording(participant, activity, data_dir))
        inv = invalid_fraction(rec["gaze"])
        row = {"participant": participant, "activity": activity,
               "invalid_fraction": inv, "excluded": inv > MAX_INVALID_FRACTION}
        if not row["excluded"]:
            row.update(recording_features(rec))
        rows.append(row)
    table = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    return table


def analysis_participants(table: pd.DataFrame) -> list[int]:
    """Participants whose A1 and A4 recordings both survived exclusion."""
    kept = table[~table["excluded"]]
    has = kept.groupby("participant")["activity"].apply(set)
    return sorted(int(p) for p, acts in has.items() if {LOW_ACTIVITY, HIGH_ACTIVITY} <= acts)


def analysis_rows(table: pd.DataFrame) -> pd.DataFrame:
    """All retained activities of the analysis participants (used for normalization)."""
    keep = table["participant"].isin(analysis_participants(table)) & ~table["excluded"]
    return table[keep].sort_values(["participant", "activity"]).reset_index(drop=True)


def retention_summary(table: pd.DataFrame) -> pd.DataFrame:
    """P11/P14: recordings, exclusions and data quality per activity."""
    g = table.groupby("activity")
    return pd.DataFrame({
        "recordings": g.size(),
        "excluded": g["excluded"].sum(),
        "invalid_fraction_median": g["invalid_fraction"].median(),
        "refits_median": g["refits"].median() if "refits" in table else float("nan"),
        "recordings_with_refit": g["refits"].apply(lambda s: int((s > 0).sum())) if "refits" in table else 0,
        "binocular_fraction_median": g["binocular_fraction"].median() if "binocular_fraction" in table else float("nan"),
    }).reset_index()


def activities_per_participant(table: pd.DataFrame) -> pd.DataFrame:
    """How many of the four activities each analysis participant keeps after exclusion."""
    rows = analysis_rows(table)
    counts = rows.groupby("participant").size()
    return counts.rename("n_retained_activities").reset_index()

"""Processing log tables (post-defense-data-processing.md §7).

Everything dropped, filled in or transformed is counted here by code, never by hand:
- recording_log_row: what each cleaning step (C1-C9) did to one recording;
- feature_log: blanks and drops for each feature;
- DECISIONS / decisions_markdown: the decisions behind the rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HIGH_ACTIVITY, LOW_ACTIVITY, PUPIL_METHOD_TAG
from flatten import SIGNALS
from preprocess import clean_blinks, invalid_fraction, trim_to_gaze_range

_NAN = float("nan")
_QC_KEYS = ("valid_time_s", "gaze_valid_fraction", "binocular_fraction", "pupil_valid_fraction",
            "refits", "artifact_samples", "n_fixations", "n_saccades")


def recording_log_row(raw_rec: dict, sig: dict | None) -> dict:
    """Counts before and after each cleaning step for one recording.

    `raw_rec` is the recording as loaded (before trimming). `sig` is recording_signals output,
    or None for an excluded recording: signal-derived keys are then NaN.
    """
    trimmed = trim_to_gaze_range(raw_rec)
    t = raw_rec["gaze"]["gaze_timestamp"].to_numpy(float)
    duration = float(t.max() - t.min()) if len(t) else 0.0
    pupil = trimmed["pupil"]
    row = {
        "gaze_rows": len(raw_rec["gaze"]),                                   # C1
        "pupil_rows_raw": len(raw_rec["pupil"]),
        "pupil_rows_after_trim": len(pupil),
        "blinks_raw": len(raw_rec["blinks"]),
        "blinks_after_trim": len(trimmed["blinks"]),
        "invalid_fraction": invalid_fraction(raw_rec["gaze"]),               # C2/C3
        "duration_s": duration,                                              # C4
        "blinks_kept": len(clean_blinks(trimmed["blinks"])),                 # C5
        "pupil_rows_3d": int(pupil["method"].astype(str).str.contains(PUPIL_METHOD_TAG).sum()),  # C7
    }
    qc = sig["qc"] if sig is not None else {}
    row.update({k: qc.get(k, _NAN) for k in _QC_KEYS})                       # C4, C6-C9
    row["excluded_gap_s"] = duration - row["valid_time_s"] if sig is not None else _NAN
    for name in SIGNALS:
        row[f"n_{name}"] = len(sig["signals"][name]) if sig is not None else _NAN
    return row


def feature_log(raw: pd.DataFrame, features: list[str], dropped: dict[str, str]) -> pd.DataFrame:
    """One row per feature: blank counts in A1/A4/all rows, and whether it was dropped and why."""
    rows = []
    a1 = raw["activity"] == LOW_ACTIVITY
    a4 = raw["activity"] == HIGH_ACTIVITY
    for f in features:
        col = raw[f] if f in raw else pd.Series(np.nan, index=raw.index)
        rows.append({
            "feature": f,
            "blank_A1": int(col[a1].isna().sum()),
            "blank_A4": int(col[a4].isna().sum()),
            "blank_all": int(col.isna().sum()),
            "dropped": f in dropped,
            "drop_reason": dropped.get(f, ""),
        })
    return pd.DataFrame(rows)


DECISIONS = [
    {"date": "2026-11-09", "item": "Order of work",
     "rule": "Extract every signal with the current cleaning rules first; decide C1-C10 and D1-D10 from the results",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "Mode",
     "rule": "Not computed: undefined for continuous data without arbitrary rounding",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "cv for gaze x/y",
     "rule": "Computed and kept, tagged 'position': gaze angles have no true zero, so cv depends on "
             "how the headset sat; report with that caveat if it ranks highly",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "Counts and durations",
     "rule": "Not features: they measure how long the activity lasted (A4 runs longer), not load; "
             "rates are used instead",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "Valid-sample %",
     "rule": "Reported, never a feature: A4 loses more data, so it would reveal the label",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "Feature-decision rules",
     "rule": "Decide before any model runs and never change a decision because of results; drop only "
             "for data quality (blanks that follow the condition, stuck values, exact copies)",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D1 Blink duration (17 features)",
     "rule": "Dropped: with the 20-value rule (D2) blank in 79 of 88 recordings (43 A1, 36 A4); "
             "8 of the 9 with values are A4, so whether a value exists reveals the label; "
             "blink_rate is kept",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D6a saccade_duration__min",
     "rule": "Dropped: one sample (4.17 ms) in 83 of 88 recordings, so it measures the tracker",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D6b Other saccade-duration statistics",
     "rule": "Kept: rounded to 4.17 ms steps but still differ between recordings; stated as a limitation",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D7 fixation_duration__min",
     "rule": "Kept: 42% of recordings at the 55 ms floor, the rest spread over 9 values",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D8 Variance (9 features)",
     "rule": "Dropped: variance is SD squared, an exact copy that would split importance with SD",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D9 Range for 6 signals",
     "rule": "Dropped for gaze speed, fixation duration and the 4 saccade signals: their minimum is "
             "fixed, so range copies max (rank correlation >= 0.999); kept for pupil and gaze x/y",
     "decided_by": "team"},
    {"date": "2026-11-09", "item": "D10 Gaze position (20 features; 22 with gaze cv)",
     "rule": "Kept and tagged 'position'; if one ranks highly, report it with a screen-content caveat",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "D2 Minimum values per signal",
     "rule": "A signal's statistics are blank when a recording has fewer than 20 values for it",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "D3 min, max and range vs recording length",
     "rule": "Kept; reported as a limitation (A4 runs longer than A1), checked against p05/p95",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "D4 Signal list",
     "rule": "Kept the 10 signals; per-eye pupil and pupil rate of change not added",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "Per-person standardization",
     "rule": "Not done: the feature table holds each feature's value as extracted; any scaling a "
             "model needs is fitted on the training people inside each LOPO fold",
     "decided_by": "team (post-defense)"},
    {"date": "2026-11-11", "item": "A2 and A3",
     "rule": "Left out of the feature table: they have no label and were only used for per-person "
             "standardization; they still count in the C9 event-detection check",
     "decided_by": "team (post-defense)"},
    {"date": "2026-11-11", "item": "C1 Line up the files in time",
     "rule": "Kept: pupil samples and blinks outside the gaze time range are removed (mainly P18 A4, "
             "whose pupil file runs up to 2 hours past the activity)",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "C2 Low-confidence samples",
     "rule": "Kept: gaze and pupil samples with confidence below 0.8 are treated as missing "
             "(6.7% of gaze samples; 0.6 or 0.7 would give 6.3-6.4%)",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "C3-C10 Remaining cleaning steps",
     "rule": "Kept as defended: >35% invalid gaze drops a recording (C3); gaps over 1 s not counted "
             "(C4); blinks 50-500 ms, merged under 100 ms (C5); gaze gaps under 75 ms filled (C6); "
             "pupil cleaning (C7); gaze as both-eye angles (C8); I-VT 45 deg/s, fixations >= 55 ms (C9); "
             "valid-sample share reported, never a feature (C10)",
     "decided_by": "team"},
    {"date": "2026-11-11", "item": "Pupil quality",
     "rule": "No extra exclusion: 7 A4 recordings keep under 80% of pupil data and P44 A4's right "
             "eye reads too small; C7 already removes impossible values, so they are kept and "
             "reported as a limitation",
     "decided_by": "team"},
    {"date": "2026-10-01", "item": "Recording exclusion (C3)",
     "rule": "Drop a recording with more than 35% invalid gaze; drop a participant missing A1 or A4",
     "decided_by": "team (defended version)"},
    {"date": "2026-10-15", "item": "P31",
     "rule": "Left out: its files are a byte-identical copy of P28's (R5-1)",
     "decided_by": "team (defended version)"},
]


def decisions_markdown() -> str:
    lines = ["| Date | Item | Rule | Decided by |", "|---|---|---|---|"]
    lines += [f"| {d['date']} | {d['item']} | {d['rule']} | {d['decided_by']} |" for d in DECISIONS]
    return "\n".join(lines) + "\n"

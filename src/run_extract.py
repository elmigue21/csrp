"""Post-defense feature extraction: every recording -> 173 flattened features -> outputs/final/.

Follows docs/post-defense-data-processing.md: the current cleaning rules (C1-C10) run as they
are, every signal is flattened into 17 statistics (§4), the feature table holds only the
labelled A1/A4 recordings with no per-person standardization (§5), and everything dropped or
changed is logged (§7). outputs/final/ is rebuilt from scratch on
every run; nothing in it is edited by hand.

Usage (local):  python src/run_extract.py [--refresh]
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from checks import sanity_check, sanity_passed
from code_preview import code_preview
from config import (
    DATA_DIR, DUPLICATE_RECORDING_IDS, FINAL_DIR, HIGH_ACTIVITY, LOW_ACTIVITY, MAX_INVALID_FRACTION,
    WORK_DIR,
)
from data import load_recording, recording_keys
from dataset import analysis_participants
from excel_previews import features_preview, raw_preview
from features import recording_signals
from final_figures import all_figures
from flatten import DROPPED, feature_dictionary, feature_names, flat_features
from preprocess import invalid_fraction
from processing_log import decisions_markdown, feature_log, recording_log_row
from raw_dictionary import raw_data_dictionary, raw_table_shapes
from run_colet import provenance

_CODE_FILES = ("data.py", "preprocess.py", "events.py", "features.py", "flatten.py",
               "processing_log.py", "run_extract.py")   # figures are redrawn every run
_SETTING_PREFIXES = ("CONFIDENCE", "MAX_INVALID", "VALID_TIME", "BLINK", "GAZE", "PUPIL",
                     "MAX_VELOCITY", "IVT", "MIN_FIXATION")
_EVENT_PREFIXES = ("gaze_speed__", "fixation_", "saccade_")
_EXCLUSION_REASON = "invalid gaze > 35%"

def _settings() -> dict:
    return {k: v for k, v in sorted(vars(config).items())
            if k.isupper() and k.startswith(_SETTING_PREFIXES)}


def _cache_key(data_dir) -> str:
    """Hash of the cleaning settings, the extraction code and the input files (path, name, size,
    modification time), so a stale cache is never reused."""
    here = Path(__file__).parent
    h = hashlib.sha1(repr(_settings()).encode())
    data_dir = Path(data_dir).resolve()
    h.update(str(data_dir).encode())
    for f in sorted(data_dir.iterdir()):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
    for name in _CODE_FILES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:10]


def build_flat_table(data_dir=DATA_DIR, work_dir=WORK_DIR, refresh=False):
    """(flat, log_recordings): one row per recording, all recordings, excluded ones included."""
    work_dir = Path(work_dir)
    key = _cache_key(data_dir)
    flat_path = work_dir / f"flat_{key}.parquet"
    log_path = work_dir / f"log_recordings_{key}.parquet"
    if flat_path.exists() and log_path.exists() and not refresh:
        return pd.read_parquet(flat_path), pd.read_parquet(log_path)
    flat_rows, log_rows = [], []
    for participant, activity in recording_keys(data_dir):
        raw = load_recording(participant, activity, data_dir)
        inv = invalid_fraction(raw["gaze"])
        excluded = inv > MAX_INVALID_FRACTION
        ids = {"participant": participant, "activity": activity}
        row = {**ids, "invalid_fraction": inv, "excluded": excluded}
        sig = None
        if not excluded:
            sig = recording_signals(raw)
            row.update(flat_features(sig))
            row.update(sig["qc"])
        flat_rows.append(row)
        log_rows.append({**ids, "excluded": excluded,
                         "exclusion_reason": _EXCLUSION_REASON if excluded else "",
                         **recording_log_row(raw, sig)})
    flat, log = pd.DataFrame(flat_rows), pd.DataFrame(log_rows)
    work_dir.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(flat_path, index=False)
    log.to_parquet(log_path, index=False)
    return flat, log


def _label(activity: pd.Series) -> pd.Series:
    return activity.map({LOW_ACTIVITY: "low", HIGH_ACTIVITY: "high"}).fillna("")


def _readme(n_participants: int, n_rows: int, n_features: int, passed: bool, n_c9: int) -> str:
    c9 = ("passed: all fixation and saccade features are kept" if passed else
          f"FAILED: {n_c9} fixation/saccade features removed (see logs/log_features.csv)")
    return f"""# Final outputs: post-defense feature extraction

Written by `src/run_extract.py`. Do not edit by hand: every run deletes and rewrites this folder.
Method: `docs/post-defense-data-processing.md`. Run details: `run_info.json`.

- **Participants:** {n_participants}
- **Rows:** {n_rows} in the feature table: each participant's A1 (`low`) and A4 (`high`) recording.
  A2 and A3 have no label and are left out (they still count in the C9 check below).
- **Features:** {n_features} model features, out of {len(feature_names())} extracted. The other
  {len(feature_names()) - n_features} were dropped by the team's data-quality decisions (D1-D10); each one's reason is
  in `logs/log_features.csv` and `features/feature_dictionary.csv`.
- **Event-detection sanity check (C9):** {c9}

| File | What it is |
|---|---|
| `features/features.csv` | One row per A1/A4 recording: participant, activity, label, then every model feature |
| `features/feature_dictionary.csv` | One plain-language line per extracted feature: signal, statistic, unit, tag, kept or dropped and why |
| `logs/log_recordings.csv` | Every recording (excluded ones too): what each cleaning step kept or removed |
| `logs/log_features.csv` | Every feature: blank values in A1 and A4, drops |
| `logs/log_decisions.md` | The decisions behind the rules, with dates |
| `logs/sanity_check.csv` | The C9 event-detection check per activity |
| `raw/raw_data_dictionary.csv` | Every column of the original COLET tables: meaning, unit, used or not, and why |
| `raw/raw_table_shapes.csv` | Size of each original table: files, columns, rows per file |
| `results/` | Model results, written by `src/run_model.py` (kept when this folder is rebuilt; `results/STALE.md` appears if the features changed since) |
| `paper_figures/` | Thesis figures, written by `src/paper_figures.py` from `results/` (kept when this folder is rebuilt) |
| `previews/raw_preview.xlsx` | The raw COLET tables in Excel: one recording (P01 A1) in detail, plus table sizes, column dictionary and ratings |
| `previews/features_preview.xlsx` | The final feature table in Excel, with the feature dictionary and the dropped features |
| `code_preview.html` | Code tour for the panel: the real code of each pipeline step, in order (open in a browser) |

**Figures** (`figures/`, PNG, redrawn every run from the files above). These are for the team to
understand the data, not figures for the paper:

| File | What it shows |
|---|---|
| `fig_01_raw_tables.png` | Size of each original COLET table |
| `fig_02_raw_columns.png` | Every original column, used or not (the raw data dictionary as a picture) |
| `fig_03_data_flow.png` | How the data shrinks from the raw files to the final feature table |
| `fig_04_feature_table.png` | The final table: features kept and dropped per signal |
| `fig_05_recording_signals.png` | One cleaned recording over time: pupil, gaze angles, eye speed, blinks |
| `fig_06_cleaning_effect.png` | Cleaning on real recordings: stray samples (C1), pupil before and after (C2, C5, C7) |
| `fig_07_a1_vs_a4.png` | Six core measures, low vs high load, one line per participant (descriptive only) |
| `fig_08_data_quality.png` | Unreadable gaze per recording, the 35% cut-off and the dropped recordings |
"""


def _write(final_dir: Path, files: dict[str, object]) -> None:
    """Write everything to a temporary folder, then swap it in with renames, so a failed run never
    leaves a half-written or half-deleted final folder. On Windows a folder holding an open file
    (e.g. a CSV open in Excel) cannot be renamed: the old folder is then kept whole."""
    tmp = final_dir.with_name(final_dir.name + ".tmp")
    old = final_dir.with_name(final_dir.name + ".old")
    for leftover in (tmp, old):
        if leftover.exists():
            shutil.rmtree(leftover, ignore_errors=True)
    for rel, content in files.items():
        path = tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, encoding="utf-8-sig")   # BOM: Excel reads UTF-8
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    if final_dir.exists():
        try:
            final_dir.rename(old)
        except OSError as err:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"Could not replace {final_dir}: close any file open from it "
                               "(for example a CSV in Excel) and run again.") from err
    tmp.rename(final_dir)
    shutil.rmtree(old, ignore_errors=True)


def _flag_stale_results(final_dir: Path) -> None:
    info = final_dir / "results" / "run_info.json"
    if not info.exists():
        return
    used = json.loads(info.read_text(encoding="utf-8")).get("features_sha256")
    now = hashlib.sha256((final_dir / "features" / "features.csv").read_bytes()).hexdigest()
    if used != now:
        (final_dir / "results" / "STALE.md").write_text(
            "# Stale results\n\nThese model results were computed from a different "
            "`features/features.csv` than the one in this folder now. Re-run "
            "`python src/run_model.py` before using them.\n", encoding="utf-8")


def main(data_dir=DATA_DIR, final_dir=FINAL_DIR, work_dir=WORK_DIR, refresh=False) -> dict:
    data_dir, final_dir = Path(data_dir), Path(final_dir)
    flat, log_recordings = build_flat_table(data_dir, work_dir, refresh)
    people = analysis_participants(flat)
    keep = flat["participant"].isin(people) & ~flat["excluded"]
    rows = flat[keep].sort_values(["participant", "activity"]).reset_index(drop=True)
    labelled = log_recordings["activity"].isin([LOW_ACTIVITY, HIGH_ACTIVITY])
    log_recordings["in_analysis"] = (log_recordings["participant"].isin(people)
                                     & ~log_recordings["excluded"] & labelled)
    log_recordings["analysis_reason"] = [
        "" if inside else
        "recording excluded (C3)" if excluded else
        "duplicate of another participant's files (R5-1)" if pid in DUPLICATE_RECORDING_IDS else
        "participant lost A1 or A4 (C3)" if pid not in people else
        "A2/A3: no label, not in the feature table"
        for inside, excluded, pid in zip(log_recordings["in_analysis"], log_recordings["excluded"],
                                         log_recordings["participant"])]

    sanity = sanity_check(rows)        # C9 checks the detector on every retained recording, A2/A3 too
    passed = sanity_passed(sanity)
    all_features = feature_names()
    c9_dropped = {} if passed else {
        f: "C9 event-detection sanity check failed" for f in all_features
        if f.startswith(_EVENT_PREFIXES) or f in ("fixation_rate", "saccade_rate")}
    dropped = {**DROPPED, **c9_dropped}
    features = [f for f in all_features if f not in dropped]

    # Only the labelled A1/A4 recordings form the feature table. The log describes every extracted
    # feature, dropped ones included, so it is built from the full table.
    rows = rows[rows["activity"].isin([LOW_ACTIVITY, HIGH_ACTIVITY])].reset_index(drop=True)
    ids = rows[["participant", "activity"]].assign(label=_label(rows["activity"]))
    table_all = pd.concat([ids, rows[all_features]], axis=1)
    table = table_all[["participant", "activity", "label", *features]]

    ann = data_dir / "annotation.csv"
    subj = data_dir / "subject_info.csv"
    feature_dict = feature_dictionary()
    raw_dict = raw_data_dictionary(data_dir)
    shapes = raw_table_shapes(log_recordings,
                              annotation_rows=len(pd.read_csv(ann)) if ann.exists() else 0,
                              subject_rows=len(pd.read_csv(subj)) if subj.exists() else 0)
    run_info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        **provenance(data_dir),
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in _settings().items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "n_participants": len(people),
        "n_rows": len(table),
        "n_features": len(features),
        "n_features_extracted": len(all_features),
        "sanity_passed": passed,
    }
    files = {
        "README.md": _readme(len(people), len(table), len(features), passed, len(c9_dropped)),
        "run_info.json": json.dumps(run_info, indent=2, default=str),
        "features/features.csv": table,
        "features/feature_dictionary.csv": feature_dict,
        "logs/log_recordings.csv": log_recordings,
        "logs/log_features.csv": feature_log(table_all, all_features, dropped),
        "logs/log_decisions.md": "# Decisions\n\n" + decisions_markdown(),
        "logs/sanity_check.csv": sanity,
        "raw/raw_data_dictionary.csv": raw_dict,
        "raw/raw_table_shapes.csv": shapes,
        **{f"figures/{name}": png for name, png in all_figures(
            data_dir, log_recordings, table, feature_dict, raw_dict, shapes,
            n_extracted=len(all_features)).items()},
    }
    files.update({
        "previews/raw_preview.xlsx": raw_preview(data_dir, shapes, raw_dict),
        "previews/features_preview.xlsx": features_preview(table, feature_dict, files["logs/log_features.csv"]),
        "code_preview.html": code_preview(final_dir / "results"),
    })
    # Model results (written by run_model.py) and the paper figures drawn from them are carried
    # over, never silently deleted. If the feature table the results were computed from has
    # changed, they are flagged as stale.
    for folder in ("results", "paper_figures"):
        old = final_dir / folder
        if old.is_dir():
            files.update({f"{folder}/{f.relative_to(old).as_posix()}": f.read_bytes()
                          for f in old.rglob("*") if f.is_file() and f.name != "STALE.md"})
    _write(final_dir, files)
    _flag_stale_results(final_dir)
    return {"n_participants": len(people), "n_rows": len(table), "n_features": len(features),
            "sanity_passed": passed}


if __name__ == "__main__":
    out = main(refresh="--refresh" in sys.argv)
    print(json.dumps(out, indent=2))

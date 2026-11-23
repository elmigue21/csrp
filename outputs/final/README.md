# Final outputs: post-defense feature extraction

Written by `src/run_extract.py`. Do not edit by hand: every run deletes and rewrites this folder.
Method: `docs/post-defense-data-processing.md`. Run details: `run_info.json`.

- **Participants:** 44
- **Rows:** 88 in the feature table: each participant's A1 (`low`) and A4 (`high`) recording.
  A2 and A3 have no label and are left out (they still count in the C9 check below).
- **Features:** 140 model features, out of 173 extracted. The other
  33 were dropped by the team's data-quality decisions (D1-D10); each one's reason is
  in `logs/log_features.csv` and `features/feature_dictionary.csv`.
- **Event-detection sanity check (C9):** passed: all fixation and saccade features are kept

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

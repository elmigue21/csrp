# Flattened Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the 171 flattened features (10 signals × 17 statistics − 2 + 3 rates) for every COLET recording, z-score them per person, and write them plus a processing log to `outputs/final/`.

**Architecture:**
- **Raw signals:** `features.py` gains `recording_signals(rec)`, which returns each signal's values, the three rates and the QC numbers. The existing `recording_features` is rebuilt on top of it, so the defended 10-feature pipeline keeps producing identical numbers.
- **Flattening:** a new `flatten.py` turns signals into named statistics and builds the feature dictionary.
- **Logging:** a new `processing_log.py` builds the log tables and the decisions list.
- **Runner:** a new `run_extract.py` runs everything and replaces `outputs/final/` on every run.

**Tech Stack:** Python 3.11, numpy 2.2, pandas 2.3, scipy 1.15, pytest.

**Spec:** `docs/post-defense-data-processing.md`

## Global Constraints

- **Order of work:** extract first. The cleaning rules C1–C10 run exactly as in the current `config.py` and `preprocess.py`, with no threshold changes (§1b).
- **Statistics, in this order:** `min, max, range, mean, median, sd, var, cv, iqr, p05, p10, p25, p75, p90, p95, skew, kurt` (17). Mode is never computed.
- **Signals, in this order (name, unit):** `pupil` mm, `gaze_x` deg, `gaze_y` deg, `gaze_speed` deg/s, `fixation_duration` ms, `saccade_amplitude` deg, `saccade_peak_velocity` deg/s, `saccade_mean_velocity` deg/s, `saccade_duration` ms, `blink_duration` ms.
- **No `cv`** for `gaze_x` and `gaze_y`.
- **Rates:** `blink_rate` (per minute of valid time), `fixation_rate` and `saccade_rate` (per second of valid gaze time), exactly as `recording_features` computes them today.
- **Feature names:** `<signal>__<statistic>` (double underscore). Rates keep their names. Total = **171** model features.
- **Never features:** counts, durations and QC values (`n_*`, `valid_time_s`, valid fractions, `refits`, `invalid_fraction`).
- **Position tag:** the gaze x/y location statistics (`mean, median, min, max, p05, p10, p25, p75, p90, p95`) get tag `position`. Every other feature gets an empty tag.
- **Statistic conventions:**
  - `sd` and `var` use ddof=1, matching the existing `pupil_sd`.
  - Percentiles use numpy's default linear method.
  - `skew` is `scipy.stats.skew(bias=False)` and `kurt` is `scipy.stats.kurtosis(fisher=True, bias=False)`.
- **Minimum number of values:** a statistic is NaN when the data is too small for its math:
  - empty input: everything is NaN;
  - `sd, var, cv, iqr`: need n ≥ 2;
  - `skew`: needs n ≥ 3;
  - `kurt`: needs n ≥ 4.

  The 20-value rule (D2) is **not** applied yet. Per-signal counts go in the log so D2 can be decided from data.
- **cv:** NaN when the mean is 0. Never ±inf.
- **Z-scoring:** reuse `normalize.per_participant_zscore` unchanged. SD = 0 for a person gives z = 0.
- **Rows:**
  - `features_raw.csv` and `features_z.csv` hold every retained activity (A1–A4) of the analysis participants (`dataset.analysis_participants`, 44 people), because z-scoring uses all four activities.
  - Columns: `participant, activity, label`, then the 171 features. `label` is `low` for A1, `high` for A4, and empty for A2/A3.
- **C9 rule:** if `checks.sanity_check` fails in any activity, every feature built from events (`gaze_speed`, `fixation_*`, `saccade_*` statistics, `fixation_rate`, `saccade_rate`) is removed from both feature tables, and `log_features.csv` records the reason.
- **Folders:**
  - **`outputs/final/`:** deleted and rewritten on every run, and only `run_extract.py` writes to it.
  - **`outputs/work/`:** holds the cache and is added to `.gitignore`.
- **Never commit:** the user commits. Plan steps only stage files.

## Review Focus

1. **Recording with zero blinks:** all `blink_duration__*` are NaN, `blink_rate` is 0.0, no warning or crash. Tested in Task 2.
2. **Signal with 1–3 values:** `sd/var/cv/iqr` NaN at n=1, `skew` NaN at n<3, `kurt` NaN at n<4. No RuntimeWarning escapes. Tested in Task 2.
3. **Constant signal, or mean 0:** `skew` and `kurt` NaN (not a scipy warning plus garbage), and `cv` NaN, not inf. Tested in Task 2.
4. **Re-run replaces the folder:** a stale file left in `outputs/final/` from an earlier run is gone after the next run. Tested in Task 4.
5. **Person whose feature never changes across activities:** z = 0, and `log_features.csv` counts that person in `n_people_sd0`. Tested in Task 4.

---

### Task 1: Raw signals per recording

**Files:**
- Modify: `src/features.py` (whole module; `recording_features` becomes a thin wrapper)
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `recording_signals(rec: dict) -> dict` returning:
  - `"signals"`: `dict[str, np.ndarray]` with exactly the 10 signal names from Global Constraints, as 1-D float arrays in the stated units, with no NaNs inside.
    - `pupil`: the finite values of the `clean_pupil` output.
    - `gaze_x` / `gaze_y`: the `az` / `el` arrays already computed.
    - `gaze_speed`: `angular_speed_deg_s(xyz, GAZE_GRID_HZ)` at finite, non-artifact samples.
    - Event signals: from `detect_events`, converted from seconds to ms.
    - `blink_duration`: `(end - start) * 1000` of `clean_blinks` output.
  - `"rates"`: `{"blink_rate", "fixation_rate", "saccade_rate"}`, same formulas as today.
  - `"qc"`: every QC key `recording_features` returns today (`valid_time_s` … `pupil_valid_fraction`), plus `"artifact_samples": int`.
- `recording_features(rec)` keeps its exact keys and values, now computed from `recording_signals`.

- [ ] **Step 1: Write the failing tests** in `tests/test_features.py`:
  - **`test_recording_signals_shape_and_units`:** on `make_recording(1, 4, rng(0), 8.0)`:
    - `set(s["signals"]) == {the 10 names}`;
    - every array is finite;
    - `240 < median(s["signals"]["fixation_duration"]) < 320`;
    - `abs(median(s["signals"]["blink_duration"]) - 150) < 20`;
    - `s["rates"]["blink_rate"] > 0`.
  - **`test_recording_features_unchanged_by_refactor`:** for that recording,
    `recording_features(rec)["pupil_mean"] == approx(np.mean(recording_signals(rec)["signals"]["pupil"]))`
    and `["fixation_duration"] == approx(np.mean(signals["fixation_duration"]))`.
- [ ] **Step 2:** Run `pytest tests/test_features.py -v`. Expected: the two new tests FAIL with ImportError.
- [ ] **Step 3:** Implement `recording_signals` and rebuild `recording_features` on it. Move code only; change no formula.
- [ ] **Step 4:** Run `pytest tests -v`. Expected: all pass, including the old `test_features.py` and `test_run_colet.py` tests, which prove the defended numbers are unchanged.

### Task 2: Flattening and feature dictionary

**Files:**
- Create: `src/flatten.py`
- Test: `tests/test_flatten.py`

**Interfaces:**
- Consumes: the `recording_signals(rec)` output from Task 1.
- Produces:
  - `STATISTICS: tuple[str, ...]`: the 17 names, in order.
  - `SIGNALS: dict[str, str]`: signal name → unit, in order.
  - `RATES: dict[str, str]`: rate name → unit.
  - `NO_CV = ("gaze_x", "gaze_y")`.
  - `POSITION_STATS = ("mean","median","min","max","p05","p10","p25","p75","p90","p95")`.
  - `summarize(values: np.ndarray, skip: tuple[str, ...] = ()) -> dict[str, float]`: the statistic name → value, without the skipped names.
  - `flat_features(sig: dict) -> dict[str, float]`: the 171 `<signal>__<stat>` and rate keys, in order.
  - `feature_names() -> list[str]`: the same 171 names, in order.
  - `feature_dictionary() -> pd.DataFrame` with columns `feature, signal, statistic, unit, tag, meaning`. `meaning` is one plain sentence, for example `pupil__p95` → "Pupil size (mm) that 95% of the recording stays below". `skew`/`kurt` have no unit. `cv` has unit "ratio".

- [ ] **Step 1: Write the failing tests** in `tests/test_flatten.py`:
  - **`test_summarize_known_values`:** for `[1, 2, 3, 4]`:
    - `min=1, max=4, range=3, mean=2.5, median=2.5`;
    - `sd≈1.2910, var≈1.6667, cv≈0.5164, iqr=1.5, p25=1.75, p95=3.85`;
    - `skew≈0.0`;
    - `kurt≈-1.2`.
  - **`test_summarize_small_n`:**
    - empty input: all 17 are NaN;
    - n=1: `mean=min=max=value`, and `sd, var, cv, iqr, skew, kurt` are NaN;
    - n=3: `skew` finite, `kurt` NaN.
  - **`test_summarize_constant_and_zero_mean`:**
    - `[5, 5, 5, 5]`: `sd=0`, and `skew` and `kurt` are NaN;
    - `[-1, 1]`: `cv` is NaN, not inf;
    - run under `warnings.simplefilter("error")`, so no RuntimeWarning escapes.
  - **`test_flat_features_count_and_names`:** on `recording_signals(make_recording(1, 4, ...))`:
    - `len == 171 == len(feature_names())`, with the same order;
    - `"gaze_x__cv" not in`;
    - `"pupil__p95" in`;
    - no name contains `mode`.
  - **`test_no_blinks_gives_nan_durations_and_zero_rate`:** `make_recording(2, 1, ...)` has no blinks, so every `blink_duration__*` is NaN and `blink_rate == 0.0`.
  - **`test_feature_dictionary`:**
    - 171 rows, unique `feature`;
    - exactly 20 rows tagged `position`, all with signal in `gaze_x`/`gaze_y`;
    - no empty `meaning`.
- [ ] **Step 2:** Run `pytest tests/test_flatten.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3:** Implement `src/flatten.py`. Guard each statistic by n (see Global Constraints) and skip scipy for n below its minimum or when `sd == 0`, so no warnings arise.
- [ ] **Step 4:** Run `pytest tests/test_flatten.py -v`. Expected: PASS.

### Task 3: Processing log builders

**Files:**
- Create: `src/processing_log.py`
- Test: `tests/test_processing_log.py`

**Interfaces:**
- Consumes: `recording_signals` from Task 1; `trim_to_gaze_range`, `invalid_fraction` and `clean_blinks` from `preprocess`.
- Produces:
  - `recording_log_row(raw_rec: dict, sig: dict | None) -> dict` (`sig` is None for an excluded recording). Keys:
    - `gaze_rows`, `pupil_rows_raw`, `pupil_rows_after_trim`, `blinks_raw`, `blinks_after_trim` (C1);
    - `invalid_fraction` (C2);
    - `duration_s` (last − first gaze timestamp), `valid_time_s`, `excluded_gap_s = duration_s − valid_time_s` (C4);
    - `blinks_kept` (C5);
    - `gaze_valid_fraction`, `binocular_fraction` (C6, C8);
    - `pupil_rows_3d`, `pupil_valid_fraction`, `refits` (C7);
    - `artifact_samples`, `n_fixations`, `n_saccades` (C9);
    - `n_<signal>` for each of the 10 signals.

    All signal-derived keys are NaN when `sig` is None.
  - `feature_log(raw: pd.DataFrame, z: pd.DataFrame, features: list[str], dropped: dict[str, str]) -> pd.DataFrame` with columns:
    - `feature, blank_A1, blank_A4, blank_all`;
    - `n_people_sd0`: number of participants whose raw values in that column are non-NaN and all equal;
    - `dropped` (bool), `drop_reason`.

    `dropped` maps feature → reason, and dropped features still get a row.
  - `DECISIONS: list[dict]` with keys `date, item, rule, decided_by`. The entries:
    - Mode not computed;
    - No `cv` for gaze x/y;
    - Counts and durations are not features;
    - Valid-sample % is not a feature;
    - Exclusion of recordings with > 35% invalid gaze;
    - P31 left out as a duplicate of P28;
    - Extract first, decide after.

    Use `decided_by = "team"` for the items the user confirmed in chat (mode, extract-first) and `"Claude (to confirm)"` for the rest.
  - `decisions_markdown() -> str`: a markdown table of `DECISIONS`.

- [ ] **Step 1: Write the failing tests** in `tests/test_processing_log.py`:
  - **`test_recording_log_row`:** on a synthetic recording,
    - `pupil_rows_after_trim <= pupil_rows_raw`;
    - `0 <= invalid_fraction <= 1`;
    - `n_fixation_duration == qc n_fixations`;
    - `excluded_gap_s >= 0`.
  - **`test_recording_log_row_excluded`:** with `sig=None`, the `n_pupil` and `n_fixations` keys are NaN and `gaze_rows` is still filled.
  - **`test_feature_log_counts`:** a hand-built 2-person × 4-activity frame where person 1's column `a` is constant:
    - `n_people_sd0 == 1` for `a`;
    - one NaN in an A1 row gives `blank_A1 == 1`;
    - `dropped={"b": "C9"}` gives `dropped == True, drop_reason == "C9"`.
- [ ] **Step 2:** Run `pytest tests/test_processing_log.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3:** Implement `src/processing_log.py`.
- [ ] **Step 4:** Run `pytest tests/test_processing_log.py -v`. Expected: PASS.

### Task 4: Extraction runner and the final folder

**Files:**
- Create: `src/run_extract.py`
- Modify: `src/config.py`: add `FINAL_DIR = Path(os.environ.get("COLET_FINAL_DIR", PROJECT_ROOT / "outputs" / "final"))` and `WORK_DIR = Path(os.environ.get("COLET_WORK_DIR", PROJECT_ROOT / "outputs" / "work"))` next to `OUT_DIR`.
- Modify: `.gitignore`: add `outputs/work/`.
- Test: `tests/test_run_extract.py`

**Interfaces:**
- Consumes: Tasks 1–3; `dataset.analysis_participants`; `data.load_recording` and `data.recording_keys`; `normalize.per_participant_zscore`; `checks.sanity_check` and `checks.sanity_passed`; config `MAX_INVALID_FRACTION` and `DUPLICATE_RECORDING_IDS`.
- Produces:
  - `build_flat_table(data_dir, work_dir, refresh=False) -> tuple[pd.DataFrame, pd.DataFrame]`, returning (flat, log_recordings).
    - **flat:** one row per recording, all 188, with `participant, activity, invalid_fraction, excluded`, then the 171 features and the QC keys.
    - **log_recordings:** one `recording_log_row` per recording, plus `participant, activity, excluded, exclusion_reason` (`"invalid gaze > 35%"` or empty) and `in_analysis`.
    - Both are cached in `work_dir` as parquet, keyed by the same settings-hash approach as `dataset._settings_hash`, with `flatten.py` and `processing_log.py` added to the hashed source.
  - `main(data_dir=DATA_DIR, final_dir=FINAL_DIR, work_dir=WORK_DIR, refresh=False) -> dict` writes the layout in spec §7.1:
    - `features/features_raw.csv`, `features/features_z.csv`, `features/feature_dictionary.csv`;
    - `logs/log_recordings.csv`, `logs/log_features.csv`, `logs/log_decisions.md`, `logs/sanity_check.csv`;
    - `README.md`, which lists each file with one plain sentence, the row and feature counts, and whether C9 passed;
    - `run_info.json` with `date`, `git_commit` (from `git rev-parse HEAD`, or `"unknown"`), `git_dirty` (bool), the `settings` (the same constants `_settings_hash` uses), `versions` (python, numpy, pandas, scipy), `n_participants`, `n_rows`, `n_features`, and `sanity_passed`.

    Returns `{"n_participants", "n_rows", "n_features", "sanity_passed"}`.
  - The CLI is `python src/run_extract.py [--refresh]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_run_extract.py`, using `write_synthetic_dataset(tmp_path/"data", n_participants=6, seconds=5.0)`:
  - **`test_extract_writes_final_folder`:**
    - every file in the layout exists;
    - `features_raw.csv` has 6×4 rows and 3 + 171 columns;
    - `label` is `low`/`high` on A1/A4 and empty elsewhere;
    - `feature_dictionary.csv` has 171 rows;
    - `run_info.json` has `n_features == 171` and the `git_commit` key.
  - **`test_extract_replaces_stale_files`:** write `final/stale.txt`, run `main`, and check the file no longer exists.
  - **`test_zscore_and_sd0_logged`:**
    - in `features_z.csv`, each participant's `pupil__mean` over their rows has mean ≈ 0;
    - `log_features.csv` has 171 rows with column `n_people_sd0`.
  - **`test_excluded_recording_logged`:** with `write_synthetic_dataset(..., low_quality={(2, 4)})`, using conftest's existing `low_quality` argument:
    - `log_recordings.csv` marks P2 A4 `excluded=True` with reason `invalid gaze > 35%`;
    - P2 is absent from `features_raw.csv`.
- [ ] **Step 2:** Run `pytest tests/test_run_extract.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3:** Implement `run_extract.py` and the config and `.gitignore` changes.
  - `main` builds everything in memory first, then `shutil.rmtree(final_dir)` and writes the files, so a crash never leaves a half-deleted folder.
  - Apply the C9 rule from Global Constraints before z-scoring.
- [ ] **Step 4:** Run `pytest tests -v`. Expected: all pass.

### Task 5: Run on the real data and check

- [ ] **Step 1:** Run `python src/run_extract.py --refresh` in the background (several minutes). Expected: `outputs/final/` is created.
- [ ] **Step 2: Check the numbers**:
  - `run_info.json`: `n_participants == 44`, `n_features == 171`, `sanity_passed == true`;
  - `features_raw.csv` has 44 × 4 rows minus excluded activities (P18 etc. keep all 4; check against `01b_activities_per_participant.csv`);
  - `log_features.csv` blank counts for `blink_duration__mean` are A1 = 13 and A4 = 0, matching the D1 numbers in the spec.
- [ ] **Step 3: Cross-check against the defended pipeline:** `features_raw.csv` `pupil__mean` equals `04_feature_table.csv` `pupil_mean` for the same rows (max abs diff < 1e-9).
- [ ] **Step 4:** Update the spec's D1/D2 numbers from the real log if they differ, then list `git add` commands for the user:

```bash
git add src/features.py src/flatten.py src/processing_log.py src/run_extract.py src/config.py .gitignore tests/test_features.py tests/test_flatten.py tests/test_processing_log.py tests/test_run_extract.py docs/post-defense-data-processing.md docs/superpowers/plans/2026-11-09-flattened-feature-extraction.md
```

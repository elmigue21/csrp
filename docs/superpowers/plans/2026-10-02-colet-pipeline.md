# COLET Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Repo rule (overrides the skill default):** never run `git commit`. Each task ends with a
> "Stage for review" step that lists `git add` commands; the user commits.
>
> **Data rule:** do **not** run anything on the real COLET data (`data/colet/parquet/`) until
> the user says so. All tests use synthetic data written to a temporary folder.

**Goal:** Rewrite `src/` for the COLET study (preprocessing → event detection → 10 features →
per-person z-score → LR vs XGBoost under LOPO → metrics, comparison, importance) and add a
Colab notebook that clones the repo from GitHub, reads the Parquet data from Google Drive and
runs the analysis stage by stage.

**Architecture:** One module per stage, each a set of pure functions over pandas/numpy, with
every threshold in `src/config.py`. `src/run_colet.py` chains them and writes every Chapter 4
table to `outputs/colet/tables/`. The notebook imports the same functions so each stage can be
checked before the next. Paths come from environment variables so the same code runs locally
and on Colab.

**Tech Stack:** Python 3.11+, numpy 2.2.4, pandas 2.3.1, scipy 1.15.2, scikit-learn 1.7.1,
xgboost 3.2.0, shap 0.51.0, pyarrow 24.0.0, matplotlib 3.10.5, pytest 9.

**Spec:** `docs/methodology-colet.md` (frozen 2026-10-02) §§2–11, with `docs/researcher-notes.md`
N18, N22–N25 and `docs/rrl-decision-log.md` (C1–C4, D-M1 revised, D-M3, D-M4, D-M8, D-N6,
V2-1–V2-4, V2-9, V3-1–V3-3, R4-1, R4-2).

## Global Constraints

- Sample invalid if `confidence < 0.8` (P2). Recording excluded if > 35% of gaze samples invalid (P3). Participant dropped if A1 or A4 is excluded.
- Valid time excludes gaps > 1 s between consecutive gaze timestamps (P4); every rate uses valid time.
- Blinks: COLET's blink events; merge blinks < 100 ms apart, then keep 50–500 ms; blink periods are missing for gaze and pupil (P5).
- Gaze: per-eye `gaze_normal0/1`, averaged; **samples with only one valid eye are missing** (decided 2026-10-02); gaps < 75 ms interpolated, longer gaps stay missing (P6, P8).
- Pupil: `diameter_3d` from `3d c++` rows only; 1.5–9 mm; MAD speed filter; both eyes averaged; gaps ≤ 250 ms filled; 4 Hz low-pass; refits counted (P7).
- Events: 5-tap differentiator velocity; reject > 1000 °/s; I-VT 45 °/s; fixations ≥ 55 ms (§5).
- Sanity check **per activity**: median fixation 150–400 ms **and** saccade:fixation 0.8–1.25; any failure → P1 uses the 5 non-event features (§5 step 7, R4-2).
- Features (exact order): `pupil_mean, pupil_sd, blink_rate, fixation_rate, fixation_duration, saccade_rate, saccade_amplitude, saccade_peak_velocity, gaze_sd_x, gaze_sd_y`. Fallback: `pupil_mean, pupil_sd, blink_rate, gaze_sd_x, gaze_sd_y`. P2: `pupil_mean, pupil_sd`.
- Per-person z-score over each participant's retained activities (all 4), label-free (§7).
- Labels: A1 = 0 (low), A4 = 1 (high); A2/A3 never enter a model (§3).
- LR: elastic net, tuned C and l1_ratio. XGBoost: max_depth 1–3, 50–300 trees, learning_rate 0.05–0.1, subsample/colsample 0.7–1.0, λ/α tuned. **Equal budget:** the same number of random-search candidates for both (§8).
- LOPO outer loop; inner `StratifiedGroupKFold` by participant; all fitted preprocessing inside folds (§9).
- Metrics: pooled ROC-AUC + participant cluster-bootstrap 95% CI; balanced accuracy, macro-F1, per-class recall, Brier, majority baseline, share of participants with p(A4) > p(A1) (§10).
- Comparison: AUC difference + bootstrap CI; paired Wilcoxon on per-participant margins; grouped permutation test vs chance (§10).
- Importance: LR coefficients; SHAP for both (TreeSHAP / linear SHAP) on held-out rows; permutation importance on pooled OOF; Kendall τ with bootstrap CI; gain in appendix only (§10).
- Ceiling rule: if both models > 0.95 AUC in P1, flag it in the results (§11).
- Never commit. Never run on real data without the user's go-ahead.

## Review Focus

- A recording with **no blinks** (common in A1) → `blink_rate` must be 0.0, not NaN. Test in Task 4.
- A recording whose valid gaze is **too short for any fixation** → event features NaN, no crash; the row stays in the table. Test in Task 4.
- A **gap > 1 s** mid-recording → excluded from valid time and no velocity computed across it. Tests in Tasks 2 and 3.
- A participant with **A2 or A3 excluded** but valid A1 and A4 → kept; z-score over the remaining activities. Test in Task 5.
- A feature that is **constant within one participant** → z-score 0.0, not inf/NaN. Test in Task 5.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/config.py` | Rewrite | Paths (env-overridable) and every threshold |
| `src/data.py` | Rewrite | Read one recording / list recordings / read annotation |
| `src/preprocess.py` | Create | P1–P8: trimming, validity, valid time, blinks, gaze directions, pupil cleaning, refits |
| `src/events.py` | Create | Velocity, I-VT, fixations, saccades |
| `src/features.py` | Create | The 10 features + QC columns for one recording |
| `src/dataset.py` | Create | Feature table for all recordings, exclusion, analysis rows, retention |
| `src/checks.py` | Create | Sanity check per activity, manipulation check, V2-13 redundancy |
| `src/labels.py` | Rewrite | A1/A4 selection and binary labels |
| `src/normalize.py` | Modify | Keep `per_participant_zscore`; remove GAZELOAD scheme dispatch |
| `src/models.py` | Rewrite | Elastic-net LR and shallow XGBoost + equal-budget search spaces |
| `src/evaluate.py` | Rewrite | LOPO with random search, metrics, bootstrap, comparison, permutation test |
| `src/importance.py` | Rewrite | Coefficients, SHAP for both, permutation importance, Kendall τ, gain |
| `src/run_colet.py` | Create | Chain the stages; write tables and `run_metadata.json` |
| `requirements-colab.txt` | Create | Pinned versions |
| `notebooks/colet_pipeline.ipynb` | Create | Colab runner (clone, install, mount Drive, staged run) |
| `tests/conftest.py` | Create | `sys.path` + synthetic COLET generator |
| `tests/test_*.py` | Create | One test file per module |
| `src/viz_style.py` | Keep | Unchanged (used by a later figures plan) |

Out of scope for this plan: Chapter 4 figures (separate plan once results exist); moving
`data/colet/convert_colet.py` into `src/` (needs the user's OK).

---

### Task 1: Config, data loader and synthetic test data

**Files:**
- Rewrite: `src/config.py`
- Rewrite: `src/data.py`
- Create: `tests/conftest.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `config.*` constants below; `data.recording_path(participant:int, activity:int, signal:str, data_dir) -> Path`; `data.load_recording(participant, activity, data_dir) -> dict[str, pd.DataFrame]` with keys `gaze`, `pupil`, `blinks`; `data.recording_keys(data_dir) -> list[tuple[int,int]]`; `data.load_annotation(data_dir) -> pd.DataFrame`; fixture-free helper `conftest.write_synthetic_dataset(root: Path, n_participants=6, seconds=8.0, seed=0, low_quality=None) -> Path`.

- [ ] **Step 1: Write `src/config.py`**

```python
"""Paths and every fixed setting of the COLET pipeline (methodology-colet.md, frozen 2026-10-02).

All thresholds were fixed after a label-blind inspection and before any model was run.
Changing one here means changing methodology-colet.md and the decision log too.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Colab points these at Google Drive; locally they default to the converted copy and
# to outputs/colet/ (both git-ignored).
DATA_DIR = Path(os.environ.get("COLET_DATA_DIR", PROJECT_ROOT / "data" / "colet" / "parquet"))
OUT_DIR = Path(os.environ.get("COLET_OUT_DIR", PROJECT_ROOT / "outputs" / "colet"))
TABLE_DIR = OUT_DIR / "tables"
CACHE_DIR = OUT_DIR / "cache"

# --------------------------------------------------------------------------- design
ACTIVITIES = (1, 2, 3, 4)
LOW_ACTIVITY = 1    # single task, no time pressure
HIGH_ACTIVITY = 4   # counting aloud + time pressure

# --------------------------------------------------------------------------- P2-P4 quality
CONFIDENCE_MIN = 0.8
MAX_INVALID_FRACTION = 0.35
VALID_TIME_MAX_GAP_S = 1.0

# --------------------------------------------------------------------------- P5 blinks
BLINK_MERGE_S = 0.100
BLINK_MIN_S = 0.050
BLINK_MAX_S = 0.500

# --------------------------------------------------------------------------- P6/P8 gaze
GAZE_GRID_HZ = 240.0
GAZE_MAX_INTERP_S = 0.075

# --------------------------------------------------------------------------- P7 pupil
PUPIL_METHOD_TAG = "3d"          # the `method` column holds e.g. "array(['3d c++'], ...)"
PUPIL_MIN_MM = 1.5
PUPIL_MAX_MM = 9.0
PUPIL_MAD_MULTIPLIER = 16.0      # Kret & Sjak-Shie 2019 dilation-speed filter default
PUPIL_GRID_HZ = 120.0
PUPIL_MAX_INTERP_S = 0.250
PUPIL_LOWPASS_HZ = 4.0
PUPIL_LOWPASS_ORDER = 2

# --------------------------------------------------------------------------- P9 events
MAX_VELOCITY_DEG_S = 1000.0
IVT_THRESHOLD_DEG_S = 45.0
MIN_FIXATION_S = 0.055
SANITY_FIXATION_MEDIAN_MS = (150.0, 400.0)
SANITY_SACCADE_FIXATION_RATIO = (0.8, 1.25)

# --------------------------------------------------------------------------- features
FEATURES = [
    "pupil_mean", "pupil_sd", "blink_rate",
    "fixation_rate", "fixation_duration",
    "saccade_rate", "saccade_amplitude", "saccade_peak_velocity",
    "gaze_sd_x", "gaze_sd_y",
]
FALLBACK_FEATURES = ["pupil_mean", "pupil_sd", "blink_rate", "gaze_sd_x", "gaze_sd_y"]
PUPIL_FEATURES = ["pupil_mean", "pupil_sd"]

# --------------------------------------------------------------------------- evaluation
RANDOM_STATE = 0
INNER_FOLDS = 5
N_ITER = 30                 # random-search candidates per model: the equal tuning budget
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200
N_PERM_IMPORTANCE = 50
CEILING_AUC = 0.95
```

- [ ] **Step 2: Write `src/data.py`**

```python
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
```

- [ ] **Step 3: Write `tests/conftest.py` (synthetic COLET generator)**

```python
"""Test setup: put src/ on the path and generate small synthetic COLET folders.

The synthetic data mimics the real Parquet layout (column names, the 2d/3d pupil row
duplication, the repr-string `method` column) and has a known structure: fixations of
FIX_S seconds separated by 10-degree saccades lasting SACC_S seconds, a pupil that is
larger in activity 4, and more blinks in activity 4. Tests compare against these.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GAZE_HZ = 250.0
PUPIL_HZ = 120.0
FIX_S = 0.30
SACC_S = 0.04
SACC_DEG = 10.0
EYE_BIAS_DEG = 5.0            # each eye bent inward by this much (cancels when averaged)
METHOD_3D = "array(['3d c++'], dtype='<U6')"
METHOD_2D = "array(['2d c++'], dtype='<U6')"


def unit_from_angles(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.column_stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def gaze_path(seconds, rng, t0=100.0):
    """Timestamps and true (az, el) in degrees: fixations joined by 10-degree saccades."""
    t = np.arange(t0, t0 + seconds, 1 / GAZE_HZ)
    az, el = np.zeros_like(t), np.zeros_like(t)
    cur = np.array([0.0, 0.0])
    phase_start, in_fix, target = t0, True, cur
    for i, ti in enumerate(t):
        elapsed = ti - phase_start
        if in_fix and elapsed >= FIX_S:
            ang = rng.uniform(0, 2 * np.pi)
            step = SACC_DEG * np.array([np.cos(ang), np.sin(ang)])
            target = np.clip(cur + step, -15, 15)
            if np.linalg.norm(target - cur) < 0.5 * SACC_DEG:
                target = cur - step
            start, in_fix, phase_start = cur.copy(), False, ti
            elapsed = 0.0
        if not in_fix:
            frac = min(elapsed / SACC_S, 1.0)
            cur = start + frac * (target - start)
            if elapsed >= SACC_S:
                cur, in_fix, phase_start = target.copy(), True, ti
        az[i], el[i] = cur
    return t, az, el


def make_recording(participant, activity, rng, seconds=8.0, low_quality=False, one_eye_frac=0.0):
    t, az, el = gaze_path(seconds, rng)
    n = len(t)
    n_blinks = 1 if activity == 1 else 4
    blink_starts = rng.uniform(t[0] + 0.5, t[-1] - 0.5, n_blinks) if activity != 1 or participant % 2 else []
    blinks = pd.DataFrame({
        "start_timestamp": np.sort(blink_starts),
        "end_timestamp": np.sort(blink_starts) + 0.15,
    })
    blinks["duration"] = blinks["end_timestamp"] - blinks["start_timestamp"]

    in_blink = np.zeros(n, bool)
    for s, e in zip(blinks.start_timestamp, blinks.end_timestamp):
        in_blink |= (t >= s) & (t <= e)
    conf = np.where(in_blink, 0.1, 0.99)
    if low_quality:
        conf[: int(0.5 * n)] = 0.2
    eye0 = unit_from_angles(az + EYE_BIAS_DEG, el)
    eye1 = unit_from_angles(az - EYE_BIAS_DEG, el)
    one_eye = rng.random(n) < one_eye_frac
    eye1[one_eye] = np.nan
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": conf})
    for k, axis in enumerate("xyz"):
        gaze[f"gaze_normal0_{axis}"] = eye0[:, k]
        gaze[f"gaze_normal1_{axis}"] = eye1[:, k]

    tp = np.arange(t[0], t[-1], 1 / PUPIL_HZ)
    base = 3.0 + 0.1 * participant + (0.4 if activity == 4 else 0.0)
    rows = []
    for eye in (0, 1):
        d = base + 0.05 * np.sin(2 * np.pi * 0.5 * tp) + rng.normal(0, 0.01, len(tp))
        pb = np.zeros(len(tp), bool)
        for s, e in zip(blinks.start_timestamp, blinks.end_timestamp):
            pb |= (tp >= s) & (tp <= e)
        for method in (METHOD_2D, METHOD_3D):
            rows.append(pd.DataFrame({
                "pupil_timestamp": tp, "eye_id": float(eye),
                "confidence": np.where(pb, 0.1, 0.99), "method": method,
                "diameter_3d": np.where(pb, 0.0, d), "model_id": 1.0,
            }))
    pupil = pd.concat(rows, ignore_index=True)
    return {"gaze": gaze, "pupil": pupil, "blinks": blinks}


def write_synthetic_dataset(root, n_participants=6, seconds=8.0, seed=0, low_quality=None):
    """Write pXX_tY_*.parquet for activities 1-4 and annotation.csv; return the folder.

    low_quality: optional set of (participant, activity) given > 35% invalid samples.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    low_quality = low_quality or set()
    ann = []
    for p in range(1, n_participants + 1):
        for a in (1, 2, 3, 4):
            rec = make_recording(p, a, rng, seconds, low_quality=(p, a) in low_quality)
            for signal, df in rec.items():
                df.to_parquet(root / f"p{p:02d}_t{a}_{signal}.parquet")
            ann.append({"participant": p, "task": a, "mean": 15 + 12 * a + rng.normal(0, 3)})
    pd.DataFrame(ann).to_csv(root / "annotation.csv", index=False)
    return root
```

- [ ] **Step 4: Write `tests/test_data.py`**

```python
from conftest import write_synthetic_dataset
from data import load_annotation, load_recording, recording_keys


def test_recording_keys_and_load(tmp_path):
    root = write_synthetic_dataset(tmp_path, n_participants=2, seconds=3)
    keys = recording_keys(root)
    assert keys == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
    rec = load_recording(1, 4, root)
    assert set(rec) == {"gaze", "pupil", "blinks"}
    assert "gaze_normal0_x" in rec["gaze"].columns


def test_annotation_renames_task(tmp_path):
    root = write_synthetic_dataset(tmp_path, n_participants=1, seconds=2)
    ann = load_annotation(root)
    assert {"participant", "activity", "mean"} <= set(ann.columns)
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: 2 passed.

- [ ] **Step 6: Stage for review**

```bash
git add src/config.py src/data.py tests/conftest.py tests/test_data.py
```

---

### Task 2: Preprocessing (P1–P8)

**Files:**
- Create: `src/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `config` constants.
- Produces:
  - `trim_to_gaze_range(rec: dict) -> dict`
  - `invalid_fraction(gaze: pd.DataFrame) -> float`
  - `valid_time_s(timestamps) -> float`
  - `clean_blinks(blinks: pd.DataFrame) -> pd.DataFrame` (columns `start`, `end`)
  - `in_intervals(t: np.ndarray, intervals: pd.DataFrame) -> np.ndarray[bool]`
  - `interp_with_gap_limit(t, values, grid, max_gap) -> np.ndarray` (values `(n,)` or `(n,k)`; NaN inside longer gaps)
  - `gaze_directions(gaze, blinks) -> tuple[np.ndarray grid_t, np.ndarray xyz (m,3)]` on a `GAZE_GRID_HZ` grid, NaN rows where missing
  - `clean_pupil(pupil, blinks, t_start, t_end) -> tuple[np.ndarray grid_t, np.ndarray diameter_mm]`
  - `count_refits(pupil) -> int`
  - `binocular_fraction(gaze) -> float`

- [ ] **Step 1: Write the failing tests `tests/test_preprocess.py`**

```python
import numpy as np
import pandas as pd

from conftest import make_recording, unit_from_angles
from preprocess import (
    clean_blinks, clean_pupil, count_refits, gaze_directions, in_intervals,
    interp_with_gap_limit, invalid_fraction, trim_to_gaze_range, valid_time_s,
)


def test_valid_time_skips_gaps_over_one_second():
    ts = np.r_[np.arange(0, 2, 0.01), np.arange(5, 6, 0.01)]   # 3 s gap
    assert abs(valid_time_s(ts) - (1.99 + 0.99)) < 0.02


def test_invalid_fraction():
    gaze = pd.DataFrame({"confidence": [0.9, 0.5, 0.79, 0.95]})
    assert invalid_fraction(gaze) == 0.5


def test_trim_drops_stray_pupil_samples():
    rec = {
        "gaze": pd.DataFrame({"gaze_timestamp": [10.0, 11.0, 12.0]}),
        "pupil": pd.DataFrame({"pupil_timestamp": [10.5, 11.5, 7000.0]}),
        "blinks": pd.DataFrame({"start_timestamp": [10.2, 7000.0], "end_timestamp": [10.4, 7000.2],
                                "duration": [0.2, 0.2]}),
    }
    out = trim_to_gaze_range(rec)
    assert out["pupil"]["pupil_timestamp"].tolist() == [10.5, 11.5]
    assert len(out["blinks"]) == 1


def test_clean_blinks_merges_then_filters():
    b = pd.DataFrame({"start_timestamp": [1.0, 1.25, 3.0, 5.0],
                      "end_timestamp": [1.2, 1.35, 3.02, 5.9]})
    out = clean_blinks(b)
    # 1.0-1.2 and 1.25-1.35 merge (gap 50 ms) into 350 ms; 20 ms and 900 ms dropped
    assert out[["start", "end"]].values.tolist() == [[1.0, 1.35]]


def test_clean_blinks_empty():
    out = clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []}))
    assert len(out) == 0 and list(out.columns) == ["start", "end"]


def test_interp_with_gap_limit():
    t = np.array([0.0, 0.05, 0.10, 0.50])
    v = np.array([0.0, 1.0, 2.0, 10.0])
    grid = np.array([0.025, 0.075, 0.3])
    out = interp_with_gap_limit(t, v, grid, max_gap=0.075)
    assert np.allclose(out[:2], [0.5, 1.5]) and np.isnan(out[2])


def test_gaze_directions_average_eyes_and_drop_one_eye_samples():
    rng = np.random.default_rng(0)
    rec = make_recording(2, 1, rng, seconds=2.0, one_eye_frac=0.0)    # P2 A1: no blinks
    t, xyz = gaze_directions(rec["gaze"], clean_blinks(rec["blinks"]))
    ok = ~np.isnan(xyz[:, 0])
    assert ok.mean() > 0.8
    # averaging the inward-biased eyes recovers the true horizontal angle (~0 at start)
    az0 = np.degrees(np.arctan2(xyz[ok][0, 0], xyz[ok][0, 2]))
    assert abs(az0) < 0.5
    rec2 = make_recording(1, 2, np.random.default_rng(0), seconds=2.0, one_eye_frac=1.0)
    _, xyz2 = gaze_directions(rec2["gaze"], clean_blinks(rec2["blinks"]))
    assert np.isnan(xyz2).all()


def test_gaze_directions_gap_over_75ms_stays_missing():
    t = np.r_[np.arange(0, 1, 0.004), np.arange(1.2, 2, 0.004)]
    v = unit_from_angles(np.zeros_like(t), np.zeros_like(t))
    gaze = pd.DataFrame({"gaze_timestamp": t, "confidence": 0.99})
    for k, axis in enumerate("xyz"):
        gaze[f"gaze_normal0_{axis}"] = v[:, k]
        gaze[f"gaze_normal1_{axis}"] = v[:, k]
    grid, xyz = gaze_directions(gaze, clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []})))
    inside_gap = (grid > 1.01) & (grid < 1.19)
    assert np.isnan(xyz[inside_gap]).all()
    assert not np.isnan(xyz[grid < 0.9]).any()


def test_clean_pupil_removes_zeros_and_keeps_level():
    rng = np.random.default_rng(1)
    rec = make_recording(2, 4, rng, seconds=4.0)
    blinks = clean_blinks(rec["blinks"])
    t0, t1 = rec["gaze"].gaze_timestamp.min(), rec["gaze"].gaze_timestamp.max()
    grid, d = clean_pupil(rec["pupil"], blinks, t0, t1)
    assert np.nanmin(d) > 1.5
    assert abs(np.nanmean(d) - (3.0 + 0.2 + 0.4)) < 0.05


def test_clean_pupil_spike_removed():
    tp = np.arange(0, 3, 1 / 120)
    rows = []
    for eye in (0, 1):
        d = np.full(len(tp), 3.0)
        d[100] = 6.0                     # impossible 3 mm jump in one sample
        rows.append(pd.DataFrame({"pupil_timestamp": tp, "eye_id": float(eye), "confidence": 0.99,
                                  "method": "array(['3d c++'], dtype='<U6')",
                                  "diameter_3d": d, "model_id": 1.0}))
    pupil = pd.concat(rows, ignore_index=True)
    empty = clean_blinks(pd.DataFrame({"start_timestamp": [], "end_timestamp": []}))
    _, d = clean_pupil(pupil, empty, 0.0, 3.0)
    assert np.nanmax(d) < 3.3


def test_count_refits():
    pupil = pd.DataFrame({"method": ["array(['3d c++'], dtype='<U6')"] * 4 + ["array(['2d c++'], dtype='<U6')"],
                          "eye_id": [0.0, 0.0, 1.0, 1.0, 0.0], "model_id": [1.0, 2.0, 5.0, 5.0, 9.0]})
    assert count_refits(pupil) == 1


def test_in_intervals():
    iv = pd.DataFrame({"start": [1.0], "end": [2.0]})
    assert in_intervals(np.array([0.5, 1.5, 2.5]), iv).tolist() == [False, True, False]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'preprocess'`.

- [ ] **Step 3: Write `src/preprocess.py`**

```python
"""Preprocessing steps P1-P8 (methodology-colet.md §4).

Every function takes and returns plain pandas/numpy objects so each step can be checked
on its own. Missing data stays NaN; nothing here fills a long gap or invents a sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from config import (
    BLINK_MAX_S, BLINK_MERGE_S, BLINK_MIN_S, CONFIDENCE_MIN, GAZE_GRID_HZ, GAZE_MAX_INTERP_S,
    PUPIL_GRID_HZ, PUPIL_LOWPASS_HZ, PUPIL_LOWPASS_ORDER, PUPIL_MAD_MULTIPLIER,
    PUPIL_MAX_INTERP_S, PUPIL_MAX_MM, PUPIL_METHOD_TAG, PUPIL_MIN_MM, VALID_TIME_MAX_GAP_S,
)

_EMPTY_BLINKS = pd.DataFrame({"start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})


def trim_to_gaze_range(rec: dict) -> dict:
    """P1: keep pupil samples and blinks inside the gaze time range.

    P18 A3/A4 carry stray pupil samples about two hours after the activity ended.
    """
    t = rec["gaze"]["gaze_timestamp"]
    t0, t1 = t.min(), t.max()
    pupil = rec["pupil"]
    pupil = pupil[(pupil["pupil_timestamp"] >= t0) & (pupil["pupil_timestamp"] <= t1)]
    blinks = rec["blinks"]
    blinks = blinks[(blinks["start_timestamp"] >= t0) & (blinks["end_timestamp"] <= t1)]
    return {"gaze": rec["gaze"], "pupil": pupil.reset_index(drop=True),
            "blinks": blinks.reset_index(drop=True)}


def invalid_fraction(gaze: pd.DataFrame) -> float:
    """P2/P3: share of gaze samples with confidence below CONFIDENCE_MIN."""
    return float((gaze["confidence"] < CONFIDENCE_MIN).mean())


def valid_time_s(timestamps) -> float:
    """P4: recording time in seconds, leaving out every gap longer than 1 s."""
    ts = np.sort(np.asarray(timestamps, dtype=float))
    if len(ts) < 2:
        return 0.0
    d = np.diff(ts)
    return float(d[d <= VALID_TIME_MAX_GAP_S].sum())


def clean_blinks(blinks: pd.DataFrame) -> pd.DataFrame:
    """P5: merge blinks less than 100 ms apart, then keep those lasting 50-500 ms.

    Merging first means a blink split in two by a single good frame counts once.
    """
    if len(blinks) == 0:
        return _EMPTY_BLINKS.copy()
    b = blinks[["start_timestamp", "end_timestamp"]].sort_values("start_timestamp").to_numpy(float)
    merged = [list(b[0])]
    for s, e in b[1:]:
        if s - merged[-1][1] < BLINK_MERGE_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = pd.DataFrame(merged, columns=["start", "end"])
    dur = out["end"] - out["start"]
    return out[(dur >= BLINK_MIN_S) & (dur <= BLINK_MAX_S)].reset_index(drop=True)


def in_intervals(t: np.ndarray, intervals: pd.DataFrame) -> np.ndarray:
    """True where t falls inside any [start, end] interval."""
    mask = np.zeros(len(t), bool)
    for s, e in zip(intervals["start"], intervals["end"]):
        mask |= (t >= s) & (t <= e)
    return mask


def interp_with_gap_limit(t, values, grid, max_gap) -> np.ndarray:
    """Linear interpolation onto `grid`, NaN wherever the bracketing samples are > max_gap apart."""
    t = np.asarray(t, float)
    values = np.asarray(values, float)
    one_d = values.ndim == 1
    v = values[:, None] if one_d else values
    out = np.full((len(grid), v.shape[1]), np.nan)
    if len(t) < 2:
        return out[:, 0] if one_d else out
    idx = np.searchsorted(t, grid)
    inside = (idx > 0) & (idx < len(t))
    left = np.where(inside, t[np.clip(idx - 1, 0, len(t) - 1)], np.nan)
    right = np.where(inside, t[np.clip(idx, 0, len(t) - 1)], np.nan)
    exact = np.isin(grid, t)
    ok = (inside & (right - left <= max_gap + 1e-9)) | exact
    for k in range(v.shape[1]):
        out[ok, k] = np.interp(grid[ok], t, v[:, k])
    return out[:, 0] if one_d else out


def gaze_directions(gaze: pd.DataFrame, blinks: pd.DataFrame):
    """P6/P8: average the two eyes' gaze directions onto a uniform grid.

    A sample is used only if confidence >= 0.8, it is outside a blink, and BOTH eyes have
    a direction. Each eye's direction alone is bent ~10 degrees inward (researcher-notes
    N25), so a one-eye sample would create a fake jump; it is treated as missing instead
    (decided 2026-10-02). Gaps up to 75 ms are interpolated; longer gaps stay NaN.
    """
    t = gaze["gaze_timestamp"].to_numpy(float)
    n0 = gaze[[f"gaze_normal0_{a}" for a in "xyz"]].to_numpy(float)
    n1 = gaze[[f"gaze_normal1_{a}" for a in "xyz"]].to_numpy(float)
    ok = ((gaze["confidence"].to_numpy() >= CONFIDENCE_MIN)
          & np.isfinite(n0).all(1) & np.isfinite(n1).all(1)
          & ~in_intervals(t, blinks))
    grid = np.arange(t.min(), t.max(), 1.0 / GAZE_GRID_HZ) if len(t) > 1 else np.array([])
    if ok.sum() < 2:
        return grid, np.full((len(grid), 3), np.nan)
    mean = (n0[ok] + n1[ok]) / 2.0
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    order = np.argsort(t[ok])
    xyz = interp_with_gap_limit(t[ok][order], mean[order], grid, GAZE_MAX_INTERP_S)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return grid, xyz


def binocular_fraction(gaze: pd.DataFrame) -> float:
    """Share of confident samples that have both eyes' directions (QC only)."""
    conf = gaze["confidence"].to_numpy() >= CONFIDENCE_MIN
    if conf.sum() == 0:
        return float("nan")
    n0 = gaze[[f"gaze_normal0_{a}" for a in "xyz"]].to_numpy(float)
    n1 = gaze[[f"gaze_normal1_{a}" for a in "xyz"]].to_numpy(float)
    both = np.isfinite(n0).all(1) & np.isfinite(n1).all(1)
    return float(both[conf].mean())


def _dilation_speed_filter(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Kret & Sjak-Shie 2019: drop samples whose dilation speed exceeds median + n*MAD."""
    if len(d) < 3:
        return np.ones(len(d), bool)
    speed = np.abs(np.diff(d)) / np.maximum(np.diff(t), 1e-6)
    s = np.maximum(np.r_[speed[0], speed], np.r_[speed, speed[-1]])
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return s <= med + PUPIL_MAD_MULTIPLIER * mad


def _lowpass_runs(d: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth low-pass on each contiguous non-NaN run long enough to filter."""
    sos = butter(PUPIL_LOWPASS_ORDER, PUPIL_LOWPASS_HZ, fs=PUPIL_GRID_HZ, output="sos")
    out = d.copy()
    finite = np.isfinite(d)
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start > 30:
            out[start:stop] = sosfiltfilt(sos, d[start:stop])
    return out


def clean_pupil(pupil: pd.DataFrame, blinks: pd.DataFrame, t_start: float, t_end: float):
    """P7: cleaned pupil diameter (mm) on a uniform grid; NaN where missing.

    3d rows only; confidence >= 0.8; outside blinks; 1.5-9 mm; dilation-speed filter per
    eye; each eye put on the grid with gaps <= 250 ms filled; the two eyes averaged where
    both exist (consistent with the both-eyes rule for gaze); 4 Hz low-pass.
    """
    grid = np.arange(t_start, t_end, 1.0 / PUPIL_GRID_HZ)
    rows = pupil[pupil["method"].astype(str).str.contains(PUPIL_METHOD_TAG)]
    per_eye = []
    for eye in (0.0, 1.0):
        e = rows[rows["eye_id"] == eye].sort_values("pupil_timestamp")
        t = e["pupil_timestamp"].to_numpy(float)
        d = e["diameter_3d"].to_numpy(float)
        keep = ((e["confidence"].to_numpy() >= CONFIDENCE_MIN)
                & (d >= PUPIL_MIN_MM) & (d <= PUPIL_MAX_MM)
                & ~in_intervals(t, blinks))
        t, d = t[keep], d[keep]
        if len(t):
            fine = _dilation_speed_filter(t, d)
            t, d = t[fine], d[fine]
        per_eye.append(interp_with_gap_limit(t, d, grid, PUPIL_MAX_INTERP_S))
    both = np.vstack(per_eye)
    mean = np.where(np.isfinite(both).all(0), both.mean(0), np.nan)
    return grid, _lowpass_runs(mean)


def count_refits(pupil: pd.DataFrame) -> int:
    """Number of 3D eye-model refits: max over eyes of (distinct model ids - 1)."""
    rows = pupil[pupil["method"].astype(str).str.contains(PUPIL_METHOD_TAG)]
    if len(rows) == 0:
        return 0
    per_eye = rows.groupby("eye_id")["model_id"].nunique() - 1
    return int(max(per_eye.max(), 0))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: all pass. If `test_clean_pupil_removes_zeros_and_keeps_level` is off by a few hundredths, check that blink periods (diameter 0) are removed before the speed filter, not after.

- [ ] **Step 5: Stage for review**

```bash
git add src/preprocess.py tests/test_preprocess.py
```

---

### Task 3: Event detection

**Files:**
- Create: `src/events.py`, `tests/test_events.py`

**Interfaces:**
- Consumes: `preprocess.gaze_directions` output `(grid_t, xyz)`.
- Produces:
  - `angular_speed_deg_s(xyz: np.ndarray, hz: float) -> np.ndarray` (NaN at edges and next to NaN)
  - `detect_events(xyz: np.ndarray, hz: float) -> dict` with keys `fixation_durations_s: np.ndarray`, `saccade_amplitudes_deg: np.ndarray`, `saccade_peak_velocities: np.ndarray`, `saccade_mean_velocities: np.ndarray`, `saccade_durations_s: np.ndarray`, `artifact_mask: np.ndarray[bool]`

- [ ] **Step 1: Write the failing tests `tests/test_events.py`**

```python
import numpy as np

from conftest import FIX_S, GAZE_HZ, SACC_DEG, gaze_path, unit_from_angles
from events import angular_speed_deg_s, detect_events

HZ = 240.0


def _grid_xyz(seconds=6.0, seed=0):
    t, az, el = gaze_path(seconds, np.random.default_rng(seed))
    grid = np.arange(t[0], t[-1], 1 / HZ)
    xyz = unit_from_angles(np.interp(grid, t, az), np.interp(grid, t, el))
    return xyz


def test_speed_of_constant_rotation():
    deg_per_s = 100.0
    az = np.arange(0, 1, 1 / HZ) * deg_per_s
    xyz = unit_from_angles(az, np.zeros_like(az))
    s = angular_speed_deg_s(xyz, HZ)
    assert np.isnan(s[:2]).all() and np.isnan(s[-2:]).all()
    assert np.allclose(s[2:-2], deg_per_s, rtol=0.01)


def test_speed_nan_next_to_gap():
    az = np.linspace(0, 5, 100)
    xyz = unit_from_angles(az, np.zeros_like(az))
    xyz[50] = np.nan
    s = angular_speed_deg_s(xyz, HZ)
    assert np.isnan(s[48:53]).all()


def test_detects_synthetic_fixations_and_saccades():
    ev = detect_events(_grid_xyz(), HZ)
    n_fix = len(ev["fixation_durations_s"])
    n_sac = len(ev["saccade_amplitudes_deg"])
    assert n_fix >= 12 and 0.8 <= n_sac / n_fix <= 1.25
    assert 0.24 <= np.median(ev["fixation_durations_s"]) <= FIX_S + 0.01
    assert 0.8 * SACC_DEG <= np.median(ev["saccade_amplitudes_deg"]) <= 1.1 * SACC_DEG


def test_short_fixations_discarded():
    # 30 ms still, then fast motion: no fixation of >= 55 ms exists
    az = np.r_[np.zeros(7), np.linspace(0, 20, 40)]
    ev = detect_events(unit_from_angles(az, np.zeros_like(az)), HZ)
    assert len(ev["fixation_durations_s"]) == 0


def test_artifact_speed_rejected():
    az = np.r_[np.zeros(60), np.full(60, 90.0)]          # 90-degree jump in one sample
    ev = detect_events(unit_from_angles(az, np.zeros_like(az)), HZ)
    assert ev["artifact_mask"].any()
    assert len(ev["saccade_amplitudes_deg"]) == 0


def test_all_nan_input():
    ev = detect_events(np.full((50, 3), np.nan), HZ)
    assert len(ev["fixation_durations_s"]) == 0 and len(ev["saccade_amplitudes_deg"]) == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'events'`.

- [ ] **Step 3: Write `src/events.py`**

```python
"""Fixation and saccade detection (methodology-colet.md §5).

Velocity: a 5-tap differentiator, h = [1, 1, 0, -1, -1] / (6 dt), applied to each
component of the unit gaze vector; for unit vectors the norm of the derivative is the
angular speed. (Velocity filtering follows Duchowski 2017; confirm the exact taps against
the book before the defense, reference check C7.)
Classification: I-VT at 45 deg/s (Salvucci & Goldberg 2000), fixations >= 55 ms (COLET),
samples above 1000 deg/s rejected as artifacts (Hausamann 2020).
"""
from __future__ import annotations

import numpy as np

from config import IVT_THRESHOLD_DEG_S, MAX_VELOCITY_DEG_S, MIN_FIXATION_S

_FIX, _SAC, _BAD = 1, 0, 2


def angular_speed_deg_s(xyz: np.ndarray, hz: float) -> np.ndarray:
    """Angular speed per sample; NaN in the first/last two samples and next to any NaN."""
    n = len(xyz)
    speed = np.full(n, np.nan)
    if n < 5:
        return speed
    d = (xyz[4:] + xyz[3:-1] - xyz[1:-3] - xyz[:-4]) * hz / 6.0
    speed[2:-2] = np.degrees(np.linalg.norm(d, axis=1))
    return speed


def _runs(labels: np.ndarray):
    """(label, start, stop) for each run of equal, non-NaN labels."""
    out, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else np.nan
        if start is not None and (np.isnan(cur) or cur != labels[start]):
            out.append((int(labels[start]), start, i))
            start = None
        if start is None and i < len(labels) and not np.isnan(cur):
            start = i
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def detect_events(xyz: np.ndarray, hz: float) -> dict:
    speed = angular_speed_deg_s(xyz, hz)
    labels = np.full(len(speed), np.nan)
    finite = np.isfinite(speed)
    labels[finite & (speed < IVT_THRESHOLD_DEG_S)] = _FIX
    labels[finite & (speed >= IVT_THRESHOLD_DEG_S)] = _SAC
    artifact = finite & (speed > MAX_VELOCITY_DEG_S)
    labels[artifact] = _BAD

    fix, amp, peak, mean_v, sac_dur = [], [], [], [], []
    for lab, start, stop in _runs(labels):
        if lab == _FIX:
            dur = (stop - start) / hz
            if dur >= MIN_FIXATION_S:
                fix.append(dur)
        elif lab == _SAC:
            before = labels[start - 1] if start > 0 else np.nan
            after = labels[stop] if stop < len(labels) else np.nan
            if before == _BAD or after == _BAD:
                continue
            a, b = xyz[max(start - 1, 0)], xyz[min(stop, len(xyz) - 1)]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            amp.append(_angle_deg(a, b))
            peak.append(float(np.nanmax(speed[start:stop])))
            mean_v.append(float(np.nanmean(speed[start:stop])))
            sac_dur.append((stop - start) / hz)
    return {
        "fixation_durations_s": np.array(fix),
        "saccade_amplitudes_deg": np.array(amp),
        "saccade_peak_velocities": np.array(peak),
        "saccade_mean_velocities": np.array(mean_v),
        "saccade_durations_s": np.array(sac_dur),
        "artifact_mask": artifact,
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_events.py -v`
Expected: all pass.

- [ ] **Step 5: Stage for review**

```bash
git add src/events.py tests/test_events.py
```

---

### Task 4: Features for one recording

**Files:**
- Create: `src/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `preprocess.*`, `events.detect_events`, `config.FEATURES`, `config.GAZE_GRID_HZ`.
- Produces: `recording_features(rec: dict) -> dict` with every name in `config.FEATURES` plus QC keys `valid_time_s, binocular_fraction, refits, n_blinks, n_fixations, n_saccades, fixation_median_ms, saccade_mean_velocity, saccade_duration_ms, gaze_valid_fraction, pupil_valid_fraction`.

- [ ] **Step 1: Write the failing tests `tests/test_features.py`**

```python
import numpy as np
import pandas as pd

from config import FEATURES
from conftest import SACC_DEG, make_recording
from features import recording_features


def test_all_features_present_and_plausible():
    rec = make_recording(1, 4, np.random.default_rng(0), seconds=8.0)
    f = recording_features(rec)
    assert set(FEATURES) <= set(f)
    assert 3.4 < f["pupil_mean"] < 3.8
    assert f["blink_rate"] > 0
    assert 2.0 < f["fixation_rate"] < 3.5            # one fixation every ~0.34 s
    assert 240 < f["fixation_duration"] < 320         # ms
    assert 0.8 * SACC_DEG < f["saccade_amplitude"] < 1.1 * SACC_DEG
    assert f["saccade_peak_velocity"] > 45
    assert f["gaze_sd_x"] > 1 and f["gaze_sd_y"] > 1


def test_no_blinks_gives_zero_rate_not_nan():
    rec = make_recording(2, 1, np.random.default_rng(3), seconds=4.0)   # even participant, A1: no blinks
    assert len(rec["blinks"]) == 0
    assert recording_features(rec)["blink_rate"] == 0.0


def test_too_short_for_events_gives_nan_not_crash():
    rec = make_recording(1, 2, np.random.default_rng(0), seconds=4.0)
    g = rec["gaze"]
    # one valid sample every 100 ms: every gap > 75 ms stays missing, so no velocity exists
    rec["gaze"] = g.assign(confidence=np.where(np.arange(len(g)) % 25 == 0, 0.99, 0.1))
    f = recording_features(rec)
    assert np.isnan(f["fixation_duration"]) and np.isnan(f["saccade_amplitude"])
    assert f["fixation_rate"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features'`.

- [ ] **Step 3: Write `src/features.py`**

```python
"""The ten whole-activity features plus QC columns for one recording (§6, P10).

Rates are per unit of valid time (P4): blinks per minute, fixations and saccades per
second. Durations are in ms, angles in degrees, velocities in deg/s, pupil in mm.
A recording with no events gets rate 0 and NaN for per-event averages.
"""
from __future__ import annotations

import numpy as np

from config import GAZE_GRID_HZ
from events import detect_events
from preprocess import (
    binocular_fraction, clean_blinks, clean_pupil, count_refits, gaze_directions,
    trim_to_gaze_range, valid_time_s,
)


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def recording_features(rec: dict) -> dict:
    rec = trim_to_gaze_range(rec)
    gaze = rec["gaze"]
    t = gaze["gaze_timestamp"].to_numpy(float)
    vt = valid_time_s(t)
    blinks = clean_blinks(rec["blinks"])

    grid, xyz = gaze_directions(gaze, blinks)
    ev = detect_events(xyz, GAZE_GRID_HZ)
    keep = np.isfinite(xyz).all(1) & ~ev["artifact_mask"]
    az = np.degrees(np.arctan2(xyz[keep, 0], xyz[keep, 2]))
    el = np.degrees(np.arctan2(xyz[keep, 1], np.hypot(xyz[keep, 0], xyz[keep, 2])))

    _, pupil = clean_pupil(rec["pupil"], blinks, t.min(), t.max())

    fix = ev["fixation_durations_s"]
    sac = ev["saccade_amplitudes_deg"]
    per_s = 1.0 / vt if vt > 0 else float("nan")
    return {
        "pupil_mean": float(np.nanmean(pupil)) if np.isfinite(pupil).any() else float("nan"),
        "pupil_sd": float(np.nanstd(pupil, ddof=1)) if np.isfinite(pupil).sum() > 1 else float("nan"),
        "blink_rate": len(blinks) * 60.0 * per_s,
        "fixation_rate": len(fix) * per_s,
        "fixation_duration": _mean(fix) * 1000.0,
        "saccade_rate": len(sac) * per_s,
        "saccade_amplitude": _mean(sac),
        "saccade_peak_velocity": _mean(ev["saccade_peak_velocities"]),
        "gaze_sd_x": float(np.std(az, ddof=1)) if len(az) > 1 else float("nan"),
        "gaze_sd_y": float(np.std(el, ddof=1)) if len(el) > 1 else float("nan"),
        # ---- QC only, never model inputs
        "valid_time_s": vt,
        "binocular_fraction": binocular_fraction(gaze),
        "refits": count_refits(rec["pupil"]),
        "n_blinks": int(len(blinks)),
        "n_fixations": int(len(fix)),
        "n_saccades": int(len(sac)),
        "fixation_median_ms": float(np.median(fix) * 1000.0) if len(fix) else float("nan"),
        "saccade_mean_velocity": _mean(ev["saccade_mean_velocities"]),
        "saccade_duration_ms": _mean(ev["saccade_durations_s"]) * 1000.0,
        "gaze_valid_fraction": float(np.isfinite(xyz).all(1).mean()) if len(xyz) else float("nan"),
        "pupil_valid_fraction": float(np.isfinite(pupil).mean()) if len(pupil) else float("nan"),
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_features.py -v`
Expected: all pass.

- [ ] **Step 5: Stage for review**

```bash
git add src/features.py tests/test_features.py
```

---

### Task 5: Feature table, exclusion, labels, normalization

**Files:**
- Create: `src/dataset.py`, `tests/test_dataset.py`
- Rewrite: `src/labels.py`
- Modify: `src/normalize.py` (remove `apply_feature_scheme`, update docstring)

**Interfaces:**
- Consumes: `data.*`, `features.recording_features`, `preprocess.invalid_fraction`.
- Produces:
  - `dataset.build_feature_table(data_dir=DATA_DIR, cache_dir=CACHE_DIR, refresh=False) -> pd.DataFrame` (one row per recording: `participant, activity, invalid_fraction, excluded` + features + QC)
  - `dataset.analysis_participants(table) -> list[int]`
  - `dataset.analysis_rows(table) -> pd.DataFrame` (retained activities of analysis participants)
  - `dataset.retention_summary(table) -> pd.DataFrame` (per activity)
  - `labels.main_rows(rows) -> pd.DataFrame` (A1 and A4 only), `labels.binary_label(rows) -> np.ndarray`
  - `normalize.per_participant_zscore(X, participants) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests `tests/test_dataset.py`**

```python
import numpy as np
import pandas as pd

from conftest import write_synthetic_dataset
from dataset import analysis_participants, analysis_rows, build_feature_table, retention_summary
from labels import binary_label, main_rows
from normalize import per_participant_zscore


def test_exclusion_and_analysis_set(tmp_path):
    root = write_synthetic_dataset(tmp_path / "d", n_participants=4, seconds=3.0,
                                   low_quality={(2, 4), (3, 2)})
    table = build_feature_table(root, cache_dir=tmp_path / "c")
    assert len(table) == 16
    assert set(table.loc[table.excluded, ["participant", "activity"]].itertuples(index=False)) == {(2, 4), (3, 2)}
    assert analysis_participants(table) == [1, 3, 4]           # P2 lost A4 -> dropped; P3 lost A2 -> kept
    rows = analysis_rows(table)
    assert len(rows[rows.participant == 3]) == 3               # A2 excluded, A1/A3/A4 kept
    ret = retention_summary(table)
    assert set(ret.columns) >= {"activity", "recordings", "excluded", "invalid_fraction_median"}


def test_cache_reused(tmp_path):
    root = write_synthetic_dataset(tmp_path / "d", n_participants=1, seconds=2.0)
    a = build_feature_table(root, cache_dir=tmp_path / "c")
    b = build_feature_table(root, cache_dir=tmp_path / "c")
    pd.testing.assert_frame_equal(a, b)


def test_labels_and_main_rows():
    rows = pd.DataFrame({"participant": [1, 1, 1, 1], "activity": [1, 2, 3, 4]})
    m = main_rows(rows)
    assert m.activity.tolist() == [1, 4]
    assert binary_label(m).tolist() == [0, 1]


def test_zscore_constant_and_partial_participants():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0, 4.0, 6.0], "b": [1, 2, 3, 4, 5, 6.0]})
    groups = pd.Series([1, 1, 1, 2, 2, 2])
    z = per_participant_zscore(X, groups)
    assert (z.loc[:2, "a"] == 0.0).all()                     # constant within P1 -> 0, not NaN/inf
    assert np.allclose(z.loc[3:, "a"], [-1, 0, 1])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataset'`.

- [ ] **Step 3: Write `src/dataset.py`**

```python
"""One feature row per recording, with the recording-exclusion rule applied (P3, P10, P14).

Features are computed once and cached, because the full set takes minutes to build.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CACHE_DIR, DATA_DIR, HIGH_ACTIVITY, LOW_ACTIVITY, MAX_INVALID_FRACTION
from data import load_recording, recording_keys
from features import recording_features
from preprocess import invalid_fraction, trim_to_gaze_range


def build_feature_table(data_dir=DATA_DIR, cache_dir=CACHE_DIR, refresh=False) -> pd.DataFrame:
    cache = Path(cache_dir) / "feature_table.parquet"
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
```

- [ ] **Step 4: Rewrite `src/labels.py`**

```python
"""Condition labels (§3, decision C2): A1 = low load (0), A4 = high load (1).

A2 and A3 never enter a model; they are used only for the per-person normalization
(all four activities) and the NASA-RTLX manipulation check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HIGH_ACTIVITY, LOW_ACTIVITY


def main_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["activity"].isin([LOW_ACTIVITY, HIGH_ACTIVITY])].reset_index(drop=True)


def binary_label(rows: pd.DataFrame) -> np.ndarray:
    return (rows["activity"].to_numpy() == HIGH_ACTIVITY).astype(int)
```

- [ ] **Step 5: Modify `src/normalize.py`** — replace the module docstring and delete `apply_feature_scheme`

Replace the whole file with:

```python
"""Per-person standardization (§7, decision D-M4).

Each feature is re-expressed as a deviation from that participant's own mean over their
retained activities (normally all four), in units of their own SD. No labels are used.

The cost, stated as a limitation: the held-out participant's own unlabelled recordings set
their "normal", and the method relies on COLET's balanced design (everyone did easier and
harder activities), so a deployed system would need a calibration session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def per_participant_zscore(X: pd.DataFrame, participants: pd.Series) -> pd.DataFrame:
    """Standardise every column within each participant.

    Constant columns for a participant (zero SD) become 0.0, i.e. "exactly this person's
    typical value".
    """
    participants = pd.Series(np.asarray(participants), index=X.index)
    grouped = X.groupby(participants, sort=False)
    centred = X - grouped.transform("mean")
    spread = grouped.transform("std").fillna(0.0)
    out = centred / (spread + _EPS)
    out = out.where(spread > _EPS, 0.0)
    return out.replace([np.inf, -np.inf], np.nan).where(X.notna())
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: all pass.

- [ ] **Step 7: Stage for review**

```bash
git add src/dataset.py src/labels.py src/normalize.py tests/test_dataset.py
```

---

### Task 6: Checks (sanity per activity, manipulation check, V2-13 redundancy)

**Files:**
- Create: `src/checks.py`, `tests/test_checks.py`

**Interfaces:**
- Consumes: analysis rows from Task 5; `data.load_annotation`.
- Produces:
  - `sanity_check(rows) -> pd.DataFrame` (per activity: `recordings, fixation_median_ms, saccade_fixation_ratio, fixation_ok, ratio_ok, passed`)
  - `sanity_passed(sanity: pd.DataFrame) -> bool`
  - `manipulation_check(annotation, participants) -> dict` (`friedman_chi2, friedman_p, n, share_a4_above_a1, median_a4_minus_a1`)
  - `redundancy_check(rows) -> pd.DataFrame` (`pair, spearman_rho, n`)

- [ ] **Step 1: Write the failing tests `tests/test_checks.py`**

```python
import numpy as np
import pandas as pd

from checks import manipulation_check, redundancy_check, sanity_check, sanity_passed


def _rows(fix_ms, n_fix, n_sac):
    return pd.DataFrame({
        "participant": [1, 1, 2, 2], "activity": [1, 4, 1, 4],
        "fixation_median_ms": fix_ms, "n_fixations": n_fix, "n_saccades": n_sac,
    })


def test_sanity_pass():
    s = sanity_check(_rows([250, 260, 240, 270], [100, 100, 100, 100], [100, 95, 105, 100]))
    assert sanity_passed(s) and s.passed.all()


def test_sanity_fails_if_one_activity_fails():
    # A4 has far more saccades than fixations (noise) -> fails, even though pooled ratio is ~1.1
    s = sanity_check(_rows([250, 260, 240, 270], [100, 60, 100, 60], [80, 100, 80, 100]))
    assert not sanity_passed(s)
    assert s.set_index("activity").loc[4, "ratio_ok"] == False  # noqa: E712


def test_manipulation_check():
    ann = pd.DataFrame({"participant": np.repeat([1, 2, 3], 4), "activity": [1, 2, 3, 4] * 3,
                        "mean": [10, 20, 30, 40, 15, 25, 35, 45, 12, 18, 33, 50]})
    out = manipulation_check(ann, [1, 2, 3])
    assert out["share_a4_above_a1"] == 1.0 and out["n"] == 3


def test_redundancy_check():
    rows = pd.DataFrame({"saccade_peak_velocity": [1, 2, 3, 4.0], "saccade_mean_velocity": [2, 4, 6, 8.0],
                         "saccade_amplitude": [1, 2, 3, 4.0], "saccade_duration_ms": [4, 3, 2, 1.0]})
    out = redundancy_check(rows).set_index("pair")
    assert out.loc["saccade_peak_velocity~saccade_mean_velocity", "spearman_rho"] == 1.0
    assert out.loc["saccade_amplitude~saccade_duration_ms", "spearman_rho"] == -1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_checks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'checks'`.

- [ ] **Step 3: Write `src/checks.py`**

```python
"""Label-blind checks run before and alongside the models.

sanity_check      -- §5 step 7 / R4-2: event detection must look normal in EVERY activity.
manipulation_check-- §3: did NASA-RTLX rise from A1 to A4 (Friedman over A1-A4)?
redundancy_check  -- V2-13: recheck the correlations behind dropping mean saccade velocity
                     (vs peak, rho 0.95) and saccade duration (vs amplitude, rho 0.83).
None of these looks at how well a feature separates A1 from A4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

from config import (
    ACTIVITIES, HIGH_ACTIVITY, LOW_ACTIVITY, SANITY_FIXATION_MEDIAN_MS,
    SANITY_SACCADE_FIXATION_RATIO,
)

REDUNDANT_PAIRS = [("saccade_peak_velocity", "saccade_mean_velocity"),
                   ("saccade_amplitude", "saccade_duration_ms")]


def sanity_check(rows: pd.DataFrame) -> pd.DataFrame:
    """Per activity: median of recording-level median fixation durations, and the ratio of
    total saccades to total fixations."""
    lo_ms, hi_ms = SANITY_FIXATION_MEDIAN_MS
    lo_r, hi_r = SANITY_SACCADE_FIXATION_RATIO
    out = []
    for act, g in rows.groupby("activity"):
        fix_ms = float(g["fixation_median_ms"].median())
        n_fix = float(g["n_fixations"].sum())
        ratio = float(g["n_saccades"].sum()) / n_fix if n_fix else float("nan")
        out.append({"activity": int(act), "recordings": len(g),
                    "fixation_median_ms": fix_ms, "saccade_fixation_ratio": ratio,
                    "fixation_ok": lo_ms <= fix_ms <= hi_ms, "ratio_ok": lo_r <= ratio <= hi_r})
    df = pd.DataFrame(out)
    df["passed"] = df["fixation_ok"] & df["ratio_ok"]
    return df


def sanity_passed(sanity: pd.DataFrame) -> bool:
    return bool(len(sanity)) and bool(sanity["passed"].all())


def manipulation_check(annotation: pd.DataFrame, participants) -> dict:
    a = annotation[annotation["participant"].isin(participants)]
    wide = a.pivot(index="participant", columns="activity", values="mean").dropna()
    stat, p = friedmanchisquare(*[wide[act] for act in ACTIVITIES])
    diff = wide[HIGH_ACTIVITY] - wide[LOW_ACTIVITY]
    return {"friedman_chi2": float(stat), "friedman_p": float(p), "n": int(len(wide)),
            "share_a4_above_a1": float((diff > 0).mean()),
            "median_a4_minus_a1": float(diff.median())}


def redundancy_check(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a, b in REDUNDANT_PAIRS:
        pair = rows[[a, b]].dropna()
        rho = spearmanr(pair[a], pair[b]).statistic if len(pair) > 2 else np.nan
        out.append({"pair": f"{a}~{b}", "spearman_rho": float(rho), "n": int(len(pair))})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_checks.py -v`
Expected: all pass.

- [ ] **Step 5: Stage for review**

```bash
git add src/checks.py tests/test_checks.py
```

---

### Task 7: Models and LOPO evaluation

**Files:**
- Rewrite: `src/models.py`, `src/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `config.N_ITER, INNER_FOLDS, RANDOM_STATE, N_BOOTSTRAP, N_PERMUTATIONS`.
- Produces:
  - `models.MODELS: dict[str, {"factory": Callable, "space": dict}]` with keys `"Logistic Regression"`, `"XGBoost"`
  - `evaluate.run_lopo(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, factory, space, n_iter=N_ITER, tune=True, params=None) -> dict` with `oof_proba, fold_models (list of {"model","train_idx","test_idx","participant"}), chosen_params`
  - `evaluate.score(y, proba) -> dict` (`roc_auc, balanced_accuracy, macro_f1, recall_low, recall_high, brier`)
  - `evaluate.participant_margins(y, proba, groups) -> pd.Series` (p(A4) − p(A1) per participant)
  - `evaluate.bootstrap_auc(y, proba, groups, n_boot=N_BOOTSTRAP, other=None) -> dict` (`auc_ci_low, auc_ci_high`, and `diff, diff_ci_low, diff_ci_high` when `other` is given; diff = other − proba)
  - `evaluate.compare_models(y, proba_lr, proba_xgb, groups, n_boot) -> dict`
  - `evaluate.majority_baseline(y, groups) -> dict`
  - `evaluate.permutation_test(X, y, groups, factory, params, n_perm=N_PERMUTATIONS) -> dict` (`observed_auc, p_value, n_perm`)
  - `evaluate.modal_params(chosen_params: list[dict]) -> dict`

- [ ] **Step 1: Write the failing tests `tests/test_evaluate.py`**

```python
import numpy as np
import pandas as pd

from evaluate import (
    bootstrap_auc, compare_models, majority_baseline, modal_params, participant_margins,
    permutation_test, run_lopo, score,
)
from models import MODELS


def _toy(n_participants=12, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(1, n_participants + 1), 2)
    y = np.tile([0, 1], n_participants)
    X = pd.DataFrame({"good": y * signal + rng.normal(0, 1, len(y)),
                      "noise": rng.normal(0, 1, len(y))})
    return X, y, groups


def test_lopo_both_models_learn_signal():
    X, y, g = _toy()
    for name, spec in MODELS.items():
        res = run_lopo(X, y, g, spec["factory"], spec["space"], n_iter=3)
        assert not np.isnan(res["oof_proba"]).any()
        assert len(res["fold_models"]) == 12
        assert score(y, res["oof_proba"])["roc_auc"] > 0.8, name


def test_score_keys_and_brier_range():
    s = score(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6]))
    assert set(s) == {"roc_auc", "balanced_accuracy", "macro_f1", "recall_low", "recall_high", "brier"}
    assert s["roc_auc"] == 1.0 and 0 <= s["brier"] <= 1


def test_margins_and_bootstrap():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.9, 0.6, 0.4])
    g = np.array([1, 1, 2, 2])
    m = participant_margins(y, p, g)
    assert np.allclose(m.loc[1], 0.7) and np.allclose(m.loc[2], -0.2)
    b = bootstrap_auc(y, p, g, n_boot=200, other=p)
    assert b["auc_ci_low"] <= b["auc_ci_high"] and b["diff"] == 0.0


def test_compare_and_baseline():
    X, y, g = _toy()
    r1 = run_lopo(X, y, g, MODELS["Logistic Regression"]["factory"], {}, tune=False)
    out = compare_models(y, r1["oof_proba"], r1["oof_proba"], g, n_boot=100)
    assert out["auc_difference"] == 0.0
    assert 0.0 <= majority_baseline(y, g)["balanced_accuracy"] <= 0.5


def test_permutation_test_detects_signal():
    X, y, g = _toy(signal=3.0)
    out = permutation_test(X, y, g, MODELS["Logistic Regression"]["factory"], {}, n_perm=19)
    assert out["p_value"] <= 0.1


def test_modal_params():
    assert modal_params([{"a": 1}, {"a": 2}, {"a": 1}]) == {"a": 1}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ImportError` (old `evaluate.py` has no `score` / `run_lopo` signature mismatch).

- [ ] **Step 3: Rewrite `src/models.py`**

```python
"""The two classifiers under comparison (§8; decisions D-N6, V2-9).

Equal tuning budget: both are tuned by random search with the same number of candidates
(config.N_ITER) over the spaces below, inside participant-grouped inner folds.
Missing values: median imputation for LR (fitted in-fold); XGBoost handles NaN natively.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import RANDOM_STATE


def make_logistic_regression() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                   max_iter=20000, random_state=RANDOM_STATE)),
    ])


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         missing=np.nan, max_depth=2, n_estimators=100, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=1, verbosity=0)


LR_SPACE = {"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": uniform(0.0, 1.0)}

XGB_SPACE = {
    "max_depth": [1, 2, 3],
    "n_estimators": randint(50, 301),
    "learning_rate": uniform(0.05, 0.05),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_lambda": loguniform(1e-2, 1e1),
    "reg_alpha": loguniform(1e-3, 1e0),
}

MODELS = {
    "Logistic Regression": {"factory": make_logistic_regression, "space": LR_SPACE},
    "XGBoost": {"factory": make_xgboost, "space": XGB_SPACE},
}
```

- [ ] **Step 4: Rewrite `src/evaluate.py`**

```python
"""Leave-one-participant-out evaluation, metrics and model comparison (§§9-10).

Outer loop: one participant (both recordings) held out per fold.
Inner loop: StratifiedGroupKFold over the training participants, random search with the
same number of candidates for both models. Scores are computed on pooled out-of-fold
probabilities; uncertainty comes from a bootstrap that resamples participants.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedGroupKFold

from config import INNER_FOLDS, N_BOOTSTRAP, N_ITER, N_PERMUTATIONS, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)


def run_lopo(X, y, groups, factory, space, n_iter=N_ITER, tune=True, params=None) -> dict:
    X = pd.DataFrame(X).reset_index(drop=True)
    y, groups = np.asarray(y), np.asarray(groups)
    oof = np.full(len(y), np.nan)
    folds, chosen = [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        est = factory()
        if params:
            est.set_params(**params)
        if tune and space:
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="roc_auc", cv=inner,
                                        random_state=RANDOM_STATE, n_jobs=1, error_score=np.nan)
            search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
            model, best = search.best_estimator_, search.best_params_
        else:
            model, best = clone(est).fit(X.iloc[train_idx], y[train_idx]), dict(params or {})
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        folds.append({"model": model, "train_idx": train_idx, "test_idx": test_idx,
                      "participant": groups[test_idx][0]})
        chosen.append(best)
    return {"oof_proba": oof, "fold_models": folds, "chosen_params": chosen}


def score(y, proba) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall_low": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "recall_high": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
    }


def participant_margins(y, proba, groups) -> pd.Series:
    """p(high) - p(low) for each participant; > 0 means A4 was ranked above A1."""
    df = pd.DataFrame({"g": groups, "y": y, "p": proba})
    hi = df[df.y == 1].groupby("g")["p"].mean()
    lo = df[df.y == 0].groupby("g")["p"].mean()
    return (hi - lo).dropna()


def bootstrap_auc(y, proba, groups, n_boot=N_BOOTSTRAP, other=None) -> dict:
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    other = None if other is None else np.asarray(other)
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    aucs, diffs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], proba[idx])
        aucs.append(a)
        if other is not None:
            diffs.append(roc_auc_score(y[idx], other[idx]) - a)
    out = {"auc_ci_low": float(np.percentile(aucs, 2.5)), "auc_ci_high": float(np.percentile(aucs, 97.5))}
    if other is not None:
        out.update({"diff": float(roc_auc_score(y, other) - roc_auc_score(y, proba)),
                    "diff_ci_low": float(np.percentile(diffs, 2.5)),
                    "diff_ci_high": float(np.percentile(diffs, 97.5))})
    return out


def compare_models(y, proba_lr, proba_xgb, groups, n_boot=N_BOOTSTRAP) -> dict:
    """XGBoost minus LR: pooled AUC difference with a participant-bootstrap CI, plus a paired
    Wilcoxon on per-participant margins (supporting test; an adaptation, see M9/M10)."""
    b = bootstrap_auc(y, proba_lr, groups, n_boot, other=proba_xgb)
    m_lr = participant_margins(y, proba_lr, groups)
    m_xgb = participant_margins(y, proba_xgb, groups)
    d = (m_xgb - m_lr).dropna()
    if len(d) >= 3 and not np.allclose(d, 0):
        stat, p = wilcoxon(m_xgb.loc[d.index], m_lr.loc[d.index])
    else:
        stat, p = np.nan, np.nan
    return {"auc_difference": b["diff"], "auc_difference_ci_low": b["diff_ci_low"],
            "auc_difference_ci_high": b["diff_ci_high"], "wilcoxon_statistic": float(stat),
            "wilcoxon_p": float(p), "n_participants": int(len(d))}


def majority_baseline(y, groups) -> dict:
    """Always predict the training-majority class under LOPO (balanced design -> 0.5)."""
    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.zeros(len(y), int)
    for train_idx, test_idx in LeaveOneGroupOut().split(y.reshape(-1, 1), y, groups):
        pred[test_idx] = int(y[train_idx].mean() > 0.5)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def modal_params(chosen_params: list[dict]) -> dict:
    """The most frequently chosen hyperparameter set across folds."""
    keys = [tuple(sorted(p.items())) for p in chosen_params]
    return dict(Counter(keys).most_common(1)[0][0]) if keys else {}


def permutation_test(X, y, groups, factory, params, n_perm=N_PERMUTATIONS) -> dict:
    """Grouped permutation test against chance (§10).

    Labels are permuted within participants (each participant's A1/A4 swapped at random),
    keeping the paired structure; hyperparameters are fixed to `params` for speed.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    observed = roc_auc_score(y, run_lopo(X, y, groups, factory, None, tune=False, params=params)["oof_proba"])
    rng = np.random.default_rng(RANDOM_STATE)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            if rng.random() < 0.5:
                idx = np.flatnonzero(groups == g)
                yp[idx] = yp[idx][::-1]
        null.append(roc_auc_score(yp, run_lopo(X, yp, groups, factory, None, tune=False, params=params)["oof_proba"]))
    p = (1 + sum(n >= observed for n in null)) / (1 + n_perm)
    return {"observed_auc": float(observed), "p_value": float(p), "n_perm": int(n_perm)}
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: all pass (the XGBoost search may take ~1 min).

- [ ] **Step 6: Stage for review**

```bash
git add src/models.py src/evaluate.py tests/test_evaluate.py
```

---

### Task 8: Feature importance

**Files:**
- Rewrite: `src/importance.py`
- Create: `tests/test_importance.py`

**Interfaces:**
- Consumes: `run_lopo` output (`fold_models` with `train_idx`, `test_idx`).
- Produces:
  - `logistic_coefficients(fold_models, names) -> pd.DataFrame`
  - `oof_shap(fold_models, X, kind: Literal["lr","xgb"]) -> np.ndarray` (rows aligned to X)
  - `shap_summary(matrix, names) -> pd.DataFrame` (`feature, mean_abs_shap, mean_shap, rank`)
  - `permutation_importance_oof(fold_models, X, y, n_repeats=N_PERM_IMPORTANCE) -> pd.DataFrame` (`feature, auc_drop_mean, auc_drop_sd`)
  - `ranking_agreement(shap_a, shap_b, groups, names, n_boot=N_BOOTSTRAP) -> dict` (`kendall_tau, ci_low, ci_high`)
  - `xgboost_gain(fold_models, names) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests `tests/test_importance.py`**

```python
import numpy as np
import pandas as pd

from evaluate import run_lopo
from importance import (
    logistic_coefficients, oof_shap, permutation_importance_oof, ranking_agreement,
    shap_summary, xgboost_gain,
)
from models import MODELS


def _fit(kind):
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(1, 11), 2)
    y = np.tile([0, 1], 10)
    X = pd.DataFrame({"good": y * 3 + rng.normal(0, 1, 20), "noise": rng.normal(0, 1, 20)})
    name = "Logistic Regression" if kind == "lr" else "XGBoost"
    res = run_lopo(X, y, g, MODELS[name]["factory"], None, tune=False)
    return X, y, g, res


def test_shap_both_models_rank_good_first():
    for kind in ("lr", "xgb"):
        X, y, g, res = _fit(kind)
        m = oof_shap(res["fold_models"], X, kind)
        assert m.shape == X.shape and not np.isnan(m).any()
        assert shap_summary(m, list(X.columns)).iloc[0]["feature"] == "good"


def test_coefficients_and_gain():
    X, y, g, res = _fit("lr")
    coef = logistic_coefficients(res["fold_models"], list(X.columns))
    assert coef.iloc[0]["feature"] == "good" and coef.iloc[0]["coefficient_mean"] > 0
    X, y, g, res = _fit("xgb")
    assert set(xgboost_gain(res["fold_models"], list(X.columns))["feature"]) == {"good", "noise"}


def test_permutation_importance_and_agreement():
    X, y, g, res = _fit("lr")
    pi = permutation_importance_oof(res["fold_models"], X, y, n_repeats=5)
    assert pi.iloc[0]["feature"] == "good" and pi.iloc[0]["auc_drop_mean"] > 0
    m = oof_shap(res["fold_models"], X, "lr")
    agree = ranking_agreement(m, m, g, list(X.columns), n_boot=50)
    assert agree["kendall_tau"] == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_importance.py -v`
Expected: FAIL with `ImportError: cannot import name 'oof_shap'`.

- [ ] **Step 3: Rewrite `src/importance.py`**

```python
"""Feature importance for both models, read from the LOPO fold models (§10; D-M8).

SHAP for BOTH models so the rankings share a scale (log-odds): TreeSHAP for XGBoost,
linear SHAP for Logistic Regression (on the in-fold imputed + scaled inputs). Each fold's
model explains only the participant it never saw. Applying SHAP to both models is our own
choice, justified by SHAP being model-agnostic (Lundberg & Lee 2017).
Permutation importance is computed on the pooled out-of-fold predictions. Gain importance
is reported in the appendix only. Importance = model reliance, not cause (Molnar 2022).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from config import N_BOOTSTRAP, N_PERM_IMPORTANCE, RANDOM_STATE


def logistic_coefficients(fold_models, names: list[str]) -> pd.DataFrame:
    coefs = np.array([f["model"].named_steps["clf"].coef_.ravel() for f in fold_models])
    out = pd.DataFrame({
        "feature": names,
        "coefficient_mean": coefs.mean(0),
        "coefficient_sd": coefs.std(0),
        "abs_coefficient_mean": np.abs(coefs).mean(0),
        "sign_consistency": np.maximum((coefs > 0).mean(0), (coefs < 0).mean(0)),
        "zeroed_share": (coefs == 0).mean(0),
    })
    return out.sort_values("abs_coefficient_mean", ascending=False).reset_index(drop=True)


def oof_shap(fold_models, X: pd.DataFrame, kind: str) -> np.ndarray:
    import shap

    X = pd.DataFrame(X).reset_index(drop=True)
    out = np.full(X.shape, np.nan)
    for f in fold_models:
        model, tr, te = f["model"], f["train_idx"], f["test_idx"]
        if kind == "xgb":
            values = shap.TreeExplainer(model).shap_values(X.iloc[te])
        else:
            pre = model[:-1]
            background = pre.transform(X.iloc[tr])
            explainer = shap.LinearExplainer(model.named_steps["clf"], background)
            values = explainer.shap_values(pre.transform(X.iloc[te]))
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        out[te] = values
    return out


def shap_summary(matrix: np.ndarray, names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature": names, "mean_abs_shap": np.nanmean(np.abs(matrix), 0),
                       "mean_shap": np.nanmean(matrix, 0)})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def permutation_importance_oof(fold_models, X, y, n_repeats=N_PERM_IMPORTANCE) -> pd.DataFrame:
    """Drop in pooled out-of-fold AUC when one feature is shuffled across all rows."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(RANDOM_STATE)

    def pooled_auc(Xm):
        p = np.full(len(y), np.nan)
        for f in fold_models:
            p[f["test_idx"]] = f["model"].predict_proba(Xm.iloc[f["test_idx"]])[:, 1]
        return roc_auc_score(y, p)

    base = pooled_auc(X)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - pooled_auc(Xp))
        rows.append({"feature": col, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def ranking_agreement(shap_a, shap_b, groups, names, n_boot=N_BOOTSTRAP) -> dict:
    """Kendall tau between the two models' mean-|SHAP| rankings, with a participant bootstrap CI."""
    groups = np.asarray(groups)

    def tau(idx):
        a = np.nanmean(np.abs(shap_a[idx]), 0)
        b = np.nanmean(np.abs(shap_b[idx]), 0)
        return kendalltau(a, b).statistic

    observed = tau(np.arange(len(groups)))
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    boots = [tau(np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    boots = np.array(boots, float)
    boots = boots[np.isfinite(boots)]
    return {"kendall_tau": float(observed), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n_features": len(names)}


def xgboost_gain(fold_models, names: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=range(len(fold_models)), columns=names)
    for i, f in enumerate(fold_models):
        for k, v in f["model"].get_booster().get_score(importance_type="gain").items():
            if k in m.columns:
                m.loc[i, k] = float(v)
    out = pd.DataFrame({"feature": names, "gain_mean": m.mean().to_numpy(),
                        "gain_sd": m.std().to_numpy(), "folds_used": (m > 0).sum().to_numpy()})
    return out.sort_values("gain_mean", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_importance.py -v`
Expected: all pass.

- [ ] **Step 5: Stage for review**

```bash
git add src/importance.py tests/test_importance.py
```

---

### Task 9: Orchestrator `run_colet.py`

**Files:**
- Create: `src/run_colet.py`, `tests/test_run_colet.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `prepare(data_dir, cache_dir, refresh=False) -> dict` with `table, rows, participants, sanity, features_used, retention`
  - `run_analysis(name: str, features: list[str], rows, n_iter, n_boot, n_perm, n_perm_imp) -> dict` with `metrics (DataFrame), comparison (dict), importance (dict of DataFrames), predictions (DataFrame)`
  - `main(data_dir=DATA_DIR, out_dir=OUT_DIR, refresh=False, n_iter=N_ITER, n_boot=N_BOOTSTRAP, n_perm=N_PERMUTATIONS, n_perm_imp=N_PERM_IMPORTANCE) -> dict`
  - Tables written to `out_dir/tables/`: `01_retention.csv, 02_sanity_check.csv, 03_manipulation_check.json, 04_feature_table.csv, 05_results.csv, 06_comparison.csv, 07_importance_<analysis>_<kind>.csv, 08_redundancy_v2_13.csv, predictions_<analysis>.csv`; plus `out_dir/run_metadata.json`.

- [ ] **Step 1: Write the failing smoke test `tests/test_run_colet.py`**

```python
import json

import pandas as pd

from conftest import write_synthetic_dataset
from run_colet import main


def test_end_to_end_on_synthetic_data(tmp_path):
    data = write_synthetic_dataset(tmp_path / "data", n_participants=8, seconds=5.0)
    out = main(data_dir=data, out_dir=tmp_path / "out", n_iter=2, n_boot=50, n_perm=3, n_perm_imp=2)
    tables = tmp_path / "out" / "tables"
    res = pd.read_csv(tables / "05_results.csv")
    assert set(res["analysis"]) == {"P1", "P2"}
    assert set(res["model"]) == {"Logistic Regression", "XGBoost"}
    assert (tables / "07_importance_P1_shap_xgb.csv").exists()
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["n_participants"] == 8 and "versions" in meta
    assert out["features_used"] in (meta["features_p1"],)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_run_colet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_colet'`.

- [ ] **Step 3: Write `src/run_colet.py`**

```python
"""Run the COLET study end to end and write every table Chapter 4 needs.

Stages (each also callable on its own from the Colab notebook):
  1. prepare        -- feature table, exclusion, retention, sanity check (decides 10 vs 5 features)
  2. run_analysis   -- P1 (10 or 5 features) and P2 (pupil only), LR vs XGBoost under LOPO
  3. outputs        -- results, comparison, importance, V2-13 check, run metadata

Usage (local):  python src/run_colet.py [--refresh]
Do not run on the real data until the team has approved it.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import manipulation_check, redundancy_check, sanity_check, sanity_passed
from config import (
    CACHE_DIR, CEILING_AUC, DATA_DIR, FALLBACK_FEATURES, FEATURES, N_BOOTSTRAP, N_ITER,
    N_PERM_IMPORTANCE, N_PERMUTATIONS, OUT_DIR, PUPIL_FEATURES,
)
from data import load_annotation
from dataset import analysis_participants, analysis_rows, build_feature_table, retention_summary
from evaluate import (
    bootstrap_auc, compare_models, majority_baseline, modal_params, participant_margins,
    permutation_test, run_lopo, score,
)
from importance import (
    logistic_coefficients, oof_shap, permutation_importance_oof, ranking_agreement,
    shap_summary, xgboost_gain,
)
from labels import binary_label, main_rows
from models import MODELS
from normalize import per_participant_zscore


def prepare(data_dir=DATA_DIR, cache_dir=CACHE_DIR, refresh=False) -> dict:
    table = build_feature_table(data_dir, cache_dir, refresh)
    rows = analysis_rows(table)
    sanity = sanity_check(rows)
    return {"table": table, "rows": rows, "participants": analysis_participants(table),
            "sanity": sanity, "retention": retention_summary(table),
            "features_used": FEATURES if sanity_passed(sanity) else FALLBACK_FEATURES}


def run_analysis(name, features, rows, n_iter=N_ITER, n_boot=N_BOOTSTRAP,
                 n_perm=N_PERMUTATIONS, n_perm_imp=N_PERM_IMPORTANCE) -> dict:
    z = per_participant_zscore(rows[features], rows["participant"])
    main = main_rows(rows.assign(**{c: z[c] for c in features}))
    X = main[features].reset_index(drop=True)
    y = binary_label(main)
    g = main["participant"].to_numpy()

    metrics, probas, fits = [], {}, {}
    base = majority_baseline(y, g)
    for model_name, spec in MODELS.items():
        res = run_lopo(X, y, g, spec["factory"], spec["space"], n_iter=n_iter)
        p = res["oof_proba"]
        probas[model_name], fits[model_name] = p, res
        row = {"analysis": name, "model": model_name, "n_samples": len(y),
               "n_features": len(features), **score(y, p), **bootstrap_auc(y, p, g, n_boot),
               "share_a4_above_a1": float((participant_margins(y, p, g) > 0).mean()),
               "majority_balanced_accuracy": base["balanced_accuracy"],
               "chosen_params_modal": json.dumps(modal_params(res["chosen_params"]), default=float)}
        row.update({f"perm_{k}": v for k, v in permutation_test(
            X, y, g, spec["factory"], modal_params(res["chosen_params"]), n_perm).items()})
        metrics.append(row)

    lr, xgb = fits["Logistic Regression"], fits["XGBoost"]
    shap_lr = oof_shap(lr["fold_models"], X, "lr")
    shap_xgb = oof_shap(xgb["fold_models"], X, "xgb")
    importance = {
        "coef_lr": logistic_coefficients(lr["fold_models"], features),
        "shap_lr": shap_summary(shap_lr, features),
        "shap_xgb": shap_summary(shap_xgb, features),
        "perm_lr": permutation_importance_oof(lr["fold_models"], X, y, n_perm_imp),
        "perm_xgb": permutation_importance_oof(xgb["fold_models"], X, y, n_perm_imp),
        "gain_xgb_appendix": xgboost_gain(xgb["fold_models"], features),
    }
    comparison = {"analysis": name,
                  **compare_models(y, probas["Logistic Regression"], probas["XGBoost"], g, n_boot),
                  **{f"shap_rank_{k}": v for k, v in ranking_agreement(shap_lr, shap_xgb, g, features, n_boot).items()}}
    predictions = pd.DataFrame({"participant": g, "activity": main["activity"].to_numpy(), "y": y,
                                "p_lr": probas["Logistic Regression"], "p_xgb": probas["XGBoost"]})
    return {"metrics": pd.DataFrame(metrics), "comparison": comparison,
            "importance": importance, "predictions": predictions}


def _versions() -> dict:
    import scipy, shap, sklearn, xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "shap": shap.__version__}


def main(data_dir=DATA_DIR, out_dir=OUT_DIR, refresh=False, n_iter=N_ITER, n_boot=N_BOOTSTRAP,
         n_perm=N_PERMUTATIONS, n_perm_imp=N_PERM_IMPORTANCE) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    prep = prepare(data_dir, out_dir / "cache", refresh)
    prep["retention"].to_csv(tables / "01_retention.csv", index=False)
    prep["sanity"].to_csv(tables / "02_sanity_check.csv", index=False)
    manip = manipulation_check(load_annotation(data_dir), prep["participants"])
    (tables / "03_manipulation_check.json").write_text(json.dumps(manip, indent=2))
    prep["table"].to_csv(tables / "04_feature_table.csv", index=False)

    results, comparisons = [], []
    for name, feats in (("P1", prep["features_used"]), ("P2", PUPIL_FEATURES)):
        out = run_analysis(name, feats, prep["rows"], n_iter, n_boot, n_perm, n_perm_imp)
        results.append(out["metrics"])
        comparisons.append(out["comparison"])
        for kind, df in out["importance"].items():
            df.to_csv(tables / f"07_importance_{name}_{kind}.csv", index=False)
        out["predictions"].to_csv(tables / f"predictions_{name}.csv", index=False)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(tables / "05_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(tables / "06_comparison.csv", index=False)
    redundancy_check(prep["rows"]).to_csv(tables / "08_redundancy_v2_13.csv", index=False)

    p1 = results[results.analysis == "P1"]
    meta = {
        "n_participants": len(prep["participants"]),
        "features_p1": prep["features_used"],
        "sanity_passed": sanity_passed(prep["sanity"]),
        "ceiling_rule_triggered": bool((p1["roc_auc"] > CEILING_AUC).all()),
        "settings": {"n_iter": n_iter, "n_boot": n_boot, "n_perm": n_perm, "n_perm_imp": n_perm_imp},
        "versions": _versions(),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return {**meta, "features_used": prep["features_used"]}


if __name__ == "__main__":
    print(json.dumps(main(refresh="--refresh" in sys.argv), indent=2))
```

- [ ] **Step 4: Run the smoke test**

Run: `python -m pytest tests/test_run_colet.py -v`
Expected: PASS (1–3 minutes).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -v`
Expected: all pass.

- [ ] **Step 6: Stage for review**

```bash
git add src/run_colet.py tests/test_run_colet.py
```

---

### Task 10: Colab notebook and pinned requirements

**Files:**
- Create: `requirements-colab.txt`, `notebooks/colet_pipeline.ipynb`, `tests/test_notebook.py`

**Interfaces:**
- Consumes: `run_colet.prepare`, `run_colet.run_analysis`, `run_colet.main`, `checks.*`, `data.load_annotation`.
- Produces: a notebook whose code cells only import names that exist in `src/`.

- [ ] **Step 1: Write `requirements-colab.txt`**

```text
numpy==2.2.4
pandas==2.3.1
scipy==1.15.2
scikit-learn==1.7.1
xgboost==3.2.0
shap==0.51.0
pyarrow==24.0.0
matplotlib==3.10.5
```

- [ ] **Step 2: Write the failing test `tests/test_notebook.py`**

```python
import ast
import importlib
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colet_pipeline.ipynb"


def test_notebook_imports_resolve():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))]
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                    "run_colet", "checks", "data", "dataset", "config"}:
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"{node.module}.{alias.name}"
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_notebook.py -v`
Expected: FAIL with `FileNotFoundError` (notebook not created).

- [ ] **Step 4: Create `notebooks/colet_pipeline.ipynb`** with the cells below, in order. Build the
JSON with this helper (put the 15 blocks below into `cells` as `(type, source)` pairs):

```python
import json
from pathlib import Path

cells = [("markdown", "..."), ("code", "...")]   # the 15 blocks below, in order
nb = {
    "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
               **({"outputs": [], "execution_count": None} if t == "code" else {})}
              for t, s in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/colet_pipeline.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
```

The cells:

Markdown 1:
```markdown
# COLET pipeline: LR vs XGBoost (Colab)

Run the cells **in order**. Each stage shows its output; check it before running the next.
Before the first run, upload the folder `parquet/` (from `data/colet/parquet/`, 847 MB) to
Google Drive, e.g. `MyDrive/csrp/colet/parquet/`.
```

Code 2 (get the code):
```python
import getpass, subprocess, os
REPO = "elmigue21/csrp"
if not os.path.exists("/content/csrp"):
    token = getpass.getpass("GitHub token (press Enter if the repo is public): ").strip()
    url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
    r = subprocess.run(["git", "clone", "--depth", "1", url, "/content/csrp"], capture_output=True, text=True)
    print("cloned" if r.returncode == 0 else r.stderr.replace(token, "***") if token else r.stderr)
else:
    print(subprocess.run(["git", "-C", "/content/csrp", "pull"], capture_output=True, text=True).stdout)
```

Code 3 (pinned packages; restarts once):
```python
import importlib.metadata as md, os, subprocess
pins = dict(l.strip().split("==") for l in open("/content/csrp/requirements-colab.txt") if "==" in l)

def installed(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None

wrong = {k: v for k, v in pins.items() if installed(k) != v}
if wrong:
    subprocess.run(["pip", "install", "-q", "-r", "/content/csrp/requirements-colab.txt"], check=True)
    print("Installed pinned versions; restarting the runtime. Then run from cell 4.")
    os.kill(os.getpid(), 9)
print("All package versions match requirements-colab.txt")
```

Code 4 (Drive + paths):
```python
from google.colab import drive
drive.mount("/content/drive")
import os, sys
os.environ["COLET_DATA_DIR"] = "/content/drive/MyDrive/csrp/colet/parquet"   # edit if needed
os.environ["COLET_OUT_DIR"] = "/content/drive/MyDrive/csrp/outputs/colet"
sys.path.insert(0, "/content/csrp/src")
print(len(os.listdir(os.environ["COLET_DATA_DIR"])), "files found (expect 566)")
```

Markdown 5: `## Stage 1: features, exclusion, retention (P0–P14)`

Code 6:
```python
from config import DATA_DIR, CACHE_DIR
from run_colet import prepare
prep = prepare(DATA_DIR, CACHE_DIR)
print("analysis participants:", len(prep["participants"]), "(expect 45)")
prep["retention"]
```

Markdown 7: `## Stage 2: label-blind checks (sanity per activity; NASA-RTLX)`

Code 8:
```python
from checks import manipulation_check
from data import load_annotation
display(prep["sanity"])
print("Features used in P1:", prep["features_used"])
manipulation_check(load_annotation(DATA_DIR), prep["participants"])
```

Markdown 9: `## Stage 3: P1 and P2 (LR vs XGBoost, LOPO). Takes a while.`

Code 10:
```python
from config import PUPIL_FEATURES
from run_colet import run_analysis
p1 = run_analysis("P1", prep["features_used"], prep["rows"])
p2 = run_analysis("P2", PUPIL_FEATURES, prep["rows"])
display(p1["metrics"]); display(p2["metrics"])
print(p1["comparison"]); print(p2["comparison"])
```

Markdown 11: `## Stage 4: feature importance (P1)`

Code 12:
```python
for kind in ("shap_lr", "shap_xgb", "coef_lr", "perm_lr", "perm_xgb"):
    print(kind); display(p1["importance"][kind])
```

Markdown 13: `## Stage 5: save everything (tables + run_metadata.json to Drive)`

Code 14:
```python
from config import OUT_DIR
from run_colet import main
meta = main(DATA_DIR, OUT_DIR)
meta
```

Markdown 15:
```markdown
Stage 5 reruns Stages 1–4 from scratch (features are cached) so that every saved table comes
from one run. Outputs are in `MyDrive/csrp/outputs/colet/tables/`.
```

- [ ] **Step 5: Run the notebook test and the full suite**

Run: `python -m pytest tests -v`
Expected: all pass.

- [ ] **Step 6: Update docs** — in `docs/HANDOFF.md` §2 add rows for `src/` (COLET pipeline, `python src/run_colet.py`), `notebooks/colet_pipeline.ipynb` and `requirements-colab.txt`; in §5 mark items 2 and 4 done (code written, **not run on real data**); in `docs/methodology-colet.md` §15 replace the "Code status" bullets with the new module list; add to `docs/methodology-colet.md` §13 the implementation choices with no published value: pupil grid 120 Hz, 2nd-order Butterworth, both-eyes rule for pupil averaging, blinks merged before the 50–500 ms filter, sanity statistic = median of recording-level medians.

- [ ] **Step 7: Stage for review**

```bash
git add requirements-colab.txt notebooks/colet_pipeline.ipynb tests/test_notebook.py docs/HANDOFF.md docs/methodology-colet.md
```

---

## After the plan (needs the user)

1. User reviews and commits; pushes to GitHub so Colab can clone it.
2. User uploads `data/colet/parquet/` to Drive.
3. **Only with the user's go-ahead:** run Stage 1–2 on the real data, check retention (45 participants) and the sanity table, then Stages 3–5.

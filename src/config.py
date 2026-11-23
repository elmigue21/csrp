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
# Post-defense extraction (post-defense-data-processing.md §7.1): only run_extract.py writes
# FINAL_DIR, and replaces it on every run; caches and scratch files go in WORK_DIR.
FINAL_DIR = Path(os.environ.get("COLET_FINAL_DIR", PROJECT_ROOT / "outputs" / "final"))
WORK_DIR = Path(os.environ.get("COLET_WORK_DIR", PROJECT_ROOT / "outputs" / "work"))

# Pina et al. 2025 (Zenodo 18669059): the extracted gaze/, trial/ and tlx/ folders, and the
# outputs of run_pina.py. Plan: docs/pina-analysis-plan.md (written before the eye data were loaded).
PINA_DATA_DIR = Path(os.environ.get("PINA_DATA_DIR", PROJECT_ROOT / "data" / "pina"))
PINA_DIR = Path(os.environ.get("PINA_OUT_DIR", PROJECT_ROOT / "outputs" / "pina"))

# --------------------------------------------------------------------------- Pina design
PINA_CONDITIONS = {0: ("90", "low"), 1: ("90", "high"), 2: ("360", "low"), 3: ("360", "high")}
PINA_MAIN_AREA = "90"            # the 360-degree area is a separate check, never pooled
PINA_WINDOW_S = 30.0             # each ~4-min segment -> 8 non-overlapping 30 s windows
PINA_N_WINDOWS = 8
# Dropped together if the fixation sanity check fails (the planned fallback)
EVENT_PREFIXES = ("gaze_speed__", "fixation_", "saccade_")

# --------------------------------------------------------------------------- design
ACTIVITIES = (1, 2, 3, 4)
LOW_ACTIVITY = 1    # single task, no time pressure
HIGH_ACTIVITY = 4   # counting aloud + time pressure

# --------------------------------------------------------------------------- data errors (R5-1)
# COLET v3 holds byte-identical gaze, pupil and blink files under P28 and P31, whose
# demographics and NASA-RTLX differ: one person's eye data is stored twice and its owner is
# unknown. The recording is kept once (as P28) for the models; both IDs are left out of the
# NASA-RTLX check.
DUPLICATE_RECORDING_IDS = (31,)
RTLX_EXCLUDED_IDS = (28, 31)

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
PUPIL_MAD_MULTIPLIER = 16.0      # default of Kret & Sjak-Shie's (2019) published code; the paper says tune it
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
# Parallel workers. Results do not depend on these (candidates are seeded; permuted labels are
# drawn first). Capped at 6 because 20 workers ran out of memory on a 24 GB laptop (2026-11-11);
# set COLET_JOBS (e.g. -1 = all cores on Colab) to change it.
SEARCH_JOBS = int(os.environ.get("COLET_JOBS", 6))   # parallel inner-search fits
PERM_JOBS = int(os.environ.get("COLET_JOBS", 6))     # parallel permutation refits
N_ITER = 30                 # random-search candidates per model: the equal tuning budget
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200
N_PERM_IMPORTANCE = 50
CEILING_AUC = 0.95

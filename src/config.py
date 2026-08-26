"""Paths, column groups and experiment settings for the GAZELOAD study."""
from pathlib import Path

# --------------------------------------------------------------------------- paths
GAZELOAD_ROOT = Path(
    r"C:\Users\John\Downloads\GAZELOAD A Multimodal Eye-Tracking Dataset for Men"
    r"\GAZELOAD A Multimodal Eye-Tracking Dataset for Men"
)
METRICS_DIR = GAZELOAD_ROOT / "04_eye-metrics"
RATINGS_XLSX = GAZELOAD_ROOT / "01_Metadata" / "Tasks_Rating.xlsx"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
CACHE_DIR = OUT_DIR / "cache"

# --------------------------------------------------------------------------- columns
LABEL_COL = "selfreport_mental_load(1-10)"

# Excluded from the feature matrix. `tasks` alone predicts the label at AUC 0.93,
# so leaving it in would let both models read the task number instead of the eyes.
# Participant_ID is corrupted in 6 files ('Hedi', 'Wassim') and is a group key, not
# a feature. Timestamps are used to build epochs, then dropped.
LEAKY_COLS = ["tasks", "Participant_ID", "timestamps_start_ms", "timestamps_end_ms"]

# Aggregated as mean + SD within each epoch.
AGG_BASE = [
    "fixation_count",
    "saccade_count",
    "saccade_amplitude_degree",
    "saccade_velocity_degree/s",
    "EyeGaze_x",
    "std_EyeGaze_x",
    "EyeGaze_y",
    "std_EyeGaze_y",
    "EyeGaze_z",
    "std_EyeGaze_z",
    "gaze_x_scene_mean",
    "gaze_y_scene_mean",
    "GTE",
    "FDI",
    "SaccRate",
    "lux_interpolated",
]

# Features derived from lighting rather than from the eyes. Lighting varies by task,
# so an ablation without them tells us how much of the signal is ambient light.
LUX_FEATURES = ["lux_interpolated_mean", "lux_interpolated_std"]

# --------------------------------------------------------------------------- epochs
WINDOW_MS = 250          # native sampling period of the published metrics
EPOCH_SECONDS = 30
EPOCH_MS = EPOCH_SECONDS * 1000
MIN_WINDOW_COVERAGE = 0.8   # keep an epoch only if >=80% of its 250 ms slots are present

# --------------------------------------------------------------------------- labels
ABSOLUTE_THRESHOLD = 4      # rating >= 4 counts as high load in the as-is labelling

# --------------------------------------------------------------------------- evaluation
RANDOM_STATE = 0
INNER_FOLDS = 5             # nested tuning, grouped by participant

# Worker count for the inner grid search. Deliberately bounded: the outer loop opens
# a new pool for each of the 26 folds of each configuration, and an unbounded pool
# churns hard enough on Windows to kill the interpreter part-way through a run.
GRID_SEARCH_JOBS = 4

LR_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

XGB_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}

"""Build eye.parquet, the input for figs.py / fe.py / impact.py.

Run this first, from this directory:
    python load.py && python figs.py && python fe.py && python impact.py

Reads the 130 released `04_eye-metrics/*_Metrics_withLux.csv` files and
concatenates them with participant/task recovered from the FILENAME, because the
`Participant_ID` column contains the strings 'Hedi' and 'Wassim' in six files
(see gazeload-eda.md §8). Point ROOT at your copy of the dataset.
"""
import glob
import os
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.environ.get(
    "GAZELOAD_ROOT",
    "/home/miguel/Downloads/GAZELOAD A Multimodal Eye-Tracking Dataset for Men",
)
# ROOT = os.environ.get(
#     "GAZELOAD_ROOT",
#     r"C:\Users\John\Downloads\GAZELOAD A Multimodal Eye-Tracking Dataset for Men"
#     r"\GAZELOAD A Multimodal Eye-Tracking Dataset for Men",
# )
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye.parquet")

files = sorted(glob.glob(os.path.join(ROOT, "04_eye-metrics", "*_Metrics_withLux.csv")))
if not files:
    raise SystemExit(f"no metrics files under {ROOT!r} -- set GAZELOAD_ROOT")

frames = []
for f in files:
    base = os.path.basename(f)
    pid, task = map(int, re.match(r"(\d+)_(\d+)_", base).groups())
    d = pd.read_csv(f)
    # coerce so the six name-corrupted files load; identity comes from the filename
    d["Participant_ID"] = pd.to_numeric(d["Participant_ID"], errors="coerce")
    d["pid"], d["task"], d["src_file"] = pid, task, base
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df.rename(columns={"selfreport_mental_load(1-10)": "load"}, inplace=True)
df.to_parquet(OUT)

print(f"{len(files)} files -> {df.shape[0]:,} windows x {df.shape[1]} cols")
print(f"groups: {df.groupby(['pid', 'task']).ngroups}  (expected 130)")
print(f"written: {OUT}")

"""Feature-direction check (review round 5, issue R5-25): plot I-VT detections on raw gaze
traces and recompute event counts per activity. Outputs go to docs/colet-eda/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import GAZE_GRID_HZ, IVT_THRESHOLD_DEG_S, MIN_FIXATION_S  # noqa: E402
from data import load_recording  # noqa: E402
from events import angular_speed_deg_s, detect_events, _runs  # noqa: E402
from preprocess import clean_blinks, gaze_directions, trim_to_gaze_range  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES = [(1, 1), (1, 4), (10, 1), (10, 4), (20, 1), (20, 4)]
WINDOW_S = 5.0


def prepare(p, a):
    rec = trim_to_gaze_range(load_recording(p, a))
    blinks = clean_blinks(rec["blinks"])
    grid, xyz = gaze_directions(rec["gaze"], blinks)
    return rec, blinks, grid, xyz


fig, axes = plt.subplots(len(CASES), 1, figsize=(14, 3.0 * len(CASES)))
for ax, (p, a) in zip(axes, CASES):
    rec, blinks, grid, xyz = prepare(p, a)
    speed = angular_speed_deg_s(xyz, GAZE_GRID_HZ)
    az = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 2]))
    el = np.degrees(np.arctan2(xyz[:, 1], np.hypot(xyz[:, 0], xyz[:, 2])))
    t = np.arange(len(xyz)) / GAZE_GRID_HZ
    start = len(t) // 3
    sel = slice(start, start + int(WINDOW_S * GAZE_GRID_HZ))
    ts = t[sel]
    ax.plot(ts, az[sel] - np.nanmedian(az[sel]), lw=0.8, label="azimuth (deg)")
    ax.plot(ts, el[sel] - np.nanmedian(el[sel]), lw=0.8, label="elevation (deg)")
    lab = np.full(len(speed), np.nan)
    f = np.isfinite(speed)
    lab[f & (speed < IVT_THRESHOLD_DEG_S)] = 1
    lab[f & (speed >= IVT_THRESHOLD_DEG_S)] = 0
    for kind, s, e in _runs(lab[sel]):
        if kind == 1 and (e - s) / GAZE_GRID_HZ >= MIN_FIXATION_S:
            ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], color="tab:green", alpha=0.15, lw=0)
        elif kind == 0:
            ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], color="tab:red", alpha=0.35, lw=0)
    ax2 = ax.twinx()
    ax2.plot(ts, speed[sel], color="0.4", lw=0.5)
    ax2.axhline(IVT_THRESHOLD_DEG_S, color="k", ls="--", lw=0.6)
    ax2.set_ylim(0, 400)
    ax2.set_ylabel("speed deg/s")
    ax.set_title(f"P{p:02d} A{a}: green = fixation >= 55 ms, red = saccade, grey = speed")
    ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "trace_check.png", dpi=90)

# Saccade/fixation counts with and without COLET-like rules, per activity.
rows = []
for p in range(1, 48):
    for a in (1, 2, 3, 4):
        try:
            rec, blinks, grid, xyz = prepare(p, a)
        except Exception:
            continue
        ev = detect_events(xyz, GAZE_GRID_HZ)
        speed = angular_speed_deg_s(xyz, GAZE_GRID_HZ)
        gt = np.isfinite(xyz).all(1).sum() / GAZE_GRID_HZ
        sac_all = ev["saccade_amplitudes_deg"]
        rows.append({
            "participant": p, "activity": a,
            "fix_rate": len(ev["fixation_durations_s"]) / gt,
            "fix_median_ms": np.median(ev["fixation_durations_s"]) * 1000,
            "sac_rate_all": len(sac_all) / gt,
            "sac_rate_ge1deg": (sac_all >= 1).sum() / gt,
            "sac_amp_median": np.median(sac_all),
            "sac_mean_vel": np.mean(ev["saccade_mean_velocities"]),
            "frac_above_ivt": np.nanmean(speed >= IVT_THRESHOLD_DEG_S),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "trace_features.csv", index=False)
print(df.drop(columns="participant").groupby("activity").mean().T.round(3))

"""COLET exploration for decisions C3 (exclusion) and C4 (features).

Reads data/colet/parquet (built by data/colet/convert_colet.py) and writes:
  docs/colet-eda/quality.csv       per-recording data quality
  docs/colet-eda/features_v0.csv   first-pass whole-activity features
  docs/colet-eda/*.png             figures
Feature settings follow the COLET paper: I-VT 45 deg/s, minimum fixation 55 ms.
This is an exploratory first pass, not the final pipeline.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "data" / "colet" / "parquet"
OUT = Path(__file__).resolve().parent

CONF = 0.8          # sample validity (Faraji 2023)
VT = 45.0           # deg/s, I-VT threshold (COLET paper)
MIN_FIX = 0.055     # s, minimum fixation (COLET paper)
MAX_GAP = 0.050     # s, a larger gap breaks an event
MAX_PV = 1000.0     # deg/s, artifact limit (Hausamann 2020)


def angular_series(g):
    """Unit gaze-direction vectors from gaze_point_3d; invalid samples -> NaN."""
    g = g.sort_values("gaze_timestamp")
    t = g["gaze_timestamp"].to_numpy()
    v = g[["gaze_point_3d_x", "gaze_point_3d_y", "gaze_point_3d_z"]].to_numpy(float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.where(n == 0, np.nan, n)
    u[g["confidence"].to_numpy() < CONF] = np.nan
    return t, u


def events(t, u):
    """I-VT on sample-to-sample angular velocity, 5-sample moving-average smoothing."""
    dot = np.clip(np.einsum("ij,ij->i", u[1:], u[:-1]), -1, 1)
    ang = np.degrees(np.arccos(dot))
    dt = np.diff(t)
    vel = ang / np.where(dt > 0, dt, np.nan)
    vel = pd.Series(vel).rolling(5, center=True, min_periods=3).mean().to_numpy()
    valid = np.isfinite(vel) & (dt <= MAX_GAP)
    is_sac = valid & (vel >= VT)
    is_fix = valid & (vel < VT)
    fix, sac = [], []
    for mask, store in ((is_fix, fix), (is_sac, sac)):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for s, e in zip(starts, ends):
            store.append((s, e + 1))
    fixations = [(t[e] - t[s], s, e) for s, e in fix if t[e] - t[s] >= MIN_FIX]
    saccades = []
    for s, e in sac:
        pv = np.nanmax(vel[s:e])
        if pv > MAX_PV or not (np.isfinite(u[s]).all() and np.isfinite(u[e]).all()):
            continue
        amp = np.degrees(np.arccos(np.clip(np.dot(u[s], u[e]), -1, 1)))
        saccades.append((t[e] - t[s], amp, pv, np.nanmean(vel[s:e])))
    return fixations, saccades


def stationary_entropy(g, bins=8):
    g = g[g["confidence"] >= CONF]
    x, y = g["norm_pos_x"].clip(0, 1), g["norm_pos_y"].clip(0, 1)
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
    p = h.ravel() / h.sum() if h.sum() else h.ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(bins * bins)) if len(p) else np.nan


def recording(pid, task):
    g = pd.read_parquet(PQ / f"p{pid:02d}_t{task}_gaze.parquet")
    pu = pd.read_parquet(PQ / f"p{pid:02d}_t{task}_pupil.parquet")
    b = pd.read_parquet(PQ / f"p{pid:02d}_t{task}_blinks.parquet")
    dur = g["gaze_timestamp"].max() - g["gaze_timestamp"].min()
    t, u = angular_series(g)
    fixations, saccades = events(t, u)
    ok = np.isfinite(u).all(axis=1)
    az = np.degrees(np.arctan2(u[ok, 0], u[ok, 2]))
    el = np.degrees(np.arcsin(np.clip(u[ok, 1], -1, 1)))
    p3 = pu[pu["method"].str.contains("3d") & (pu["confidence"] >= CONF)]
    p3 = p3[p3["diameter_3d"].between(1.5, 9)]
    bd = b["duration"].to_numpy() * 1000 if len(b) else np.array([])
    bd_ok = bd[(bd >= 50) & (bd <= 500)]
    fd = np.array([f[0] for f in fixations]) * 1000
    sa = np.array(saccades) if saccades else np.empty((0, 4))
    q = {
        "participant": pid, "task": task, "duration_s": dur,
        "gaze_invalid_frac": float((g["confidence"] < CONF).mean()),
        "pupil3d_invalid_frac": float(1 - len(p3) / max(1, pu["method"].str.contains("3d").sum())),
        "blinks_total": len(bd), "blinks_implausible": int(len(bd) - len(bd_ok)),
        "n_fixations": len(fd), "n_saccades": len(sa),
    }
    f = {
        "participant": pid, "task": task,
        "fixation_rate_hz": len(fd) / dur,
        "fixation_dur_median_ms": np.median(fd) if len(fd) else np.nan,
        "fixation_dur_sd_ms": np.std(fd) if len(fd) > 1 else np.nan,
        "saccade_rate_hz": len(sa) / dur,
        "saccade_amp_median_deg": np.median(sa[:, 1]) if len(sa) else np.nan,
        "saccade_peakvel_mean_dps": np.mean(sa[:, 2]) if len(sa) else np.nan,
        "saccade_vel_mean_dps": np.mean(sa[:, 3]) if len(sa) else np.nan,
        "saccade_dur_median_ms": np.median(sa[:, 0]) * 1000 if len(sa) else np.nan,
        "blink_rate_per_min": len(bd_ok) / (dur / 60),
        "blink_dur_mean_ms": bd_ok.mean() if len(bd_ok) else np.nan,
        "pupil_mean_mm": p3["diameter_3d"].mean(),
        "pupil_sd_mm": p3["diameter_3d"].std(),
        "gaze_disp_x_deg": np.std(az) if len(az) else np.nan,
        "gaze_disp_y_deg": np.std(el) if len(el) else np.nan,
        "gaze_entropy_stationary": stationary_entropy(g),
    }
    return q, f


def main():
    qs, fs = [], []
    for pid in range(1, 48):
        for task in range(1, 5):
            q, f = recording(pid, task)
            qs.append(q)
            fs.append(f)
        print(f"participant {pid} done", flush=True)
    q = pd.DataFrame(qs)
    f = pd.DataFrame(fs)
    q.to_csv(OUT / "quality.csv", index=False)
    f.to_csv(OUT / "features_v0.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for task, c in zip(range(1, 5), ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
        ax.hist(q.loc[q.task == task, "gaze_invalid_frac"] * 100, bins=40, range=(0, 60),
                alpha=0.6, label=f"A{task}", color=c)
    for th in (20, 30, 35, 40):
        ax.axvline(th, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Gaze samples with confidence < 0.8 (%)")
    ax.set_ylabel("Recordings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "c3_invalid_samples.png", dpi=150)

    feats = [c for c in f.columns if c not in ("participant", "task")]
    corr = f[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "c4_feature_correlation.png", dpi=150)
    print("finished")


if __name__ == "__main__":
    main()

"""Label-blind check of round-4 review R4-1 (gaze_point_3d vs gaze_normal0/1).

Reads data/colet/parquet and writes docs/colet-eda/check_r4_1.csv (one row per recording).
Counts and spreads per recording only; no model, no A1-vs-A4 comparison.
Results: researcher-notes N25.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = str(ROOT / "data" / "colet" / "parquet")


def angles(x, y, z):
    az = np.degrees(np.arctan2(x, z))
    el = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return az, el


rows = []
for f in sorted(glob.glob(os.path.join(PARQUET, "p*_t*_gaze.parquet"))):
    name = os.path.basename(f).replace("_gaze.parquet", "")
    d = pd.read_parquet(f)
    d = d[d["confidence"] >= 0.8]
    if len(d) == 0:
        continue
    p = d[["gaze_point_3d_x", "gaze_point_3d_y", "gaze_point_3d_z"]].to_numpy()
    n0 = d[["gaze_normal0_x", "gaze_normal0_y", "gaze_normal0_z"]].to_numpy()
    n1 = d[["gaze_normal1_x", "gaze_normal1_y", "gaze_normal1_z"]].to_numpy()
    ok0 = ~np.isnan(n0).any(axis=1)
    ok1 = ~np.isnan(n1).any(axis=1)
    both = ok0 & ok1

    # 3D gaze point
    depth = np.linalg.norm(p, axis=1)
    az3, el3 = angles(p[:, 0], p[:, 1], p[:, 2])

    # mean of the two normals (binocular), else the valid eye
    nm = np.where(both[:, None], (n0 + n1) / 2, np.where(ok0[:, None], n0, n1))
    azn, eln = angles(nm[:, 0], nm[:, 1], nm[:, 2])
    azb, elb = angles(*((n0 + n1) / 2)[both].T) if both.any() else (np.nan, np.nan)

    # vergence between the two eyes' directions
    cosv = np.sum(n0[both] * n1[both], axis=1) / (
        np.linalg.norm(n0[both], axis=1) * np.linalg.norm(n1[both], axis=1))
    verg = np.degrees(np.arccos(np.clip(cosv, -1, 1)))

    rows.append(dict(
        rec=name,
        n=len(d),
        frac_z_le0=np.mean(p[:, 2] <= 0),
        depth_med_mm=np.median(depth),
        z_med_mm=np.median(p[:, 2]),
        sd_x_3d=np.nanstd(az3), sd_y_3d=np.nanstd(el3),
        sd_x_norm=np.nanstd(azn), sd_y_norm=np.nanstd(eln),
        sd_x_norm_bino=np.nanstd(azb), sd_y_norm_bino=np.nanstd(elb),
        frac_mono=1 - both.mean(),
        verg_med_deg=np.median(verg) if len(verg) else np.nan,
    ))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_r4_1.csv"), index=False)
pd.set_option("display.width", 200)
print("recordings:", len(r))
print("recordings with any z<=0:", (r.frac_z_le0 > 0).sum())
print("median of per-recording median depth (mm):", r.depth_med_mm.median())
print("range of per-recording median z (mm):", r.z_med_mm.min(), r.z_med_mm.max())
print("median vergence between eyes (deg):", r.verg_med_deg.median(),
      "IQR", r.verg_med_deg.quantile([.25, .75]).tolist())
print("median monocular share:", r.frac_mono.median(), "max:", r.frac_mono.max())
for c in ["sd_x_3d", "sd_y_3d", "sd_x_norm", "sd_y_norm", "sd_x_norm_bino", "sd_y_norm_bino"]:
    print(f"{c}: median {r[c].median():.1f}  max {r[c].max():.1f}  n>30 {(r[c] > 30).sum()}")
print("\nrecordings with 3D spread x > 30 deg:")
print(r[r.sd_x_3d > 30][["rec", "frac_z_le0", "sd_x_3d", "sd_x_norm", "sd_x_norm_bino", "frac_mono"]]
      .round(2).to_string(index=False))
print("\ntop 5 by normal-based spread x:")
print(r.nlargest(5, "sd_x_norm")[["rec", "sd_x_norm", "sd_x_norm_bino", "frac_mono", "verg_med_deg"]]
      .round(2).to_string(index=False))

"""Is the bottleneck feature engineering, or sample size?

Learning curve over the number of TRAINING PARTICIPANTS. If performance has
plateaued by 25 participants, more data of the same kind will not help and the
limit is measurement/construct. If it is still climbing, n is the binding
constraint and no amount of feature engineering substitutes for it.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(0)
d = pd.read_parquet("eye.parquet").sort_values(["pid", "task", "timestamps_start_ms"])

# ---- rebuild the S2 feature set (cleaned + rates + log-robust + dynamics) ----
art = (d.saccade_amplitude_degree > 90) | (d["saccade_velocity_degree/s"] > 900)
d["is_artifact"] = art.astype(int)
d["gaze_invalid"] = (d.EyeGaze_z < 0).astype(int)
d["amp_c"] = d.saccade_amplitude_degree.where(~art)
d["vel_c"] = d["saccade_velocity_degree/s"].where(~art)
d["gaze_disp"] = np.sqrt(d.std_EyeGaze_x ** 2 + d.std_EyeGaze_y ** 2)
d["t_frac"] = d.groupby(["pid", "task"]).cumcount() / d.groupby(["pid", "task"]).timestamps_start_ms.transform("size")

g = d.groupby(["pid", "task"])
F = pd.DataFrame(index=g.size().index)
F["artifact_rate"] = g.is_artifact.mean()
F["gaze_invalid_rate"] = g.gaze_invalid.mean()
F["fix_rate_hz"] = g.fixation_count.mean() * 4
F["sacc_rate_hz"] = g.saccade_count.mean() * 4
F["frac_no_fix"] = g.fixation_count.apply(lambda s: (s == 0).mean())
F["frac_no_sacc"] = g.saccade_count.apply(lambda s: (s == 0).mean())
for c, nm in [("amp_c", "amp"), ("vel_c", "vel"), ("FDI", "fdi"), ("gaze_disp", "disp")]:
    lg = np.log1p(d[c])
    F[f"{nm}_logmed"] = lg.groupby([d.pid, d.task]).median()
    F[f"{nm}_logiqr"] = (lg.groupby([d.pid, d.task]).quantile(.75)
                         - lg.groupby([d.pid, d.task]).quantile(.25))
F["fix_per_sacc"] = g.fixation_count.sum() / g.saccade_count.sum().replace(0, np.nan)
F["scene_x_sd"] = g.gaze_x_scene_mean.std()
F["scene_y_sd"] = g.gaze_y_scene_mean.std()
for c in ["fixation_count", "saccade_count", "gaze_disp"]:
    F[f"{c}_slope"] = g.apply(lambda s, c=c: np.polyfit(s.t_frac, s[c].fillna(0), 1)[0])
F["fix_acf1"] = g.fixation_count.apply(lambda s: s.autocorr(1))

y = g.load.first()
pid = y.index.get_level_values("pid").values
# per-participant feature standardisation (label-free, legitimate)
Fz = F.groupby(level="pid").transform(lambda s: (s - s.mean()) / (s.std() if s.std() else np.nan)).fillna(0)

RS = dict(random_state=0, max_iter=120, early_stopping=False)


def lopo_rho(X, yy, groups, train_pids):
    """Mean within-participant Spearman rho on held-out participants."""
    rhos = []
    for held in np.unique(groups):
        if held in train_pids:
            continue
        tr = np.isin(groups, train_pids)
        te = groups == held
        if tr.sum() < 20 or te.sum() < 3:
            continue
        m = HGR(**RS).fit(X[tr], yy[tr])
        p = m.predict(X[te])
        r = pd.Series(p).corr(pd.Series(yy[te].values), method="spearman")
        if np.isfinite(r):
            rhos.append(r)
    return np.mean(rhos) if rhos else np.nan


print("=== LEARNING CURVE: within-participant rho vs number of training participants ===")
print("    (each point = mean over 8 random participant subsets, tested on held-out people)\n")
all_pids = np.unique(pid)
Xv, yv = Fz.values, y
for k in [5, 8, 12, 16, 20, 25]:
    vals = []
    for rep in range(8):
        tr = RNG.choice(all_pids, size=k, replace=False)
        v = lopo_rho(Xv, yv, pid, tr)
        if np.isfinite(v):
            vals.append(v)
    print("  train on %2d participants -> rho = %+.3f  (SD %.3f over subsets)"
          % (k, np.mean(vals), np.std(vals)))

print("\n=== how many participants would a stable estimate need? ===")
# SD of the per-participant rho, used to size the CI on the mean
per = []
logo = LeaveOneGroupOut()
for tr, te in logo.split(Xv, yv, pid):
    m = HGR(**RS).fit(Xv[tr], yv.values[tr])
    r = pd.Series(m.predict(Xv[te])).corr(pd.Series(yv.values[te]), method="spearman")
    if np.isfinite(r):
        per.append(r)
per = np.array(per)
print("  LOPO per-participant rho: mean %+.3f, SD %.3f, n=%d folds" % (per.mean(), per.std(), len(per)))
print("  95%% CI on the mean: [%+.3f, %+.3f]  (width %.3f)"
      % (per.mean() - 1.96 * per.std() / np.sqrt(len(per)),
         per.mean() + 1.96 * per.std() / np.sqrt(len(per)),
         2 * 1.96 * per.std() / np.sqrt(len(per))))
print("  participants needed for a CI half-width of 0.10: %d"
      % int(np.ceil((1.96 * per.std() / 0.10) ** 2)))
print("  participants needed for a CI half-width of 0.05: %d"
      % int(np.ceil((1.96 * per.std() / 0.05) ** 2)))
print("  fraction of participants with rho > 0: %.0f%% (%d of %d)"
      % (100 * (per > 0).mean(), (per > 0).sum(), len(per)))

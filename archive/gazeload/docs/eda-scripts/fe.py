import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GroupKFold

df = pd.read_parquet("eye.parquet").sort_values(["pid", "task", "timestamps_start_ms"])
cv = GroupKFold(5)
RS = dict(random_state=0, max_iter=120, early_stopping=False)


def score(X, y, gp, name, model=None):
    """Report R2 and MAE on out-of-fold predictions, plus within-participant rank corr."""
    m = model or HGR(**RS)
    p = cross_val_predict(m, X, y, cv=cv, groups=gp, n_jobs=5)
    ss = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = np.abs(y - p).mean()
    d = pd.DataFrame({"pid": gp.values, "y": y.values, "p": p})
    wr = d.groupby("pid").apply(lambda g: g.y.corr(g.p, method="spearman")).mean()
    print("  %-52s R2=%+.3f  MAE=%.2f  within-pid rho=%+.2f" % (name, ss, mae, wr))
    return ss


# ---------- window-level cleaning (FE step 1) ----------
d = df.copy()
art = (d.saccade_amplitude_degree > 90) | (d["saccade_velocity_degree/s"] > 900)
d["is_artifact"] = art.astype(int)
d["gaze_invalid"] = (d.EyeGaze_z < 0).astype(int)
nrm = np.sqrt(d.EyeGaze_x ** 2 + d.EyeGaze_y ** 2 + d.EyeGaze_z ** 2)
d["norm_bad"] = ((nrm - 1).abs() > 0.01).astype(int)
# physiologically valid saccades only
d["sacc_amp_clean"] = d.saccade_amplitude_degree.where(~art)
d["sacc_vel_clean"] = d["saccade_velocity_degree/s"].where(~art)
# angular gaze coords (more meaningful than raw x/y/z)
d["gaze_az"] = np.degrees(np.arctan2(d.EyeGaze_x, d.EyeGaze_z.clip(lower=1e-6)))
d["gaze_el"] = np.degrees(np.arcsin(d.EyeGaze_y.clip(-1, 1)))
d["gaze_disp"] = np.sqrt(d.std_EyeGaze_x ** 2 + d.std_EyeGaze_y ** 2)
# time-on-task (0..1 within session) -- fatigue / drift
d["t_frac"] = d.groupby(["pid", "task"]).cumcount() / d.groupby(["pid", "task"]).timestamps_start_ms.transform("size")
# window-to-window gaze shift (scanpath velocity in scene pixels)
d["scene_shift"] = np.sqrt(d.groupby(["pid", "task"]).gaze_x_scene_mean.diff() ** 2 +
                           d.groupby(["pid", "task"]).gaze_y_scene_mean.diff() ** 2)

RAW = ['fixation_count', 'saccade_count', 'saccade_amplitude_degree', 'saccade_velocity_degree/s',
       'EyeGaze_x', 'std_EyeGaze_x', 'EyeGaze_y', 'std_EyeGaze_y', 'EyeGaze_z', 'std_EyeGaze_z',
       'gaze_x_scene_mean', 'gaze_y_scene_mean', 'GTE', 'FDI', 'SaccRate', 'blink_flag_any',
       'lux_interpolated']
g = d.groupby(["pid", "task"])
y = g.load.first()
gp = pd.Series(y.index.get_level_values("pid"), index=y.index)

print("=== reference points ===")
print("  %-52s R2=%+.3f  MAE=%.2f" % ("predict the global mean", 0.0, (y - y.mean()).abs().mean()))
A = g[RAW].agg(['mean', 'std', 'median'])
A.columns = ['_'.join(c) for c in A.columns]
score(A, y, gp, "S0  raw 51 aggregates (current best guess)")

# ---------- S1: cleaned + distributional + dynamics ----------
def build(dd):
    gg = dd.groupby(["pid", "task"])
    f = pd.DataFrame(index=gg.size().index)
    # quality / data-loss rates -- turn artifacts into explicit features
    f["artifact_rate"] = gg.is_artifact.mean()
    f["gaze_invalid_rate"] = gg.gaze_invalid.mean()
    f["norm_bad_rate"] = gg.norm_bad.mean()
    # event rates per second (250ms windows -> x4)
    f["fix_rate_hz"] = gg.fixation_count.mean() * 4
    f["sacc_rate_hz"] = gg.saccade_count.mean() * 4
    f["blink_rate_hz"] = gg.blink_flag_any.mean() * 4
    # structural zeros as their own signal (NOT imputed)
    f["frac_no_fixation"] = gg.fixation_count.apply(lambda s: (s == 0).mean())
    f["frac_no_saccade"] = gg.saccade_count.apply(lambda s: (s == 0).mean())
    # robust location + spread on log-transformed skewed vars, artifacts removed
    for c, nm in [("sacc_amp_clean", "amp"), ("sacc_vel_clean", "vel"), ("FDI", "fdi"),
                  ("gaze_disp", "disp"), ("scene_shift", "shift")]:
        lg = np.log1p(dd[c])
        f[f"{nm}_logmed"] = lg.groupby([dd.pid, dd.task]).median()
        f[f"{nm}_logiqr"] = lg.groupby([dd.pid, dd.task]).quantile(.75) - lg.groupby([dd.pid, dd.task]).quantile(.25)
    # fixation/saccade balance -- the classic load ratio
    f["fix_per_sacc"] = gg.fixation_count.sum() / gg.saccade_count.sum().replace(0, np.nan)
    # gaze concentration: how tightly gaze clusters in the scene
    f["scene_x_sd"] = gg.gaze_x_scene_mean.std()
    f["scene_y_sd"] = gg.gaze_y_scene_mean.std()
    f["gaze_az_sd"] = gg.gaze_az.std()
    f["gaze_el_med"] = gg.gaze_el.median()
    # DYNAMICS: drift over time-on-task (slope of z-scored metric vs t_frac)
    for c in ["fixation_count", "saccade_count", "gaze_disp"]:
        f[f"{c}_slope"] = gg.apply(lambda s, c=c: np.polyfit(s.t_frac, s[c].fillna(0), 1)[0]
                                   if s[c].notna().sum() > 10 else np.nan)
    # burstiness / temporal irregularity of fixation counts
    f["fix_lag1_acf"] = gg.fixation_count.apply(lambda s: s.autocorr(1))
    f["fix_cv"] = gg.fixation_count.std() / gg.fixation_count.mean().replace(0, np.nan)
    f["session_len_s"] = gg.size() * 0.25
    return f


F = build(d)
score(F, y, gp, "S1  cleaned + rates + log-robust + dynamics (%d feats)" % F.shape[1])

# ---------- S2: within-participant z-scoring of FEATURES ----------
Fz = F.groupby(level="pid").transform(lambda s: (s - s.mean()) / (s.std() if s.std() else np.nan))
score(Fz.fillna(0), y, gp, "S2  S1 z-scored within participant")

# ---------- S3: within-participant target normalisation ----------
yz = y.groupby(level="pid").transform(lambda s: (s - s.mean()) / (s.std() if s.std() else 1))
score(F, yz, gp, "S3  raw features -> within-pid z-scored TARGET")
score(Fz.fillna(0), yz, gp, "S4  z-scored features -> z-scored target  [both]")

# ---------- S5: rank within participant (ordinal, distribution-free) ----------
yr = y.groupby(level="pid").rank(pct=True)
score(Fz.fillna(0), yr, gp, "S5  z-scored features -> within-pid percentile rank")

# ---------- S6: small regularised linear model (n=130 favours few params) ----------
lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                    RidgeCV(alphas=np.logspace(-2, 3, 30)))
score(Fz.fillna(0), yz, gp, "S6  S4 features, RidgeCV instead of gradient boosting", lin)

# ---------- S7: top-k features only (curse of dimensionality at n=130) ----------
corr = pd.concat([Fz.fillna(0), yz.rename("y")], axis=1).corr()["y"].drop("y").abs().sort_values(ascending=False)
print("\n  top 10 features by |corr| with within-pid z-scored load:")
for k, v in corr.head(10).items():
    print("     %-24s %.3f" % (k, v))
for k in (5, 8, 12):
    score(Fz[corr.head(k).index].fillna(0), yz, gp, f"S7  top-{k} features only, RidgeCV", lin)

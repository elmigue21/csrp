import pandas as pd, numpy as np, matplotlib, warnings, os
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = r"C:\dev\csrp\docs\eda-figures"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, GOOD = "#d03b3b", "#0ca30c"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "legend.frameon": False,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def fin(ax, t, s=None, xl=None, yl=None):
    ax.set_title(t, loc="left", pad=22 if s else 10)
    if s:
        ax.text(0, 1.02, s, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel(xl or "")
    ax.set_ylabel(yl or "")


def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, n), bbox_inches="tight")
    plt.close(fig)


df = pd.read_parquet("eye.parquet")
g = df.groupby(["pid", "task"])
lab = g.load.first().reset_index()

# FIG 1 -- effective sample size + target distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
vc = lab.load.value_counts().sort_index()
axes[0].bar(vc.index, vc.values, width=0.42, color=S1)
for x, v in vc.items():
    axes[0].text(x, v + 0.4, str(v), ha="center", fontsize=7.5, color=INK2)
axes[0].set_xticks(range(1, 11))
axes[0].set_ylim(0, 26)
fin(axes[0], "Target distribution: 130 labels, not 135,499",
    "self-reported mental load, one value per (participant, task) session",
    "reported load (1-10)", "sessions")
bp = axes[1].boxplot([lab[lab.task == t].load for t in range(1, 6)], patch_artist=True,
                     widths=0.5, medianprops=dict(color=SURF, lw=2),
                     flierprops=dict(marker="o", ms=4, mfc=S2, mec="none"))
for b in bp["boxes"]:
    b.set(facecolor=S1, edgecolor="none", alpha=.85)
axes[1].set_xticklabels([f"Task {i}" for i in range(1, 6)])
fin(axes[1], "Load rises with task, as designed",
    "Friedman chi2=67.1, p=9.5e-14; Spearman rho=0.58", "", "reported load")
save(fig, "01_target.png")

# FIG 2 -- variance decomposition + rater bias
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
parts = ["Participant\n(rater bias)", "Task\n(the manipulation)", "Residual"]
vals = [35.4, 37.3, 27.3]
axes[0].barh(parts, vals, color=[CRIT, S1, GRID], height=.5)
for i, v in enumerate(vals):
    axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
axes[0].set_xlim(0, 46)
axes[0].invert_yaxis()
axes[0].xaxis.set_major_formatter(PercentFormatter())
axes[0].grid(axis="y", visible=False)
fin(axes[0], "Who you asked matters as much as what you asked",
    "variance in reported load, decomposed (n=130)", "share of total variance")
pm = lab.groupby("pid").load.mean().sort_values()
axes[1].bar(range(26), pm.values,
            color=[CRIT if v < 2.5 or v > 6 else S1 for v in pm.values], width=.6)
axes[1].axhline(lab.load.mean(), color=INK2, ls="--", lw=1.2)
axes[1].text(-0.4, lab.load.mean() + .25, f"grand mean {lab.load.mean():.1f}",
             ha="left", fontsize=7.5, color=INK2)
axes[1].set_xticks([])
axes[1].set_ylim(0, 8.1)
fin(axes[1], "Per-participant mean load spans 1.8 to 7.4",
    "each bar = one participant averaged across all 5 tasks",
    "26 participants (sorted)", "mean reported load")
save(fig, "02_rater_bias.png")

# FIG 3 -- leakage: lux is a task fingerprint, and the split decides the answer
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
lx = g.lux_interpolated.mean().reset_index()
bp = axes[0].boxplot([lx[lx.task == t].lux_interpolated for t in range(1, 6)],
                     patch_artist=True, widths=0.5, medianprops=dict(color=SURF, lw=2),
                     flierprops=dict(marker="o", ms=4, mfc=S1, mec="none"))
for b in bp["boxes"]:
    b.set(facecolor=S2, edgecolor="none", alpha=.85)
axes[0].set_xticklabels([f"Task {i}" for i in range(1, 6)])
fin(axes[0], "Ambient light separates the tasks almost perfectly",
    "lux alone predicts which task: 39.5% vs 20% chance", "", "mean lux in session")
names = ["random-row 5-fold\n(INVALID)", "GroupKFold by participant\n(honest)"]
r2 = [0.047, -0.263]
bars = axes[1].bar(names, r2, color=[CRIT, S1], width=.42)
axes[1].axhline(0, color=INK2, lw=1)
for b, v in zip(bars, r2):
    axes[1].text(b.get_x() + b.get_width() / 2, v + (0.018 if v > 0 else -0.042),
                 f"R2 = {v:+.3f}", ha="center", fontsize=9, color=INK2)
axes[1].set_ylim(-0.34, 0.14)
fin(axes[1], "The split decides the answer",
    "identical features and model, only the CV scheme differs", "", "cross-validated R2")
save(fig, "03_leakage.png")

# FIG 4 -- implausible saccade values
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
a = df.saccade_amplitude_degree.dropna()
axes[0].hist(np.log10(a), bins=90, color=S1)
axes[0].axvline(np.log10(90), color=CRIT, ls="--", lw=1.5)
axes[0].text(np.log10(90) + .07, axes[0].get_ylim()[1] * .82,
             "90 deg\n(anatomical max)", fontsize=7.5, color=CRIT)
axes[0].set_xticks([0, 1, 2, 3])
axes[0].set_xticklabels(["1", "10", "100", "1,000"])
fin(axes[0], "16% of saccade amplitudes exceed 90 deg",
    "largest recorded: 3,516 deg, nearly 10 full rotations",
    "saccade amplitude (deg, log scale)", "windows")
v = df["saccade_velocity_degree/s"].dropna()
axes[1].hist(np.log10(v), bins=90, color=S1)
axes[1].axvline(np.log10(900), color=CRIT, ls="--", lw=1.5)
axes[1].text(np.log10(900) + .07, axes[1].get_ylim()[1] * .74,
             "900 deg/s\n(physiological\nceiling)", fontsize=7.5, color=CRIT)
axes[1].set_xticks([1.5, 2, 2.5, 3, 3.5, 4])
axes[1].set_xticklabels(["30", "100", "300", "1k", "3k", "10k"])
fin(axes[1], "11% of peak velocities are physiologically impossible",
    "largest recorded: 16,179 deg/s", "saccade peak velocity (deg/s, log scale)", "windows")
save(fig, "04_implausible.png")

# FIG 5 -- missingness + degenerate features
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
m = (df.drop(columns=["Participant_ID"]).isna().mean() * 100)
m = m[m > 0].sort_values()
axes[0].barh(m.index, m.values, color=S2, height=.5)
for i, v in enumerate(m.values):
    axes[0].text(v + .5, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
axes[0].set_xlim(0, 30)
axes[0].grid(axis="y", visible=False)
fin(axes[0], "All missingness is structural, not random",
    "NaN appears exactly when the underlying event count is zero",
    "% of windows missing")
deg = {"blink flag == 0": 99.6, "GTE == 0": 98.8, "saccade_count == 0": 23.9,
       "fixation_count == 0": 11.7}
k = list(deg)[::-1]
vv = [deg[x] for x in k]
axes[1].barh(k, vv, color=[CRIT if x > 90 else S1 for x in vv], height=.5)
for i, val in enumerate(vv):
    axes[1].text(val + 1, i, f"{val:.1f}%", va="center", fontsize=8, color=INK2)
axes[1].set_xlim(0, 112)
axes[1].grid(axis="y", visible=False)
fin(axes[1], "Two features are effectively constant",
    "GTE and the blink flag carry almost no information", "% of windows at zero")
save(fig, "05_missing_degenerate.png")

# FIG 6 -- group-level correlation with the target
FEAT = ['fixation_count', 'saccade_count', 'saccade_amplitude_degree',
        'saccade_velocity_degree/s', 'EyeGaze_x', 'std_EyeGaze_x', 'EyeGaze_y',
        'std_EyeGaze_y', 'EyeGaze_z', 'std_EyeGaze_z', 'gaze_x_scene_mean',
        'gaze_y_scene_mean', 'GTE', 'FDI', 'SaccRate', 'blink_flag_any',
        'lux_interpolated']
agg = g[FEAT].mean()
agg['load'] = g.load.first()
gc = agg.corr()['load'].drop('load').sort_values()
fig, ax = plt.subplots(figsize=(6.8, 5))
ax.barh(gc.index, gc.values, color=[S2 if v < 0 else S1 for v in gc.values], height=.6)
ax.axvline(0, color=INK2, lw=1)
for i, v in enumerate(gc.values):
    ax.text(v + (0.008 if v > 0 else -0.008), i, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7.5, color=INK2)
ax.set_xlim(-0.30, 0.32)
ax.grid(axis="y", visible=False)
fin(ax, "No eye metric correlates strongly with reported load",
    "session-level Pearson r, n=130 (|r| < 0.23 throughout)",
    "correlation with reported load")
save(fig, "06_group_corr.png")

# FIG 7 -- honest baselines
fig, ax = plt.subplots(figsize=(7.6, 3.6))
n = ["predict the\nglobal mean", "eye features\n(51 aggregates)", "eye features,\nlux removed",
     "task label\nonly", "task label\n+ eye features"]
v = [0.0, -0.151, -0.197, 0.304, 0.066]
bars = ax.bar(n, v, color=[GRID, CRIT, CRIT, GOOD, S1], width=.5)
ax.axhline(0, color=INK2, lw=1)
for b, x in zip(bars, v):
    ax.text(b.get_x() + b.get_width() / 2, x + (0.014 if x >= 0 else -0.032),
            f"{x:+.3f}", ha="center", fontsize=8.5, color=INK2)
ax.set_ylim(-0.27, 0.37)
fin(ax, "Eye features alone do not beat guessing the mean",
    "GroupKFold by participant, n=130 sessions", "", "cross-validated R2")
save(fig, "07_baselines.png")

# FIG 8 -- within-participant normalization recovers signal
fig, ax = plt.subplots(figsize=(6.4, 3.6))
n2 = ["raw load\n(rater bias intact)", "within-participant\nz-scored load"]
v2 = [-0.151, 0.170]
bars = ax.bar(n2, v2, color=[CRIT, GOOD], width=.38)
ax.axhline(0, color=INK2, lw=1)
for b, x in zip(bars, v2):
    ax.text(b.get_x() + b.get_width() / 2, x + (0.012 if x >= 0 else -0.028),
            f"{x:+.3f}", ha="center", fontsize=9, color=INK2)
ax.set_ylim(-0.23, 0.24)
fin(ax, "Removing rater bias flips the sign of the result",
    "same 51 eye features, same GroupKFold split, target rescaled per participant",
    "", "cross-validated R2")
save(fig, "08_normalization.png")

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)

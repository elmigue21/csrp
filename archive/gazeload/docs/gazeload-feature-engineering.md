# GAZELOAD — Feature Engineering Plan

Companion to [gazeload-eda.md](gazeload-eda.md). Every recommendation here is
**measured**, not proposed: each block was built, then scored with
`GroupKFold(5)` by participant on the 130 session-level samples.

> **Scope.** This is a *regression* study on the raw 1–10 rating at session level
> (n = 130), run to find which transformations carry signal. The main project
> (`methodology.md`) does binary classification on 1,087 30-second epochs with
> leave-one-participant-out, so the R² values here are **not comparable** to its
> ROC-AUCs — read them as relative evidence about feature construction, not as
> competing scores. Sections 1–6 transfer directly to the epoch pipeline; §7
> (per-participant standardisation) is already implemented there as the
> `within_person` feature scheme. Pipeline-level fixes are in
> **[gazeload-data-quality-actions.md](gazeload-data-quality-actions.md)**.

---

## Measured results

Target: `selfreport_mental_load(1-10)`. Model: `HistGradientBoostingRegressor`
(RidgeCV where noted). Split: `GroupKFold(5)` on `pid`.
`ρ` = mean **within-participant** Spearman correlation between predicted and
actual load — the metric that survives rater bias.

| Step | Feature set | R² | MAE | ρ |
|---|---|---|---|---|
| — | predict the global mean | 0.000 | 2.00 | — |
| **S0** | raw 51 aggregates (mean/std/median of the 17 shipped columns) | −0.196 | 2.15 | +0.27 |
| **S1** | cleaned + rates + log-robust + dynamics (29 features) | −0.062 | 2.09 | +0.46 |
| **S2** | S1, z-scored **within participant** | **+0.107** | **1.82** | **+0.52** |
| S3 | raw features → within-participant z-scored target | +0.023 | 0.70 | +0.34 |
| S4 | z-scored features → z-scored target | *+0.417* | *0.56* | *+0.58* |
| S5 | z-scored features → within-participant percentile rank | *+0.376* | *0.18* | *+0.61* |
| S6 | S4 features, RidgeCV instead of boosting | +0.141 | 0.71 | +0.29 |

**S2 is the deployable result**: R² +0.107 and MAE 1.82 against the mean's 2.00,
with within-person ρ = 0.52. Going from S0 to S2 is a swing of **+0.30 R²** on
identical data, model, and split — pure feature engineering.

*S4 and S5 are italicised because they are not deployable scores.* They normalise
the **target** using the participant's own five ratings, which you do not have for
a new user. They are useful as a **ceiling**: they say the features can rank a
person's own sessions well (ρ = 0.58–0.61), and that between-subject rating bias
is what caps the absolute-load task.

The distinction that matters:

| Normalisation | Needs | Verdict |
|---|---|---|
| z-score **features** within participant | that person's eye data only (a calibration window) | **legitimate** — this is S2 |
| z-score **target** within participant | that person's own load ratings | **not available at prediction time** — ceiling analysis only |

---

## The seven changes that produced the gain

### 1. Cut the artifact population before aggregating (biggest single win)

21% of windows with a saccade carry impossible kinematics (median amplitude 359°,
69% above 900°/s). Averaging over them destroys the real distribution — a mean
amplitude of 98° is a mixture statistic describing nothing.

```python
art = (d.saccade_amplitude_degree > 90) | (d["saccade_velocity_degree/s"] > 900)
d["sacc_amp_clean"] = d.saccade_amplitude_degree.where(~art)
d["sacc_vel_clean"] = d["saccade_velocity_degree/s"].where(~art)
d["is_artifact"]    = art.astype(int)       # keep the rate as a feature
```

Retain `artifact_rate`, `gaze_invalid_rate` (`EyeGaze_z < 0`) and `norm_bad_rate`
(non-unit gaze vector) as **explicit data-quality features**. They vary 9.6×
across participants and let the model discount unreliable sessions instead of
silently absorbing their noise.

### 2. Convert counts to rates, and encode the structural zeros

The NaNs are MNAR-by-construction (`saccade_amplitude` is null exactly when
`saccade_count == 0`), so imputation invents data. Encode the absence directly:

```python
f["fix_rate_hz"]       = g.fixation_count.mean() * 4      # 250 ms -> Hz
f["sacc_rate_hz"]      = g.saccade_count.mean() * 4
f["frac_no_fixation"]  = g.fixation_count.apply(lambda s: (s == 0).mean())
f["frac_no_saccade"]   = g.saccade_count.apply(lambda s: (s == 0).mean())
```

`frac_no_fixation` lands in the **top 3** features (|r| = 0.251). The "nothing
happened" rate is more informative than any average of what did.

### 3. Log-transform then take robust statistics

Amplitude skew is 4.32, velocity 4.73, FDI 3.04. A mean over that is dominated by
the tail; a mean over a *bimodal* one is meaningless. Use median and IQR on
`log1p`, computed after step 1:

```python
lg = np.log1p(dd[c])
f[f"{nm}_logmed"] = lg.groupby([dd.pid, dd.task]).median()
f[f"{nm}_logiqr"] = lg.groupby([dd.pid, dd.task]).quantile(.75) \
                  - lg.groupby([dd.pid, dd.task]).quantile(.25)
```

`vel_logmed` (0.212) and `amp_logiqr` (0.211) both make the top 8; their raw-mean
equivalents in S0 do not.

### 4. Add temporal dynamics — the 250 ms grid is being wasted

S0 collapses each session to a mean and throws away the time axis. Windows are
contiguous and exactly 250 ms apart, so drift and burstiness are free:

```python
d["t_frac"] = d.groupby(["pid","task"]).cumcount() \
            / d.groupby(["pid","task"]).timestamps_start_ms.transform("size")
# slope of the metric against time-on-task -> fatigue / disengagement
f["saccade_count_slope"] = g.apply(lambda s: np.polyfit(s.t_frac, s.saccade_count, 1)[0])
f["fix_lag1_acf"]        = g.fixation_count.apply(lambda s: s.autocorr(1))
f["fix_cv"]              = g.fixation_count.std() / g.fixation_count.mean()
```

**`saccade_count_slope` is the single strongest feature in the whole set
(|r| = 0.287)** — how gaze behaviour *changes* over a session beats any static
average of it. `fixation_count_slope` also makes the top 6.

### 5. Re-express gaze geometry in angles and dispersion

Raw `EyeGaze_x/y/z` are Cartesian components of a direction vector — a tree has to
reconstruct the geometry from three correlated columns. Give it the geometry:

```python
d["gaze_az"]   = np.degrees(np.arctan2(d.EyeGaze_x, d.EyeGaze_z.clip(lower=1e-6)))
d["gaze_el"]   = np.degrees(np.arcsin(d.EyeGaze_y.clip(-1, 1)))
d["gaze_disp"] = np.sqrt(d.std_EyeGaze_x**2 + d.std_EyeGaze_y**2)
f["scene_x_sd"] = g.gaze_x_scene_mean.std()      # scanpath spread in the scene
f["scene_shift"] = np.sqrt(g.gaze_x_scene_mean.diff()**2 + g.gaze_y_scene_mean.diff()**2)
```

`scene_x_sd` reaches |r| = 0.251 (top 4): **gaze concentration** carries the signal,
while the raw scene *mean* that topped the S0 correlation table is mostly a
head-pose artifact.

### 6. The classic load ratio

```python
f["fix_per_sacc"] = g.fixation_count.sum() / g.saccade_count.sum().replace(0, np.nan)
```

|r| = 0.229. Under load, gaze shifts toward fewer, longer fixations — the ratio
expresses that directly, and neither count does alone.

### 7. Z-score features within participant — the step that flips the sign

This is what turns S1 (−0.062) into S2 (**+0.107**), and it is legitimate because
it uses no labels:

```python
Fz = F.groupby(level="pid").transform(lambda s: (s - s.mean()) / s.std())
```

Each participant becomes their own control, which removes the between-subject
component in *feature* space the way the EDA showed rater bias needs removing in
target space. In deployment this corresponds to a per-user calibration window —
collect a few minutes of that person's eye data, then score relative to their own
baseline. Budget for that in the product design; it is not optional here.

---

## What to drop

| Drop | Why |
|---|---|
| `timestamps_start_ms`, `timestamps_end_ms` | device uptime; identifies the session perfectly (η² = 1.000) |
| `Participant_ID`, `tasks` | identifiers, not inputs |
| `SaccRate` | exactly `4 × saccade_count` for every row |
| `lux_interpolated` | task fingerprint — classifies the task at 39.5% vs 20% chance; a back-door path to the target |
| `GTE` | 0.0 in 98.8% of windows, 9 distinct values |
| `blink_flag_any` | fires in 0.4% of windows; under-triggering ~10× |
| participant 07 (all 5 sessions) | 72–97% of windows have gaze behind the head |

Removing lux costs nothing on the honest split (it *helped* the leaky one) — this
is exactly what a leakage-driven feature does.

---

## Keep the feature count small

n = 130. S0's 51 features overfit badly, and adding them to the task label dropped
R² from 0.304 to 0.066. The 29-feature S1/S2 set is already at the edge; 15–20 is
safer. Two guardrails:

- **Regularise or stay shallow.** RidgeCV on the same features (S6) underperforms
  boosting here, so the non-linearity is real — but constrain depth and use
  `min_samples_leaf` in the tens, not the default.
- **Select inside the CV loop.** The top-k rows I ran (S7) picked features using
  correlations computed on *all* 130 samples, so their numbers are optimistically
  biased and I have excluded them from the results table. Any real selection must
  sit inside a `Pipeline` fitted per fold.

---

## Is more feature engineering the answer? No — and here is the evidence

Two different questions hide inside "is the data enough":

### Can a model extract signal? — Yes, and it has plateaued

Learning curve over the number of **training participants** (S2 feature set,
tested on held-out people, 8 random subsets per point):

| Training participants | Within-participant ρ | SD across subsets |
|---|---|---|
| 8 | +0.205 | 0.125 |
| 12 | +0.184 | 0.162 |
| 16 | +0.194 | 0.145 |
| 20 | +0.232 | 0.159 |
| 25 | +0.375 | *0.528* |

**Flat from 8 to 20 participants.** The 25-participant point is not trustworthy —
with 25 in training only one person is held out, so each replicate is a
single-participant test, hence the 0.528 SD. Read the reliable range: the curve is
level.

That is the important result. **More participants of the same kind would not
substantially improve extraction, and neither will more features.** The
S0 → S2 gain (+0.30 R²) already collected what construction can offer; past that,
every added block made things worse (RidgeCV −, top-k −, features added to the task
label 0.304 → 0.066). At 130 labels the model is capacity-limited, not
information-limited. The ceiling is **measurement quality and construct validity**,
not engineering effort.

### Can you report the result precisely? — No, badly underpowered

Leave-one-participant-out, 26 folds:

- Per-participant ρ: **mean +0.412, SD 0.446**
- 95% CI on the mean: **[+0.241, +0.583]** — width **0.343**
- **81% of participants (21/26) show ρ > 0**

The direction is robust; the magnitude is not. To narrow that CI to ±0.10 would
take **77 participants**; to ±0.05, **306**.

### What this means for the write-up

| Claim | Supported? |
|---|---|
| "Eye-movement features carry a consistent within-person signal about workload" | **Yes** — 21/26 participants positive |
| "Logistic regression and XGBoost differ / do not differ by X" | **Yes**, if you report CIs and treat overlapping intervals as inconclusive |
| "The within-person effect is ρ ≈ 0.41" | **No** — CI spans 0.24–0.58 |
| "This system detects cognitive load" | **No** — loses to the task label alone |
| "More features would improve this" | **No** — measured and rejected |

Report the SD across the 26 folds everywhere, not just the pooled headline, and
state the CI width as a limitation. That is a defensible thesis result: a
well-executed comparison with an honest power statement beats an overclaimed
effect.

## Highest-value additions not in the current release

1. **Pupil diameter.** `01_Metadata/pupil.py` already computes
   `average_PupilDiameter_mm_norm` (lux-normalised) and writes
   `*_pupil_final.csv` — **but no pupil files ship with the dataset.** Pupil
   dilation is the strongest established ocular index of cognitive load; the
   normalisation for the lux confound is already implemented. Obtaining these
   files would likely outweigh every transformation above.
2. **Fixation *durations*.** Only counts are shipped. Mean and SD of fixation
   duration are more standard load indices than count, and
   `metrics_extraction.py` detects the fixations already — they just are not
   exported per-event.
3. **Recomputed `GTE` over 2–5 s windows.** The metric is sound; the 250 ms window
   is too short for a grid transition to occur, which is why it is 98.8% zero.
4. **A working blink channel.** Blink rate and duration are established load
   indices, and the shipped flag is unusable.

---

## Reference implementation

The full scored experiment is `fe.py` in the analysis scratchpad; the feature
builder is the `build()` function. Wrap the final version in an sklearn
`Pipeline` (imputer → scaler → model) so the per-participant z-scoring and any
selection are fitted per fold, then hand off to model evaluation with
**MAE + within-participant Spearman ρ** and both baselines (global mean, and
task-label-only at R² = 0.304) reported alongside.

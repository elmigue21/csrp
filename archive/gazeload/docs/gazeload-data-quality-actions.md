# GAZELOAD — Data-Quality Actions for the Current Pipeline

Actionable deltas between the EDA ([gazeload-eda.md](gazeload-eda.md)) and what
`docs/methodology.md` + `src/` currently implement.

**Validation of scope:** rebuilding 30 s epochs with `EPOCH_MS = 30000` and
`MIN_WINDOW_COVERAGE = 0.8` reproduces **exactly 1,087 epochs**, matching
methodology.md §3.3. The findings below therefore apply to the pipeline as built,
not to a different reading of the data.

## What the methodology already gets right

The EDA independently confirms every one of these, which is worth stating plainly
because they are the decisions that usually go wrong:

- n = 130 independent labels, not 135,499 windows (§6.1) — confirmed exactly.
- Leave-one-participant-out, no random split (§6.2) — my measurement: a random-row
  split flips R² from −0.263 to +0.047, i.e. it inverts the sign, not just the
  magnitude.
- Participant identity from the filename, `Participant_ID` corrupted with `Hedi`
  and `Wassim` in six recordings (§3.1) — confirmed, same six files.
- `tasks` excluded as leaky (§3.2) — confirmed; task label alone reaches R² = 0.304.
- Structural missingness kept as `*_missing_frac` rather than imputed (§3.4) — confirmed
  mechanism.
- GTE zero in 98.8% of windows, blink flag firing in 0.43% (§3.3) — confirmed to
  the decimal; 250 ms is too short, 30 s epochs are the right fix.
- Per-participant standardisation is label-free and legitimate, but implies a
  calibration recording (§5) — confirmed, and it is the single largest source of
  gain I measured (R² −0.062 → +0.107).
- Pupillometry absent from the release (§2.2) — confirmed; `pupil.py` writes
  `*_pupil_final.csv` but no such file ships.

## Five findings not currently handled

### 1. `saccade_amplitude_degree_mean` does not measure saccade amplitude

No plausibility filter exists anywhere in `src/data.py`, so `AGG_BASE` averages
over a **bimodal mixture**: 21% of windows with a saccade carry impossible
kinematics (median 359°, up to 3,516°; 69% of them above 900°/s). The two
populations are the same windows, so this is one failure mode.

**Root cause — a missing termination criterion in the authors' detector.** In
`01_Metadata/metrics_extraction.py`, `detect_saccades` opens an event when angular
velocity exceeds `SAC_VEL_THR = 35`°/s and closes it only when velocity drops back
below that threshold. There is a `SAC_MIN_MS = 15` floor but **no maximum-duration
criterion**. Amplitude is then accumulated as `np.sum` of per-sample angular steps
— *path length along the trajectory*, not net displacement — so any stretch that
never falls back under 35°/s (smooth pursuit, tracking noise, post-blink recovery)
becomes one enormous run-on "saccade" whose amplitude grows without bound. That
also explains why the impossible amplitudes and impossible velocities are the same
windows: `peak_vel` is a max over the same over-long run.

Confirmed in the data via implied duration (`amplitude / peak_velocity`):

| | Implied duration, median | p95 | > 150 ms |
|---|---|---|---|
| Plausible events (≤ 90°) | **53.8 ms** — textbook | 298 ms | 13.2% |
| Artifact events (> 90°) | **283.3 ms** | 805 ms | **68.8%** |

Real saccades last under ~100 ms. The artifact population implies events 5× longer.

**The rest of the channel is sound.** The *main sequence* — the standard validity
test in the eye-tracking methodology literature, where peak velocity must rise
predictably with amplitude — holds well once the run-on events are removed, and
fails inside them:

| Subset | log-log R² | Spearman ρ |
|---|---|---|
| Plausible saccades (≤ 20°, ≤ 900°/s) | **0.735** | **+0.858** |
| Artifact events (> 90°) | 0.158 | +0.350 |

Median peak velocity climbs monotonically across amplitude bins
(0–2° → 49.5°/s, 2–4° → 72.4, 4–6° → 101.4, 6–8° → 125.4, 8–10° → 144.3,
10–15° → 169.2, 15–20° → 193.0). That is a genuine main sequence.

So this is **not** "the saccade data is unusable" — it is "79% of it is valid and
the other 21% is a separable, mechanically explained detector failure." After
filtering you have a defensible saccade channel, and the filter is justified by a
published validity criterion rather than by my judgement of what looks plausible.

At the 30 s epoch scale:

| Feature | Raw (as built) | After cleaning | corr(raw, cleaned) |
|---|---|---|---|
| `saccade_amplitude_degree_mean` | median **78.9°** | median **10.97°** | **−0.250** |
| `saccade_velocity_degree/s_mean` | median **352.1°/s** | median **119.8°/s** | **−0.024** |

The correlations are the point. The raw epoch feature is **not a noisy version** of
mean saccade amplitude — at r = −0.25 it is anti-correlated with it, behaving as an
artifact-rate proxy with the sign reversed. For velocity, r = −0.02 means no
relationship at all. **479 of 1,087 epochs (44.1%) have a mean amplitude that is
itself anatomically impossible** (> 90°).

Any Chapter 4 statement attributing importance to "saccade amplitude" is currently
describing tracking dropout rate, not oculomotor behaviour.

**Fix** — in `build_epochs`, before aggregating:

```python
art = (df["saccade_amplitude_degree"] > 90) | (df["saccade_velocity_degree/s"] > 900)
df["saccade_amplitude_degree"] = df["saccade_amplitude_degree"].where(~art)
df["saccade_velocity_degree/s"] = df["saccade_velocity_degree/s"].where(~art)
```

Then add `artifact_frac = art.mean()` per epoch as a derived feature — the artifact
rate varies 4.7%–44.7% across participants and is worth keeping explicitly rather
than leaving it smuggled inside the amplitude mean. Note this raises
`sacc_missing_frac`, so recompute it after the filter.

The 90° / 900°/s cut is a proxy for the real criterion, which is duration. If you
re-derive events from the raw recordings, add `SAC_MAX_MS = 150` to
`detect_saccades` instead — that is the standard fix and it removes the artifacts
at source. Filtering the released windows is the option available without the
`.vrs` files.

**Unresolved: peak velocity may be undersampled.** `preprocess_gaze` linearly
interpolates gaze onto a `TARGET_HZ = 120` grid via `np.interp`. Linear
interpolation cannot recover velocity detail finer than the *native* sampling
interval, and Aria's MPS eye-gaze output is documented at a substantially lower
rate than 120 Hz. If so, `SAC_MIN_MS = 15` events (two samples on the 120 Hz grid)
are interpolation artifacts rather than measured saccades, and absolute peak
velocities are compressed — consistent with the observed 49.5°/s median for 0–2°
saccades, which is low against published norms. The monotonic main sequence says
the *ordering* is preserved, so relative comparisons survive. Verify the native
rate against the raw recordings before quoting absolute velocities; if it is well
below 120 Hz, treat velocity as ordinal and say so in the limitations.

### 2. Three of the 37 features are exact duplicates

`SaccRate == 4 × saccade_count` for every row in the dataset, so with both in
`AGG_BASE`:

| Pair | Epoch-level correlation |
|---|---|
| `SaccRate_mean` ↔ `saccade_count_mean` | **1.000000** |
| `SaccRate_std` ↔ `saccade_count_std` | **1.000000** |
| `fdi_missing_frac` ↔ `fixation_zero_frac` | **1.000000** (identical in 1,087/1,087 epochs) |

The third arises because `FDI` is NaN exactly when `fixation_count == 0`, which
makes those two derived features the same column computed two ways.

This matters specifically for the study's comparison. Perfect collinearity makes
**logistic regression coefficients arbitrary** — the shared effect is split across
the duplicate pair in a way that depends on the solver, not the data — and it
**splits XGBoost gain and SHAP attribution** between interchangeable columns,
halving the apparent importance of saccade rate. Both are Chapter 4 deliverables
(`importance.py`).

**Fix** — drop `SaccRate` from `AGG_BASE` and drop `fdi_missing_frac` from
`derived`. The feature count becomes 34, and the interpretation becomes
well-defined.

### 3. Invalid gaze vectors are not filtered, and participant 07 is unusable

`EyeGaze_z < 0` — a unit gaze vector pointing behind the head — occurs in **9.0%**
of all windows, and **15.5%** of gaze vectors are not unit-length. Neither is
checked.

**Participant 07 is the extreme case: all five recordings have 72–97% invalid
windows** (median invalid fraction 0.97 across their 47 epochs). Their gaze
direction features are noise, yet they carry equal weight as one of 26 LOPO folds
— roughly 3.8% of the outer-loop estimate, and one fold's per-participant score in
the mean ± SD table.

Excluding participant 07 costs **47 of 1,087 epochs (4.3%)**. Across the corpus,
72 epochs (6.6%) exceed 50% invalid gaze.

**Fix** — add `gaze_invalid_frac` and `gaze_norm_bad_frac` as epoch features, and
pre-register exclusion of participant 07 as a documented quality criterion in §3.
State the threshold before seeing results so it is not a post-hoc choice. Report
the headline both ways if the exclusion changes anything.

### 4. Lighting is in the headline configuration

Config A — the `[HEADLINE]` — sets `include_lux=True`. Lighting is not a weak
nuisance variable here; it is close to a task label:

- Mean session lux by task: **T1 426, T2 212, T3 420, T4 188, T5 265**.
- **`lux_interpolated` alone classifies the task at 39.5% against a 20% chance
  baseline** — essentially all of the 40.7% reachable with the full feature set.
- Its direct correlation with the rating is negligible (ρ = 0.035, p = 0.70), so a
  correlation scan will not flag it.

That last point is why this is easy to miss. Because `tasks` is excluded for
predicting the label at AUC 0.93, and lighting reconstructs the task at nearly
twice chance, **lux is a back-door path to the same information the front-door
exclusion was designed to block.** The existing B/F ablations measure this, which
is good — but the ablation is currently framed as a robustness check on the
headline rather than as the headline.

**Fix** — promote a lux-free configuration to `[HEADLINE]` and report the
lux-inclusive one as the contrast, reversing the current framing. The §3.2
exclusion table should list `lux_interpolated` with the reasoning above.

### 5. One prose inaccuracy to correct

methodology.md §3.4 and the comment in `src/data.py` both state that `FDI` is
missing when `fixation_count < 2`. It is missing when **`fixation_count == 0`**:

```
FDI NaN & fixation_count == 0  : 15,822
FDI NaN & fixation_count == 1  : 0
FDI notnull & fixation_count == 1 : 17,013
```

The **code is correct** (it uses `FDI.isna()` directly, not a count threshold) — only
the stated mechanism is wrong. It needs fixing because it is an empirical claim in
a methods chapter, and because it is what makes finding #2's duplicate
non-obvious. Likewise §3.3 describes `fdi_missing_frac` as "fewer than two
fixations"; it is "no fixation", which is `fixation_zero_frac` under another name.

## Suggested order

1. **#2 and #5** — remove duplicates, fix the prose. Minutes, and #2 changes
   Chapter 4's importance tables.
2. **#1** — plausibility filter. Largest effect on what the saccade features mean.
3. **#4** — swap which configuration is the headline. A framing change, already
   implemented.
4. **#3** — gaze validity features and the participant 07 decision. Pre-register
   the threshold.

None of these change the study design; #1 and #2 change what the results *mean*,
and #4 changes which number is quoted as the finding.

## Also worth noting in limitations

Two items from the EDA that affect no code but belong in the write-up:

- **The event log lost its trial labels.** The only `Action` strings in the corpus
  are `action`, `Start of experiment`, `End of experiment`, and
  `Start/End action: Error`. The literal `"Error"` stands where an action name
  should be in **78 of 130 recordings** (tasks 3–5 only). Trial-level segmentation
  is therefore impossible with the released data — worth stating, since it forecloses
  the finer-grained analysis a reader might ask for.
- **Stream durations disagree.** The eye stream is > 30 s shorter than the event log
  in 12 recordings, worst `13_2` at **425 s missing** (327 s of eye data against a
  752 s log). Three recordings have duplicate start/end markers (`13_2`, `13_3`,
  `18_1`). The 80% coverage rule silently drops the affected epochs rather than
  flagging the recordings, so the loss is currently invisible in the outputs.
- **Demographics.** The distribution is titled "…for Men"; `Tasks_Rating.xlsx`
  records 16 male and 10 female participants. Resolve before citing the sample
  composition.

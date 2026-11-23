# Post-defense data processing

**Status:** draft, 2026-11-09. Written after the proposal defense. Decisions marked
**To decide** are still open. The extraction is built (`src/run_extract.py`) and has run
once with the current rules; its outputs are in `outputs/final/` (§7.1).

**Related files:**
- `methodology-colet.md`: the defended version. This doc replaces its §4 (preprocessing),
  §6 (features) and the "fixed feature set" rule. Everything else there still stands
  unless this doc says otherwise.
- `researcher-notes.md`: background notes referred to as N1, N21, etc.

---

## 0. Why this doc exists

**Panel feedback:** our dataset preparation was judged incorrect. Each eye signal was
squeezed into one or two numbers (for example, pupil diameter became only `pupil_mean` and
`pupil_sd`). The panel asked us to **flatten each signal into many summary statistics**:
mean, SD, median, min, max, percentiles, and so on.

**The underlying problem:** the earlier flattening was done for us, and we could not explain
it when asked. So this doc has two jobs:
1. Describe the new processing: clean the data again, then flatten it into as many features
   as possible.
2. Make every step something the team chose and can explain in plain words. Each step has a
   **"How to explain it"** line for exactly that purpose.

**The new idea in one sentence:** create as many features as possible, then let the
feature-importance analysis tell us which ones matter, and test models with the top 10, 20,
50 or 100 features.

## 1. Scope

**In scope (this doc):** raw COLET files → cleaned signals → flattened features →
one feature table (A1/A4 only, no per-person standardization, §5), plus a log of everything dropped or changed. All
final outputs go in `outputs/final/` (§7.1).

**Out of scope (modeling stage, later doc):** choosing the top-N features, model tuning,
LOPO evaluation, SHAP. Section 8 keeps notes for that stage so they aren't lost.

**Unchanged from the defended version:**

| Item | Value |
|---|---|
| Dataset | COLET (Zenodo 7766785 v3), 47 released participants |
| Labels | A1 = low load, A4 = high load; A2 and A3 only for the manipulation check (not in the feature table, §5) |
| Unit | One row per whole activity (no time windows; activities are too short, N5) |
| Main analysis size | 44 participants × 2 activities = **88 rows**. P06 and P17 drop out at step C3; P31's files are a byte-identical copy of P28's, so P31 is left out (R5-1, `config.DUPLICATE_RECORDING_IDS`) |

**Note:** `methodology-colet.md` and the thesis draft still say 45 participants / 90 rows in
places. The pipeline has used 44 / 88 since the P31 duplicate was found. Those documents need
the same correction.

## 1a. What the raw data looks like

Each recording (one participant × one activity) is **three tables**. Example: P01, A1
(34 s).

**Gaze table: 8,205 rows × 21 columns.** One row per moment, about 242 per second.

| gaze_timestamp | confidence | gaze_normal0_x | gaze_normal0_y | gaze_normal0_z | gaze_normal1_x | gaze_normal1_y | gaze_normal1_z | … |
|---|---|---|---|---|---|---|---|---|
| 5410.5517 | 0.9995 | −0.2353 | −0.4202 | 0.8764 | 0.2776 | −0.3913 | 0.8774 | … |
| 5410.5558 | 0.9997 | −0.2348 | −0.4203 | 0.8765 | 0.2776 | −0.3913 | 0.8774 | … |

| Columns | What they are | Used? |
|---|---|---|
| `gaze_timestamp` | Time in seconds | Yes |
| `confidence` | How sure the tracker is (0–1) | Yes (C2) |
| `gaze_normal0/1_x/y/z` | Direction each eye points | **Yes: gaze x/y, gaze speed, fixations, saccades** |
| `norm_pos_x/y` | Gaze position in the head camera's image | No (camera image, not the screen) |
| `gaze_point_3d_x/y/z` | Estimated 3D point being looked at | No (depth is broken, R4-1) |
| `eye_center0/1_3d_x/y/z` | Where each eyeball sits | No |
| `world_index`, `base_data` | Internal bookkeeping | No |

**Pupil table: 16,406 rows × 34 columns.** Each measurement appears twice: once from
the tracker's **2d** method (pixels) and once from its **3d** method (mm). Eye 0 and eye 1
alternate.

| pupil_timestamp | eye_id | method | confidence | diameter | diameter_3d | model_id | … |
|---|---|---|---|---|---|---|---|
| 5410.5538 | 1 | 2d c++ | 1.0000 | 40.76 | — | — | … |
| 5410.5538 | 1 | 3d c++ | 0.9996 | 40.76 | **3.94** | 72 | … |
| 5410.5579 | 0 | 3d c++ | 0.9997 | 38.73 | **3.78** | 53 | … |

| Columns | What they are | Used? |
|---|---|---|
| `pupil_timestamp`, `eye_id` | Time, and which eye | Yes |
| `method` | 2d or 3d measurement | Yes (3d rows only) |
| `confidence` | How sure the tracker is | Yes (C2) |
| `diameter_3d` | **Pupil size in mm** | **Yes: the pupil signal** |
| `model_id` | ID of the tracker's 3D eye model | Only to count refits (QC) |
| `diameter` | Pupil size in camera pixels | No (depends on camera distance) |
| `ellipse_*`, `sphere_*`, `circle_3d_*`, `theta`, `phi`, `projected_sphere_*`, `model_confidence`, `norm_pos_x/y` | The tracker's internal eye-model geometry | No |

**Blinks table: one row per blink** (P01 A1 has 2), 10 columns. We use `start_timestamp`
and `end_timestamp`, which become blink rate and blink duration.

**In short:** of the 65 raw columns, we use about 14. Cleaning (§3) and flattening (§4)
turn these three tables into **one row** of features per recording.

## 1b. Order of work (decided 2026-11-09)

1. **Extract first.** Run the full extraction with the current cleaning rules (C1–C10 as
   written below) and **every** signal, including blink duration. Nothing is dropped at this
   stage except what the cleaning rules already remove.
2. **Then decide.** Using the extracted table and the processing log, the team goes through
   C1–C10 and D1–D4 and marks each Keep or Change.
3. **Re-run** with any changes. The log records what each change did.

## 2. How to read each step

Every cleaning step and every feature decision uses the same layout:

- **What:** what the step does.
- **Why:** the problem it solves, with a source where we have one.
- **Current rule:** what the defended pipeline did (our starting point, not a given).
- **Effect:** what it removed or changed, in numbers (filled in from the processing log once
  the code runs).
- **How to explain it:** one or two sentences you could say to the panel.
- **Status:** **Keep** (team confirmed), **Change** (new rule written in), or **To decide**.

## 3. Cleaning (raw data → clean signals)

All steps started as **To decide**. The team reviewed them on 2026-11-11 and kept all ten
as they are (C1–C10). Thresholds must be set **without looking at A1 vs A4 differences**. They are
chosen from data quality alone, before any model runs.

### C1. Line up the files in time
- **What:** cut the pupil and blink data to the time range covered by the gaze data.
- **Why:** some files contain stray samples recorded long after the activity ended (P18 A3/A4
  have pupil samples about 2 hours later).
- **Current rule:** trim pupil and blinks to the first and last gaze timestamp. A blink is
  kept only if it starts and ends inside that range.
- **Effect (88 analysis recordings, `log_recordings.csv`):**
  - Pupil rows: 2,118,776 → 2,105,748 (13,028 removed, 0.6%). 12,868 of them come from one
    recording, P18 A4: its gaze covers 51 s, but its pupil file carries samples from 12 to
    115 minutes after the activity ended. The other 60 recordings lost 4 rows each, which is
    one camera frame at the very edge (2 eyes × 2 pupil methods).
  - Blinks: 915 → 909. All 6 removed blinks are P18 A4's, logged 17 to 115 minutes after the
    activity ended.
  - Outside the analysis, P18 A3 lost 24,070 pupil rows the same way.
- **How to explain it:** "We only keep data recorded while the activity was happening. In one
  recording, the pupil file contained samples from up to two hours after the activity ended,
  and those were removed."
- **Status:** **Keep** (team, 2026-11-11).

### C2. Mark low-confidence samples as invalid
- **What:** each sample has a confidence score from the eye tracker (0 to 1). Samples below a
  cut-off are treated as missing.
- **Why:** low confidence means the camera could not see the pupil properly, so the value is
  unreliable.
- **Current rule:** confidence < 0.8 is invalid (sources PP5–PP8; rated STRONG in
  `methodology-colet.md` §8b).
- **Effect (88 analysis recordings, 1,059,327 gaze samples):**
  - 6.7% of gaze samples are below 0.8 (pupil: 6.8%). Per recording: median 2.2%, highest
    23.1% (P44 A4).
  - 91.5% of samples sit at 0.9 or above, and the rest mostly sit very low (1.7% are exactly 0).
    So the exact cut-off barely matters:

    | Cut-off | Gaze samples marked invalid | Highest recording |
    |---|---|---|
    | 0.6 | 6.3% | 22.7% |
    | 0.7 | 6.4% | 22.8% |
    | **0.8 (current)** | **6.7%** | **23.1%** |
    | 0.9 | 8.5% | 30.8% |

  - A4 loses more samples than A1 (median 6.9% vs 0.9%). That is why the valid-sample share is
    reported but never used as a feature (C10): it would give away the label.
- **How to explain it:** "The tracker tells us how sure it is about each sample. We ignore
  moments when it couldn't see the eye clearly."
- **Status:** **Keep** (team, 2026-11-11).

### C3. Drop recordings that are mostly bad
- **What:** if too much of a recording is invalid, drop the whole recording. A participant
  who loses A1 or A4 drops out of the main analysis.
- **Why:** statistics from a recording that is mostly missing don't describe the person.
- **Current rule:** drop if more than 35% of gaze samples are invalid (Nenna 2023, HR4).
  This removes P06 A3, P06 A4 and P17 A4, so P06 and P17 drop out.
- **Effect:** 3 of 184 recordings dropped (P31's copies not counted): P06 A3 (57% invalid),
  P17 A4 (49%) and P06 A4 (39%). The worst recording kept is P17 A3 at 24%. Nothing sits
  between 24% and 39%, so any cut-off from 25% to 38% drops exactly the same recordings:

  | Cut-off | Recordings dropped | People who leave the analysis |
  |---|---|---|
  | 25% / 30% / **35% (current)** | 3 | P06, P17 |
  | 40% | 2 | P17 |
  | 50% | 1 | none |

- **How to explain it:** "If more than a third of a recording was unreadable, we didn't
  trust it."
- **Status:** **Keep** (team, 2026-11-11).

### C4. Measure usable recording time
- **What:** work out how long each recording actually had usable data, leaving out gaps
  longer than 1 s.
- **Why:** rates ("blinks per minute") must be divided by the time we could actually see the
  eye, not the raw clock time.
- **Current rule:** gaps over 1 s are excluded (P18 A3: 12 s; P18 A4: 3.7 s; P16 A4: 1.4 s).
- **Effect (88 analysis recordings):** only 2 recordings have a gap over 1 s: P18 A4 (3.7 s of
  51 s) and P16 A4 (1.4 s of 31 s). Together 5.1 s out of 4,376 s.
- **How to explain it:** "When the tracker lost the eye for more than a second, we don't
  count that time."
- **Status:** **Keep** (team, 2026-11-11).

### C5. Clean the blink list
- **What:** use COLET's own blink events, keep only realistic ones, and blank out gaze and
  pupil during blinks.
- **Why:** very short or very long "blinks" are usually tracking errors. During a blink the
  eye is closed, so gaze and pupil values are meaningless.
- **Current rule:** keep blinks 50–500 ms; merge blinks less than 100 ms apart (PP6, PP10,
  CF2).
- **Effect (88 analysis recordings):** 909 blinks after C1 → 768 kept.
- **How to explain it:** "A real blink lasts between a twentieth and half a second. Anything
  else we treat as noise, and we ignore the eye data while the eye is shut."
- **Status:** **Keep** (team, 2026-11-11).

### C6. Fill short gaps in gaze
- **What:** fill short gaps in the gaze signal by drawing a straight line between the points
  on either side. Longer gaps stay missing.
- **Why:** tiny dropouts are normal and filling them keeps the signal continuous. Long gaps
  can't be guessed.
- **Current rule:** fill gaps shorter than 75 ms (PP7, PP11).
- **How to explain it:** "If the tracker blinked out for a split second, we connect the dots.
  Anything longer we leave empty."
- **Status:** **Keep** (team, 2026-11-11).

### C7. Clean the pupil signal
- **What:**
  1. Use the 3D pupil diameter in mm.
  2. Remove values outside 1.5–9 mm (physically impossible).
  3. Remove sudden jumps that are faster than a pupil can really change (MAD speed filter).
  4. Average the two eyes.
  5. Fill gaps up to 250 ms.
  6. Smooth with a 4 Hz low-pass filter.
- **Why:** raw pupil data contains spikes from blinks, eyelid edges and camera errors (PP9,
  PP11, NM7, CF2).
- **Current rule:** as listed above. We also count how often the tracker re-fitted its 3D eye
  model per recording (reported, not used as a feature).
- **How to explain it:** "We removed pupil sizes that are physically impossible or change
  faster than a real pupil can, then smoothed out the remaining jitter."
- **Status:** **Keep** (team, 2026-11-11).

### C8. Convert gaze to angles
- **What:** turn each eye's gaze direction into angles (left/right and up/down, in degrees)
  and average the two eyes. If only one eye is valid, the sample counts as missing.
- **Why:** COLET's 3D gaze point is unreliable. Its depth is wrong and some points are behind
  the camera (R4-1). Each eye's own direction is reliable. Switching to one eye when the
  other drops out would create fake jumps (N25).
- **Current rule:** as described, using `gaze_normal0/1`.
- **Effect (88 analysis recordings):** both eyes were valid in 99% of samples in the median
  recording (lowest 88%).
- **How to explain it:** "We measure where each eye points as an angle and average the two.
  We don't use the tracker's estimate of how far away the gaze lands, because that estimate
  was broken."
- **Status:** **Keep** (team, 2026-11-11).

### C9. Detect fixations and saccades
- **What:** split the gaze signal into **fixations** (the eye is still) and **saccades** (the
  eye jumps), using eye speed.
- **Why:** COLET doesn't include these events, so we have to detect them ourselves (N1).
- **Current rule:**
  - Speed above 45°/s = saccade; below = fixation (I-VT, the method the COLET paper cites).
  - Fixations must last at least 55 ms.
  - Speeds above 1000°/s are treated as errors.
  - Sanity check per activity: median fixation 150–400 ms, and the saccade:fixation count
    ratio between 0.8 and 1.25.
  - If the check fails in **any** activity, all fixation and saccade features (S4–S9 and
    the fixation and saccade rates) are removed **for everyone**, not just for the failing
    recordings. Removing them only where the check failed would leave blanks that could line
    up with A4, which is the same leak as D1. This is the defended rule
    (`methodology-colet.md` §5 step 7). The defended run passed this check.
- **How to explain it:** "When the eye moves faster than 45 degrees per second it's jumping;
  slower than that, it's looking at something. This is the standard method, and the same one
  the COLET authors used."
- **Effect (88 analysis recordings):** the sanity check passes in all four activities (median
  fixation 212–235 ms; saccade:fixation ratio 0.93–0.99), so all fixation and saccade features
  are kept. 12,237 fixations and 11,675 saccades in total.
- **Status:** **Keep** (team, 2026-11-11).

### Pupil quality: kept and reported as a limitation (decided 2026-11-11)
- C3 looks only at gaze confidence. 7 analysis recordings keep less than 80% of their pupil
  data after C7, and all 7 are A4.
- **P18 A4 (38% usable pupil):** the pupil data was saved less often than usual, so many gaps
  are longer than the 250 ms fill limit. Values look believable (about 2.8 mm).
- **P44 A4 (43% usable pupil):** the right eye reads about 1.7 mm against 2.75 mm for the left.
  A 1 mm difference between eyes is not realistic, so the 3D eye model for that eye was
  probably wrong. Half of its values fall below the 1.5 mm floor.
- This matters more without per-person standardization (§5), because the model sees raw pupil
  sizes.
- **Options:** (A) keep and report as a limitation; (B) drop P44 entirely (43 participants);
  (C) blank only the bad pupil features. C is not recommended: the blanks would appear only in
  A4, the same leak as D1.
- **Between-eye check, all 88 recordings** (`outputs/work/pupil_between_eye_check.csv`): P44 A4
  is **not** unique. The left and right eye's median pupil size differ by more than 1 mm in
  7 recordings (5 A4, 2 A1). The largest is P36 A1: 3.6 mm vs 7.3 mm. The typical difference
  is 0.27 mm. So "the eyes disagree" is common in this dataset and cannot single out P44 on its
  own. What is unusual about P44 A4 is that 45% of its right-eye values fall below the 1.5 mm
  floor (next worst: P21 A4 at 12%; every other recording under 10%).
- **How to explain it:** "C3 checks gaze quality only. Seven high-load recordings kept less
  than 80% of their pupil data, and in one (P44, high load) the tracker misjudged the size of one
  eye. The cleaning step already removes physically impossible pupil sizes, so what remains is
  usable. We kept these recordings and report this as a limitation, because there was no
  quality rule that singled them out fairly."
- **Status:** **Keep and report** (team, 2026-11-11). No participant is dropped for pupil
  quality; P44 stays.

### C10. Quality report (not a feature)
- **What:** the valid-sample percentage for every recording, reported by activity.
- **Why:** A4 recordings lose more data than A1 (median 7.3% vs 0.9% invalid). Readers need
  to see that. It is **never** used as a feature, because the model would learn "more missing
  data = high load".
- **How to explain it:** "High-load recordings lose more data, so 'how much was lost' would
  give away the answer. We report it but never give it to the model."
- **Status:** **Keep** (team, 2026-11-11).

## 4. Flattening (clean signals → features)

### 4.1 The two kinds of signal

- **Continuous signals** have a value at every moment (about 240 per second). We summarize
  all of a recording's values.
- **Event signals** have one value per event (one duration per fixation, one amplitude per
  saccade). We summarize all of a recording's events.

### 4.2 Signals to flatten

| # | Signal | Kind | Unit | Was in the defended set as |
|---|---|---|---|---|
| S1 | Pupil diameter | Continuous | mm | `pupil_mean`, `pupil_sd` |
| S2 | Gaze angle, left/right | Continuous | deg | `gaze_sd_x` |
| S3 | Gaze angle, up/down | Continuous | deg | `gaze_sd_y` |
| S4 | Gaze speed | Continuous | deg/s | (not used) |
| S5 | Fixation duration | Event | ms | `fixation_duration` (mean only) |
| S6 | Saccade amplitude | Event | deg | `saccade_amplitude` (mean only) |
| S7 | Saccade peak velocity | Event | deg/s | `saccade_peak_velocity` (mean only) |
| S8 | Saccade mean velocity | Event | deg/s | dropped (duplicate of S7) |
| S9 | Saccade duration | Event | ms | dropped (tied to S6) |
| S10 | Blink duration | Event | ms | dropped (leak, D-M3); see decision D1 |

**Status:** **Keep this list** (team, 2026-11-11, decision D4). S4, S8 and S9 are back in
because the goal is now "as many as possible". Duplicates are no longer a reason to drop a feature; the importance analysis
deals with them (§8).

### 4.3 Statistics (applied to every signal)

| Statistic | Plain meaning |
|---|---|
| `min` | Smallest value |
| `max` | Largest value |
| `range` | max − min |
| `mean` | Average |
| `median` | Middle value |
| `sd` | Typical distance from the average |
| `var` | SD squared |
| `cv` | SD ÷ mean (spread relative to size) |
| `iqr` | Spread of the middle half (75th − 25th percentile) |
| `p05`, `p10`, `p25`, `p75`, `p90`, `p95` | Value below which 5%, 10%, 25%, 75%, 90%, 95% of values fall |
| `skew` | Lopsidedness: is there a long tail on one side? |
| `kurt` | Peakedness: how many extreme values are there? |

17 statistics in total.

**What `cv` is (coefficient of variation).** It answers: "how much does this jump around,
compared with its usual size?"
- Example: a pupil is usually **4 mm** and wobbles by about **0.4 mm**, so
  cv = 0.4 ÷ 4 = **10%**.
- **Why it's useful:** it lets us compare people with different pupil sizes. A 6 mm pupil
  wobbling by 0.6 mm has a bigger SD, but the same cv (10%), so it is really just as steady.

**`cv` for gaze x/y: kept, tagged `position`** (team, 2026-11-11).
- Gaze direction is measured as "how far left/right (or up/down) of the camera's straight
  ahead". Left is a negative number, right is a positive one. That zero point depends on how
  the headset sat, not on the eye.
- cv divides by the average. When the average is near 0, the result swings wildly:

  | Recording | Average gaze x | SD | cv |
  |---|---|---|---|
  | 1 | +0.1° | 5° | +50 |
  | 2 | −0.1° | 5° | −50 |

- **On our 88 recordings:** gaze left/right cv runs from −48 to +35 and changes sign in 7
  recordings; gaze up/down cv runs from −1.4 to −0.1. Pupil cv, for comparison, runs from
  0.02 to 0.17.
- **Why keep them anyway:** the goal is "everything in", and the importance analysis decides.
  They are tagged `position` with the other gaze-location statistics (§4.6), because like
  those they depend on where the headset's zero sits.
- **If one ranks highly:** report it with the caveat that it may reflect headset position
  rather than load. Without per-person standardization (§5), headset offsets are not removed.
- **How to explain it:** "cv divides by the average. Gaze angles are measured from the
  headset's straight-ahead, so their average, and therefore their cv, depends partly on how
  the headset sat. We kept them, tagged them, and treat them with care if they rank highly."

**Mode is excluded** (decided). These signals are continuous, so an exact value almost
never repeats and the mode is undefined without arbitrary rounding.

### 4.4 Rates (one number per recording, not flattened)

| Feature | Definition |
|---|---|
| `blink_rate` | Blinks per minute of usable time (C4) |
| `fixation_rate` | Fixations per second of time with a valid gaze direction |
| `saccade_rate` | Saccades per second of time with a valid gaze direction |

**Counts and total durations are not features** (for example, "number of blinks" or
"recording length"). A3 and A4 recordings are longer than A1, so counts would mostly measure
how long the activity took, not load (median 51 s in A4 vs 37 s in A1). This is unchanged
from the defended version. **Confirmed by the team, 2026-11-11.**
- **How to explain it:** "A longer activity has more blinks just because it lasted longer, so
  we used blinks per minute instead of the number of blinks."

### 4.5 Naming and size

- **Naming:** `<signal>__<statistic>`, for example `pupil__p95`,
  `fixation_duration__median`, `gaze_x__iqr`.
- **Size:** 10 signals × 17 statistics, plus 3 rates = **173 features extracted** on 88 rows.
  After the team's data-quality decisions (§6, D1–D10), **140 are model features**; the other
  33 stay in the logs with their reason. (Until 2026-11-11, `cv` for gaze x/y was not computed:
  171 extracted, 138 kept.)

### 4.6 Gaze position features: kept, but tagged

- The location statistics of gaze x/y (`mean`, `median`, `min`, `max`, the individual
  percentiles, and `cv`, 22 in all) describe **where on the screen** someone looked. They may reflect the screen
  layout of each activity, or how the headset sat, rather than load.
- **Decided:** keep them, because the goal is "everything in". They get the tag
  `position` in the feature table.
- **Why the tag matters:** if one of them ranks highly, we say in the results that it may
  reflect the task layout, before a panelist says it for us.
- The **spread** statistics of gaze (`sd`, `var`, `iqr`, `range`, `skew`, `kurt`) are not
  tagged. They describe how much the eyes moved around.

## 5. No per-person standardization (changed 2026-11-11)

**Decided 2026-11-11 (team, post-defense): the feature table holds each feature's value as
extracted. There is no per-person z-score, and A2 and A3 are left out.**

- **What the defended version did:** for each feature, it rewrote every value as "how far
  above or below this person's own normal", where the normal was the person's average over
  all four activities (A1–A4). That step is now removed.
- **What a per-person z-score is** (kept here so it can be explained if the panel asks):

  | | A1 | A2 | A3 | A4 | Person's normal (mean) | Typical swing (SD) |
  |---|---|---|---|---|---|---|
  | Person A, raw | 5.0 | 5.2 | 5.4 | 5.8 | 5.35 | 0.34 |
  | Person B, raw | 2.0 | 2.2 | 2.4 | 2.8 | 2.35 | 0.34 |
  | Both, z-score | −1.02 | −0.44 | +0.15 | +1.32 | 0 | 1 |

  The z-score is (value − person's normal) ÷ person's typical swing. For Person A in A1 that
  is (5.0 − 5.35) ÷ 0.34 = −1.02, "about one swing below their own normal". It makes two people
  with different natural pupil sizes look the same.
- **What the feature table holds now:** 88 rows, each participant's A1 (`low`) and A4
  (`high`) recording, with every feature in its own unit (mm, deg, ms, …).
- **Why A2 and A3 are gone:** only A1 and A4 have labels. A2 and A3 were in the table only to
  help work out each person's normal for the z-score. Without the z-score they have no use in
  the feature table. They still count in the C9 event-detection check (§3), which checks that
  the fixation/saccade detector behaves normally on every recording, and they stay in
  `log_recordings.csv` with the reason "A2/A3: no label, not in the feature table".
- **Scaling for the model is a separate thing:** logistic regression needs features on a
  common scale, because its penalty would otherwise treat a feature in mm differently from one
  in deg/s. That scaling is fitted on the training people only, inside each LOPO fold, and
  belongs to the modeling stage (§8). It does not use the test person's data.
- **How to explain it:** "The feature table keeps each feature's measured value for the
  labelled low-load (A1) and high-load (A4) activities. Any scaling the model needs is learned
  from the training participants only."

## 6. Open decisions

**Rules for every keep/drop decision (agreed 2026-11-09):**
1. **Decide before any model runs, and never change a decision because of the results.** If
   a kept feature later ranks highly and looks suspicious, report it with a warning; don't
   remove it. Removing features after seeing results steers the results toward what we
   expect (like grading your own exam and throwing out the questions you got wrong).
2. **Drop only for data-quality reasons:** blanks that follow the condition, values that are
   stuck (the same for almost everyone), or exact copies. Never drop a feature for "not
   predicting well"; the importance analysis inside each fold decides that (§8).
3. **Stuck vs rounded:** drop a feature if almost everyone has the same value (e.g.
   `saccade_duration__min`, 94% identical). Keep it if it is rough but still differs
   between people (e.g. the saccade-duration percentiles, D6b).
- **How to explain it:** "All feature exclusions were decided on data-quality grounds before
  any model was trained, and none was revisited after seeing results."

**D1. Blink duration (S10)**: **Decided 2026-11-09 (team): (a) drop all 17
blink-duration features; keep `blink_rate`.**
- **The problem:** in our current data, A1 recordings have a median of **1 blink**:
  - 13 of 44 have none;
  - 29 have fewer than 2;
  - 38 have fewer than 4.

  A4 has a median of 10.5 blinks. With the 20-value minimum from D2, blink-duration
  statistics would be blank in 43 of 44 A1 recordings and 36 of 44 A4 recordings. Most
  blink-duration statistics would therefore be blank or meaningless in A1 and filled in for
  A4. (Counts confirmed by the first extraction run.) The model would learn "blank = low
  load". That is the same problem that got blink duration dropped before (D-M3), now in 17
  columns instead of one.
- **Evidence from the extraction run** (88 A1/A4 rows, `outputs/final/features/features.csv`):

  | | A4 (high) | A1 (low) |
  |---|---|---|
  | `blink_duration__mean` blank | 0 | 13 |
  | `blink_duration__mean` has a value | 44 | 31 |
  | `blink_duration__kurt` blank | 7 | 38 |
  | `blink_duration__kurt` has a value | 37 | 6 |

  A rule that looks at no eye data at all ("blank → low load, value → high load") is right
  **65%** of the time on `__mean`, **81%** on `__sd` and **85%** on `__kurt`. The blanks alone
  carry the answer.
- **Precise wording:** this is a *shortcut*, not leakage in the strict sense of test data
  reaching training. A blank really means "this person barely blinked", which is real
  behaviour that `blink_rate` already measures. The damage is to feature importance (our
  main deliverable): SHAP would credit "blink duration" for what is really blink *count*.
  Filling the blanks does not help. XGBoost learns "blank → low" by itself, and for logistic
  regression a filled-in value becomes the marker, because only A1 rows get it.
- **How to explain it:** "Blink duration is blank whenever there are too few blinks to
  compute it. With our 20-value minimum it is blank in 79 of 88 recordings, and 8 of the 9
  recordings that do have a value are high load. Even before that rule, a rule using only
  'is it blank?' predicted the load level 85% of the time without any eye data. Keeping it would have
  made the importance analysis credit blink duration for what is really blink count, which
  blink rate already measures."
- **Don't say "dropped because it had many blanks".** A panelist can answer "then fill them
  in". The reason is that the blanks **follow the load condition**.
- **Paper wording (Chapter 3), if (a) is chosen:**
  > "Blink-duration statistics were excluded. They are undefined when a recording contains
  > too few blinks, and this occurred almost exclusively in low-load recordings (e.g., the
  > mean was undefined in 13 of 44 A1 recordings and 0 of 44 A4 recordings; kurtosis in 38
  > vs 7). Missingness was therefore tied to the condition: a rule using only whether the
  > value was missing classified the load level correctly in up to 85% of cases. Imputation
  > does not remove this dependence, and keeping the features would have credited blink
  > duration with information that actually reflects blink frequency, which is captured by
  > blink rate. A similar exclusion was made by Hogervorst et al. (2014)."
- **Options:**
  - (a) Drop all blink-duration statistics. Keep `blink_rate`, which carries the blink
    information without the gap.
  - (b) Keep them and fill blanks with a fixed value. The leak stays: the model learns the
    fill value.
  - (c) Keep only blink-duration `mean` and `median`, and only for recordings with at least
    the minimum number of blinks (D2). Most A1 rows would still be blank.
- **My recommendation:** (a), recorded in the log with these counts.
- **Why this is not the reason the panel rejected:** the old feature set dropped features for
  being "too many" or "duplicates", and the panel rejected that thinking. Blink duration is
  dropped for a different reason: keeping it would make the results **fake**. The model
  would score well by spotting blank cells, not by reading anyone's eyes.
- **Blinking is still in the model:** `blink_rate` is defined for every recording (0
  blinks = rate 0), so it keeps the blink information without the blanks.
- **Sources:**
  - Hogervorst et al. 2014 (EM9): discarded segments where blink duration was undefined.
  - Kapoor & Narayanan 2023: leakage makes ML results look better than they are.
  - Blink duration *is* a known load measure (COLET; Tao 2019), and we say so. The problem
    is only that our A1 recordings barely contain any blinks.
- **How to explain it:** "Blink duration needs blinks. Most low-load recordings had zero or
  one blink. With the 20-value minimum, almost only high-load recordings have a value at all
  (8 of 9). The model would have learned 'a value exists means high load', a shortcut, not a
  finding. We kept blink rate instead, which works
  even with zero blinks."

**D2. Minimum number of values per statistic**: **Decided 2026-11-11 (team): at least 20.**
- **The problem:** some statistics need a minimum number of values to mean anything.
  `skew` needs at least 3, `kurt` at least 4, and the 5th/95th percentiles are just the
  min/max when there are fewer than about 20 values.
- **Measured** (88 analysis rows, from the defended pipeline's
  `outputs/colet/tables/04_feature_table.csv`): fixations and saccades are fine. Every
  A1/A4 recording has at least 56 fixations and 56 saccades. Continuous signals have thousands of samples. So in practice
  this only affects blink duration (D1).
- **Rule:** a signal's statistics are computed only if the recording has at least 20 values
  for that signal; otherwise they are all left blank, and the blanks are counted in
  `log_features.csv` (`flatten.MIN_VALUES`). With blink duration dropped (D1), this changes
  nothing in the current data.
- **How to explain it:** "A statistic like the 95th percentile means nothing if there are only
  a handful of values, so a signal needs at least 20 values in a recording before we describe it.

**D3. min, max and range depend on recording length**: **Decided 2026-11-11 (team): keep, report
as a limitation.**
- **The problem:** the longer you record, the more likely you are to catch one extreme value.
  A4 recordings are longer than A1 (median 51 s vs 37 s). So `max`, `min` and `range` can
  rise in A4 partly just because A4 lasted longer.
- **Options:**
  - (a) Keep them and say so in the limitations.
  - (b) Also compute them on a fixed-length slice of every recording (for example the first
    30 s), as a check.
- **Decision:** (a). `p05` and `p95` are the length-robust versions of `min` and `max`; if the
  extremes end up in the top features, compare them with `p05`/`p95` when reporting.
- **How to explain it:** "The longer you record, the more likely you catch one extreme value,
  and high-load activities ran longer. So we treat min, max and range with care and check them
  against the 5th and 95th percentiles, which are not affected by length the same way."

**D4. Confirm the signal list (§4.2)**: **Decided 2026-11-11 (team): keep the 10 signals as
they are.**
- Considered and not added: each eye's pupil separately (P36 and P44 show one eye's reading can
  be far off, and averaging the two eyes softens that) and the pupil's rate of change (a new
  signal that would need its own definition and sources).
- **How to explain it:** "A signal is one thing we measure over time, such as pupil size. We
  used the ten standard eye-tracking measures and summarized each into 17 statistics."


**D5. Go through cleaning steps C1–C10**: **Decided 2026-11-11 (team): all ten kept** (§3).
The pupil-quality question raised during the review is kept and reported as a limitation (§3).

**D6a. `saccade_duration__min`**: **Decided 2026-11-09 (team): drop.**
- **What it is:** the shortest eye jump (saccade) in a recording.
- **The problem:** the eye tracker can't measure anything shorter than one sample, which is
  **4.167 ms** at 240 samples per second. So the shortest jump is almost always exactly one
  sample:

  | Value | Recordings (of 88 A1/A4) |
  |---|---|
  | 4.167 ms (1 sample) | 83 |
  | 8.333 ms (2 samples) | 5 |

  Everyone gets the same number, so the feature can't tell anyone apart. It measures the
  equipment, not the person. (Analogy: a stopwatch that only shows whole seconds gives
  almost every runner a fastest lap of "1 second".)
- **Why drop it instead of letting the model ignore it:**
  - **Main reason (certain):** it describes the tracker's sampling speed, not the person's
    eyes, so it does not belong in a study of people's mental load.
  - **Second reason (a risk, not a certainty):** the 5 recordings at 8.33 ms differ by
    chance. A model *might* latch onto them and make the feature look "important". We can't
    know in advance, and must not check by looking at the labels.
  - Keeping it would also be defensible: it is 1 feature of 153 and would probably be
    ignored. Dropping it is the cleaner choice because it is explainable in one sentence.
- **Why only `min`:** the other saccade-duration statistics use all of a recording's
  saccades, so they still vary between people. Some are coarse too (D6b).
- **How to explain it:** "The shortest saccade in a recording is limited by the sampling
  rate: one sample is 4.17 ms. In 83 of 88 recordings the minimum was exactly one sample, so
  the feature reflected the equipment, not the participant, and was removed."

**D6b. The other saccade-duration statistics**: **Decided 2026-11-09 (team): keep all of
them.** Coarseness is stated as a limitation.
- **What they are:** summaries of how long each eye jump lasts in a recording: the middle
  jump (`median`), a typical short jump (`p10`, `p25`), a typical long jump (`p75`, `p90`),
  and so on.
- **The problem:** the tracker measures time in 4.17 ms steps, so every value is a multiple
  of about 4 ms (4, 8, 12, 17, 21, 25, 29… ms). The values are **rounded**:

  | Statistic | Different values (of 88) | Recordings sharing the most common value |
  |---|---|---|
  | `median` | 8 | 44 (25 ms) |
  | `p10` | 8 | 46 (12.5 ms) |
  | `p25` | 8 | 51 (16.67 ms) |
  | `p75` | 9 | 46 (33.33 ms) |
  | `iqr` | 8 | 34 (12.5 ms) |
  | `p05` | 16 | 49 (8.33 ms) |
  | `p90` | 17 | 25 (37.5 ms) |
  | `p95` | 26 | 22 (41.67 ms) |
  | `max` | 23 | 11 (58.33 ms) |
  | `mean` (for comparison) | 86 | almost all different |

- **Why keep them, unlike `min` (D6a):** `min` was **stuck**, the same for 94% of
  recordings. These are rounded but still **differ between people**: the median is 25 ms for
  half the recordings and 21 or 29 ms for most of the rest. Any cut-off for "too coarse"
  would be arbitrary and hard to defend. If they don't help, the importance analysis ranks
  them low; if one ranks highly, treat it with suspicion.
- **How to explain it:** "Saccade durations are measured in steps of one sample (4.17 ms),
  so their percentiles take only a few values. Unlike the minimum, which was identical in
  94% of recordings, these still differed between recordings, so they were kept and the
  coarseness is reported as a limitation."
- **Source:** `outputs/final/features/features.csv`.

**D7. `fixation_duration__min`**: **Decided 2026-11-09 (team): keep.**
- **What it is:** the shortest "look" (fixation) in a recording. The detection rule counts a
  fixation only if it lasts at least 55 ms, so the lowest possible value is 58.33 ms (14
  samples).
- **The data** (88 A1/A4 recordings):

  | Shortest look | Recordings |
  |---|---|
  | 58.33 ms (lowest possible) | 37 |
  | 62.5 ms | 22 |
  | 66.67 ms | 9 |
  | 70.83 ms | 8 |
  | 75–100 ms (6 values) | 12 |

- **Why keep it, unlike `saccade_duration__min` (D6a):** 42% of recordings sit at the floor,
  against 94% for the shortest saccade, and the rest spread over 9 values. Under rule 3 it
  is rounded, not stuck.
- **How to explain it:** "The shortest fixation is bounded by the 55 ms detection rule, and
  42% of recordings sat at that floor. Unlike the shortest saccade, which was identical in
  94% of recordings, it still varied across the rest, so it was kept."

**D8. Variance (`__var`)**: **Decided 2026-11-09 (team): drop variance, keep SD.** This
removes 9 features (`blink_duration__var` already went with D1).
- **What it is:** variance = SD × SD. Both measure how much a signal wobbles; variance is
  just a bigger-looking number for the same wobble (like a height in metres and in
  centimetres).
- **The data:** for all 9 signals, `__sd` and `__var` put the 88 recordings in exactly the
  same order (rank correlation 1.000).
- **Why it's a problem:** the main result is a ranking of which features matter most. When
  two columns hold the same information, the importance analysis splits the credit between
  them, so both look less important than the wobble really is:

  | Keep both | Keep only SD |
  |---|---|
  | pupil SD 50 points + pupil variance 50 points | pupil SD 100 points |

  Dropping variance keeps the ranking honest and loses no information.
- **Why SD and not variance:** SD is in the same unit as the signal (mm, ms, deg).
- **How to explain it:** "Variance is SD squared, so it carries exactly the same information.
  Keeping both would have split the importance of one property across two columns. We kept
  SD because it is in the same unit as the measurement."

**D9. Range where it copies max**: **Decided 2026-11-09 (team): drop `__range` for 6
signals; keep it for pupil, gaze x and gaze y.** Dropped: `gaze_speed__range`,
`fixation_duration__range`, `saccade_amplitude__range`, `saccade_peak_velocity__range`,
`saccade_mean_velocity__range`, `saccade_duration__range`.
- **What it is:** range = biggest value − smallest value.
- **The problem:** for these 6 signals the smallest value is basically **fixed**, so
  range = max − (a fixed number) and carries nothing new:

  | Signal | Smallest value | Why it's fixed | Range vs max (rank correlation) |
  |---|---|---|---|
  | Eye speed | ~0 °/s | The eye is always still at some moment | 1.000 |
  | Fixation duration | ~58 ms | Shorter looks are not counted (55 ms rule) | 0.9998 |
  | Saccade peak / mean speed | ~46 °/s | Only movements over 45 °/s count as saccades | 0.9996 / 0.9993 |
  | Saccade duration | ~4 ms | One sample; the tracker can't measure shorter | 0.9992 |
  | Saccade amplitude | ~0.5° | The smallest jumps are always tiny | 0.9992 |

  Example (eye speed): values 0, 2, 5, 40, 200, 370 give max = 370 and
  range = 370 − 0 = 370, the same number.
- **Why keep range for pupil and gaze:** their smallest value changes from person to
  person (e.g. pupil 3.0–4.0 mm gives max 4.0, range 1.0), so range and max differ (rank
  correlation 0.67–0.72).
- **Why drop the copies:** same reason as variance (D8): two columns with the same
  information split the importance credit.
- **How to explain it:** "For signals whose minimum always sits near zero or near a
  detection threshold, the range was nearly identical to the maximum (rank correlation
  ≥ 0.999), so only the maximum was kept. For pupil size and gaze angle the range carried
  different information and was kept."

**D10. Gaze position (20 features tagged `position`)**: **Decided 2026-11-09 (team): keep
all 20, tagged.** (On 2026-11-11 gaze `cv` was added to the tag, making 22; §4.3.)
- **What they are:** the location statistics of gaze left/right and up/down (`mean`,
  `median`, `min`, `max`, `p05`–`p95`). They say *where on the screen* someone looked.
- **The worry:** where someone looks could depend on the screen layout of a task rather than
  on mental load.
- **Why keep them in COLET:** every activity used the same kind of visual-search images,
  drawn at random from one set of 21, and a chin rest held the head still. (When D10 was
  decided, per-person z-scoring also removed headset offsets; it was dropped on 2026-11-11,
  §5, so that reason no longer applies.)
- **Rule 1 applies:** if one ranks highly, report it with the warning that it may reflect
  where content appeared on screen. Do not remove it.
- **How to explain it:** "Gaze-position statistics were kept because all activities used the
  same type of visual-search images drawn from one set, with a chin rest. They are tagged in the feature dictionary, and any that rank highly are
  reported with the caveat that they may reflect screen content rather than load."

**Summary of the feature decisions (2026-11-09):**

| Decision | Features | Result |
|---|---|---|
| D1 Blink duration: blanks follow the condition | 17 | Dropped |
| D6a `saccade_duration__min`: stuck at one sample | 1 | Dropped |
| D6b Other saccade-duration statistics: rounded but vary | 14 | Kept |
| D7 `fixation_duration__min`: rounded but varies | 1 | Kept |
| D8 Variance: exact copy of SD | 9 | Dropped |
| D9 Range where it copies max | 6 | Dropped |
| D10 Gaze position (+ gaze `cv`, 2026-11-11) | 22 | Kept, tagged |
| **Total** | **173 extracted** | **33 dropped, 140 kept** |

**Confirmed by the first extraction run (2026-11-09):** the D1 blink counts (13 A1
recordings with no blinks, 29 with fewer than 2, 38 with fewer than 4) match
`logs/log_features.csv` exactly.

## 7. Processing log

**Rule:** everything that is dropped, filled in or transformed is written down
automatically by the code, never by hand.

### 7.1 The final folder: `outputs/final/`

Every final output goes in one folder. If it is in `outputs/final/`, it is the version we
report. If it isn't there, it isn't final.

**Rules:**
- **Only the pipeline writes here.** Nothing in this folder is edited by hand.
- **Each run replaces the whole folder**, so old and new files never get mixed.
- **Every run records when and from what** in `run_info.json`: date, git commit, and the
  settings used.
- **Intermediate files stay out** (caches, partial tables, test runs). They go in
  `outputs/work/`.
- **The defended results stay where they are** in `outputs/colet/`, untouched, so we can
  always show what changed since the defense.

**Layout:**

```
outputs/final/
├── README.md               what each file is, in plain words
├── run_info.json           date, git commit, settings
├── features/
│   ├── features.csv        one row per A1/A4 recording (88 rows): what the model uses
│   └── feature_dictionary.csv
├── logs/
│   ├── log_recordings.csv
│   ├── log_features.csv
│   ├── log_decisions.md
│   └── sanity_check.csv    the C9 event-detection check per activity
├── raw/
│   ├── raw_data_dictionary.csv   every column of the original COLET tables
│   └── raw_table_shapes.csv      files, columns and rows of each original table
├── figures/              PNGs for the team to understand the data (not for the paper)
└── (modeling stage adds results/ here later)
```

**Figures (`figures/`, drawn by `src/final_figures.py` from the files above).** These are
working figures for the team to understand the data. They are **not** paper figures.

| File | What it shows | Use it to explain |
|---|---|---|
| `fig_01_raw_tables.png` | Size of each original table: files, columns, rows | "What did the raw data look like?" |
| `fig_02_raw_columns.png` | Every original column, marked used / quality check only / not used | The raw data dictionary at a glance |
| `fig_03_data_flow.png` | Raw files → cleaning → signals → flatten → A1/A4 → final table, with the size at each stage | "How did 6.8 million rows become 88?" |
| `fig_04_feature_table.png` | Features kept and dropped per signal | "Where do the 140 features come from?" |
| `fig_05_recording_signals.png` | One cleaned recording over time (pupil, gaze angles, eye speed with the 45 deg/s line, blinks) | What a signal is (D4) and how fixations/saccades are found (C9) |
| `fig_06_cleaning_effect.png` | (a) P18 A4's stray pupil samples removed by C1; (b) pupil before and after cleaning | What cleaning actually does |
| `fig_07_a1_vs_a4.png` | Six core measures, low vs high load, one line per participant | What the data shows, descriptively |
| `fig_08_data_quality.png` | Unreadable gaze per recording, the 35% cut-off, the 3 dropped recordings | C3 and why data loss is never a feature (C10) |

- The example recording in figures 05 and 06(b) is chosen by a fixed rule: the analysed A4
  recording with the median share of invalid gaze (currently P08 A4).
- Figure 07 is **descriptive only**. Its six measures are the defended thesis's core
  measures, fixed before the plot was drawn, not picked for how well they separate A1 from
  A4. Feature selection still happens inside each LOPO fold (§8).
- **How to explain figure 07:** "Each line is one person. Most lines go up from low to high
  load for blink rate and pupil size. This only describes the data; the model decides which
  features matter."

**What each file holds:**
- `feature_dictionary.csv`: one row per feature with its name, signal, statistic, unit,
  tag (`position` or not), and a one-line plain meaning (for example: `pupil__p95` = "pupil
  size that 95% of the recording stays below, in mm"). This is the cheat sheet for
  explaining any feature the panel asks about.
- `log_recordings.csv`: one row per recording (all 188, excluded ones too) with:
  - C1: gaze rows, pupil rows and blinks before and after trimming to the gaze time range;
  - C2/C3: share of low-confidence gaze samples, whether the recording was dropped, and why;
  - C4: recording length, usable time, and time lost to gaps over 1 s;
  - C5: blinks kept after merging and the 50–500 ms rule;
  - C6–C8: share of valid gaze directions, share of samples with both eyes, share of valid
    pupil values, 3D pupil rows, pupil model refits;
  - C9: samples rejected as too fast, fixation and saccade counts;
  - how many values each of the 10 signals had (`n_pupil`, `n_blink_duration`, …), which
    is what decision D2 needs;
  - `in_analysis`: whether the recording is in the feature table, and `analysis_reason`:
    why not (recording excluded, participant lost A1 or A4, P31 duplicate, or A2/A3 with no
    label).

  Not yet counted: how many gaze and pupil samples were filled by interpolation (C6, C7).
- `log_features.csv`: one row per feature with:
  - number of blank values in A1, in A4 and overall;
  - whether it was dropped, and why.
- `log_decisions.md`: each decision from this doc with its date, the rule chosen, and who
  decided.
- `raw_data_dictionary.csv`: every column of the 5 original tables (gaze 21, pupil 34,
  blinks 10, annotation 9, subject_info 5) with its type, unit, an example value, its plain
  meaning, whether we use it, and why not. §1a summarizes it.
- `raw_table_shapes.csv`: for each original table, the number of files and columns, and the
  smallest, median, largest and total number of rows.

**Already known entries:**

| What | Effect | Reason |
|---|---|---|
| Recordings P06 A3, P06 A4, P17 A4 dropped | P06 and P17 leave the main analysis | Over 35% invalid gaze (C3) |
| Mode | Not computed | Undefined for continuous data (§4.3) |
| Counts and durations | Not features | Measure activity length, not load (§4.4) |
| Valid-sample % | Reported, not a feature | Would leak the label (C10) |
| Blink duration statistics | Pending D1 | Blank in most A1 recordings |

## 8. Notes for the modeling stage (not decided here)

The modeling decisions now have their own doc: `post-defense-modeling-plan.md` (MD1–MD4).

**Picking the top features must happen inside each LOPO fold.**
- **Wrong way:** rank all 140 features using all 88 rows, keep the top 20, then run LOPO.
  The ranking has already seen the test person, so accuracy comes out too high. Panels
  catch this.
- **Right way:** in each fold, rank features using only the 43 training people, keep the
  top N, train, then test on the held-out person.
- **Bonus result:** report how often each feature made the top N across the 44 folds (for
  example, "pupil SD was picked in 44 of 44 folds").
- This replaces the defended rule that "the feature set is fixed in advance"
  (`methodology-colet.md` §6).

**Regularization.**
- **The problem:** with 140 features and 88 rows, some features will separate A1 from A4 in
  these 88 rows by pure luck. A model without limits will lean on them and then fail on new
  people.
- **The fix:** regularization makes the model pay a cost for every feature it relies on. It
  uses a feature only if that feature helps consistently, and shrinks the rest toward zero.
- **Where it stands:** both models already have it (LR uses an elastic-net penalty, and
  XGBoost has built-in penalties, both tuned inside each fold). With about 14× more features,
  the tuning matters more.

**Duplicate features.** Many columns are near-copies (`sd` and `var`, `mean` and `median`).
Duplicates split the importance between them, so each looks less important than the
underlying signal really is. The modeling doc should say how importance is grouped, for
example by summing per signal.

## 9. What changed since the defense

| Area | Defended version | Now |
|---|---|---|
| Features per signal | 1–2 (mean, sometimes SD) | 17 statistics |
| Number of features | 10 | 173 extracted, 140 model features after D1–D10 |
| Feature set | Fixed in advance | Generated broadly; top-N chosen inside each fold (modeling stage) |
| Duplicate features | Dropped (saccade mean velocity, saccade duration) | Kept |
| Gaze position | Excluded | Kept, tagged `position` |
| Per-person z-score | Every feature, over A1–A4 | Not done; scaling for the model is fitted inside each LOPO fold (§5) |
| Rows in the feature table | 176 (A1–A4; A2/A3 only for the z-score) | 88 (A1 and A4) |
| Cleaning | Done by a tool, not reviewed by the team | Each step reviewed and confirmed by the team (C1–C10) |
| Record of changes | Spread across notes | One processing log, written by the code |
| Where outputs live | Mixed into `outputs/colet/` | Final outputs only in `outputs/final/` (§7.1); defended outputs left as they were |

# Proposed Methodology — COLET Version

**Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

Status: **proposal for team review**. Nothing has been downloaded, coded or run. It supersedes
`methodology-replan.md` (the GAZELOAD version), which is kept for reference. Source IDs (`EM7`,
`X2`, …) refer to `rrl-master-list.md`. Items marked **[CHECK]** must be confirmed against
the COLET full paper or the downloaded data before they are final.

Last updated: 2026-10-01

---

## 1. The study in one paragraph

We compare Logistic Regression and XGBoost for binary cognitive-load detection from
eye-tracking features computed from COLET's raw gaze, pupil and blink data. Labels come from
the experimental design, with NASA-TLX as the manipulation check. Both models are evaluated
under leave-one-participant-out (LOPO) validation with nested, participant-grouped tuning.
Feature importance is compared across the two models with LR coefficients, TreeSHAP and
permutation importance. As a secondary contribution, we show how much COLET's published
accuracies depend on random (subject-dependent) splits.

**Gap this fills (verified):**
- No published study on COLET uses XGBoost or SHAP.
- No published study on COLET uses peer-reviewed participant-held-out validation.
- No study in this domain computes SHAP for both LR and XGBoost.

Sources: §9b, §4b of the master list; X1–X24.

## 2. Dataset: COLET

| Item | Value | Source |
|---|---|---|
| Descriptor | Ktistakis *et al.*, *Comput. Methods Programs Biomed.* 224:106989, 2022 (peer-reviewed) | EM7 |
| Access | Zenodo record 7766785 (v3), CC BY 4.0, about 791 MB, open | §9b |
| Participants | 47 retained (56 recorded; 9 excluded for poor recordings or eye disease) | §9b (via Fuhl 2023) |
| Design | 2×2 within-subject: multitasking × time pressure, giving 4 activities and 188 recordings | §9b |
| Task | Visual-search puzzles (5 images per activity); secondary task is counting backwards by 4 | §9b |
| Setting | Lab, chin rest, 80 cm from a 24" screen | §9b |
| Eye tracker | Pupil Labs Pupil Core, binocular, 240 Hz **[CHECK]** | §9b (secondary sources) |
| Released data | Raw gaze, pupil and blink streams plus annotations (`.mat`) | §9b |
| Self-report | NASA-TLX per activity (mean of 6 subscales) | §9b |

**Why COLET over GAZELOAD:**
- the descriptor is peer-reviewed;
- 4 independent groups have reused it in 5 studies;
- pupil and raw gaze are available;
- N is larger (47 vs 26).

**Limitations to state:**
- It is a screen-based lab task with a chin rest. Cite DS1–DS3 for the limits of screen-based
  setups.
- It is not industrial or wearable.

**[CHECK] after download:**
- activity durations (this sets how many windows each activity yields);
- whether a rest or baseline recording exists;
- per-sample confidence values;
- which participants are already excluded;
- whether the stimulus images differ by activity, since image luminance would affect pupil (§5).

## 2b. What the COLET paper confirms (full PDF read, 2026-10-01)

The PDF is in the repository root: `1-s2.0-S0169260722003716-main.pdf`.

**Confirmed facts**

| Item | From the paper |
|---|---|
| Participants | 56 recruited; 9 excluded (7 by vision criteria, 2 for poor recordings). 47 analysed: 26 F / 21 M, age 32 ± 8 (18–47). |
| Tracker | Pupil Core, **binocular 240 Hz**, accuracy 0.60°, precision 0.02. Chin and head rest. |
| Display | 24" LCD, 1280×720, viewed from 80 cm. |
| Lighting | Controlled photopic room light: 400 lx at the cornea with the screen off, 450 lx with a blank screen. |
| Activities | A1 = no time pressure + single task; A2 = time pressure + single; A3 = no time pressure + multitask; A4 = time pressure + multitask. |
| Trials | Each activity = **5 puzzle images**, randomly drawn from 21. Activities in **random order**, 2-minute breaks between them. |
| Time pressure | **Only a verbal instruction** ("as quickly as possible"), with no countdown or time limit. The authors judge it **weak**. |
| Secondary task | Counting **aloud** backwards from 1000 by 4. |
| Self-report | NASA **RTLX** (unweighted) after each activity. Mean scores: A1 19.4, A2 29.2, A3 44.8, A4 52.2. **All pairs differ significantly** (Bonferroni p < 0.014). |
| Duration | **Not reported** as a number. The authors say trials were too short for trial-level analysis. Reaction time differs by condition (multitasking about +29 s; time pressure about −11 s). |
| Their processing | Gaze converted to degrees of visual angle; **5-tap FIR velocity filter**; **I-VT at 45°/s**; **minimum fixation 55 ms**. Pupil in mm. No confidence threshold stated. |
| Luminance check | Regressed each participant's pupil size on image brightness (HSV V channel). Only 2 of 47 participants correlated, so they judged luminance effects small. |
| Features | 28, computed **per whole activity**: fixation, saccade, blink and pupil statistics, plus their variation, skewness and kurtosis. |
| Models | GNB, RF, linear SVM, sklearn GradientBoosting (labelled "EGB"), kNN, NB, **LR**, DT. Random search with 1000 iterations. **No XGBoost.** |
| Validation | **Random 80/20 split + 5-fold CV**. Features were **selected by ANOVA on all the data** and min-max scaled. Both are leakage risks. |
| Key results | **A1 vs A4: kNN 0.86, LR 0.85**. Single vs multitask: kNN 0.90. A3 vs A4: 0.60 (near chance). NASA low vs high: GNB 0.88. |

**Consequences for our plan**

1. **The primary label is directly precedented.** A1 vs A4 is the COLET paper's own binary
   task, with LR at 0.85 under a random split. We report how it holds up under LOPO.
   - The authors already ran the manipulation check (A1 < A2 < A3 < A4 on RTLX).
   - We replicate it; we don't need to invent it.
2. **Time pressure is weak** (the authors' own judgment, and A3 vs A4 ≈ chance). So:
   - **S1 (multitask vs single) is the second key analysis.** It is the dominant, well-induced
     factor.
   - A1 vs A4 stays primary as the clean low-vs-high contrast.
3. **New confound: the secondary task means talking.**
   - Blink frequency rises from about 0.05/s to 0.24/s under multitasking (Table 4 of the
     paper).
   - Blink rate also rises during conversation (ET5, Bentivoglio 1997: 17/min at rest,
     26/min while talking).
   - So part of the "multitask" eye signal may be **speech**, not workload.
   - **Mitigation:** run a sensitivity model without blink features, report the result, and
     discuss it in Ch. 4.
4. **Activities are short, and their length differs by condition.**
   - 30 s windows may give only a few windows per activity, with more windows in multitask
     activities.
   - **Revised plan:** decide the unit **after** measuring durations (P0).
     - (a) whole-activity features, as in COLET (one sample per person per activity); or
     - (b) shorter non-overlapping windows (e.g. 10 s; Fuhl used 10–30 s on COLET).
   - Whole-activity features with only 2 samples per person (A1, A4) give an undefined
     per-person AUC. Windows allow per-person metrics.
   - Activity duration itself differs by condition, so it must **not** be a feature.
5. **Event detection:** use COLET's own settings as primary, so results are comparable with
   the original:
   - 5-tap FIR velocity filter, I-VT 45°/s, minimum fixation 55 ms.
   - I-DT (1.0°, 100 ms) becomes the sensitivity check.
6. **Luminance:** cite the authors' check (2 of 47 correlated) and repeat it. Pupil features
   stay in the core set, with a pupil-free sensitivity model.
7. **Leakage in the original protocol:**
   - random split;
   - ANOVA feature selection on all data;
   - scaling before the split.

   This strengthens our secondary contribution (S3): show how much these steps inflate
   results on COLET.

## 2c. P0 data inventory (measured from the downloaded data, 2026-10-01)

**Data handling:**
- **Source:** Zenodo COLET_v3.zip (MD5 verified), decoded with `mat-io`.
- **Converted to:** `data/colet/parquet/` (564 files, 847 MB).
- **Script:** `data/colet/convert_colet.py`; per-recording results are in
  `data/colet/inventory.csv`.
- **Not committed:** `data/` is git-ignored.

**Median per activity (47 participants)**

| | A1 (single, no pressure) | A2 (single, pressure) | A3 (multi, no pressure) | A4 (multi, pressure) |
|---|---|---|---|---|
| **Duration (s)** | **37.0** (16.8–75.8) | **29.2** (11.3–62.7) | **61.9** (24.6–129.8) | **51.0** (27.7–141.0) |
| Gaze rate (Hz) | 242 | 240 | 243 | 243 |
| Gaze samples with confidence < 0.8 | 0.9% | 0.7% | 7.5% | 7.5% |
| Pupil rate per eye (Hz) | 242 | 240 | 243 | 243 |
| Blink rate (/min) | 2.1 | **0.0** | **13.3** | **15.3** |
| Blink duration, median (ms) | 193 | 189 | 220 | 226 |
| NASA-RTLX mean | 19.4 | 28.7 | 45.0 | 51.9 |

**What this settles**

1. **Sampling:** **240 Hz per eye** (pupil rows per eye ≈ 240/s); gaze samples every 4 ms.
   Timestamps have gaps up to about 200 ms (blinks).
2. **Activities are very short:** 11–141 s, median 29–62 s.
   - **30 s windows are not feasible.** The median A2 activity holds zero full 30 s windows.
   - 10 s windows give a median of only 2–6 per activity.
3. **Duration differs strongly by condition.** Multitask activities last about twice as long.
   Duration (and anything that scales with it, like counts) **must not be a feature**, and
   windowing would give unequal sample counts by class.
4. **The blink confound is confirmed in the data.**
   - Blink rate is about 0–2/min in single-task activities and about 13–15/min in multitask
     activities, where participants count aloud.
   - Low-confidence samples rise to 7.5% under multitasking for the same reason.
   - The **no-blink sensitivity model** (§2b point 3) is essential.
5. **Pupil:**
   - Every sample appears twice, once from the `2d c++` method and once from `3d c++`.
     `diameter_3d` exists only on the 3d rows.
   - **Use the 3d rows for `diameter_3d` (mm).**
6. **Data quality:** only 3 recordings have > 30% of gaze samples below 0.8 confidence (P06
   A3 and A4, P17 A4). Apply the pre-set exclusion rule (P1) to them.
7. **RTLX matches the paper** (A1 < A2 < A3 < A4), so the manipulation check replicates.

**Decision needed: unit of analysis** (replaces the 30 s window plan in §5 / P9)

| Option | Samples (A1 vs A4) | Pros | Cons | Support |
|---|---|---|---|---|
| **(a) Whole activity (recommended primary)** | 94 (47 × 2) | Same unit as the COLET paper, so results are directly comparable. No unequal window counts. Rates are computed over the whole activity, so they are more stable. | Small N for XGBoost. Each LOPO fold tests 2 samples, so per-person AUC is 0 or 1 ("did the model rank this person's A4 above their A1?"). Report pooled AUC plus the share of correctly ranked participants. | EM7, Fuhl (COLET), EM9 (block-level) |
| (b) 10 s windows, first *k* windows per activity (equal count) | 47 × 2 × k (k ≈ 2) | More samples; per-person metrics defined | Fewer blinks per window (noisy rates); fixed *k* discards data; still short | WN4, Fuhl (10–30 s on COLET), ET5 |
| (c) 10 s windows, all of them | Unequal: multitask has about 2× more | Most data | Class imbalance comes from duration, not load; must use class weights | — |

**Decided (2026-10-01):** **(a) whole activity** is the unit of analysis. Option (b) stays
available as a sensitivity analysis (S6). This replaces the 30 s windows in §5 and P9.

## 3. Labels

**Decided (2026-10-01, C2):** the primary target is **two levels, A1 (low) vs A4 (high)**.

**Talking confound (C2b):** the team chose to follow the RRL.
- Blinks stay as workload features.
- There is no dedicated speech control, as in the COLET paper and Pluchino 2023.
- §2b point 3 and the "no-blink sensitivity model" / "A1 vs A2 check" below are therefore
  **optional**, not part of the plan.
- Consider one sentence in the limitations noting that the secondary task was spoken.

Comparison of the options considered:

| | (a) A1 vs A4 | (b) Single vs multitask (A1+A2 vs A3+A4) | (c) A1 vs A2 (check only) |
|---|---|---|---|
| What it asks | Lowest vs highest load | Secondary task present vs absent | Time pressure only, **no talking** |
| NASA-RTLX gap | 19 → 52 (largest) | about 24 → 48 | 19 → 29 (small) |
| Samples | 94 (2 per person) | **188** (4 per person, 2 per class) | 94 |
| Per-person AUC | 0 or 1 only | Defined (2 vs 2) | 0 or 1 only |
| Talking confound | Yes (A4 has counting aloud) | Yes (all multitask) | **None** |
| COLET paper result | LR 0.85, kNN 0.86 | kNN 0.90, RF 0.88 | NB 0.80 |
| Literature fit | "Drop the middle", low vs high (Hogervorst) | Condition factor, all data used | Isolates load from speech |

**Primary target: designed load, binary.** Options under C2 are A1 vs A4 or single vs
multitask; the text below describes A1 vs A4.
- **Low** = no multitasking and no time pressure.
- **High** = multitasking **and** time pressure.
- **Left out of the primary analysis:** the two single-factor activities. This "drop the middle"
  approach follows Hogervorst (LB5); condition-based labels follow LB4–LB8 and EM5.
- **Result:** one low and one high activity per person, a balanced 1:1 design.

**Secondary targets:**
1. **Multitasking vs none** (all 4 activities). The COLET authors found multitasking affected
   17 eye features, against 7 for time pressure (EM7).
2. **NASA-TLX labels** as COLET studies use them: high ≥50 vs <50.
   - This allows comparison with prior COLET results (§9b).
   - Expect it to be harder: Gado found self-report labels much less learnable than condition
     labels (LB4).

**How the labels work (confirmed 2026-10-01):**
- **Every A1 recording = low; every A4 recording = high,** for all participants.
- Labels come from the **experimental condition**, not from each person's rating (8/12 twins
  do this).
- **Role of each activity:**
  - A1, A4: the primary analysis.
  - A2, A3: left out of the primary analysis. They are used in the manipulation check, and
    remain available for optional extra analyses (single vs multitask; A1 vs A2).
- **Individual-level check (measured):**
  - **All 47 of 47 participants rated A4 higher than A1** on NASA-RTLX.
  - Median difference 31.7 points (min 1.6, max 67.5); none equal or reversed.
  - Suggested Ch. 3 sentence: *"All 47 participants rated A4 as more demanding than A1
    (NASA-RTLX; median difference 31.7 points), supporting the use of A1 and A4 as low- and
    high-load conditions."*
- **Per-person normalization:** computed over all 4 activities (label-free; a steadier
  average). To be confirmed when the preprocessing is coded.

**Manipulation check:**
- A Friedman test (or repeated-measures ANOVA) of NASA-TLX across the 4 activities.
- Report whether the high activity is rated higher than the low one within participants.
- Sources: LB6, LB7, LB12.

## 3b. Preprocessing plan (raw data → clean signals → windows)

COLET releases **raw** Pupil Core exports per participant × activity (`data_v3.mat`, fields
`data{1..47}.task{1..4}`). They contain:
- **`gaze`:** `norm_pos_x/y`, `confidence`.
- **`pupil`:** `diameter` (px), `diameter_3d` (mm), `confidence`, `model_confidence`.
- **`blinks`:** `start_timestamp`, `duration`, `confidence`.
- **`annotation`:** NASA-RTLX.

**No fixations or saccades are provided, and no rest/baseline recording exists** (Zenodo README,
read directly). Every step below is ours and must be stated in Chapter 3. All thresholds are
fixed **before** looking at results.

New source IDs (PP1–PP16) are listed in `rrl-colet-list.md` §F.

| Step | What we do (parameters) | Why | Support |
|---|---|---|---|
| **P0. Inventory** | Per participant × activity: duration, sample count, **effective sampling rate measured from timestamps** (the 240 Hz stream is fused from two 120 Hz eye cameras running in anti-phase, so timing is irregular), gaps. | Catch corrupt or short recordings; do not assume 240 Hz. | PP7 (Faraji 2023) |
| **P1. Recording exclusion** | Keep COLET's 9 exclusions. Exclude any recording with < 70% valid samples after P2 (pre-set; design choice). Report all drops. | Bad recordings add noise and bias. | §9b; PP7 |
| **P2. Sample-quality filter** | Gaze/pupil samples with **confidence < 0.8 = invalid** (sensitivity check at the vendor's 0.6). For pupil data, also require good `model_confidence`. Gaze outside the screen = invalid. | Low-confidence samples are tracking failures (blinks, lost pupil). | PP7 (Faraji 2023), PP8 (Hausamann 2020); PP5 (Kassner 2014, defines confidence); PP6 (Ehinger 2019, independent accuracy check) |
| **P3. Blinks** | Start from COLET's blink events. Keep plausible durations (about 50–500 ms; drop implausibly long "blinks"). Merge blinks < 100 ms apart. Features: blink rate (/min) and mean duration. Treat blink periods as missing for gaze and pupil. | The Pupil Labs detector sometimes reports 20 s+ blinks; blinks themselves are a workload feature. | PP10 (Hershman 2018), PP6 (Ehinger 2019), CF2 (Steinhauer 2022: typically ~200 ms); CL7 |
| **P4. Gap handling** | Linearly interpolate **gaze** gaps < 75 ms; pad ±50–100 ms around longer gaps; leave long gaps missing. | Short gaps are noise; filling long gaps would invent data. | PP7 (Faraji 2023), PP11 (Kret & Sjak-Shie 2019) |
| **P5. Pupil cleaning** | Use **`diameter_3d` (mm)**, not px. Feasible range 1.5–9 mm. Remove dilation-speed outliers (MAD). Drop 50 ms around gaps > 75 ms. Average both eyes. Interpolate gaps ≤ 250 ms. **4 Hz zero-phase low-pass**. | Standard, published pupil pipeline; the 3D model reduces gaze-angle error. | PP11 (Kret & Sjak-Shie 2019), PP9 (Petersch & Dierkes 2022), NM7 (Mathôt 2018), CF2 |
| **P6. Gaze → degrees** | Convert `norm_pos` to degrees of visual angle using 80 cm distance, a 24" screen, 1280×720 [CHECK in the PDF]. | Event thresholds are defined in degrees. | ET1, PP4 (Holmqvist 2011) |
| **P7. Gaze smoothing** | Light Savitzky–Golay filter before velocity calculation. | Prevents noise creating false saccades. | EM5 |
| **P8. Event detection** (see §3c) | **Primary: COLET's own settings** (2D gaze → degrees; Duchowski 5-tap FIR velocity filter; **I-VT 45°/s, minimum fixation 55 ms**) for comparability with the original paper. Must reproduce COLET Table 4 before use. **Sensitivity: I-DT, dispersion 1.0°, minimum 100 ms** (suits static stimuli with a chin rest). Reject velocities > 1000°/s as artifacts. Report all parameters. | Fixations and saccades are not in the data. No algorithm is universally best, and results depend on thresholds, so report them and test sensitivity. | PP1 (Salvucci & Goldberg 2000), PP3 (Andersson 2017), PP2 (Komogortsev 2010), PP12 (Birawo 2022: 0.5–1°, 100–200 ms), PP13 (Trabulsi 2021: 30°/s, 60 ms), PP8 |
| **P9. Windowing** | Non-overlapping **30 s** windows within each activity, by elapsed time. Drop windows with < 80% valid samples. Each window inherits its activity's label. (Prior COLET work used 50% overlap with a random split, which we avoid.) | Rates need multi-second windows; no overlap and participant-wise splits avoid leakage. | EM8, EM9, HR1; Fuhl 2023 (§C #23); EV3, EV2 |
| **P10. Features** | Per window: fixation duration and rate; saccade amplitude, peak velocity and rate; blink rate and duration; pupil mean and SD; gaze dispersion (SD x/y); **stationary and transition gaze entropy** on a fixed screen grid (e.g. 3×3 matching the puzzle's 9 boxes, or 100×100 px), normalized by H_max. | Literature-backed workload features; entropy formulas as published. | §4; CL7; PP14 (Shiferaw 2018), ET17, PP15 (Krejtz 2015) |
| **P11. Window QC** | Valid-sample % per window kept as a QC column, **not** a feature. Report retention by participant and condition. | Data loss that differs by condition would be a confound. | PP7, LB13 |
| **P12. Luminance check** | Compute mean pixel intensity and contrast of the 21 stimulus images and test that they do not differ by condition. Lighting was controlled (reported: 400 lx screen off / 450 lx blank screen; [CHECK]). Optional covariate: local luminance at fixation. | The light reflex can exceed cognitive effects on pupil size. | CF2 (Steinhauer 2022), CF3 (Pfleging 2016), CF5 |
| **P13. Fold-aware steps** | Per-person normalization (§6), then scaling, imputation and correlation pruning **fitted inside each training fold**. | Prevents preprocessing leakage. | EV2, LB13 |
| **P14. Retention report** | Table of samples, windows and recordings kept at each step, by participant and condition. | Transparency. | LB13 |

**Confirmed from the Zenodo README:**
- 47 participants × 4 tasks;
- gaze, pupil (px and mm), blinks, NASA-RTLX;
- 21 stimulus images;
- **no fixation/saccade events and no baseline recording.**

**Reported by Fuhl 2023 (secondary source; verify in the COLET PDF):**
- activities ran in **random order** (good: no fixed time-on-task confound);
- chin rest;
- 1280×720 display;
- controlled lighting.

**Still [CHECK] in the COLET PDF:**
- activity durations and time-pressure limits;
- COLET's own confidence and event-detection settings;
- whether 240 Hz is per eye or fused (measure it from the timestamps either way).

## 3c. Event detection: what researchers need to know (noted 2026-10-01)

**Why it is needed:**
- COLET provides only raw gaze positions (about 240 per second), not fixations or saccades.
- Fixation and saccade features (CL7; ET8–ET10) exist only after we **detect** these events
  from the positions.
- This is standard practice in eye-tracking research, and it involves choices that change
  the numbers.

**How it works (plain terms):**
- Slow eye movement = fixation (holding on a point); fast eye movement = saccade (a jump).
- Speed is measured in degrees of visual angle per second.

| Step | Purpose | Source |
|---|---|---|
| 1. Convert gaze positions to degrees of visual angle (screen geometry: 80 cm, 24", 1280×720) | Eye movement is an angle | Duchowski (ET1; COLET cites the 2003 chapter "Eye movement analysis") |
| 2. Velocity with a 5-tap FIR filter | Smooths tracker jitter that would look like movement | Duchowski (COLET ref [48]) |
| 3. I-VT: below 45°/s = fixation, above = saccade | Separates holds from jumps | Salvucci & Goldberg 2000 (PP1); threshold "as in" Andersson 2017 (PP3) |
| 4. Keep fixations ≥ 55 ms | Very short holds are usually noise | COLET (its source Zaidawi 2021 is an **arXiv preprint**); supported by Trabulsi 2021 (60 ms, PP13) |

**RRL precedent (detection from raw gaze):**

| Study | Method | Settings |
|---|---|---|
| COLET paper (EM7) | I-VT + 5-tap FIR | 45°/s, 55 ms |
| Božak 2026 (EM5) | I-DT + Savitzky–Golay | not recorded |
| GAZELOAD paper (DS0) | I-VT | 30°/s, 60 ms, merge 75 ms |
| Upasani 2024 (ET6) | I-VDT | 75°/s, 1°, 80 ms |
| Trabulsi 2021 (PP13) | I-VT, settings evaluated | 30°/s, 60 ms |
| Hausamann 2020 (PP8) | Velocity from raw gaze | > 1000°/s rejected |

Method sources:
- Salvucci & Goldberg 2000 (the algorithms);
- Andersson 2017 (no single best algorithm; standard detectors work well for static
  stimuli);
- Komogortsev 2010 (results depend on thresholds, so **report the settings**).

**Things the team must be aware of:**
1. **Thresholds change the results.** Studies use 30–75°/s and 55–100 ms. We use COLET's
   settings for comparability, and report them with an I-DT sensitivity check.
2. **The first-pass exploration used a hand-rolled detector with unsourced details:**
   - 3D gaze direction instead of 2D positions;
   - a moving average instead of the FIR filter;
   - no minimum saccade length;
   - a 50 ms gap rule.

   Its fixation and saccade numbers did **not** match COLET Table 4 (saccade amplitude 2.6°
   vs 14°). **Those numbers must not be used.** The final pipeline replaces the detector with
   the sourced recipe above.
3. **Validation rule:** the rebuilt detector must reproduce COLET Table 4 approximately before
   its features are used (A1: fixation about 273 ms, saccade amplitude about 14°, saccade rate
   about 1.7/s).
4. **Open gaps:**
   - (a) Duchowski's exact filter coefficients still need checking in the book.
   - (b) COLET does not say how it converted the camera's normalized coordinates to screen
     degrees, so our conversion is a stated design choice.
   - (c) COLET's 55 ms source is a preprint.
5. **Some RRL studies probably used their tracker's built-in event detection** (Tobii,
   EyeLink); our notes do not record it. COLET provides no built-in events, so we must run
   detection ourselves.

## 4. Feature extraction from raw data

Raw data are available, so we can compute the classic workload measures that GAZELOAD lacked.

**Data-quality steps (thresholds stated in advance):**
- Drop samples below a Pupil Labs confidence threshold. **[CHECK]** the value; this is a
  stated design choice.
- Interpolate short gaps only.
- Report the proportion of data retained per participant.

**Event detection:**
- A velocity-threshold algorithm (I-VT) or dispersion-threshold algorithm (I-DT), with the
  parameters reported.
- EM5 uses I-DT; GAZELOAD used I-VT at 30°/s.
- **[TO SOURCE]** a peer-reviewed reference for the algorithm.

**Features** (grades from the R3 feature audit and CL7):

| Feature | Literature support |
|---|---|
| Pupil diameter (mean, SD), baseline-corrected **[CHECK]** | STRONG (CL7 79%; CF5; NM7) |
| Fixation duration (mean, SD) | STRONG (CL7 73%) |
| Blink rate, blink duration | STRONG (CL7 71% / 58%; ET12) |
| Saccade amplitude, peak velocity | ADEQUATE (ET8–ET10) |
| Fixation rate, saccade rate | ADEQUATE (CL7, ET20, ET21) |
| Gaze dispersion (SD of x, y) | STRONG (ET14–ET16) |
| Stationary and transition gaze entropy | ADEQUATE (ET6, ET17, EM24) |

**Rules:**
- Leave out absolute gaze-position features (EM5).
- Prune features with |r| > 0.80 inside each training fold (EM5, FI13).

## 5. Windows and confounds

**Windows:**
- **Primary:** non-overlapping **30 s** windows within each activity, each inheriting the
  activity's label. Precedent: EM8, EM9, HR1.
- **Sensitivity:** whole-activity features (as in the COLET paper) and 10 s / 60 s windows.
  Fuhl used 10–30 s windows on COLET and found 25 s best.
- **No overlapping windows.** Overlap combined with random folds caused likely leakage in
  Wibirama 2025 (§9b).
- The final window length depends on activity duration. **[CHECK]**

**Pupil luminance confound:**
- The light reflex can exceed cognitive effects on pupil size (CF2, CF5).
- If puzzle images differ in brightness between activities, pupil size partly encodes the
  stimulus.
- **Mitigations:**
  - Check stimulus luminance per activity.
  - Report a **pupil-free sensitivity model**. EM5's pupil-free ablation came within about
    3 points of its full model.
  - Interpret pupil SHAP with care.

## 6. Per-person normalization

**Resolved:** the Zenodo README shows **no rest/baseline recording**, so the second row below
applies.
- **Primary:** per-person z-score (or subtractive centering, per NM7) of each feature over all
  of that person's activities. It is label-free, and every person has all 4 conditions.
- **Sensitivity:** no normalization.

**Original options table (kept for reference):**

| If… | Primary normalization | Support |
|---|---|---|
| A rest/baseline recording exists | Baseline calibration: z-score each feature on the person's baseline only | NM5 (leak-free; rest baseline works), NM4, NM7 |
| No baseline exists | Per-person z-score over all of that person's activities. It is label-free and transductive, and the design is balanced (every person has all 4 conditions). | Common practice: EM5, LB7, EM19, NM4, NM6 |

**Always report:**
- **No normalization** as a sensitivity analysis.
- **Explicitly:** if the transductive option is used, it relies on unlabeled data from the
  test participant. Tognotti found this can inflate scores by 3–13 points (NM5, EV2).

## 7. Models

| Model | Configuration | Support |
|---|---|---|
| Logistic Regression | Standardize (fold-fitted) + **L2** penalty, C tuned. Elastic-net variant if coefficients are read as importance. | M1, M8, X2, X4, X5 |
| XGBoost | Shallow, regularized: max_depth 2–6, learning_rate 0.05–0.1, 100–500 trees, subsample/colsample 0.7–1.0, λ/α tuned | M2, X2, X14, X21 |

- **Equal tuning budget** and identical folds for both models (EV7).
- **Missing values:** LR gets median imputation + indicators; XGBoost uses native NaN handling
  (MD1, MD2).
- **Framing:** expect XGB ≥ LR, with the size of the gap uncertain (§4b synthesis; M3, M4).

## 8. Validation

- **Outer loop:** LOPO, 47 folds (EV3, EV4, EM16, EM17, X6, X7).
- **Inner loop:** GroupKFold by participant, used for tuning only (EV7, EV8, X14, X22).
- **All preprocessing is fitted inside the training fold:** scaling, imputation, pruning
  (EV2, LB13).
- **Secondary analysis (contribution):** re-run the same pipeline with COLET's original random
  80/20 split.
  - This shows how much subject-dependent splitting inflates results (EV3, EV5).
  - The one subject-level COLET result reported AUC 0.67 (Haag, preprint, §9b).

## 9. Metrics, statistics, importance

Carried over from `methodology-replan.md` §8–9.

**Metrics:**
- ROC-AUC, balanced accuracy, macro-F1, per-class recall, and a majority baseline (EM5, X7).
- Brier score (M14).
- Reported pooled and per participant, with bootstrap CIs (EV10).

**Model comparison:**
- Paired Wilcoxon over 47 participants, with rank-biserial effect size and a bootstrap CI of
  ΔAUC (EV12).
- Optional: Bayesian equivalence test (M11).
- Grouped permutation test against chance (FI11).

**Importance:**
- LR standardized coefficients (FI12).
- XGB out-of-fold TreeSHAP (FI3).
- Permutation importance for both models (FI4, FI5).
- Kendall τ between the two models' rankings.
- Top-k stability across folds (FI14).
- Grouped importance for correlated features (FI10).
- Univariate Friedman check (EM5).
- Gain importance in the appendix only (FI6).

## 10. Pre-specified analysis plan

| # | Analysis | Role |
|---|---|---|
| P1 | Designed low vs high · LOPO · 30 s · core features · chosen normalization · LR vs XGB | **Primary** |
| S1 | Multitasking vs none (all 4 activities) | Secondary target |
| S2 | NASA-TLX labels (≥50) | Comparability with prior COLET work |
| S3 | Random 80/20 split (COLET protocol) | Shows the effect of leakage |
| S4 | No normalization; alternative normalization | Sensitivity |
| S5 | Pupil-free model | Luminance-confound check |
| S6 | Whole-activity features; 10 s / 60 s windows | Window sensitivity |

**Rule:** P1 is the headline result whatever scores highest.

## 11. What carries over from the current code

- **Reusable with small changes:** `src/models.py`, `src/evaluate.py`, `src/importance.py`,
  `src/figures.py`. The LOPO, nested tuning, Wilcoxon and SHAP logic are dataset-independent.
- **Must be rewritten:**
  - `src/data.py`: load COLET `.mat`, quality filter, event detection, windowing.
  - `src/labels.py`: condition labels and TLX labels.
  - `src/config.py`: paths and parameters.
  - `src/normalize.py`: baseline option.

## 12a. RRL support matrix

- **Columns:** each methodology decision, the sources that support it (IDs from
  `rrl-master-list.md`), and the overall strength.
- **STRONG** = 2+ strong, claim-matched sources, at least one in eye-tracking workload.
- **ADEQUATE** = support exists but is indirect (another modality) or rests on one strong
  source.
- **DESIGN CHOICE** = no source sets the value; we state and justify it.

| # | Decision | Eye-tracking workload sources | Method / general sources | Strength | Note |
|---|---|---|---|---|---|
| 1 | Compare LR (interpretable baseline) vs XGBoost | X1, X2, X3, X12, X13, EM1 | M1, M2, M3, M4, M5, M6 | **STRONG** | XGB ≥ LR in all 7 head-to-heads; the gap is uncertain |
| 2 | Use COLET (public, peer-reviewed) | EM7; 5 reuse studies (§9b) | DS6, DS7 | **STRONG** | Screen-based lab: limitation (DS1–DS3) |
| 3 | Eye-tracking-only features | CL7, EM5 (pupil-free ablation), EM17, EM18 | ET1 | **STRONG** | — |
| 4 | Labels from the experimental design (low vs high) | LB4, LB5, LB7, LB9, EM5, EM16, EM17, HR1–HR4 | LB6, LB8, LB12 | **STRONG** | 8/12 twins use condition labels |
| 5 | Drop the middle (single-factor) conditions | LB5 | LB7 (alternative: merge) | ADEQUATE | One strong precedent |
| 6 | NASA-TLX as manipulation check, not primary label | LB4, EM5 | CL4, LB6, LB8, LB10, LB11, LB12 | **STRONG** | — |
| 7 | NASA-TLX labels as a secondary analysis | EM7 and all COLET reuse studies | LB4 (expect it to be harder) | **STRONG** | Allows comparison with prior COLET results |
| 8 | Compute features from raw gaze (event detection: I-DT 1.0°/100 ms; I-VT sensitivity) | EM5 (I-DT), PP12, PP13 | PP1, PP2, PP3 | **STRONG** | Report parameters; run a sensitivity check |
| 8b | Sample-quality filter (confidence ≥ 0.8), gap handling, blinks | PP6, PP7, PP8 | PP5, PP10, CF2 | **STRONG** | 0.8 from two peer-reviewed Pupil Core studies; 0.6 is vendor guidance only |
| 8c | Pupil preprocessing (mm, range, MAD, 4 Hz low-pass, subtractive) | PP9 | PP11, NM7, CF2 | **STRONG** | — |
| 8d | Gaze entropy computation (grid, normalized) | PP14 | ET17, PP15 | **STRONG** | — |
| 8e | Stimulus luminance check | — | CF2, CF3 | **STRONG** | — |
| 9 | Pupil diameter features | CL7 (79%), EM5, X15 | CF5, NM7 | **STRONG** | Luminance caveat (row 16) |
| 10 | Fixation duration, blink rate/duration | CL7, ET12, ET13, ET22 | — | **STRONG** | — |
| 11 | Saccade amplitude/velocity, fixation and saccade rates | CL7, ET8–ET10, ET20, ET21 | — | ADEQUATE | Direction of effect varies by task |
| 12 | Gaze dispersion and gaze entropy | ET14–ET16, ET6, ET17, EM24 | — | **STRONG** (dispersion) / ADEQUATE (entropy) | — |
| 13 | Drop absolute gaze position; prune \|r\| > 0.80 | EM5 | FI7, FI10, FI13 | **STRONG** | Threshold taken from EM5 |
| 14 | 30 s non-overlapping windows | EM8, EM9, HR1; Fuhl on COLET (§9b) | WN4 | ADEQUATE | Exact value is a design choice; counter-evidence WN3 |
| 15 | No overlapping windows across folds | §9b (Wibirama leakage) | EV5, EV6 | **STRONG** | — |
| 16 | Pupil luminance check + pupil-free sensitivity model | EM5 (pupil-free ablation) | CF1–CF5 | **STRONG** | — |
| 17 | Per-person normalization | EM5, EM19, EM15, LB7, HR3 | NM1, NM4, NM5, ET2–ET4 | **STRONG** | Choice of baseline vs transductive depends on [CHECK] |
| 18 | Report no-normalization and transductive sensitivity | — | NM5, EV2, LB13 | ADEQUATE | Leakage argument from physiological signals |
| 19 | Missing values: impute + indicator (LR), native NaN (XGB) | — | MD1, MD2, MD3, M2 | **STRONG** | — |
| 20 | LR with L2 / elastic net | X2, X4, X5 | M8 | **STRONG** | — |
| 21 | XGBoost shallow and regularized; hyperparameter ranges | X2, X14 | M2, X21, X16 | **STRONG** | — |
| 22 | Equal tuning budget, nested grouped tuning | X14, X22 | EV7, EV8, EV9 | **STRONG** | — |
| 23 | Leave-one-participant-out outer loop | EM5, EM16, EM17, X6, X7 | EV3, EV4, EV9 | **STRONG** | High variance noted (EV9) |
| 24 | Preprocessing fitted inside training folds | X1, X7 | EV2, LB13 | **STRONG** | — |
| 25 | Random-split re-run to show leakage on COLET | §9b (COLET results under random vs subject-level splits) | EV3, EV5 | **STRONG** | Secondary contribution |
| 26 | ROC-AUC + balanced accuracy + majority baseline | EM5, X7 | EV11, EV13 | **STRONG** | — |
| 27 | Calibration (Brier) | — | M13, M14 | ADEQUATE | General ML evidence |
| 28 | Wilcoxon over participants + effect size + bootstrap CI | — | EV12, EV10, M9, M10 | **STRONG** | Adaptation of Demšar noted |
| 29 | Optional Bayesian equivalence test | — | M11, M12 | ADEQUATE | — |
| 30 | LR coefficients as importance | X5, X13, X19 | FI12 | **STRONG** | — |
| 31 | Out-of-fold TreeSHAP for XGB | X7, X16, X22, FI15 | FI1, FI3 | **STRONG** | — |
| 32 | Permutation importance for both models | — | FI4, FI5, FI11 | **STRONG** | — |
| 33 | Gain importance only in an appendix | — | FI3, FI6 | **STRONG** | — |
| 34 | Ranking agreement + fold stability + grouped importance | X5 | FI5, FI9, FI10, FI14 | **STRONG** | — |
| 35 | Univariate cross-check of top features | EM5 | — | ADEQUATE | One precedent |
| 36 | Pre-specified primary analysis | — | EV2, LB13 | ADEQUATE | Reporting good practice |

**Design choices with no source for the exact value** (state them plainly in Chapter 3):
- the 30 s window length;
- the 70% recording-validity and 80% window-validity cut-offs;
- the |r| > 0.80 pruning threshold (borrowed from EM5);
- which conditions form "low" and "high".

## 12. Next steps

1. **Adviser sign-off** on switching from GAZELOAD to COLET.
2. Download COLET (Zenodo 7766785) and obtain the descriptor PDF (open access at the
   publisher).
3. Resolve every **[CHECK]**:
   - activity duration;
   - baseline recording;
   - confidence values;
   - stimulus luminance;
   - exclusions;
   - tracker spec.
4. Source a peer-reviewed reference for the event-detection algorithm.
5. Team decisions:
   - primary label (designed low vs high, recommended, vs multitasking factor);
   - normalization option;
   - core vs exploratory features.
6. Only then: change `src/` and run P1 and the sensitivity analyses.
7. Rewrite Chapters 1–3 for COLET, then Chapters 4–5 from the new results.

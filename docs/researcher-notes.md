# Researcher Notes

Things the research team must understand and be able to explain to the adviser or panel:
judgment calls, known weaknesses, and traps in the data or the method. Each note says what
it is, why it matters, and what we do about it. Decisions themselves are recorded in
`rrl-decision-log.md`, and the method in `methodology-colet.md`.

Dataset: **COLET** (Ktistakis et al., 2022). Last updated: 2026-10-01.

---

## N0. The dataset and how we use it

- **What COLET is:**
  - 47 participants solved visual-search puzzles ("find the squares with a chandelier") on a
    screen while a Pupil Core eye tracker recorded their eyes at 240 Hz.
  - A chin rest kept their heads still.
  - After each activity they rated their workload with NASA-RTLX (0–100).
- **The main task, a picture puzzle like a CAPTCHA:**
  - A photo of an indoor scene (kitchen, classroom, garage, …) is split into a **3 × 3 grid**.
  - An instruction names an object ("Choose the squares in which pouffes are located").
  - Participants select every square that contains it. This is a visual-search task.
  - There are 21 photos (in `data/colet/images/`); each activity uses 5, drawn at random.
- **Secondary task: counting backwards out loud** from 1000 in steps of 4 ("1000, 996,
  992, …") while solving the puzzles.
- **Time pressure:** a spoken instruction to finish "as quickly as possible" instead of "at a
  comfortable pace". There was no timer.
- **Session:**
  1. Consent and vision tests.
  2. Seated with a chin rest, 80 cm from a 24" screen, wearing the Pupil Core tracker.
  3. One practice puzzle.
  4. The four activities in random order, with 2-minute breaks.
  5. NASA-RTLX after each activity.
- **The four activities** (a 2 × 2 design: time pressure × secondary task). Each activity is
  5 puzzle images, presented in random order:

| Activity | What participants did | Median length | NASA-RTLX (mean) | Our label |
|---|---|---|---|---|
| **A1** | Solve puzzles normally (single task, no time pressure) | 37 s | 19.4 | **Low load** |
| A2 | Solve puzzles "as quickly as possible" (time pressure) | 29 s | 28.7 | Not used in main analysis |
| A3 | Solve puzzles while counting backwards aloud from 1000 by 4 (multitask) | 62 s | 45.0 | Not used in main analysis |
| **A4** | Time pressure **and** counting aloud | 51 s | 51.9 | **High load** |

- **What we compare:** A1 (low) vs A4 (high), the easiest and the hardest (decision C2).
  - Using only the extremes and leaving out the middle follows Hogervorst 2014, Gado 2023,
    Wu 2020 and the COLET paper itself (LR 0.85 on A1 vs A4).
  - A2 and A3 are used only for the manipulation check (all four ratings: A1 < A2 < A3 < A4)
    and for optional extra analyses.
- **One sample per activity:** each person's whole A1 recording becomes one "low" example, and
  their whole A4 recording one "high" example (decision C1).
- **Testing:** leave-one-participant-out. Train on everyone except one person, test on that
  person, and repeat for every person.
- See also N2 (labels), N3 (talking in A3/A4), N4 (weak time pressure), N5 (activity length).

## N0b. Why A1 = low load and A4 = high load (justification)

**The decision:** A1 (single task, no time pressure) is labelled **low load** and A4
(multitask + time pressure) **high load**, for every participant. A2 and A3 are left out of
the main analysis. Six lines of evidence support it:

**1. The experiment was designed that way.**
- COLET is a 2 × 2 design with two load factors: a secondary task, and time pressure.
- A1 has **neither** factor and A4 has **both**, so they are the two ends of the designed
  load range.
- Comparing the extremes gives the clearest low-vs-high contrast.

**2. Participants felt it that way (manipulation check).**
- **COLET paper:** NASA-RTLX rose A1 < A2 < A3 < A4 (19.4, 29.2, 44.8, 52.2), and **every
  pair differed significantly** (Bonferroni-adjusted).
- **Our data, per person:** **all 47 of 47 participants rated A4 above A1** (median
  difference +31.7 points; none equal or reversed).

**3. Performance agrees.**
- The COLET paper reports more mistakes, longer completion times and worse inverse-efficiency
  scores in the multitask activities.
- So A4 was harder by behaviour, not only by rating.

**4. It is how the closest studies label workload (condition labels).**
- **8 of the 12 closest eye-tracking studies** use the experimental condition as the label,
  not self-report.
- Examples: Božak 2026, ADABase (Oppelt 2023), WAUC (Albuquerque 2020), Hogervorst 2014,
  Fridman 2018.
- Schmidt 2019 recommends using study conditions as labels and questionnaires to verify them.
  That is exactly our setup: condition = label, NASA-RTLX = check.

**5. Using only the extremes and dropping the middle is a known approach.**
- Hogervorst 2014 classified 0-back vs 2-back and left out 1-back.
- Gado 2023 and Wu 2020 also used only low vs high extremes.
- **The COLET paper itself ran A1 vs A4 as a binary task** (LR 0.85, kNN 0.86), so our
  result is directly comparable.

**6. Why not label by each person's own rating instead.**
- Self-report numbers aren't comparable across people:
  - "high between-subject variability" (Hart & Staveland 1988);
  - "limited absolute validity" of rating numbers (Marinescu 2018);
  - subjective and physiological measures often disagree (Matthews 2015; Hancock & Matthews
    2019).
- Gado 2023 tested both on the same eye features: **condition labels were learnable (F1
  0.69), self-report labels were not (F1 0.35, below chance).**

**Why A2 and A3 are left out:**
- They are the middle conditions.
- Time pressure was only a spoken instruction, and the COLET authors call it weak; A3 vs A4
  was near chance (0.60) in their results. Including them would blur the low/high boundary.
- They are still used in the manipulation check (all four ratings) and for optional extra
  analyses.

**Limits of this choice (be ready to explain):**
1. **The label is the task's demand, not each person's feeling.** One participant's A1–A4
   rating gap was only 1.6 points; they are still labelled low and high.
2. **A4 includes talking** (counting aloud), which affects blinks and data quality (N3).
3. **Because time pressure was weak, the A1–A4 contrast is driven mostly by the secondary
   task.** Our "high load" is in practice dual-task load with a spoken secondary task.

**Sources:**
- COLET paper (EM7); Božak 2026 (EM5); Oppelt 2023 (LB7); Albuquerque 2020 (LB6);
  Hogervorst 2014 (LB5/EM9); Fridman 2018 (LB9); Schmidt 2019 (LB8); Gado 2023 (LB1/LB4);
  Wu 2020 (EM14).
- Hart & Staveland 1988 (CL4); Marinescu 2018 (CL6); Matthews 2015 (LB10); Hancock &
  Matthews 2019 (LB11).
- Full citations: `rrl-colet-list.md`.

## N1. Fixations and saccades are calculated, not given

- **What:** COLET only contains raw gaze positions, about 240 per second. Fixations (the eye
  holding still) and saccades (quick jumps) must be **detected** from those positions with an
  algorithm. This is standard practice (COLET, Božak 2026, GAZELOAD, Upasani 2024,
  Trabulsi 2021).
- **How:**
  1. Convert gaze positions to degrees.
  2. Compute eye speed with a smoothing filter.
  3. Below **45°/s** = fixation; above = saccade.
  4. Keep fixations of at least **55 ms**.

  Sources: Duchowski; Salvucci & Goldberg 2000; Andersson 2017; COLET's own settings.
- **Why it matters:** the settings change the results. Studies use 30–75°/s and 55–100 ms.
  We use COLET's settings so our features are comparable with theirs, and we report them
  (Komogortsev 2010).
- **Watch out:**
  - The first exploration used a quick hand-rolled detector with unsourced details. Its
    fixation and saccade numbers did **not** match COLET's paper (e.g. saccade size 2.6° vs
    14°). **Do not use those numbers.**
  - The final detector is checked with the lean sanity check (N23, V2-2). COLET's Table 4 is
    compared **descriptively only** (N18).
  - COLET's source for the 55 ms rule is a preprint, so cite Trabulsi 2021 alongside it.
- **Details:** `methodology-colet.md` §5.

## N2. Labels come from the activity, not from how each person felt

- **What:** every A1 recording is "low load" and every A4 recording is "high load", based on
  the experiment design (condition labels, as in 8/12 of the closest studies).
- **Check:** all 47 participants rated A4 as more demanding than A1 on NASA-RTLX (median
  +31.7 points).
- **Watch out:** one participant's gap was only 1.6 points. They are still labelled low and
  high, because the label reflects the task's demand, not the exact feeling. This is normal
  for condition labels, but be ready to explain it.

## N3. The hard activities include talking

**Talking and counting are not eye measures,** but they affect the eyes indirectly. The
secondary task has two parts:

1. **Counting backwards** is mental arithmetic. This is the **intended** extra load, which we
   want the eyes to reveal.
2. **Saying it out loud** is a physical action. It is not mental load, but it changes the
   eye data anyway:
   - **More blinks:** people blink more while speaking (Bentivoglio 1997: 17 → 26/min).
   - **Jittery gaze and lost samples:** jaw and face movement shakes the head and tracker.
     This can look like more and faster saccades. Invalid samples rise from 0.9% to 7.3%
     (N15).
   - **Possibly bigger pupils,** from the general arousal of speaking.

So the eye changes in A4 = thinking harder (wanted) + talking (side effect), and the tracker
cannot separate them.

Analogy: measuring stress from heart rate while also asking the person to walk.

This comes from how COLET designed the task (a silent version would have avoided it), not
from our method.

**Details:**

- **What:** in A3 and A4, participants count backwards **out loud**. Talking makes people
  blink more by itself (Bentivoglio 1997). Blink rate jumps from about 2 to about 15 per
  minute.
- **Why it matters:** a model may partly detect "talking", not only "high mental load".
- **What we do:** we follow the RRL (decision C2b). Blinks stay as workload features, as in
  the COLET paper and Pluchino 2023, which did not separate the two.
- **Recommended:** one sentence in the limitations saying the secondary task was spoken.

## N4. Time pressure was weak

- **What:** time pressure was only a spoken instruction ("as quickly as possible"), with no
  timer. The COLET authors themselves judged it weak; A3 vs A4 was near chance in their
  results.
- **Why it matters:** most of the A1 → A4 difference comes from the counting task, not the
  time pressure.

## N5. One sample per whole activity (decision C1)

**Terms:**
- An **activity** is one of the four puzzle tasks (A1–A4).
- A **recording** is one person's eye data during one activity (47 × 4 = 188 recordings).
- A **window** is a fixed-length slice of a recording (e.g. 30 s), used to turn one recording
  into several examples.

**The decision:** each recording becomes **one example**. Its features (fixation rate, pupil
mean, blink rate, …) are computed over the whole activity:
```
Participant 1, A1 (37 s) → 1 example → "low"
Participant 1, A4 (51 s) → 1 example → "high"
```

**Why not windows:**
- Activities are short: 11–141 s, with a median of 37 s (A1), 29 s (A2), 62 s (A3) and
  51 s (A4).
- The typical A2 holds **no** full 30 s window. 10 s windows give only 2–6 per activity, and
  rates such as blinks become noisy.
- Multitask activities last **about twice as long**, so windowing would give the "high" class
  more examples because the task took longer, not because the load was higher.
- The 30 s windows from the earlier GAZELOAD plan therefore do not carry over.

**Support:**
- The **COLET paper** itself uses whole activities, and says trials were too short to
  analyse separately.
- Whole-task or whole-block features are also used by Kaczorowska 2021 (90–180 s task
  parts), Wu 2020 and Shafiei 2025 (whole exercises).
- Short windows (2.5–10 s) are also common (Božak 2026: 3 s). They are kept as an optional
  sensitivity check.

**Consequences:**
- **Main analysis size:** 45 participants × 2 activities = **90 samples** (see N8, N9).
- **Per-person result:** each held-out person has one low and one high sample, so the result
  is "ranked correctly or not" (N8).
- **No length-based features:** activity length, and anything that grows with it (raw
  counts), must not be a feature, or the model would learn "longer = harder". Use rates
  (per second or per minute) instead.

## N6. The original COLET results are probably inflated

- **What:** the COLET paper used a random 80/20 split, so the same person could appear in
  training and testing. It also selected features (ANOVA) and scaled them on all the data
  before splitting. These are known leakage risks (Saeb 2017; Kapoor & Narayanan 2023).
- **Why it matters:** our leave-one-participant-out results will likely be **lower** than
  their LR 0.85. That is expected and more honest, not a failure.
- **What we do:** analysis S3 first reproduces COLET's protocol exactly, then removes one leak
  at a time, to show which step inflates results.
- **Note:** no peer-reviewed participant-held-out result on COLET was found, so we don't cite
  a "true" COLET baseline. (A preprint reporting one was removed under our peer-reviewed-only
  rule.)

## N7. Per-person standardization (decision D-M4)

**Decision (2026-10-01): we use per-person standardization.**

**What it is (plain terms):**
- Each person's eye numbers are rewritten as "how unusual is this **for them**":
  (value − that person's average) ÷ that person's spread, a z-score.
- It is done separately for every person and every feature, using that person's 4
  activities.
- Then **one model learns from all people together**.
- Nobody is grouped (no "small-pupil group"). Each person is only compared with their own
  normal.

**Why it's needed:**
- Differences between people are much bigger than the workload effect.
- Example: pupil size rises about 0.3 mm from A1 to A4, but normal pupil size varies by
  millimetres between people.
- Without it, the model mostly learns **who** has big pupils, not **who is working hard**.
- Everyone has a stable personal eye "signature" (Bargary 2017; Poynter 2013).

**It is not overfitting.** The model is never trained on the test person. The issues are
these two:
1. **A small hint from the test person's own data.** Their "normal" is computed from their 4
   activities (labels are not used), so the test is somewhat easier than meeting a total
   stranger. Tognotti 2026 measured 3–13 points of inflation from this kind of step.
2. **Bias toward COLET's balanced design.** It works because everyone did 2 easier and 2
   harder activities. A real user who only did hard tasks would have a "high normal", and
   their hard task would look normal. So in real use it needs a calibration step.

**The options considered:**

| Option | What | Pros | Cons |
|---|---|---|---|
| (a) Report both | With and without, side by side | Most transparent; measures the bias | More to report |
| **(b) With only (chosen)** | Per-person standardization | Standard in the closest studies; lets the workload signal show | Open to the "easier test" critique; relies on the balanced design |
| (c) Without only | Raw features, global scaling | Strictest, a "total stranger" test | Likely weak results; ignores what most RRL studies do |
| (d) Rest-baseline calibration | "Normal" from a separate calm recording | Avoids the bias | **Not possible:** COLET has no rest recording |

**What the RRL did:**

| Approach | Studies |
|---|---|
| Per-person standardization on own data (like ours) | **Božak 2026** ("participant-referenced"), **ADABase** (Oppelt 2023), Gado 2023, Shafiei 2025, Rizzo 2022, Nerella 2026, Albuquerque 2022; Hefron 2018 (**EEG**, a method precedent only) |
| Separate rest/baseline recording | Appel 2023, Pluchino 2023, Upasani 2024, Wozniak 2024, Aygun 2022, Laut 2026 |
| Global scaling only | COLET paper, Choi & Nam 2026, He 2025 |
| Compared with vs without | Albuquerque 2022 (none 0.649 / per-person 0.708 / baselines 0.60–0.64); Fdez 2021; Tognotti 2026 |
| Warned about the bias | Tognotti 2026 (inflation); Kapoor & Narayanan 2023 (preprocessing with test data = leakage) |

**What to say to a panel:**
> "We standardized each participant's features relative to their own average, as in the
> closest prior studies (Božak 2026; Oppelt 2023; Gado 2023). This removes stable individual
> differences in eye behaviour. It uses each participant's unlabelled recordings and relies
> on COLET's balanced design, so in practice it corresponds to a calibrated user; we state
> this as a limitation."

**Limitation sentence (Chapter 3 or 5):** per-person standardization uses each
participant's own unlabelled data and assumes a mix of low- and high-load recordings per
person. Deployment would therefore require a calibration session.

## N8. The sample is small

- **What:** after exclusions (decision C3): **45 participants, 90 samples** (one A1 and one A4
  each).
- **Why it matters:**
  - Results from leave-one-participant-out vary a lot from person to person (Varoquaux 2017,
    2018), so report per-participant results and confidence intervals, not only one overall
    number.
  - Each held-out person has one low and one high sample, so their individual result is
    simply "ranked correctly or not".

## N9. Excluding bad data (decision C3)

**The rules (fixed before any model results were seen):**

| Rule | Value | Source |
|---|---|---|
| (a) A gaze sample is **invalid** if the tracker's confidence is below… | **0.8** | Faraji 2023; Hausamann 2020 |
| (b) Exclude a **recording** if its share of invalid gaze samples is above… | **35%** | Nenna 2023 (dropped trials with > 35% missing data) |
| (c) If a person loses A1 or A4, drop that **person** from A1 vs A4 | Yes | Each person needs one low and one high sample to be tested fairly |
| (d) Ignore **blink events** shorter than 50 ms or longer than 500 ms | Yes | Steinhauer 2022 (blinks typically ~200 ms, can exceed half a second); Hershman 2018 |

**What it removes:**
- **Recordings:** 3 of 188. P06 A3 (57% invalid), P17 A4 (49%), P06 A4 (39%). All others are
  ≤ 24%.
- **People from the main analysis:** 2 (P06, P17), because their A4 is excluded. Their A1
  recordings were fine.
- **Main analysis after exclusion: 45 participants, 90 samples** (96% of the data kept).
- **Blink events:** 211 of 1,992 (11%) are ignored as implausible.

**Why 35% and not lower:**
- Single-task activities are almost clean (median 0.9% invalid). Multitask ones lose more
  (median 7.5%), because talking and blinking cause tracking dropouts.
- A strict cut such as 20% would remove 13 recordings, **mostly high-load ones**. That would
  bias which high-load data survive.
- At 35%, only clearly broken recordings go.

**Watch out:** the invalid share is higher in high-load activities for a reason connected to
the task (talking). That is part of N3, not a data error.

## N10. Pupil size and screen brightness

- **What:** pupil size reacts to light as well as to mental effort (Steinhauer 2022; Mathôt
  2018). The puzzle images differ in brightness.
- **What's known:** COLET checked this. Only 2 of 47 participants showed pupil size following
  image brightness. Lighting was controlled (400–450 lx).
- **Still to do:** our own stimulus-brightness check (pinned, C8).

## N11. Results can change across computers

- **What:** in the earlier GAZELOAD runs, the same code and seed gave different numbers on two
  machines (XGBoost behaving slightly differently). One "significant" result disappeared.
- **What we do:** pin package versions, run the final experiments in one environment
  (e.g. Colab), and record the versions in the thesis.

## N12. How the data were converted

- **What:** COLET's MATLAB file stores data as MATLAB "tables", which standard Python tools
  can't read. We used the open-source `mat-io` package to decode it, then saved every table
  as a Parquet file without changing any values.
- **Why it matters:** be able to describe the data-preparation chain. It is reproducible:
  `data/colet/convert_colet.py`, from the Zenodo file (MD5 verified).

## N13. The gaze positions are in camera coordinates, not screen coordinates

- **What:** COLET's `norm_pos_x/y` gaze positions are relative to the **head-mounted
  world-camera image**, not the screen.
  - In the data, gaze covers only about 0.4–0.6 of the image width.
  - Matching the positions to the tracker's 3D gaze direction implies a camera field of view
    of about **94° × 53°**.
- **Why it matters:**
  - Converting these positions with the *screen* size would give wrong angles. That would
    make the fixation and saccade features wrong (review M1).
  - The COLET paper doesn't say how it converted them, so its saccade numbers (median
    amplitude about 14°) may not be reproducible exactly.
- **What we do (D-M1, revised after round-4 review R4-1):** use each eye's own gaze direction
  (`gaze_normal0/1`), averaged over the two eyes. The 3D gaze point and these camera positions
  both depend on a faulty depth estimate (N18, N25).

## N14. Two timing problems in the raw data

- **D1, stray pupil samples:** P18's A3 and A4 pupil streams contain samples about **2 hours
  after** the activity ended (7,250 s and 6,955 s of pupil data vs 68 s and 51 s of gaze).
  **Fix:** trim pupil and blink data to each recording's gaze time range.
- **D2, long gaps:** P18 A3 has a 12 s gap in gaze, P18 A4 3.7 s, and P16 A4 1.4 s. **Fix:**
  compute rates over **valid time** (gaps over 1 s excluded), not elapsed time. P16 A4 and
  P18 A4 are in the main analysis.
- **Why it matters:** both would distort durations, blink rates and pupil statistics if left
  in.

## N15. Data loss and missing values differ by condition

- **Data loss:**
  - Invalid gaze samples: median 0.9% in A1 vs 7.3% in A4.
  - Higher in A4 for 41 of 45 participants (Wilcoxon p = 4.8×10⁻⁸).
  - This is consistent with talking during A4 (N3).
- **Missing values:**
  - Blink duration is missing in 13 of 45 A1 recordings (no blinks) and in 0 of 45 A4
    recordings.
  - A model could learn "missing = low load" from that alone (review M3).
- **What we do:**
  - Data-quality share is reported by condition, never used as a feature.
  - **Decided (D-M3): the blink-duration feature is dropped;** blink rate is kept. This
    follows Hogervorst 2014, which discarded segments whose blink duration was undefined.
    Without it, a model could learn "no blink duration = A1".

## N16. Ethics and licence

- **Ethics:** COLET was approved by the FORTH Ethics Committee (110/12-02-2021). Our study is
  a secondary analysis of public, de-identified data.
- **Licence:** the data are CC BY 4.0 (Zenodo record 7766785); the article is CC BY-NC-ND.
- Cite the dataset and the paper.
- Suggested Chapter 3 sentence: *"This study is a secondary analysis of the publicly
  available, de-identified COLET dataset (CC BY 4.0), whose collection was approved by the
  Ethics Committee of the Foundation for Research and Technology – Hellas (110/12-02-2021)."*

## N17. What happened to the eyes as load increased (COLET paper, Tables 4–5)

| Eye measure | A1 → A4 | Change | Main driver | RRL expectation | Match? |
|---|---|---|---|---|---|
| Pupil size | 3.49 → 3.82 mm | Bigger | Counting aloud and time pressure | Bigger (Tao 2019; Mathôt 2018) | ✅ |
| Blink rate | 3 → 14 /min | About 5× more | Counting aloud | Up with mental load, down with visual focus (Recarte 2008) | ✅, but talking also raises blinks (N3) |
| Blink duration | 206 → 212 ms (229 in A3) | Slightly longer | Counting aloud | Longer (Tao 2019) | ✅ |
| Saccades per second | 1.7 → 3.4 | About 2× more | Counting aloud | Mixed, often lower (Tao 2019) | ⚠ |
| Saccade speed, average / peak | 146 → 268 / 217 → 357 °/s | Much faster | Counting aloud | Slower (Di Stasi 2010) | ❌ Opposite |
| Saccade duration | 15 → 19 ms | Slightly longer | Counting aloud | — | — |
| Saccade size | 14.1° → 14.1° | No change | — | Smaller (May 1990) | ⚠ |
| Fixations per second | 2.50 → 2.35 | Slightly fewer | Counting lowered it; time pressure raised it | Task-dependent (Liu 2022) | ⚠ |
| Fixation duration | 273 → 254 ms | Shorter | Time pressure (hurrying) | Longer (Tao 2019) | ❌ Opposite |

**In plain words:** under high load, the eyes **opened wider**, **blinked far more**, made
**more and faster jumps**, and paused a little less on each spot. Jump size didn't change.
Most changes came from the counting-aloud task. Time pressure mainly shortened fixations and
slightly enlarged the pupil.

**What it means:**
- Pupil and blinks behaved as the literature predicts.
- Faster saccades and shorter fixations go the "wrong" way, probably because of:
  - talking (N3);
  - hurrying under time pressure;
  - COLET's undisclosed gaze-to-angle conversion (N13).
- In Chapter 4, explain the observed direction rather than assuming the textbook one.
- When reading the feature-importance results, a top rank for blinks or saccade speed may
  partly reflect talking.

**Caution:** these are the COLET authors' numbers. Our rebuilt pipeline must confirm the
directions in our own data. Our pupil and blink values already match theirs.

## N18. How we turn gaze into angles (decision D-M1)

**The problem:**
- Fixations and saccades are measured in **degrees**: how far the eye turned.
- COLET's 2D gaze positions (`norm_pos_x/y`) are positions on the **head-mounted camera's
  image**, not on the screen (N13).
- So they cannot be converted with the screen size and viewing distance, as screen-tracker
  studies do.

**Revised after round-4 review (R4-1).** The first choice, the tracker's 3D gaze point
(`gaze_point_3d`), turned out to be broken in COLET (N25):
- To place a 3D point, the tracker guesses **how far away** you are looking, from how much the
  eyes turn inward. Its typical guess is **11 cm**; the screen is **80 cm** away.
- In 39 of 188 recordings it sometimes puts the point **behind the head**, which flips the
  direction (P37: gaze "spread" of 102°).
- The 2D camera positions (`norm_pos`) are made from that same point, so they share the error.

**What we do now:**

| Part | What | Why it is defensible |
|---|---|---|
| **Main method** | Use each eye's own **gaze direction** (`gaze_normal0/1`), averaged over the two eyes. The angle between two consecutive directions is how far the eye moved, in degrees. | A direction needs no distance guess. Averaging cancels each eye's inward bias (N25). Removes the 102° artifact (max spread 9.4°). **RRL:** distance guesses from the eyes' turning are unreliable at about 80 cm (Hooge 2019) and on Pupil Core specifically (Velisar & Shanidze 2024); eye speed from the angle between direction vectors (Kothari 2020). **Our own choice:** no study names `gaze_normal` itself. |
| **One eye missing** (pending team OK) | Treat the sample as missing; the P6 gap rule handles it | Each eye alone is about 10° off, so switching eyes would create fake jumps (N25) |
| **Check: sanity check** | Median fixation 150–400 ms and saccade:fixation 0.8–1.25, in every activity (N23) | Fails → 5-feature fallback |
| **COLET comparison** | Put our averages next to COLET's Table 4 and explain differences; main-sequence plausibility described, not tested | COLET never described its conversion, so exact replication isn't possible. Comparing openly is honest. |

**Dropped:** the cross-check between two conversions (and its ±15% rule). Both came from the
same faulty point, so agreeing would prove nothing.

**What to say to a panel:**
> "COLET's 3D gaze point relies on a depth estimate that was implausible at the 80 cm viewing
> distance (median 113 mm) and sometimes placed gaze behind the camera. We therefore computed
> angles from the tracker's per-eye gaze directions, averaged across the two eyes, which do not
> depend on depth. The detected events passed a pre-stated sanity check in every activity."

**Watch out:**
- The main-sequence source (Di Stasi 2011) was read at abstract level only. Read the full
  text before the defense.
- Our earlier first-pass numbers came from a rough version of this method, without the
  checks. They must not be used (N1).
- Expect our saccade numbers to differ from COLET's (their median saccade is about 14°). That
  is explained, not a failure.

## N19. Required vs optional analyses (decision D-N5)

**Required (the title needs only this):**
- **P1:** LR vs XGBoost on eye-tracking features, A1 (low) vs A4 (high), leave-one-participant-out,
  with feature importance.

**Also required (added for round-2 review V2-1):**
- **P2, pupil-only model:** the same as P1, but using only pupil mean and pupil SD.
- **Why:** talking during counting aloud affects up to 6 of the 10 features: blink rate, saccade
  rate, amplitude and peak velocity, gaze spread x/y (N3, N22). Pupil is the most established
  workload signal (Tao 2019: 79%) and the least affected by talking.
- **Precedent:** pupil-only models in Rolon-Merette 2026, Appel 2018 and Stolte 2020.
  (Karunathilake 2025 was dropped as support: abstract only, K-means labels; V3-5.)
- **How to read it:**
  - P2 close to P1: the result isn't just talking.
  - P2 lower but above chance: part talking, part genuine load.
  - P2 at chance: the full result was mostly talking (an honest finding).
- **Caveats:**
  - With only 2 features, LR and XGBoost will likely perform similarly.
  - Pupil is still exposed to arousal and to 3D-model refits (N22), so the refit rule (V2-3)
    matters for P2.

**Optional:** none of these is required. Each answers a likely panel question. Decide later
which, if any, go in the paper; anything not done can be mentioned as future work.

| Optional analysis | Panel question it answers |
|---|---|
| S1: single vs multitask (A1+A2 vs A3+A4) | "Time pressure was weak. Isn't this really counting vs no counting?" (D-M5 chose to keep it) |
| S2: NASA-RTLX labels, COLET's bins (low 0–29 vs high 50–100) | "How do your results compare with the COLET paper?" |
| S3: COLET replication (COLET's method vs ours; optional ladder) | "Why is your score lower than COLET's 85%?" (D-M7) |
| S4: no per-person standardization | "Doesn't standardization make the test easier?" (D-M4) |
| S5: pupil-free model | "Isn't pupil size affected by screen brightness?" |
| No-blink model | "Isn't the model just detecting talking?" (N3) |
| I-DT event detection instead of I-VT | "Do the results depend on the detection algorithm?" (N1) |
| Confidence threshold 0.6 instead of 0.8 | "Do the results depend on the data-quality cut-off?" |
| A1 vs A2 | "Can load be detected without any talking?" |

**Why keep the list short:**
- Running many versions invites the "you tried many things until one worked" critique.
- Each analysis costs team time.
- Božak 2026 kept a small, focused set, and the Demirezen 2024 checklist says to pre-specify
  what is run.

## N20. Feature importance: LR weights and SHAP (decisions D-N6, D-M8)

**What feature importance is:** which eye measures each model relied on most. It is in the
title, so it must be explained.

**LR weights (elastic net, D-N6):**
- LR gives every feature a weight; a bigger weight means more important.
- With only 90 samples, weights can grow too large (overfitting), so a penalty keeps them
  modest.
- **Elastic net** also sets useless features to zero and handles related features (e.g.
  saccade rate and speed) more cleanly.
- This follows Kaczorowska 2021/2022, which used elastic-net LR weights as importance for
  eye-tracking workload.

**SHAP (D-M8):**
- SHAP splits each prediction into how much each feature pushed it. Averaged over all
  predictions, that is each feature's importance.
- XGBoost has no weights, so SHAP is the standard tool for it (Lundberg 2020; Božak 2026).
- **We also run SHAP on LR**, so both models are measured with the same tool, on the same
  scale. LR weights are still reported.

**Watch out:**
- **No RRL study did SHAP for both models.** The RRL used weights for LR (Kaczorowska) and
  SHAP for trees (Božak, Shafiei, Xu).
- Our justification is the SHAP method itself: it works for any model (Lundberg & Lee 2017).
- Present it as an addition that makes the comparison like-for-like, not as standard practice.
- Importance means what the model **relied on**, not what causes workload (Molnar 2022).

## N21. The feature set and why some RRL features were left out (decision C4)

**The 10 features** (one value per recording, computed over the whole activity):
1. pupil size (mean);
2. pupil size variability (SD);
3. blink rate;
4. fixation rate;
5. fixation duration;
6. saccade rate;
7. saccade amplitude;
8. saccade peak velocity;
9. gaze spread, horizontal;
10. gaze spread, vertical.

They cover the core features used across the RRL: pupil size, blink rate, fixation duration
and rate, saccade rate, amplitude and velocity (COLET; Tao 2019; Božak 2026; He 2025).
Gaze spread follows Božak 2026 (fixation dispersion among the top features) and Tao's
"fixation spread".

**Left out, and why.** These are not bad workload measures; the reasons are specific to our
data.

| Feature | Used by | Why left out | Support |
|---|---|---|---|
| Blink duration | COLET, Hogervorst 2014, Tao 2019 | Missing whenever there is no blink: 13/45 A1 vs 0/45 A4. It would reveal the label. Blinking is still covered by blink rate. | Hogervorst 2014 (discarded undefined blink duration); Kapoor & Narayanan 2023 (leakage) |
| Mean saccade velocity | COLET, Tao, Rahman 2021 | Near-duplicate of peak velocity (ρ = 0.95) | Dormann 2013; Strobl 2008; Molnar 2022; Božak 2026 (\|r\| > 0.80 pruning) |
| Saccade duration | COLET, Tao | Tied to amplitude (ρ = 0.83, main sequence) | Same as above |
| Variation, skewness, kurtosis of each measure | COLET only | About 18 extra features for 90 samples (overfitting risk); not among Tao's 13 workload measures; hard to interpret | Kaczorowska 2021 (6–8 of 20 features improved results); Božak 2026 (488 → 58); Varoquaux 2018 |
| Gaze entropy | Upasani 2024, Wu 2020 | **Kept as exploratory, not dropped.** Direction is inconsistent (Diaz-Piedra 2019 vs Di Stasi 2016) and it is not in Tao's list. | Shiferaw 2019 |

**Chapter 3 sentence:**
> "Ten features were retained. Blink duration was excluded because it is undefined when no
> blink occurs, which happened only in low-load recordings and would have revealed the label
> (cf. Hogervorst et al., 2014; Kapoor & Narayanan, 2023). Mean saccade velocity and saccade
> duration were excluded as near-duplicates of peak velocity and amplitude (ρ = 0.95 and
> 0.83; cf. Božak et al., 2026; Dormann et al., 2013). Higher-order distribution statistics
> used by Ktistakis et al. (2022) were not included, to keep the feature set small relative
> to the sample size and limited to measures with established workload evidence (Tao et al.,
> 2019)."

**Note:** the correlations (0.95, 0.83) come from the first-pass exploration. Recheck them
once the rebuilt pipeline (N18) produces final features.

## N22. Round-2 data checks: blinks vs data loss, and pupil-model refits

Run 2026-10-01 for the round-2 review (V2-1, V2-3). Label-blind except that results are shown by
activity, as for the data-quality figures.

**Blink rate vs data loss (V2-1):**
- Blinks themselves create low-confidence samples, so the raw correlation (ρ 0.60–0.78 within
  each activity) is partly automatic.
- The fairer check is data loss **outside** blink periods (±100 ms):

| Activity | Invalid samples outside blinks (median) | ρ with blink rate |
|---|---|---|
| A1 | 0.2% | 0.44 |
| A2 | 0.3% | 0.31 |
| A3 | 3.1% | 0.44 |
| A4 | 2.5% | 0.52 |

- Outside blinks, data loss is about **10× higher in the counting-aloud activities**. That is
  tracking loss from talking, not blinks.
- People who blink more also lose more data (moderate ρ). Blink rate therefore partly reflects
  tracking quality, which supports the reviewer's concern that talking affects several
  features.

**Pupil 3D-model refits (V2-3):**
- Pupil Core sometimes refits its 3D eye model during a recording, and `diameter_3d` can jump
  when it does.

| Activity | Recordings with > 1 model (of 45) | Max models in one recording |
|---|---|---|
| A1 | 8 | 4 |
| A2 | 11 | 2 |
| A3 | 21 | 5 |
| A4 | 19 | 5 |

- Refits are **more common in multitask activities** (A4 > A1 in 17 participants, A4 < A1 in 3;
  Wilcoxon p = 0.001). Likely cause: talking moves the headset.
- So pupil-size steps from refits could differ by condition. A handling rule is needed
  (decision V2-3).

## N23. Scope rule and the four "lean" decisions (round-2 review)

**The scope rule:**
- The RRL tells us what is **acceptable**, not what is **required**.
- We only include steps needed for valid **P1** (LR vs XGBoost, A1 vs A4, 10 features) and
  **P2** (pupil-only).
- Everything else the literature or reviewers suggest becomes **optional, a limitation, or
  future work**.
- **For a panel:** "we prioritized a focused, fully specified analysis that answers the
  research question, and list further checks as future work."

### V2-3. Pupil-model refits (lean)

**The problem:**
- The tracker measures pupil size using a 3D model of the eyeball. Sometimes it **rebuilds**
  that model mid-recording, and pupil size can then **jump** although the eye didn't change.
- Refits happen more in the counting-aloud activities (19–21 vs 8–11 of 45 recordings; N22).

**What we do:**
- Keep the speed-based outlier filter already in the plan (Kret & Sjak-Shie 2019). It removes
  impossibly fast changes, i.e. the moment of the jump.
- **Count** refits per recording and report them.

**What we don't do:** per-segment pupil statistics. That would also correct the level shift
after a jump, but no RRL study did it.

**Limitation sentence:** "Pupil Core occasionally re-fits its 3D eye model during a recording,
which can shift pupil-size estimates; refits were more frequent in multitask activities and
are reported per recording."

**Second limitation sentence, for P2 (V3-1, decided in round 4):** "The pupil-only model (P2)
is not fully independent of the spoken secondary task: Pupil Core's 3D eye-model refits, which
can shift pupil-size estimates, were more frequent in the multitask activities (N22)." The
refit-free P2 run is not done (it would add an analysis); it can be named as future work.

**RRL:** no study discusses refits. Kret & Sjak-Shie 2019, Mathôt 2018 and Steinhauer 2022 give
the general pupil-cleaning guidance.

### V2-2. Checking the fixation/saccade detection (lean)

**The problem:**
- We detect fixations and saccades ourselves (N1). We need a simple way to show it worked.
- The first pass found more saccades than fixations, a sign of noise. In real viewing they
  alternate: look, jump, look, jump.

**What we do:** one sanity check after detection.
1. Is the median fixation duration in the normal range, **150–400 ms**? (COLET reports
   273 ms.)
2. Are there roughly **as many saccades as fixations**? Pass if the ratio is **0.8–1.25**
   (V3-3).

Both checks are run **for each activity separately** (R4-2). Talking noise is mostly in A3/A4,
so a check on everything mixed could pass while A4 alone fails. (The ±15% cross-check between
two conversions was removed in round 4; N18.)

**If both pass in every activity,** we use the fixation and saccade features. **If it fails,** P1 runs on the
5 features that don't need detection: pupil mean, pupil SD, blink rate, gaze spread x/y.

**What we don't do:**
- a third conversion method (`gaze_normal0/1`);
- a formal main-sequence slope test.

**RRL:**
- Komogortsev 2010 and Andersson 2017: detection should be evaluated, and results depend on
  the settings.
- Typical values: Salvucci & Goldberg 2000, Trabulsi 2021, COLET.
- Fallback to a feature subset: Božak 2026 (fixation-only model).
- Writing these as explicit pass/fail rules is our own addition, and so is the 0.8–1.25
  tolerance: no RRL source sets it.

### V2-9. XGBoost settings (just a setting)

**The problem:** XGBoost builds many decision trees. Deep trees with only 88 training samples
can **memorize** the data (overfitting).

**What we do:** search only shallow trees, **depth 1–3**, with **50–300 trees**. LR gets the
same tuning budget.

**RRL:**
- Walocha 2025 (small dataset): tuning most often chose depth 2.
- Shafiei 2025 searched down to depth 1.
- Cawley & Talbot 2010 warn against over-tuning on small data.
- Deep settings (e.g. Nerella 2026, depth 7) come from much larger datasets.

### V2-4. Possible "ceiling" (one sentence)

**The problem:**
- Blinks rise a lot from A1 to A4 (about 3 → 14 per minute in COLET). With per-person
  standardization, P1 might be almost perfect for **both** models.
- Then LR and XGBoost can't differ, and the comparison in the title is a tie at the top.

**What we do:** write this rule **before** seeing results: *"If both models exceed 0.95 AUC in
P1, the model comparison is interpreted mainly from P2."*
- P2 (pupil-only) is harder, so the models have room to differ there.

**RRL:**
- Near-perfect results happen (COLET: 0.98 on A2 vs A4).
- Top-performing models often tie (Kaczorowska 2021: LR = RF at about 96%; Christodoulou
  2019).
- Pre-stating the interpretation is our own addition. It is one sentence and protects against
  "you changed the story after seeing results".

### What we decided not to do (limitations or future work)
- Per-segment pupil handling (V2-3).
- A third gaze-conversion method and a formal main-sequence test (V2-2).
- A required COLET replication (V2-5): it stays optional, and "showing COLET's inflation" is
  removed from the stated contributions.
- A required no-standardization run (V2-8): it stays optional; if asked, cite Tognotti 2026
  and N7.
- Optional analyses S1–S5 and robustness runs (N19).

## N24. Round-3 review: keeping the scope from growing

**The test we used:** for every round-3 item, ask "is it needed to get a **valid P1/P2
result**?" (scope rule, N23). If not, it doesn't get added.

**Approved (2026-10-01); none of these adds an analysis:**

| Item | What changed | Why it isn't scope growth |
|---|---|---|
| V3-3 | Sanity check now has numbers: saccade:fixation **0.8–1.25** (a ±15% agreement rule was also added, then removed in round 4; N18) | The check already existed; without numbers it can't be applied |
| V3-4 | Gaze spread is computed after the existing > 1000°/s rejection (**superseded in round 4:** it did not remove the artifact; N25) | Gaze spread is already a feature; this reuses a step we already have |
| V3-2 | Under the ceiling rule, importance is still reported from P1 | Importance from P1 is already computed |
| V3-5 | Weak source dropped; Stolte added to the list; Hefron marked EEG | Makes claims smaller, not bigger |
| V3-6 | Status and wording fixes | Text only |

**Example of what we avoided:** the reviewer's preferred fix for V3-1 was an extra P2 run on
participants without refits. That is a new analysis on a smaller group, so it was **not**
approved. Round 4: the minimum fix (one limitation sentence, N23 V2-3) was chosen instead.

**For a panel:** "Reviewer suggestions were accepted when they made an existing step fully
specified, and recorded as limitations or future work when they would add new analyses."

## N25. Round-4 check: the gaze input (R4-1)

Run 2026-10-01 to verify the round-4 reviewer's numbers before changing the method. Label-blind:
counts and spreads per recording only; no model; nothing compared between A1 and A4. Samples
with confidence ≥ 0.8. Script: `colet-eda/check_r4_1.py`; output `colet-eda/check_r4_1.csv`.

**1. The 3D gaze point is broken (reviewer confirmed):**

| Check | Reviewer | Ours |
|---|---|---|
| Recordings with gaze points behind the camera (z ≤ 0) | 39 / 188 | **39 / 188** |
| Typical depth guess (median of recording medians) | 111 mm | **113 mm** (true: 800 mm) |
| Range of per-recording median z | −1064 to +484 mm | **−1064 to +484 mm** |
| Recordings with horizontal spread > 30° | 6 | **6** (P37 A1–A4, P09 A3, P47 A3) |
| P37 A4 horizontal spread | 102° | **102.1°** |

**2. The per-eye directions fix it:**

| Gaze input | Horizontal spread: median / max | Vertical: median / max | Recordings > 30° |
|---|---|---|---|
| 3D gaze point (old) | 4.6° / **102.1°** | 3.6° / 15.7° | 6 |
| Two eyes averaged, one eye when the other is missing (reviewer's rule) | 4.0° / 15.1° | 3.1° / 13.2° | 0 |
| **Two eyes averaged, both valid only (proposed)** | **3.5° / 9.4°** | **2.8° / 10.2°** | 0 |

**3. The catch: each eye alone is biased.**
- The two eyes' directions converge by **about 25°** (median; middle half 18–29°). At 80 cm,
  with about 6 cm between the eyes, the real value is about 4–5°.
- So each eye's direction is bent inward by about 10°. Averaging the two cancels most of it.
- Switching to one eye when the other drops out makes the gaze **jump about 10°**, which looks
  like a saccade. Examples: P02 A3 spread 11.0° (reviewer's rule) vs 3.3° (both eyes only);
  P37 A4 12.9° vs 4.8°.
- Samples with only one valid eye: median **0.8%** per recording, max **19%** (P06 A3, already
  excluded by C3).
- **Proposed (pending team OK):** use both-eye samples only; one-eye samples are missing, and
  the existing P6 gap rule handles them. No new step.

**Why this matters for a panel:** the method change rests on our own numbers, not only on the
reviewer's.

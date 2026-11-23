# Changes from v4-2-1 to v4-4

v4-2-1 = v4-2 with the two "were not tested / not run" sentences replaced by the A3 vs A4 and
A1 vs A2 results. v4-4 = v4-3 (post-hoc analyses added) plus the second review's fixes.
All new results are labelled post hoc (added after the main results; exploratory). The main
analysis (A1 vs A4, per-person z-score, P1 and P2) and its numbers are unchanged.

## Chapter 1

- **Scope:** new sentence saying further robustness analyses were added after the results, are not
  part of the pre-specified plan, and are reported in Chapter 4 and Appendix B.
- **Scope:** new sentence on A2-baseline centering (0.90–0.92 with all features, post hoc).
- **Scope:** "has been proven to increase the blink rate" → "has been reported to increase".
- **Conceptual framework:** "participant-disjoint LOPO" → LOPO "in which no test participant's
  labels are used in training".
- (The A3 vs A4 / A1 vs A2 sentence was already in v4-2-1; unchanged.)

## Chapter 3

- **Research design:** "differences in performance can be attributed to the models" → "mainly
  reflect the models, although the two models are tuned over different sets of settings".
- **Analysis plan:** Table 7 now lists three groups (required, optional listed in advance, post hoc),
  with a matching lead-in sentence and a sentence that post-hoc analyses do not change the
  pre-specified ones.
- **Validation:** reference [55] (Varoquaux 2017) cited for grouped inner cross-validation; it was
  previously never cited.
- **Implementation:** the robustness analyses are named as post-hoc checks alongside the
  shuffled-label control.

## Chapter 4

- **Intro:** lists the post-hoc section and says those analyses are labelled as such.
- **Model comparison:** "neither model was consistently better" → no evidence XGBoost performed
  better; Logistic Regression scored equal or higher in most settings, with the interval excluding
  zero in three uncorrected comparisons (no blink rate, 10 s windows, raw pupil A1 vs A2). The
  no-blink result is now "the only pre-specified comparison" with an interval excluding zero.
- **New section "Post-hoc robustness analyses":**
  - A2-baseline centering: P1 0.903 / 0.921 (comparable to or modestly above raw 0.853 / 0.881);
    A2 is a time-pressure activity, not a rest baseline.
  - Baselines: blink rate alone 0.963 (z) / 0.883 (raw); blink + pupil mean 0.973 / 0.970;
    default hyperparameters 0.991 / 0.977. Raw blink alone is slightly above the raw 10-feature
    LR (0.853), intervals overlapping.
  - Speaking vs time pressure: A3 vs A4 0.60–0.66 (raw 0.44–0.53); A1 vs A2 0.68–0.77 (raw
    0.50–0.64); worded as "time pressure is hard to detect during speech", and "suggest" (not
    "indicate") that most of the separation comes from speaking.
  - **New Table 15** (post-hoc A1 vs A4 analyses) and **new Table 16** (contrasts by speaking and
    time pressure, with 95% CIs).
- **Discussion:** new paragraph: A2 reference still gives 0.90–0.92; with speech held constant
  detection is weak; the data cannot separate mental load from speech.

## Chapter 5

- **Summary of findings:** item 3 adds "no evidence that XGBoost performed better"; item 4
  softened (pupil signal present between the silent activities but weak when both involve
  speaking); new item 6 summarizing the post-hoc results.
- **Conclusions:** new sentence that, with speaking held constant, time pressure was barely
  detectable.
- **Limits paragraph:** "participant-disjoint training" → "no test participant's labels are used
  in training".
- **Limitations:**
  - Speech bullet now points to Table 16; the blink/invalid table is renumbered **Table 15 →
    Table 17**.
  - Standardization bullet: adds the A2-baseline result (0.90–0.92) and that A2 is not a rest
    recording.
  - Blink-detector bullet: adds that very low rates in silent activities (A2 median 0.0) may partly
    reflect missed blinks.
  - "Only the two extreme activities were classified" → "…in the main analysis".
  - **New bullet:** activity order, inferred from timestamps; A4 followed A1 for 18 of 44;
    predictions unrelated to position (all p ≥ 0.19); fixation rate drifts (Holm p = 0.024).
  - **New bullet:** the robustness analyses were added later, not corrected for multiple
    comparisons, and (except A2-baseline centering) use the same standardization.
- **Recommendations:**
  - Rec. 2 replaced: classify A1 vs A3 and A2 vs A4 (speaking only) to complete the comparison.
  - Rec. 3: calibration from a separate, **neutral** baseline, since A2 involved time pressure.
  - Practitioner paragraph: "Silent tasks were not tested" → silent tasks examined only post hoc
    (A1 vs A2, 0.68–0.77); now points to Recommendation 1 only.

## Appendix

- **New Appendix B, post-hoc windowed analysis:** non-overlapping 10 s windows (our own choice),
  153 A1 and 230 A4 windows from 44 participants, with balanced accuracy. **New Table B1.**
  P1 0.928 / 0.901 (XGB − LR −0.027, CI −0.049 to −0.009).

## Size

- v4-2-1: about 55 pages (as v4-2), 16 tables. v4-4: 62 pages, 19 tables, 8 figures, no comments, every
  reference cited.

## Not changed

- Title, objectives, main results and figures.
- Citation numbering order ([24] before [25]; [29], [32] introduced late).
- Equivalence test (spoken answer only) and A2+A3 standardization (notes only).

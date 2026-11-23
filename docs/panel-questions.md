# Likely Panel Questions and Answers

**Thesis:** Comparative Analysis of Logistic Regression and XGBoost for Cognitive Load Detection Using Eye-Tracking Features with Feature Importance Analysis (COLET)
**Sources:** `docs/CSRP-Chapter-1-to-5-draft-COLET-v4-3.md` (the paper) and `docs/superpowers/plans/2026-10-02-colet-pipeline.md` (the plan; "Task N" below)
**Order:** Level 1 is the shallowest (basic understanding); Level 5 is the deepest (attacks on the validity of the result).

Each entry has the question, a direct answer you can say out loud, and where the answer is found.

---

## Level 1: Basics (what and why)

**Q1. What is your study about, in one sentence?**
We compare Logistic Regression and XGBoost for telling low from high cognitive load using eye-tracking features from the public COLET dataset, test both on people held out from training, and explain which features each model relied on.
*Where:* Ch 1, Purpose and Description; General Objective.

**Q2. What is cognitive load, and how did you define it?**
It is the mental effort a task requires. We do not measure it directly; we use COLET's task conditions as labels: A1 (single task, no time pressure) = low, A4 (counting backwards aloud under time pressure) = high.
*Where:* Ch 1, Project Context; Operational Definition of Terms ("Cognitive Load"); Ch 3, Labels and Unit of Analysis.

**Q3. Why eye tracking instead of questionnaires or EEG?**
Questionnaires such as NASA-TLX are given after the task and depend on self-judgement; EEG and heart sensors need specialised, intrusive equipment. Eye tracking records pupil, blinks, fixations and saccades during the task. A review of 91 studies found pupil diameter significant in 79% of the studies that measured it, fixation duration in 73% and blink rate in 71%.
*Where:* Ch 1, Project Context; Ch 2, Cognitive Load and Its Measurement [5].

**Q4. Why these two models?**
Logistic Regression is the interpretable linear baseline; XGBoost is a boosted tree ensemble that can capture non-linear effects and interactions. The question is whether the extra flexibility pays off on small tabular eye data. Prior direct comparisons disagree.
*Where:* Ch 1, Project Context; Ch 2, Logistic Regression and XGBoost; Synthesis.

**Q5. What dataset did you use, and how big is it?**
COLET v3 (Zenodo 7766785, CC BY 4.0): 47 released participants, Pupil Labs Pupil Core, four activities in a 2 × 2 design (time pressure × counting aloud). After exclusions we analysed 44 participants and 88 recordings (one A1 and one A4 each).
*Where:* Ch 3, Dataset (Table 2); Data Preprocessing; Ch 4, Data Quality and Retention.

**Q6. What is your research gap?**
No study in our search applied XGBoost or SHAP to COLET. Most COLET studies used splits not grouped by participant; the one participant-independent study (Fenoglio et al.) used neither LR, XGBoost nor feature importance. Studies also explained only one model.
*Where:* Ch 2, Synthesis (Table 1); Ch 1, Project Context.

**Q7. What are the ten features?**
Pupil mean, pupil SD, blink rate, fixation rate, fixation duration, saccade rate, saccade amplitude, saccade peak velocity, gaze SD horizontal, gaze SD vertical.
*Where:* Ch 3, Feature Extraction (Table 5); plan Global Constraints and `src/config.py` (`FEATURES`), Task 4.

**Q8. What is your main result?**
Both models separated A1 from A4 almost perfectly: ROC-AUC 0.992 (LR) and 0.980 (XGBoost), balanced accuracy 0.966 and 0.920, both ranked A4 above A1 for all 44 people. XGBoost did not beat Logistic Regression.
*Where:* Ch 4, Main Analysis (Table 12); Ch 5, Summary of Findings 2–3.

**Q9. What is the significance of the study?**
It gives a fully specified, reproducible, participant-independent comparison on a public dataset and adds evidence on when a simple interpretable model is enough. We do not claim a deployable tool; this is an early lab-based step.
*Where:* Ch 1, Significance of the Study.

---

## Level 2: Data and preprocessing (how)

**Q10. Why only A1 and A4? What about A2 and A3?**
We compare the two extremes of the workload manipulation, following prior studies. A2 and A3 are not classified in the main analysis, but they are used in the manipulation check and in each person's standardization.
*Where:* Ch 3, Labels and Unit of Analysis (Table 3); plan Global Constraints ("A2/A3 never enter a model"), Task 5 (`src/labels.py`).

**Q11. How do you know A4 is actually harder than A1?**
Manipulation check on NASA-RTLX: Friedman χ²(3) = 82.72, p < 0.001, n = 43; all 43 rated A4 higher than A1 (median difference 31.7 points). RTLX is used only to validate the labels, not as the label.
*Where:* Ch 4, Manipulation Check (Table 8, Figure 5); plan Task 6 (`src/checks.py`, manipulation check).

**Q12. Why condition labels instead of self-reported workload as labels?**
Ratings are affected by how each person uses the scale, so a condition label is cleaner for classification. Ratings confirm the conditions differ.
*Where:* Ch 3, Labels and Unit of Analysis [34].

**Q13. Why one sample per whole activity instead of time windows?**
The activities are short (11–141 s), and the COLET authors used the same unit. As a post-hoc check we also ran 10 s windows: P1 standardized ROC-AUC 0.928 (LR) and 0.901 (XGBoost).
*Where:* Ch 3, Labels and Unit of Analysis; Appendix B (Table B1).

**Q14. How did you open the COLET file?**
It is one 3.8 GB MATLAB file with nested table objects. After checking the checksum we decoded it once with the open-source mat-io package and flattened it into 564 Parquet files (188 recordings × 3 signals) plus CSVs for NASA-RTLX and participant info. No values were changed.
*Where:* Ch 3, Dataset (Figure 3); plan Task 1 (`src/data.py`).

**Q15. Why did you drop participants? Who?**
Recordings with more than 35% invalid samples were excluded: P06 A3 (57.4%), P06 A4 (38.8%), P17 A4 (48.6%). P06 and P17 then had no valid A1–A4 pair, so both were dropped. P31 was dropped because its eye files are byte-identical to P28's in all four activities.
*Where:* Ch 3, Data Preprocessing (Table 4 and text below it); Ch 4, Data Quality and Retention; plan Global Constraints (P3), Task 5 (`src/dataset.py`).

**Q16. What is the confidence threshold and why 0.8?**
Samples with tracker confidence below 0.8 are invalid, following Pupil Labs data-quality guidance.
*Where:* Ch 3, Table 4 ("Sample validity") [30], [65]; `src/config.py` `CONFIDENCE_MIN`; plan Task 2.

**Q17. How did you clean the pupil signal?**
3D-model diameter in mm, kept between 1.5 and 9 mm; dilation-speed outlier removal (median + 16 MAD, the default of Kret & Sjak-Shie's code); both eyes averaged; gaps up to 250 ms filled; 4 Hz low-pass filter; pupil-model refits counted.
*Where:* Ch 3, Table 4 ("Pupil") [29], [68], [69]; plan Global Constraints (P7), Task 2 (`clean_pupil`, `count_refits`).

**Q18. How did you handle blinks?**
We used the tracker's blink events, merged blinks less than 100 ms apart, kept those lasting 50–500 ms, and treated blink periods as missing gaze and pupil.
*Where:* Ch 3, Table 4 ("Blinks") [67], [68]; plan Global Constraints (P5), Task 2 (`clean_blinks`).

**Q19. How did you detect fixations and saccades?**
I-VT at 45°/s with a 55 ms minimum fixation, both taken from the COLET authors. Velocity comes from a five-point smoothed difference of unit gaze vectors on a 240 Hz grid; velocities above 1000°/s are rejected as artifacts.
*Where:* Ch 3, Fixation and Saccade Detection [18], [31], [43], [73], [74]; plan Task 3 (`src/events.py`).

**Q20. Why per-eye gaze direction instead of the tracker's 3D gaze point?**
The 3D point depends on an estimate of viewing distance from vergence, which is not reliable for this kind of tracker. In COLET the median estimated distance was about 113 mm against a true 800 mm, and some points were behind the camera. So we computed angles from each eye's direction vector, averaged across eyes.
*Where:* Ch 3, text after Table 4 [70], [71], [72].

**Q21. Why are samples with only one valid eye treated as missing?**
Each eye's direction alone was biased inward by about 10°, so a single-eye sample would create a fake jump. Treating it as a gap follows the practice of treating unreliable samples as missing. It removed only about 1% of samples.
*Where:* Ch 3, text after Table 4; Ch 4, Table 9; plan Global Constraints ("decided 2026-10-02").

**Q22. How do you know the event detection worked?**
A sanity check per activity, set before modelling: median fixation must be 150–400 ms and saccades per fixation 0.8–1.25. All four activities passed (212.5–235.4 ms; 0.93–0.99), so we used all ten features instead of the five-feature fallback.
*Where:* Ch 3, Fixation and Saccade Detection; Ch 4, Event-Detection Sanity Check (Table 10); plan Task 6.

**Q23. Why did you drop blink duration, saccade duration and mean saccade velocity?**
Blink duration is undefined when there is no blink, and that happened only in A1 (13 of 44), so its missingness would reveal the label. Mean saccade velocity and saccade duration nearly duplicate peak velocity and amplitude (ρ = 0.99 and 0.80 in the final features).
*Where:* Ch 3, Feature Extraction (paragraph after Table 5); plan Task 6 (V2-13 redundancy, `08_redundancy_v2_13.csv`).

**Q24. Did you control for lighting, since the pupil responds to light?**
Room lighting was controlled at 400–450 lx. Image brightness differs, but the COLET authors found pupil size following brightness in only 2 of 47 participants. We applied no further correction and list it as a limitation.
*Where:* Ch 3, Feature Extraction (last paragraph); Ch 5, Limitations.

**Q25. Were there missing feature values? How did you impute?**
None of the ten features had missing values, so imputation played no role. The pipeline still had median imputation in-fold for LR and native missing handling for XGBoost.
*Where:* Ch 4, Data Quality and Retention (last sentence); Ch 3, Model Development.

---

## Level 3: Models, validation and metrics

**Q26. What is LOPO, and why use it?**
Leave-one-participant-out: in each of 44 folds, both recordings of one person are the test set and the other 43 people train the model. If the same person appears in train and test, the model can learn the person instead of the condition and look better than it is.
*Where:* Ch 3, Validation Procedure; Ch 2, Participant-Independent Validation [14], [53]; plan Task 7 (`src/evaluate.py`).

**Q27. How did you tune hyperparameters without leakage?**
Nested: inside each LOPO training set, random search with 30 candidates per model, scored by ROC-AUC in a 5-fold inner CV grouped by participant (`StratifiedGroupKFold`). In-fold scaling and imputation are fit on training participants only.
*Where:* Ch 3, Validation Procedure [33], [54]; `src/config.py` (`N_ITER = 30`, `INNER_FOLDS = 5`); plan Task 7.

**Q28. What hyperparameters did you search?**
LR: elastic net, C log-uniform 10⁻³–10², L1 ratio 0–1. XGBoost: depth 1–3, 50–300 trees, learning rate 0.05–0.1, row/column subsampling 0.7–1.0, L2 0.01–10, L1 0.001–1.
*Where:* Ch 3, Model Development (Table 6); plan Task 7 (`src/models.py`).

**Q29. Is giving both models 30 candidates fair? XGBoost has more parameters.**
It is equal in number of tries, not equal in coverage; we say this in the paper. In practice tuning added nothing at n = 88: default settings scored 0.991/0.977 in P1 and 0.852/0.819 in P2, equal to or above the tuned values.
*Where:* Ch 3, Model Development; Ch 4, Model Comparison; Post-hoc Robustness (Table 15, "Default hyperparameters").

**Q30. Why elastic net and shallow trees?**
With 86 training samples per fold, coefficients and deep trees overfit. Elastic net stabilises correlated coefficients; shallow trees are what a nested-LOPO workload study found selected on small data.
*Where:* Ch 3, Model Development [8], [50], [51]; Ch 2, Logistic Regression and XGBoost.

**Q31. Why ROC-AUC as the primary metric?**
It measures how well a model ranks high-load above low-load across all thresholds, so it does not depend on a chosen cut-off. We also report balanced accuracy, macro-F1, per-class recall, Brier score, a majority baseline (0.500) and the share of people with A4 ranked above A1.
*Where:* Ch 3, Performance Metrics; Ch 4, Table 12; plan Global Constraints (§10).

**Q32. How were the confidence intervals computed?**
A bootstrap that resamples participants (2000 resamples), so both recordings of a person stay together.
*Where:* Ch 3, Performance Metrics [28]; `src/config.py` `N_BOOTSTRAP`; plan Task 7.

**Q33. How did you test that the models beat chance?**
A grouped permutation test (200 permutations) swaps each person's two labels at random. Both models gave p = 0.005, the smallest possible with 200 permutations. It uses default parameters, so the p-value refers to the untuned pipeline.
*Where:* Ch 3, Performance Metrics [82]; Ch 4, Main Analysis; P2 section.

**Q34. What is the P2 analysis and why do it?**
P2 repeats the analysis with only pupil mean and pupil SD. A4 involves speaking, which raises blink rate, so P2 checks whether detection survives without the most speech-sensitive feature. Result: LR 0.831, XGBoost 0.812, both above chance.
*Where:* Ch 3, Analysis Plan (Table 7); Ch 4, Pupil-Only Analysis (P2).

**Q35. What is the "ceiling rule"?**
Stated before seeing results: if both models exceed 0.95 ROC-AUC in P1, the model comparison is read mainly from P2, because near-perfect models tie. Both exceeded it, so the comparison was taken from P2. Feature importance is still reported from P1.
*Where:* Ch 3, Analysis Plan; Ch 4, Main Analysis (paragraph after Figure 7); plan Global Constraints (§11), `CEILING_AUC`.

---

## Level 4: Interpreting the results

**Q36. So which model is better?**
Neither was shown to be better. In P2 the AUC difference (XGBoost − LR) was −0.019 (95% CI −0.086 to 0.047), and in P1 −0.012 (−0.030 to 0.002). Both intervals include zero. The upper bound says XGBoost's advantage, if any, is at most 0.002 in P1 and 0.047 in P2.
*Where:* Ch 4, Model Comparison (Table 13); Ch 5, Conclusions.

**Q37. Then can you say the two models are equal?**
No. We set no equivalence margin in advance, so we can only say no difference was detected at this sample size, not that they perform the same.
*Where:* Ch 4, Model Comparison; Ch 5, Conclusions and Limitations.

**Q38. Your Wilcoxon test gave p = 0.007 in favour of XGBoost. Doesn't that contradict you?**
That test is on each person's margin, p(A4) − p(A1), which depends on how extreme each model's probabilities are. LR was strongly regularised in P2 (median C = 0.15), which pulls its probabilities toward 0.5. So the margins measure confidence, not separation. Pooled ROC-AUC was the pre-specified primary metric and was kept as the main comparison.
*Where:* Ch 3, Performance Metrics; Ch 4, Model Comparison (Table 13 and text).

**Q39. Is there any setting where one model clearly wins?**
Logistic Regression was significantly better when blink rate was removed (difference −0.043, CI −0.084 to −0.007), the only pre-specified comparison whose interval excludes zero, and with 10 s windows (−0.027, CI −0.049 to −0.009, post hoc). Without per-person standardization, XGBoost's point estimates were higher, but not significantly.
*Where:* Ch 4, Model Comparison (second paragraph); Appendix B (Table B1).

**Q40. Which features mattered most?**
By SHAP, both models' top three were blink rate, mean pupil diameter and fixation rate; ranking agreement Kendall's τ = 0.69 (95% CI 0.60–0.73). Permutation importance agreed on blink rate (AUC drop 0.30 for LR, 0.27 for XGBoost); pupil mean mattered much less to XGBoost (0.04 vs 0.12).
*Where:* Ch 4, Feature Importance (Figure 8); plan Task 8 (`src/importance.py`, `07_importance_*.csv`).

**Q41. What direction did each feature push?**
From LR coefficients: higher blink rate (+5.34) and larger pupil (+3.85) push toward high load; higher fixation rate (−2.28) and wider gaze spread (−1.34, −1.14) push toward low load. These kept the same sign in at least 89% of folds. They are conditional on the other features, not single-feature directions.
*Where:* Ch 4, Feature Importance (Table 14).

**Q42. Why use three importance methods?**
Coefficients show direction for LR only; SHAP puts both models on the same scale; permutation importance is a model-agnostic check on held-out predictions. Built-in XGBoost gain is in the appendix only because built-in tree importance can be biased.
*Where:* Ch 3, Feature Importance Analysis; Appendix A (Table A1).

**Q43. Why does XGBoost gain rank saccade rate second but SHAP doesn't?**
Gain measures improvement in the training objective at splits and can be biased toward certain features; SHAP and permutation importance work on held-out predictions. That is why gain is only reported in the appendix and our conclusions rest on SHAP and permutation.
*Where:* Ch 4, Feature Importance; Appendix A.

**Q44. Your features moved in different directions from the literature. Why?**
Longer fixations under load match the literature, but saccades became slightly larger and faster and gaze spread increased, unlike earlier reports, and saccade rate fell where COLET reported a rise. Likely causes are our gaze-to-angle conversion, event-detection settings and speaking during A4. The cause was not tested.
*Where:* Ch 4, Eye-Tracking Features by Condition (Table 11); Discussion (last paragraph).

**Q45. COLET's own LR got 0.85 accuracy; you got 0.966. Why are you higher?**
The protocols differ, so the numbers are not directly comparable. Their split was not described as participant-independent and feature selection used the whole dataset. Our gain most likely comes from per-person standardization and whole-activity averaging. Without standardization our LR balanced accuracy was 0.739.
*Where:* Ch 4, Discussion (third paragraph); Ch 2, The COLET Dataset and Its Prior Use.

---

## Level 5: Deep and critical questions (validity)

**Q46. 0.99 is suspiciously high. Is there leakage?**
We checked three ways. (1) A shuffled-label run (post hoc, 40 runs) dropped ROC-AUC to 0.46–0.53, so the pipeline only works with true labels. (2) All fitted preprocessing and tuning sit inside the folds. (3) The task is easy for a simple reason: blink rate alone gives 0.97 standardized, and 42 of 44 people blinked more in A4. The one thing that does use test-person data is the per-person standardization, which we disclose (next question).
*Where:* Ch 4, Shuffled-Label Check; Ch 3, Validation Procedure; `docs/leakage-audit.md`.

**Q47. Your standardization uses the test person's own A1 and A4. Isn't that leakage?**
It uses no labels, but it does use the test person's unlabeled recordings, including the two being classified, so it makes the test easier than meeting a stranger. We state that directly. Without it, ROC-AUC falls to 0.853–0.881 (P1) and 0.611–0.649 (P2). Post hoc, subtracting only each person's A2 recording, which keeps A1 and A4 out of the reference, still gave 0.903–0.921. A real system would need a calibration session per user. The adviser approved this as the main analysis.
*Where:* Ch 3, Per-Person Standardization; Ch 4, Table 12 ("not standardized") and Post-hoc Robustness (Table 15, "A2-baseline centering"); Ch 5, Limitations; plan Task 5 (`src/normalize.py`).

**Q48. Isn't A4 different from A1 by speaking, not just load? Are you detecting talking?**
Yes, the conditions mix dual-tasking, speaking and time pressure, so we claim detection of COLET's task condition, not cognitive load separated from speech. Evidence: blink rate tracks speaking (median 1.7, 0.0, 11.7, 11.7 per minute in A1–A4). With speaking held constant (A3 vs A4), time pressure was barely detectable (0.60–0.66); between the silent activities (A1 vs A2) it reached 0.68–0.77. Pupil-only P2 still reached 0.81–0.83, so the result does not rest on blinks alone.
*Where:* Ch 1, Scope and Limitations; Ch 4, Post-hoc Robustness (Table 16); Discussion; Ch 5, Limitations (Table 17).

**Q49. Is the pupil-only analysis really free of speech effects?**
Not fully. The tracker refitted its pupil model about twice as often in the counting activities (38 vs 19 recordings), and the pupil can respond to speaking itself. We state this as a limitation and recommend computing pupil features per stable model segment.
*Where:* Ch 4, Data Quality (Table 9); P2 section; Ch 5, Limitations and Recommendation 4.

**Q50. With 88 samples and a ceiling effect, can you answer the "does flexibility help" question at all?**
Only partly. P1 is at the ceiling, and in P2 mean pupil diameter carries almost all the information and XGBoost chose single-split trees in 37 of 44 folds, so P2 cannot test interactions. Our conclusion is limited to: on this small, low-dimensional problem, XGBoost showed no advantage.
*Where:* Ch 4, Model Comparison (end of first paragraph); Main Analysis (chosen settings).

**Q51. Your exclusions removed only counting-activity recordings. Doesn't that bias the sample?**
Yes, that is a possible selection effect: all three excluded recordings (P06 A3, P06 A4, P17 A4) came from counting activities, so the retained people may be the ones easier to track while talking. It is listed as a limitation.
*Where:* Ch 4, Data Quality and Retention; Ch 5, Limitations.

**Q52. What happened with P28 and P31?**
Their released gaze, pupil and blink files are byte-identical in all four activities, but their demographics and RTLX ratings differ. So one person's recordings were stored twice and the owner cannot be determined. We kept them once (as P28) and left both IDs out of the RTLX analyses (hence n = 43 there). The run was repeated once, with no setting changed, after this was found.
*Where:* Ch 3, Data Preprocessing (text below Table 4); Ch 3, Implementation; Ch 4 intro and Manipulation Check.

**Q53. Were your thresholds chosen after looking at the results?**
No. All thresholds were fixed in one config file after a label-blind data-quality inspection and before any model was trained. Analyses added after the results (shuffled labels, A2 baseline, other contrasts, windows, simple baselines, order) are labelled post hoc and reported separately, with no multiple-comparison correction.
*Where:* Ch 3, Research Design and Implementation; Ch 4, Post-hoc Robustness (intro); `src/config.py` docstring; plan header ("frozen 2026-10-02").

**Q54. Which settings have no published source?**
Our own choices: 35% recording exclusion, the 1 s valid-time gap rule, the 0.8–1.25 sanity tolerance, using each eye's gaze direction, treating one-eye samples as missing, merging blinks before the 50–500 ms range, the 120 Hz pupil grid and second-order Butterworth filter, the 240 Hz gaze grid and the five-point velocity filter. They are labelled as such.
*Where:* Ch 3, Design Choices.

**Q55. The 45°/s threshold came from COLET. Was it validated for your tracker?**
No. It matches the minimum peak velocity of human-coded saccades in Andersson et al., but it was not validated for a 240 Hz Pupil Core and we ran no threshold sensitivity analysis. The sanity check passing is our evidence that the events are plausible.
*Where:* Ch 3, Fixation and Saccade Detection; Ch 5, Limitations.

**Q56. Blinks were more frequent in A4. Couldn't they split fixations and bias the fixation features?**
Yes. A blink or a gap over 75 ms inside a fixation splits it, and segments under 55 ms are dropped, so fixation counts could rise or fall in A4. The gaze-time denominator corrects for missing time but not for splitting. The effect was not quantified.
*Where:* Ch 3, Design Choices (last paragraph); Ch 5, Limitations.

**Q57. A4 recordings are longer than A1. Does duration affect the features?**
Rates are normalised by valid time, but recording length (median 51 vs 37 s) may affect how stable the SD features (gaze spread, pupil SD) are. Listed as a limitation.
*Where:* Ch 5, Limitations.

**Q58. Could activity order (practice or fatigue) explain the result?**
The order is not in the released data; we inferred it from timestamps post hoc. Order varied (A4 followed A1 for 18 of 44), and neither model's predicted probabilities were related to activity position (all p ≥ 0.19). Fixation rate rose slightly with position (Holm-corrected p = 0.024), so a small practice or fatigue effect on that feature cannot be ruled out.
*Where:* Ch 5, Limitations (activity order); `docs/colet-eda/extra_7_activity_order.py`.

**Q59. Do the ten features add anything over blink rate alone?**
Very little. LR on standardized blink rate alone reached 0.963; blink rate plus pupil mean reached 0.973 (LR) and 0.970 (XGBoost). The ten-feature models added about 0.01–0.02.
*Where:* Ch 4, Shuffled-Label Check (last sentences); Post-hoc Robustness (Table 15).

**Q60. Your bootstrap CIs: don't they ignore that the 44 LOPO models share training data?**
Yes, that is a known limitation of the participant bootstrap over pooled out-of-fold predictions: it resamples test participants but does not account for overlapping training sets, so the intervals may be somewhat narrow. We did not compute a corrected variance. Prepare this as a spoken answer.
*Where:* Not in the paper; raised in `docs/panel-review-handoff.md` (item 7).

**Q61. The sanity check used all participants. Isn't that held-out data influencing a decision?**
Yes, the choice between ten features and the five-feature fallback used all participants. It is an unsupervised quality check that never uses labels, and we list it as a limitation.
*Where:* Ch 5, Limitations.

**Q62. Would your recommendation hold for a silent high-load task?**
Not necessarily. In silent time pressure (A2), median blink rate dropped to 0.0 from 1.7 in A1, and visual tasks are known to reduce blinking, so the blink effect may reverse. Our practitioner recommendation is limited to tasks like this one (speaking under time pressure) with a per-user baseline.
*Where:* Ch 5, Recommendations (last paragraph); Table 17.

**Q63. What would you do next?**
(1) Repeat on a dataset with a silent high-load task. (2) Classify A1 vs A3 and A2 vs A4 to measure the speech effect directly. (3) Calibrate from a separate neutral baseline. (4) Handle pupil-model refits. (5) Extend to more load levels and real-time windows.
*Where:* Ch 5, Recommendations 1–5.

**Q64. Can someone reproduce your results?**
Yes. Python 3.11 with pinned versions (NumPy 2.2.4, pandas 2.3.1, SciPy 1.15.2, scikit-learn 1.7.1, XGBoost 3.2.0, SHAP 0.51.0), every threshold in one config file, fixed seed (0), a Colab notebook with the same pins, and tests on synthetic data with known structure before running on COLET.
*Where:* Ch 3, Implementation; plan Tasks 1, 9 and 10 (`src/config.py`, `src/run_colet.py`, `requirements-colab.txt`, `notebooks/colet_pipeline.ipynb`).

---

## Quick number sheet

| Item | Value | Where |
|---|---|---|
| Participants / samples | 44 / 88 | Ch 4, Data Quality |
| P1 ROC-AUC LR / XGB | 0.992 / 0.980 | Table 12 |
| P2 ROC-AUC LR / XGB | 0.831 / 0.812 | Table 12 |
| No standardization P1 / P2 | 0.853–0.881 / 0.611–0.649 | Table 12 |
| A2-baseline centering P1 | 0.903 / 0.921 | Table 15 |
| Blink rate alone (standardized) | 0.963 | Table 15 |
| A3 vs A4 / A1 vs A2 | 0.60–0.66 / 0.68–0.77 | Table 16 |
| AUC difference P1 / P2 (XGB − LR) | −0.012 / −0.019, both CIs include 0 | Table 13 |
| No-blink difference | −0.043 (−0.084 to −0.007) | Ch 4, Model Comparison |
| Kendall's τ (SHAP) | 0.69 (0.60–0.73) | Ch 4, Feature Importance |
| Shuffled-label ROC-AUC | 0.46–0.53 | Ch 4, Shuffled-Label Check |
| Friedman (NASA-RTLX) | χ²(3) = 82.72, p < 0.001, n = 43 | Ch 4, Manipulation Check |

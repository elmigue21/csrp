# CHAPTER 1|THE PROBLEM AND ITS BACKGROUND
## PROJECT CONTEXT
Cognitive load is the mental effort required by a task, drawing on an individual's limited processing capacity {1}, {2}. It matters in ergonomics and human factors because mental demands are linked to performance and safety in complex tasks {3}. In this study, cognitive load is a label for task conditions: searching a virtual scene alone is low load, and searching while also responding to an auditory task is high load. Related terms such as mental workload and mental effort are treated as the same idea.

Cognitive load is usually measured with questionnaires such as the NASA Task Load Index (NASA-TLX) {4} or with physiological signals {5}. Questionnaires are given after the task and depend on self-assessment {6}, and physiological sensors need extra equipment. Eye trackers are now built into virtual reality (VR) headsets, so pupil size, gaze direction and blinks can be recorded during a task without additional sensors {bielsa}. A systematic review of 91 studies found pupil diameter significant in 79% of the studies that measured it and blink rate in 71% {5}.

Logistic Regression is a linear model with interpretable coefficients and a common baseline {7}, {8}. XGBoost is a boosted tree algorithm that can model non-linear relationships and feature interactions {9}, {10}. Studies comparing the two reached different results {11}, {12}, {13}, many evaluated models on data from the same people used for training, which overestimates performance for new users {14}, {15}, and none of the closest studies reviewed here checked whether a model does better than the single best feature (Chapter 2).

An earlier version of this study used the COLET dataset {18}. There, blink rate alone matched both models, and the high-load condition required speaking, which raises blinking {23}. The study was therefore repeated on the public dataset of Pina et al. {pina}, whose high-load condition adds an auditory task without speaking, and a comparison with a single-feature baseline was added. The analysis plan and its pass criteria were written before the eye data were loaded (Appendix A). After the planned analysis, blink rate was excluded from the main analysis because the headset's blink signal was found to measure tracking loss rather than blinking (Chapter 3); the planned results with blink rate are also reported.
## PURPOSE AND DESCRIPTION
This study compares Logistic Regression and XGBoost for detecting a high-load task condition from VR eye-tracking features, and tests whether either model does better than the single best feature.

Each four-minute task segment is cut into eight 30-second windows, and each cleaned eye signal in a window is summarized with the same statistics (for example, mean, median, standard deviation and percentiles). Both models are evaluated with leave-one-participant-out validation, so every prediction is made for a person not used in training {14}. Feature selection and tuning are done inside each training fold. Performance is reported as accuracy, precision, recall, F1-score, Brier score and ROC-AUC, and SHAP values {19}, {20} and selection counts show which features each model relied on.
## OBJECTIVES OF THE STUDY
### General Objective
The general objective of this study is to compare Logistic Regression and XGBoost for detecting a high-load task condition from VR eye-tracking features, and to identify the features each model relies on.
### Specific Objectives
Specifically, the study aims:

1.	To extract window-level eye-tracking features from the Pina et al. dataset by summarizing each cleaned eye signal with 17 statistics, for binary classification of low versus high load.

2.	To train and evaluate Logistic Regression and XGBoost under leave-one-participant-out validation, with feature selection and hyperparameter tuning done inside each training fold.

3.	To compare the two models with each other and with a one-feature baseline chosen inside each fold, using accuracy, precision, recall, F1-score, the Brier score and ROC-AUC.

4.	To analyze feature importance using SHAP values for both models and the number of folds in which each feature was selected.
## SIGNIFICANCE OF THE STUDY
Because eye trackers are built into current VR headsets, detecting high task load from eye data alone is a possible building block for adaptive training and education applications {bielsa}, in line with Sustainable Development Goals 3, 4 and 9 {24}. As a small study of one VR task with modest accuracy, it does not show that such applications are feasible; it tests how far eye features alone go and against what simple baseline.

For researchers, the study offers an evaluation on a public dataset in which every step that learns from data is fitted on training participants only {15}, the pass criteria were set before analysis, and each model is compared with the single best feature. Because the dataset is open, the analysis can be reproduced.
## SCOPE AND LIMITATIONS
The study uses the eye-tracking recordings of 36 participants in the Pina et al. dataset {pina}. The main analysis classifies the 90° search area, low versus high load; the 360° search area is analyzed separately as a secondary check. Each participant contributes 16 windows (8 low, 8 high), giving 576 windows.

The study uses eye-tracking data only and is limited to Logistic Regression and XGBoost. The gaze data are measured relative to the head, so fixation and saccade features are used only if a pre-set check shows they are reliable. Blink rate is excluded from the main analysis because, in this headset, it measures tracking loss. The high-load condition adds an auditory task with button presses, so the study detects a task condition, not cognitive load separated from that task. The sample is small {28}. Real-time detection and testing on other VR tasks are outside the scope. Self-reported workload is used only to confirm that the conditions differed.
## CONCEPTUAL FRAMEWORK
The conceptual framework is a sequential pipeline that turns raw eye-tracking data into classification results and feature importance (Figure 1).

![Conceptual framework of the study. **[TODO: redraw for Pina: COLET → Pina et al.; recordings → 30 s windows; add the one-feature baseline.]**](@docx:figure1)

Gaze and pupil data are taken from the dataset, cleaned and cut into windows labeled low or high load. Each cleaned signal is summarized into the eye-tracking feature set, and the same features are given to Logistic Regression, XGBoost and a one-feature baseline. All three are trained and tested with leave-one-participant-out validation {33}. The held-out predictions feed the performance evaluation and the feature analysis (SHAP and selection count).
## OPERATIONAL DEFINITION OF TERMS
**Accuracy –** The share of windows classified correctly. The two classes are equal in size.

**Blink Rate –** The number of blinks per minute of valid recording time.

**Brier Score –** The mean squared difference between predicted probabilities and actual outcomes; lower is better.

**Cognitive Load –** The mental effort required to perform a task. In this study it is defined by the task condition: visual search alone is low load, and visual search with the auditory task is high load.

**F1-Score –** The harmonic mean of precision and recall, macro-averaged over the two classes.

**Feature Selection –** Keeping only the most useful features. Features are ranked by their single-feature ROC-AUC on the training participants, and the top N are kept.

**Fixation / Saccade –** A period in which the eyes stay relatively still / a rapid eye movement between fixations.

**Gaze Angle –** The horizontal or vertical direction of gaze relative to the head, in degrees.

**Leave-One-Participant-Out (LOPO) Validation –** Holding out all windows of one participant for testing, training on the others, and repeating for every participant.

**One-Feature Baseline –** Logistic Regression on the single feature that best separates the classes in the training participants of each fold.

**Precision / Recall –** Of the windows predicted as one class, the share that truly belong to it / of the windows that truly belong to one class, the share predicted as that class.

**ROC-AUC –** How well a model ranks high-load above low-load windows across all thresholds.

**Search Area –** The part of the virtual scene where objects can appear: 90° in front of the participant or 360° around them.

**Selection Count –** The number of folds, out of 36, in which a feature was among the selected top N.

**SHAP –** A method that divides each prediction into the contribution of each feature; applied to both models.

**Window –** A 30-second part of a task segment; the unit of analysis.

# CHAPTER 2|REVIEW OF RELATED LITERATURE AND STUDIES
## COGNITIVE LOAD AND ITS MEASUREMENT
Cognitive load describes the share of a person's limited mental capacity that a task requires {1}, {2}, and mental workload is studied in ergonomics because of its relationship with performance and safety {3}. The NASA Task Load Index rates six dimensions of workload after a task {4}, and its unweighted "raw" form is common {34}. Different workload measures are each sensitive to load but do not always agree {6}.
## EYE-TRACKING INDICATORS OF COGNITIVE LOAD
The pupil dilates with increasing demand but also responds strongly to light {35}. Blink rate is sensitive to workload but depends on the task, and blinking increases during speech {23}, {36}. Saccade amplitude has been reported to decrease under load {37}, fixation measures relate to perceptual and cognitive load {40}, and gaze tends to concentrate in a smaller area under higher load {41}, {42}. Fixations and saccades are detected from raw gaze, commonly with a velocity threshold (I-VT) {31}; the results depend on the algorithm and settings, which were evaluated on recordings with a still head, so the output should be checked {43}, {44}.
## EYE TRACKING IN VIRTUAL REALITY
In head-mounted eye tracking, the eyes and head move together, so gaze measured relative to the head mixes eye and head movement {72}. Pupil size has been used to assess workload in VR human–robot tasks, where it was corrected against each participant's baseline {66}, and gaze has been used to predict cognitive load in augmented reality {12}. Blink rate has been reported to rise with mental load but fall with visual load {36}, so a task that adds both kinds of demand may change it in either direction. Pina et al. released eye, physiological and questionnaire data from participants who searched a VR scene with and without an auditory secondary task, in a 90° and a 360° search area {pina}. Bielsa et al. analyzed the same experiment statistically and found that high load reduced saccade amplitude and how far the eyes moved from the center, and increased pupil size {bielsa}. Because the gaze is recorded relative to the headset, they cast gaze rays from the headset's position and orientation into the scene before detecting fixations and saccades {bielsa}.
## LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD CLASSIFICATION
Logistic Regression models the probability of a class from a weighted sum of features {7}; the elastic net combines L1 and L2 penalties and handles correlated features {50}, and has been used for eye-tracking workload classification {8}. XGBoost trains regularized decision trees in sequence {9}, and shallow trees were mostly selected in a small nested-LOPO study {51}. Direct comparisons disagree: XGBoost was ahead in some studies {11}, {12}, no significant difference was found with eye-only features in driving {13}, and in clinical prediction machine learning showed no consistent benefit over Logistic Regression {26}.
## PARTICIPANT-INDEPENDENT VALIDATION AND FEATURE IMPORTANCE
If data from the same person are in both the training and test sets, accuracy is overestimated {14}, {53}. Testing on new people is harder because the model must work despite individual differences, such as in pupil size or gaze habits, that it has not seen {14}; this is why participant-independent results are usually lower than results on people seen in training. Leakage also occurs when feature selection or scaling is fitted before splitting {15}, and tuning must be nested {33}, {54}. With small samples, uncertainty must be reported {28}. Filter methods rank each feature by a simple score {guyon}, and the ranking must be repeated inside every training fold {15}. SHAP assigns each feature a share of every prediction {19}, {20}; how often a feature is selected shows how stable the selection is {83}, and importance shows what a model relied on, not a cause {62}.
## SYNTHESIS
**Table 1.** Summary of the closest related studies.

| Study | Data | Models | Validation | Feature importance |
|---|---|---|---|---|
| Bielsa et al. {bielsa} | Pina et al. (VR) | Statistical analysis, no classifier | – | – |
| Rolon-Merette et al. {11} | Own n-back data | LR, XGBoost, others | 10 held-out participants | – |
| Shao et al. {13} | Own driving data (26 drivers) | LR, XGBoost, others | Random 80/20 split | SHAP |
| Božak et al. {16} | Own data | Feature-based models | Leave-one-subject-out | SHAP |
| This study | Pina et al. (VR) | LR, XGBoost, one-feature baseline | Leave-one-participant-out | SHAP for both models, selection count |

Whether a boosted ensemble improves on a regularized linear model under participant-independent evaluation remains open {11}, {13}, {26}, and the studies reviewed did not test whether either model does better than the single best feature. The Pina et al. data have been analyzed statistically {bielsa} but, among the studies found, not with these two classifiers. This study fills that gap.

# CHAPTER 3|METHODOLOGY
## RESEARCH DESIGN
The study uses a quantitative, comparative design based on secondary analysis of a public dataset. Both models are trained on the same features, folds and number of candidate settings. The contrast, unit of analysis, exclusion rules, event-detection check and pass criteria were written down before the eye data were loaded (Appendix A). The one change made after the planned analysis was the exclusion of blink rate, described below; no other contrasts, windows or feature sets were tried for the main result.
## DATASET
The study uses the dataset of Pina et al. from Zenodo (record 18669059), released under a CC BY 4.0 license {pina} (Table 2).

**Table 2.** Description of the dataset.

| Item | Description |
|---|---|
| Participants | 36 with eye-tracking data (23 male, 13 female; age 31.9 ± 11.3 years, range 22–62) |
| Headset | Varjo XR-4 with built-in eye tracking at 200 Hz {bielsa} |
| Task | Find a highlighted object in a virtual scene as fast as possible {bielsa} |
| Conditions | Search area (90° or 360°) × load (visual search alone, or also pressing a controller button on hearing an odd number) {bielsa}; one continuous segment of about four minutes each |
| Used here | Per-eye gaze direction, pupil diameter (mm) and validity; trial timing; NASA-TLX |

Table 3 shows the data before and after processing. Of the 23 columns of the raw eye-tracking files, nine are used: the two timestamps, the gaze and per-eye validity flags, each eye's gaze direction and each eye's pupil diameter.

**Table 3.** Shape of the data before and after processing.

| Stage | Rows | Columns | One row is |
|---|---|---|---|
| Raw eye-tracking files (36) | 13,185,280 | 23 | One 5 ms sample |
| Raw trial logs (36) | 21,964 | 5 | One object found |
| Window table | 1,152 (36 × 4 segments × 8 windows) | 140 features | One 30 s window |
| Model input, 90° (main) | 576 (288 low, 288 high) | 48 features | One 30 s window |
| Model input, 360° (check) | 572 | 48 features | One 30 s window |
## LABELS AND UNIT OF ANALYSIS
In the 90° search area, the segment without the auditory task is low load and the segment with it is high load. The 360° segments are analyzed separately and never pooled with the main result. Each segment is cut into eight non-overlapping 30-second windows from the start of its first trial, giving 16 windows per participant. Windows are used because fewer objects are found under high load, so trials differ in number and length between conditions. The order of the four conditions differed between participants: all 24 possible orders occurred, and in the 90° area 18 participants did the low-load segment first and 18 the high-load segment first.

As a manipulation check, each participant's raw NASA-TLX for the combined task is compared with that for the visual task alone with a Wilcoxon signed-rank test.
## DATA PREPROCESSING
The cleaning steps of the earlier COLET version are reused unchanged where they apply (Table 4).

**Table 4.** Preprocessing steps.

| Step | Procedure | Source |
|---|---|---|
| Time alignment | Headset timestamps converted to Unix time to match the trial log | Design choice |
| Sample validity | A sample is valid if the gaze and both eyes are marked valid by the headset | Design choice |
| Window exclusion | Windows with more than 35% invalid samples excluded | Adapted from {66} (applied there to trials) |
| Blinks | Periods with both eyes invalid; merged if less than 100 ms apart, then kept if 50–500 ms | {67}, {68} |
| Gaze gaps | Gaps shorter than 75 ms linearly interpolated | Design choice |
| Pupil | 1.5–9 mm range; dilation-speed outlier removal; both eyes averaged; gaps up to 250 ms filled; 4 Hz low-pass filter | {29}, {68}, {69} |
| Gaze direction | Each eye's direction vector averaged across both eyes, converted to horizontal and vertical angles and resampled to 240 Hz | {70}, {72} |
## FIXATION AND SACCADE DETECTION
Fixations and saccades are detected with I-VT (45°/s, fixations shorter than 55 ms discarded) {31}, {43}. These settings were evaluated with a still head {43}, and the gaze here is measured relative to a moving head {72}, so a pre-set check is applied: the median fixation duration must lie between 150 and 400 ms. If it fails, all gaze-speed, fixation and saccade features are dropped.
## FEATURE EXTRACTION
Each cleaned signal in a window is summarized with 17 statistics: minimum, maximum, range, mean, median, standard deviation, variance, coefficient of variation, interquartile range, the 5th, 10th, 25th, 75th, 90th and 95th percentiles, skewness and kurtosis. Blink, fixation and saccade rates are added as single features. As in the COLET version, variances (the squares of the standard deviations), blink-duration statistics, the ranges of event signals and the minimum saccade duration are removed, leaving 140 features. If the event check fails, 49 remain: 16 statistics each of pupil diameter, horizontal gaze angle and vertical gaze angle, plus blink rate. With blink rate excluded (below), the main analysis uses 48.
## EXCLUSION OF BLINK RATE
This step is a deviation from the plan in Appendix A and was decided after the planned analysis had been run. The headset does not report blinks, so a blink is counted when both eyes are marked invalid at the same time (Table 4). The detected rate was far below that of healthy adults at rest {23}, and it was higher wherever the share of invalid samples was higher: under high load and in the 360° area, where people turn their heads more. The share of invalid samples alone classified the conditions as well as blink rate (Chapter 4). Blink rate therefore measures tracking loss rather than blinking, and because the high-load condition adds button presses, it may reflect movement rather than load. It was excluded for the same reason as the fixation and saccade features: the signal does not measure what its name says. The planned analysis with blink rate is reported in Chapter 4.
## MODEL DEVELOPMENT AND VALIDATION
Inside each training fold, features are ranked by |ROC-AUC − 0.5| and the top N are kept {guyon}; N is tuned from {5, 10, 20, 40, 80, 140}, and values above the number of available features use all of them. Logistic Regression uses standardization and an elastic-net penalty (C log-uniform 10⁻³–10², L1 ratio 0–1) {8}, {50}. XGBoost uses shallow regularized trees (depth 1–3, 50–300 trees, learning rate 0.05–0.1, subsampling 0.7–1.0, L2 0.01–10, L1 0.001–1) {9}, {51}. The one-feature baseline keeps N = 1 and fits Logistic Regression without tuning.

Models are evaluated with leave-one-participant-out validation over the 36 participants: in each fold, the 16 windows of one participant form the test set and the 560 windows of the other 35 the training set {14}. Settings are tuned by random search with 30 candidates per model, scored by ROC-AUC in a five-fold inner cross-validation grouped by participant {33}, {54}, {55}.
## PERFORMANCE METRICS AND STATISTICAL ANALYSIS
Accuracy, macro precision, recall for each class and macro F1-score are computed at a 0.5 threshold on the pooled held-out predictions {78}, together with the Brier score {79}, ROC-AUC {78} and the number of participants whose high-load windows received a higher mean probability than their low-load windows. Each metric has a 95% confidence interval from a bootstrap that resamples participants (2000 resamples) {28}, and differences between models use the same resamples, so they are paired.

A permutation test checks performance against chance {82}: each participant's labels are swapped at random (which segment is high load) 200 times, and the whole pipeline, feature selection included, is rerun with the settings each model chose most often on the real labels. The p-value compares the permuted results with the same fixed-setting pipeline on the real labels; the smallest possible value is 1/201 ≈ 0.005.

The pass criteria were set before analysis on ROC-AUC: (a) above chance: the ROC-AUC confidence interval above 0.5 and permutation p < 0.05; (b) features work together: the model minus the one-feature baseline has a confidence interval above 0; (c) importance is spread (descriptive). The same comparisons are also reported for accuracy, precision and F1-score.
## FEATURE IMPORTANCE ANALYSIS
SHAP values are computed on the held-out participant of each fold (linear SHAP for Logistic Regression, TreeSHAP for XGBoost) {19}, {20}; a feature not selected in a fold counts as zero there. Each feature's and each signal's share of the total mean absolute SHAP value is reported, with selection counts {83}. As planned (Appendix A), each model is refit without the signal of its own top feature. Importance is read as what each model relied on, not as a cause of cognitive load {62}.

Two checks were added after the planned analysis, following an external review, and are labeled post-hoc: a one-feature baseline on the share of invalid samples in each window, to test whether blink rate measures tracking loss, and the main predictions split by condition order. A further check rotated the gaze by the headset's rotation before event detection, to test whether head movement explains the failed event check.
## IMPLEMENTATION, ETHICS AND DESIGN CHOICES
The pipeline was implemented in Python 3.11 with NumPy 2.2.4, pandas 2.3.1, SciPy 1.15.2, scikit-learn 1.7.1, XGBoost 3.2.0 and SHAP 0.51.0, with fixed random seeds. The dataset is public and partially anonymized; according to its authors, it was collected following the ethics regulation of the Comité de Ética de la Investigación de la Comunidad Autónoma de Aragón (CEICA) {pina}. No new data were collected.

Some settings have no published source for their exact value and are the researchers' own choices, fixed before analysis: the 30-second window, the validity rule, the time alignment, the 75 ms gap limit, the 35% exclusion threshold, the minimum of 20 values per signal and the candidate values of N.

# CHAPTER 4|RESULTS AND DISCUSSION
## MANIPULATION CHECK AND DATA QUALITY
All 36 participants rated the combined task higher than the visual task alone (raw NASA-TLX, 0–20 scale: mean 11.31 against 7.25; median difference 3.92; Wilcoxon p < 0.001), so the labels reflect a difference in reported workload, although the conditions also differ in the auditory task.

No 90° window exceeded the 35% invalid-sample limit (median 0.3% invalid), so all 576 windows were kept; each window held about 6,000 samples. The event-detection check failed (Table 5), so all gaze-speed, fixation and saccade features were dropped.

**Table 5.** Event-detection check.

| Search area | Windows | Median fixation duration (ms) | Passed |
|---|---|---|---|
| 90° | 576 | 665 | No |
| 360° | 572 | 125 | No |

Blink rate, as detected, had a median of 0.0 per minute under low load and 2.0 under high load in the 90° area, and 69% of low-load and 47% of high-load windows had no detected blink. The share of invalid samples was also higher under high load (median 0.03% against 0.50%), and blink rate was higher in the 360° area (median 4.0 and 6.0 per minute). On its own, the share of invalid samples reached an accuracy of 0.590 and a ROC-AUC of 0.610 (95% CI 0.571–0.655), no different from blink rate (ROC-AUC difference 0.029, −0.012 to 0.067). Blink rate was therefore excluded (Chapter 3), and the main analysis used 48 features: 16 statistics each of pupil diameter, horizontal gaze angle and vertical gaze angle.
## CLASSIFICATION RESULTS
Logistic Regression classified 56.8% of windows from new participants correctly (F1-score 0.568, ROC-AUC 0.606, 95% CI 0.549–0.662; permutation p = 0.005), so it detected the high-load condition above chance. XGBoost did not: its accuracy was 0.524 and its ROC-AUC confidence interval included 0.5 (0.443–0.579; permutation p = 0.055). The one-feature baseline chose the standard deviation of horizontal gaze in all 36 folds and reached an accuracy of 0.559 (Table 6).

**Table 6.** Results for the 90° search area under leave-one-participant-out validation (576 windows, 48 features; 95% CI in brackets).

| Metric | Logistic Regression | XGBoost | One-feature baseline |
|---|---|---|---|
| Accuracy | 0.568 (0.523–0.613) | 0.524 (0.470–0.580) | 0.559 (0.517–0.604) |
| Precision, macro | 0.568 (0.523–0.614) | 0.524 (0.469–0.580) | 0.559 (0.518–0.605) |
| Recall, low | 0.587 (0.503–0.674) | 0.542 (0.427–0.656) | 0.559 (0.444–0.670) |
| Recall, high | 0.549 (0.448–0.649) | 0.507 (0.396–0.618) | 0.559 (0.444–0.674) |
| F1-score, macro | 0.568 (0.520–0.613) | 0.524 (0.469–0.578) | 0.559 (0.515–0.602) |
| Brier score | 0.245 (0.230–0.261) | 0.288 (0.262–0.315) | 0.247 (0.240–0.254) |
| ROC-AUC | 0.606 (0.549–0.662) | 0.509 (0.443–0.579) | 0.574 (0.520–0.634) |
| High above low (participants) | 25 of 36 | 21 of 36 | 23 of 36 |
| Permutation test p | 0.005 | 0.055 | – |
| Features selected (median N) | 48 (all) | 40 | 1 (horizontal gaze SD, 36 of 36) |
## MODEL COMPARISON AND ONE-FEATURE BASELINE
Logistic Regression did better than XGBoost on every metric (Table 7): accuracy by 0.043 and ROC-AUC by 0.097, with confidence intervals excluding zero. Neither model did better than the one-feature baseline: Logistic Regression was 0.009 ahead on accuracy and 0.032 on ROC-AUC, and XGBoost was behind, with every interval including zero. Criterion (b) was therefore not met. Table 8 summarizes the pre-set criteria.

**Table 7.** Paired differences (95% CI).

| Comparison | Accuracy | F1-score, macro | ROC-AUC |
|---|---|---|---|
| LR − baseline | 0.009 (−0.045 to 0.059) | 0.009 (−0.046 to 0.059) | 0.032 (−0.026 to 0.086) |
| XGBoost − baseline | −0.035 (−0.099 to 0.026) | −0.035 (−0.099 to 0.027) | −0.065 (−0.140 to 0.001) |
| XGBoost − LR | −0.043 (−0.089 to −0.002) | −0.043 (−0.090 to −0.002) | −0.097 (−0.155 to −0.038) |

**Table 8.** Pass criteria set before analysis.

| Criterion | Result | Met |
|---|---|---|
| (a) Above chance | LR: ROC-AUC CI 0.549–0.662, p = 0.005; XGBoost: CI 0.443–0.579, p = 0.055 | LR yes; XGBoost no |
| (b) Better than one feature | ROC-AUC +0.032 (LR) and −0.065 (XGBoost), both CIs include 0 | No |
| (c) Importance spread | Top feature 10% in both models; top five 41% (LR) and 40% (XGBoost); three signals ≥ 5% in both | Descriptive only (no pass/fail rule) |
## FEATURE IMPORTANCE
No single feature carried either model (Table 9). In both models the top feature held about 10% of the SHAP total and the top five about 40%, and all three signals contributed: horizontal gaze 37%, vertical gaze 35% and pupil 28% in Logistic Regression, and horizontal gaze 42%, pupil 35% and vertical gaze 22% in XGBoost. As planned, each model was refit without the signal of its own top feature: vertical gaze for Logistic Regression and pupil for XGBoost. Without vertical gaze, Logistic Regression fell to an accuracy of 0.536 and a ROC-AUC of 0.554 (95% CI 0.485–0.621), which includes chance; without pupil, XGBoost stayed at chance (accuracy 0.517, ROC-AUC 0.517). Logistic Regression therefore needed the vertical gaze signal, though no single feature of it. With accuracy near 0.6, SHAP shares describe what the models used and should not be over-read.

**Table 9.** Top features by SHAP share (selection count out of 36) and share per signal.

| | Logistic Regression | XGBoost |
|---|---|---|
| 1 | Vertical gaze, minimum, 10% (36) | Pupil, 5th percentile, 10% (36) |
| 2 | Horizontal gaze, 5th percentile, 9% (36) | Horizontal gaze, SD, 9% (36) |
| 3 | Vertical gaze, 90th percentile, 8% (30) | Pupil, minimum, 8% (35) |
| 4 | Pupil, SD, 7% (36) | Vertical gaze, minimum, 7% (30) |
| 5 | Pupil, IQR, 6% (31) | Horizontal gaze, minimum, 6% (30) |
| Share per signal | Horizontal gaze 37%, vertical gaze 35%, pupil 28% | Horizontal gaze 42%, pupil 35%, vertical gaze 22% |
## PLANNED ANALYSIS WITH BLINK RATE
With blink rate included, as in the plan (49 features), Logistic Regression reached an accuracy of 0.592 (ROC-AUC 0.620) and XGBoost 0.595 (0.597), both above chance (permutation p = 0.005), with no difference between them. The one-feature baseline chose blink rate in all 36 folds and reached an accuracy of 0.611 (ROC-AUC 0.581), so neither model did better than it. XGBoost depended on blink rate: blink rate was its top feature (20% of the SHAP total), and without it XGBoost fell to chance, while Logistic Regression did not.
## SECONDARY AND SENSITIVITY ANALYSES
In the 360° search area (572 windows; 4 excluded; 48 features), Logistic Regression reached an accuracy of 0.568 (F1 0.567, ROC-AUC 0.590) and XGBoost 0.607 (F1 0.604, ROC-AUC 0.641). Here XGBoost did better than Logistic Regression (ROC-AUC difference 0.051, 95% CI 0.012–0.086). The one-feature baseline again chose the standard deviation of horizontal gaze and reached 0.600 (F1 0.599, ROC-AUC 0.618); neither model did better than it, so criterion (b) was not met there either.

As a sensitivity check decided after the planned results, the 90° analysis with blink rate was rerun keeping the 91 failed event features (140 in total). Accuracy rose to 0.630 (Logistic Regression) and 0.651 (XGBoost), but one feature alone, the SD of fixation duration, reached 0.616, and event features took 82% of Logistic Regression's SHAP total. Because these features failed the check, this is not the main result.

In the post-hoc checks, the condition order was balanced (18 participants each way). ROC-AUC was 0.592 (Logistic Regression), 0.499 (XGBoost) and 0.561 (baseline) for the participants who did low load first, and 0.621, 0.522 and 0.590 for those who did high load first. Rotating the gaze by the headset's rotation roughly halved the median gaze speed in the 360° area (41 to 22°/s) but did not make the event check pass (median fixation 672 ms at 90° and 127 ms at 360°).
## DISCUSSION
Without blink rate, Logistic Regression detected the high-load condition in new participants above chance but modestly (accuracy 0.568), and XGBoost did not. Logistic Regression did better than XGBoost in the main 90° area, but XGBoost did better in the 360° area, so neither model was consistently better, in line with studies that found no consistent advantage {13}, {26}. Neither model did better than the standard deviation of horizontal gaze alone, but that feature was weak (ROC-AUC 0.574), and no single feature carried the result: the top feature held about 10% of the SHAP total in both models, and pupil, horizontal gaze and vertical gaze all contributed. The load signal in these features is therefore weak and spread out rather than concentrated in one feature, although Logistic Regression lost its above-chance result without the vertical gaze signal. With 36 participants, the confidence intervals of the differences are about ±0.05 wide, so a small real difference cannot be told apart from none.

The directions agree with Bielsa et al. {bielsa}: under high load, the 5th percentile of pupil size was larger in 28 of 36 participants and the horizontal gaze spread was smaller in 25 of 36.

As in the COLET version, the planned single best feature was blink rate, and again it was tied to something other than load: speaking there, and tracking loss here. Detected blinks were far fewer than real blinking {23} and rose with the share of invalid samples. Blink rate rose under high load, the direction reported for added mental load {36}, but because the high-load condition adds button presses, the rise may come from movement. A blink signal from a VR headset should therefore be checked against tracking quality before it is used as a load feature.

Condition order was balanced, so it cannot produce the main result, and the difference between participants who did high or low load first was small. The event check failed in both areas. Gaze measured relative to the head mixes eye and head movement {72}, but correcting for head rotation did not make the check pass, and at 90°, where the head barely turns, the fixations were too long rather than too short, so head movement does not explain that failure; its cause was not identified. Bielsa et al. detected fixations after mapping gaze into the 3D scene with the headset's position and orientation {bielsa}, which needs the scene geometry that is not part of the released data.

# CHAPTER 5|CONCLUSION AND RECOMMENDATION
## SUMMARY OF FINDINGS
1.	The event-detection check failed, and blink rate was found to measure tracking loss, so 48 features (pupil, horizontal and vertical gaze) were used for 36 participants and 576 windows.

2.	In the 90° search area, Logistic Regression reached an accuracy of 0.568 (95% CI 0.523–0.613), precision 0.568, F1-score 0.568 and ROC-AUC 0.606, above chance (p = 0.005). XGBoost reached 0.524 (0.470–0.580), 0.524, 0.524 and 0.509, not above chance.

3.	Logistic Regression did better than XGBoost in the 90° area (accuracy +0.043, ROC-AUC +0.097), but XGBoost did better in the 360° area (ROC-AUC +0.051).

4.	Neither model did better than the one-feature baseline (horizontal gaze SD: accuracy 0.559, F1-score 0.559, ROC-AUC 0.574); all differences had confidence intervals including zero.

5.	No single feature carried the result: the top feature held about 10% of SHAP attributions in both models, and all three signals contributed. Without the vertical gaze signal, Logistic Regression fell to chance (accuracy 0.536).

6.	In the planned analysis with blink rate, both models reached about 0.59 accuracy and blink rate alone 0.611; blink rate classified no better than the share of invalid samples.
## CONCLUSIONS
Eye-tracking features from a VR headset can detect a high-load task condition in new people above chance with Logistic Regression, but modestly, and XGBoost did not do so in the main analysis. Which model did better depended on the search area, so neither is consistently better, and neither did better than the single best feature, a weak measure of horizontal gaze spread. The load signal in these features is weak and spread across pupil and gaze rather than carried by one feature. The headset's blink signal measured tracking loss rather than blinking, and should be checked before blink rate is used.
## LIMITATIONS
•	The high-load condition adds an auditory task with button presses, so the results reflect a task condition, not cognitive load alone.

•	Fixation and saccade features could not be used: the event check failed, also after correcting for head rotation, and the scene geometry needed to map gaze into the scene is not released.

•	Blink rate was excluded after the planned analysis had been run, a deviation from the plan; the planned results are reported.

•	Pupil size also responds to the brightness of the scene, which was not controlled.

•	The sample is small (36 participants), all windows of a condition come from one segment, so the 576 windows are not independent (the confidence intervals resample participants), and only one VR task was studied.

•	No equivalence margin was set, so "no difference" does not mean the models are equal.

•	The checks on tracking loss, condition order and head rotation were added after the planned analysis.
## RECOMMENDATIONS
1.	Map gaze into the scene with the headset's position and orientation and the scene geometry before detecting fixations and saccades, as Bielsa et al. did {bielsa}.

2.	Repeat the comparison with more participants, so that small differences between models and the single best feature can be detected.

3.	Use a headset or method that reports blinks directly, check any blink signal against tracking quality, and use a secondary task without a motor response.

# APPENDIX A|ANALYSIS PLAN
The plan below was written on 13 November 2026, before any eye-tracking data were loaded. It was not deposited with a timestamp at the time: it was first committed to the project repository on 23 November 2026, together with the results, so its date rests on the researchers' own record. The cleaning steps, thresholds and event check had been built earlier on the COLET dataset, and the change of dataset was itself prompted by the COLET results. Two deviations from the plan are reported in Chapters 3 and 4: blink rate was excluded from the main analysis after the planned analysis had been run, and an earlier draft removed the same signal from both models in the refit (item 8c) instead of each model's own top signal. Before writing it, only the trial files had been inspected: each condition is one continuous four-minute segment, trials (one per object found) last about one second, and fewer trials occur under high load.

1.	Contrast (main): 90° search area, low load versus high load. Secondary, reported separately and never pooled with or chosen over the main contrast: 360°.

2.	Unit: each four-minute segment cut into eight non-overlapping 30-second windows from the first trial start; 16 windows per participant (8 low, 8 high).

3.	Signals, mapped onto the unchanged feature code: gaze from each eye's forward vector, valid if the gaze and both eyes are valid; pupil diameter (mm) per eye; blinks as runs with both eyes invalid, then the existing rule (merge gaps under 100 ms, keep 50–500 ms).

4.	Exclusions: windows with more than 35% invalid samples. Event check on the median fixation duration (150–400 ms); if it fails, gaze-speed, fixation and saccade features are dropped.

5.	Features: the same 17 statistics per signal, no per-person standardization.

6.	Models: in-fold top-N selection, elastic-net Logistic Regression and XGBoost, 30-candidate random search, five-fold grouped inner cross-validation, leave-one-participant-out.

7.	Reported: every metric (ROC-AUC with participant-bootstrap 95% CI, accuracy, balanced accuracy, precision, recall per class, macro F1, Brier, participants with high above low).

8.	Pass tests: (a) above chance: ROC-AUC CI lower bound above 0.50 and label-permutation p < 0.05 (labels swapped per participant, selection inside, settings fixed at the most-chosen values); (b) features work together: model minus in-fold one-feature baseline has a CI above 0; (c) importance spread (descriptive): SHAP share of the top 1, 3 and 5 features, number of signals with at least 5% share, selection counts, and the models refit without the top feature's signal.

9.	Whatever the outcome, all of the above are reported. No other contrasts, windows or feature sets are tried.

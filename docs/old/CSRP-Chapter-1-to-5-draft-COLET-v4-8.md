# CHAPTER 1|THE PROBLEM AND ITS BACKGROUND
## PROJECT CONTEXT
Cognitive load is the mental effort required by a task, drawing on an individual's limited processing capacity {1}, {2}. It matters in ergonomics and human factors because mental demands are linked to performance and safety in complex tasks {3}, and monitoring it is of growing interest in education, healthcare, transportation and human–computer interaction. In this study, cognitive load is a label for task conditions: COLET's easiest activity (A1) is low load, and its hardest activity, which adds counting aloud under time pressure (A4), is high load. Related terms such as mental workload and mental effort are treated as the same idea.

Cognitive load is usually measured with questionnaires such as the NASA Task Load Index (NASA-TLX) {4} or with physiological signals such as electroencephalography and cardiac activity {5}. Questionnaires are given after the task and depend on self-assessment, and different workload measures do not always agree {6}. Physiological sensors need specialized equipment that is intrusive and hard to use outside the laboratory. Eye tracking is a less intrusive alternative that records pupil size, blinks, fixations and saccades during the task. A systematic review of 91 studies found pupil diameter significant in 79% of the studies that measured it, fixation duration in 73% and blink rate in 71% {5}.

Machine learning can combine eye-tracking measures into models that classify cognitive load. Logistic Regression is a linear model with interpretable coefficients and a common baseline {7}, {8}. XGBoost is a boosted tree algorithm that can model non-linear relationships and feature interactions in tabular data {9}, {10}. Comparing the two shows whether the extra flexibility of an ensemble improves on a simple, interpretable model.

Studies comparing the two models used different datasets, labels, features and validation, and reached different results {11}, {12}, {13}. Many evaluated models on data from the same people used for training, which overestimates performance for new users {14}, {15}, and few explained the features used by both models {16}, {17}.

This study compares Logistic Regression and XGBoost for cognitive load detection on the public COLET dataset {18}, which contains raw gaze, pupil and blink data from 47 participants performing visual-search tasks under four conditions. Both models are tested on participants excluded from training, and both are explained with the same method (SHAP).
## PURPOSE AND DESCRIPTION
This study develops and evaluates a machine learning framework that detects cognitive load from eye-tracking features, and compares Logistic Regression and XGBoost for classifying low versus high load.

Raw eye-tracking data are taken from the COLET dataset {18}. The single task without time pressure (A1) is labeled low load, and the dual task with time pressure (A4) is labeled high load. The recordings are cleaned following published guidelines where they exist, with the remaining settings listed as design choices, and fixations and saccades are detected with the settings of the COLET authors. Each cleaned eye signal is then summarized with 17 statistics (for example, mean, median, standard deviation and percentiles), giving 140 features per recording after copies and unusable statistics are removed. Both models are evaluated with leave-one-participant-out validation, so every prediction is made for a person not used in training {14}. Inside each training fold, the features are ranked and only the best N are kept, with N chosen by tuning.

The study also identifies the eye-tracking features each model relied on, using SHAP values for both models {19}, {20} and the number of folds in which each feature was selected.
## OBJECTIVES OF THE STUDY
### General Objective
The general objective of this study is to compare Logistic Regression and XGBoost for cognitive load detection using eye-tracking features, and to identify the features each model relies on.
### Specific Objectives
Specifically, the study aims:

1.	To extract whole-activity eye-tracking features from the COLET dataset by summarizing each cleaned eye signal with 17 statistics, for binary classification of low (A1) versus high (A4) cognitive load.

2.	To train and evaluate Logistic Regression and XGBoost under leave-one-participant-out validation, with feature selection and hyperparameter tuning done inside each training fold.

3.	To compare the two models using ROC-AUC, accuracy, precision, recall, F1-score and the Brier score, against a majority-class baseline.

4.	To determine how performance changes with the number of selected features.

5.	To analyze feature importance using SHAP values for both models and the number of folds in which each feature was selected.
## SIGNIFICANCE OF THE STUDY
This study examines how well an interpretable model and an ensemble model can distinguish workload conditions from eye movements alone. Education, healthcare (where pupil size and blink rate have been used to assess workload in surgery {17}, {25}), safety-critical work {3} and human–computer interaction could one day use such monitoring, in line with Sustainable Development Goals 3, 4 and 9 {24}. As a small laboratory study, it is only an early step. The comparison also adds evidence, as in clinical prediction {26}, on when an interpretable model is enough.

For researchers, the study offers a documented evaluation on a public dataset in which every step that learns from data (feature selection, scaling and tuning) is fitted on training participants only, a practice that prevents overly optimistic results {15}. Because COLET is open, the analysis can be reproduced and extended.
## SCOPE AND LIMITATIONS
The study compares Logistic Regression and XGBoost for cognitive load detection using eye-tracking features from the COLET dataset {18}, which contains raw Pupil Core recordings of 47 participants in four activities. Only the two extreme activities are classified: A1 (single task, no time pressure) as low load and A4 (counting backwards aloud under time pressure) as high load. After excluding low-quality recordings and one duplicated set of recordings, 44 participants and 88 recordings are analyzed, one sample per whole activity.

The scope covers preprocessing, fixation and saccade detection, extraction of summary-statistic features, feature selection, model training and tuning, leave-one-participant-out evaluation and feature importance. The study uses eye-tracking data only, is limited to Logistic Regression and XGBoost, and uses a laboratory task with a chin rest, so the findings may not transfer to real-world settings {27}. The sample of 44 participants is small, which widens the uncertainty of the results {28}.

The high-load activity involves speaking, which raises the blink rate {23}. The experiment does not separate talking from workload, so the study detects COLET's task conditions, which mix dual-tasking, speaking and time pressure, and not cognitive load separated from speech. Real-time detection, hardware implementation and testing on another dataset are outside the scope. Self-reported workload is used only to confirm that the activities differed in workload.
## CONCEPTUAL FRAMEWORK
The conceptual framework is a sequential machine learning pipeline that turns raw eye-tracking data into classification results and feature importance (Figure 1).

![Conceptual framework of the study.](@docx:figure1)

Gaze, pupil and blink measurements are taken from the COLET dataset and preprocessed to remove low-confidence data and to detect fixations and saccades {30}, {31}. In the Experimental Configurations stage, A1 is labeled low load and A4 high load. Each cleaned signal is summarized into the Eye-Tracking Feature Set. The same features are given to a regularized linear model (Logistic Regression) and a non-linear ensemble (XGBoost) {9}. Both are trained and tested with Leave-One-Participant-Out (LOPO) validation, in which the test participant's data are never used for training, feature selection or tuning {33}. The held-out predictions feed two analyses: Performance Evaluation and Feature Analysis (SHAP and selection count). The framework yields three deliverables: a comparison of model performance, a ranking of the important eye-tracking features, and tuned cognitive load classification models.
## OPERATIONAL DEFINITION OF TERMS
**Accuracy –** The share of recordings classified correctly. The two classes are equal in size, because each participant has one low-load and one high-load recording.

**Binary Classification –** Assigning each observation to one of two classes. In this study, each recording is classified as low (A1) or high (A4) cognitive load.

**Blink Rate –** The number of blinks per minute of valid recording time.

**Brier Score –** The mean squared difference between predicted probabilities and actual outcomes; lower is better.

**Cognitive Load –** The mental effort required to perform a task. In this study it is defined by the COLET task condition: A1 is low load and A4 is high load. Because A4 also involves speaking, the models classify task conditions, not a directly measured mental state.

**COLET Dataset –** A public dataset of raw eye-tracking recordings from 47 participants performing visual-search tasks under four workload conditions; the only data source of this study.

**Elastic Net –** A penalty for regression models that combines L1 and L2 regularization; used to keep Logistic Regression stable with a small sample.

**Eye Tracking –** The measurement of eye movements and pupil size with a camera-based tracker; here, a Pupil Labs Pupil Core.

**F1-Score –** The harmonic mean of precision and recall. Both the per-class and the macro-averaged F1-score are reported.

**Feature Selection –** Keeping only the most useful features. In this study, features are ranked by their single-feature ROC-AUC on the training participants, and the top N are kept.

**Fixation –** A period in which the eyes stay relatively still on one location.

**Gaze Angle –** The horizontal (left/right) or vertical (up/down) direction of gaze in degrees. Its location statistics, such as minimum and mean, describe where people looked.

**I-VT (Velocity-Threshold Identification) –** An algorithm that labels eye samples as fixation or saccade depending on whether eye speed is below or above a threshold; here, 45°/s.

**Leave-One-Participant-Out (LOPO) Validation –** Holding out all data of one participant for testing, training on the others, and repeating for every participant.

**Logistic Regression –** A classification model that estimates the probability of a class from a weighted sum of input features; the interpretable model of this study.

**NASA-RTLX –** The raw, unweighted NASA Task Load Index; used only to confirm that the activities differed in workload.

**Precision –** Of the recordings predicted as one class, the share that truly belong to it.

**Pupil Diameter –** The size of the pupil in millimetres, estimated by the tracker's 3D eye model.

**Recall –** Of the recordings that truly belong to one class, the share predicted as that class.

**ROC-AUC –** The area under the Receiver Operating Characteristic curve: how well a model ranks high-load above low-load recordings across all thresholds; the primary metric.

**Saccade –** A rapid eye movement between fixations.

**Selection Count –** The number of LOPO folds, out of 44, in which a feature was among the selected top N.

**SHAP (SHapley Additive exPlanations) –** A method that divides each prediction into the contribution of each feature; applied to both models.

**Summary Statistic –** A single number describing a signal over a whole recording, such as its mean, median, standard deviation, minimum, maximum, percentiles, skewness or kurtosis.

**XGBoost –** An implementation of gradient-boosted decision trees; the ensemble model compared with Logistic Regression.

# CHAPTER 2|REVIEW OF RELATED LITERATURE AND STUDIES
This chapter reviews cognitive load and its measurement, eye-tracking indicators of workload, the COLET dataset, Logistic Regression and XGBoost for workload classification, participant-independent validation and feature selection, and feature importance, and ends with the research gap.
## COGNITIVE LOAD AND ITS MEASUREMENT
Cognitive load describes the share of a person's limited mental capacity that a task requires {1}. In cognitive load theory, working memory is limited, so tasks that exceed it impair learning and performance {2}. In ergonomics, mental workload is studied because of its relationship with performance and safety {3}.

Workload is measured with subjective ratings, performance measures and physiological signals. The NASA Task Load Index is the most widely used rating scale {34}; it rates six dimensions of workload after a task {4}, and its unweighted "raw" form is common {34}. Ratings are given after the task, and different workload measures are each sensitive to load but do not always agree {6}. Physiological measures are recorded during the task, and eye measures are among the most widely used {5}.
## EYE-TRACKING INDICATORS OF COGNITIVE LOAD
Pupil diameter is the most established eye indicator of mental effort. The pupil dilates with increasing demand but also responds strongly to light, so luminance must be controlled {35}. Blink rate is also sensitive to workload but depends on the task: mental work increases blinking while visual tasks decrease it {36}, and blinking increases during speech {23}. Saccade amplitude has been reported to decrease under load {37}, saccadic peak velocity has been proposed as a workload index {38}, {39}, fixation rate and duration relate to both perceptual and cognitive load {40}, and gaze tends to concentrate in a smaller area under higher load {41}, {42}.

Fixations and saccades must be detected from raw gaze samples. The velocity-threshold algorithm (I-VT) labels samples as saccades when eye speed exceeds a threshold {31}. Detection results depend on the algorithm and thresholds, so settings should be reported and the output checked {43}, {44}.
## THE COLET DATASET AND ITS PRIOR USE
COLET was published by Ktistakis et al. as an open dataset for cognitive workload estimation from eye tracking {18}. Forty-seven participants solved visual-search puzzles in four activities that crossed time pressure with a secondary task of counting backwards aloud. Eye movements were recorded with a Pupil Labs Pupil Core tracker, and participants rated each activity with the NASA-RTLX. The authors summarized pupil, blink, fixation and saccade signals with statistics including higher-order ones, and Logistic Regression separated A1 from A4 with an accuracy of 0.85. That result used a 20% hold-out set and five-fold cross-validation not described as participant-independent, with feature selection on the full dataset. An earlier conference paper described the approach {45}.

Later studies used COLET with random splits of windows {46}, five-fold cross-validation on overlapping windows {47} and a 70/15/15 split of recordings {48}; none of these splits were grouped by participant, so data from the same person could appear in both training and test sets. Fenoglio et al. evaluated on unseen users with user-grouped cross-validation, using Gaussian Naive Bayes, random forest and neural networks {49}. None of these studies applied Logistic Regression and XGBoost together or reported feature importance.
## LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD CLASSIFICATION
Logistic Regression models the probability of a class from a weighted sum of features {7}. With small samples, its coefficients can be unstable, so regularization is used; the elastic net combines L1 and L2 penalties, handles correlated features and can set the weights of unhelpful features to zero {50}. Kaczorowska et al. used elastic-net Logistic Regression for three-level workload classification from eye-tracking features {8}.

XGBoost trains decision trees in sequence with regularization against overfitting {9}, and tree ensembles work well on tabular data {10}. Walocha et al. tuned XGBoost with nested leave-one-participant-out validation and found that shallow trees were mostly selected in their small dataset {51}, and Shafiei et al. used XGBoost and SHAP with EEG and eye tracking in surgery {17}.

Direct comparisons disagree. On held-out participants, Rolon-Merette et al. found XGBoost above chance while Logistic Regression, given raw time series, stayed near chance {11}. Nerella et al. compared both models in augmented reality {12}, and Shao et al. found no significant ROC-AUC difference with eye-only features in driving {13}. In clinical prediction, a systematic review found no consistent benefit of machine learning over Logistic Regression {26}.
## PARTICIPANT-INDEPENDENT VALIDATION AND FEATURE SELECTION
If records from the same person are in both the training and test sets, a model can recognize the person rather than the condition, and accuracy is overestimated; subject-independent validation better reflects use on new people {14}, {53}. Leakage also occurs when steps such as feature selection or scaling are fitted on the whole dataset before splitting {15}. Tuning must likewise be done inside the training data, through nested cross-validation {33}, {54}. With small samples, cross-validated estimates have large error bars, so uncertainty must be reported {28}. Individual differences in eye behavior are large and stable {32}, which makes generalizing to new people harder.

Summarizing each eye signal with many statistics gives far more features than samples. Filter methods rank each feature by a simple score and keep the best ones {guyon}; to avoid optimistic results, the ranking must be repeated inside every training fold {15}, {54}.
## FEATURE IMPORTANCE ANALYSIS
SHAP assigns each feature a share of every prediction based on Shapley values and can be applied to any model {19}; TreeSHAP computes these values efficiently for tree ensembles such as XGBoost {20}. When features are selected inside each fold, how often a feature is selected shows how stable the selection is {83}. Importance shows what a model relied on, not a causal effect {62}.

Božak et al. combined condition labels, leave-one-subject-out validation and SHAP for eye-tracking workload inference {16}, and Xu et al. cautioned that SHAP rankings from models trained on all data are exploratory {63}. These studies explained a single model or model family: Logistic Regression studies used coefficients {8}, and tree-based studies used SHAP {17}.
## SYNTHESIS
Table 1 summarizes the closest studies.

**Table 1.** Summary of the closest related studies.

| Study | Data | Models | Validation | Feature importance |
|---|---|---|---|---|
| Ktistakis et al. {18} | COLET | Several classifiers incl. LR, k-NN and GNB | 20% hold-out and 5-fold CV, not described as participant-independent | – (ANOVA feature selection only) |
| Fuhl et al. {46} | COLET | Classifiers on a pupil feature | Random split of windows | – |
| Fenoglio et al. {49} | COLET | GNB, RF, neural networks | 4-fold user-grouped CV | – |
| Wibirama et al. {47} | COLET | Deep learning | 5-fold CV on overlapping windows | – |
| Dell'Acqua et al. {48} | COLET | Large language models | 70/15/15 split of recordings | – |
| Rolon-Merette et al. {11} | Own n-back data | LR, XGBoost, SVM, CNN, Transformer | 10 held-out participants | – |
| Kaczorowska et al. {8} | Own data | Elastic-net LR and others | Repeated random train-test splits | LR coefficients |
| Božak et al. {16} | Own data | Feature-based models | Leave-one-subject-out | SHAP |
| Shafiei et al. {17} | EEG + eye | XGBoost | Random split | SHAP |
| This study | COLET | LR (elastic net), XGBoost | Leave-one-participant-out | SHAP for both models, selection count |

Direct comparisons disagree on whether XGBoost improves on Logistic Regression {11}, {13}, and in clinical prediction it often does not {26}. Whether a boosted ensemble improves on a regularized linear model for small eye-tracking datasets under participant-independent evaluation therefore remains open. Among the studies found, none applied XGBoost or SHAP to COLET, the only participant-independent COLET evaluation {49} used neither model nor feature importance, and the others used splits not grouped by participant {18}, {46}, {47}, {48}. This study compares the two models on COLET under leave-one-participant-out validation and explains both with the same method.

# CHAPTER 3|METHODOLOGY
This chapter describes the research design, dataset, labels, preprocessing, feature extraction, models, validation, metrics and feature importance analysis (Figure 2).

![Overview of the methodology.](outputs/final/paper_figures/fig2_method.png)
## RESEARCH DESIGN
The study uses a quantitative, comparative design based on secondary analysis of a public dataset. Both models are trained and evaluated on the same features, with the same folds and the same number of candidate settings, so that differences in performance mainly reflect the models. All cleaning thresholds were fixed after a data-quality inspection and before any model was trained.
## DATASET
The study uses COLET, version 3, from Zenodo (record 7766785), released under a CC BY 4.0 license {18} (Table 2).

**Table 2.** Description of the COLET dataset.

| Item | Description |
|---|---|
| Participants | 56 recruited; 9 excluded by the authors; 47 released (26 female, 21 male; age 32 ± 8 years) |
| Setting | Laboratory with chin and head rest; 24-inch screen (1280 × 720) at 80 cm; controlled lighting (400–450 lx) |
| Eye tracker | Pupil Labs Pupil Core, binocular; gaze ≈ 242 Hz, pupil ≈ 121 Hz per eye (measured; nominal 240 Hz) |
| Task | Visual-search puzzles, five images per activity; activities in random order with 2-minute breaks |
| Design | 2 × 2: time pressure × secondary task (counting backwards aloud from 1000 in steps of 4) |
| Released data | Raw gaze, pupil and blink exports per participant and activity; NASA-RTLX per activity; no fixation or saccade events |

The dataset is a single 3.8 GB MATLAB file with nested tables per participant, activity and signal. After its checksum was verified, it was decoded once with the open-source mat-io package and flattened into one table per participant, activity and signal (Figure 3): 564 Parquet files (188 recordings × 3 signals) and two CSV files with the NASA-RTLX ratings and participant information. No values were changed.

![Structure of the COLET data before and after flattening.](@docx:figure3)
## LABELS AND UNIT OF ANALYSIS
Labels come from the experimental conditions: A1 (single task, no time pressure) is low load and A4 (counting aloud under time pressure) is high load. The visual-search task is the same in all activities; A4 adds a spoken counting task, so the two conditions also differ in speech. A2 and A3 are left out of the classification, following studies that compare the extremes of a workload manipulation {25}, {64}. Condition labels are used rather than self-reports because ratings depend on how each participant uses the scale {34}.

The NASA-RTLX ratings {18}, {34} are used as a manipulation check: a Friedman test compares the four activities, and each participant's A4 rating is compared with their A1 rating. The activities are short (11 to 141 s), so the unit of analysis is one sample per whole activity, as in the COLET study {18}, giving two samples per participant (Table 3).

**Table 3.** COLET activities and their role in this study.

| Activity | Condition | Median duration | Role |
|---|---|---|---|
| A1 | Single task, no time pressure | 37 s | Low load |
| A2 | Single task, time pressure | 29 s | Manipulation check |
| A3 | Counting aloud, no time pressure | 62 s | Manipulation check |
| A4 | Counting aloud and time pressure | 51 s | High load |
## DATA PREPROCESSING
The raw signals are cleaned before features are computed (Figure 4, Table 4).

![Preprocessing and feature extraction.](outputs/final/paper_figures/fig4_preprocessing.png)

**Table 4.** Preprocessing steps.

| Step | Procedure | Source |
|---|---|---|
| Time alignment | Pupil and blink data trimmed to each recording's gaze time range | Data check |
| Sample validity | Samples with tracker confidence below 0.8 treated as invalid | {30}, {65} |
| Recording exclusion | Recordings with more than 35% invalid samples excluded; participants missing A1 or A4 dropped (P06, P17); duplicated recordings kept once (P31 dropped) | {66} |
| Valid time | Blink rate computed over valid recording time (gaps longer than 1 s excluded); fixation and saccade rates over time with a valid gaze direction | Data check |
| Blinks | Tracker blink events merged if less than 100 ms apart, then kept if 50–500 ms; blink periods treated as missing | {67}, {68} |
| Gaze gaps | Gaps shorter than 75 ms linearly interpolated; longer gaps left missing | {30} |
| Pupil | 3D-model diameter in mm; 1.5–9 mm range; dilation-speed outlier removal; both eyes averaged; gaps up to 250 ms filled; 4 Hz low-pass filter | {29}, {68}, {69} |
| Gaze direction | Each eye's unit gaze vector, averaged across both eyes and resampled to 240 Hz | {70}, {71}, {72} |
| Quality report | Share of valid samples reported per recording; never used as a feature | {30} |

Three of 188 recordings were excluded (P06 A3, P06 A4 and P17 A4), so P06 and P17 lack a valid A1–A4 pair and are dropped. The released files of P28 and P31 are byte-identical in all four activities although their demographic data and ratings differ; the recordings are kept once (as P28), and both IDs are left out of the NASA-RTLX analysis. This leaves 44 participants and 88 samples.

Gaze angles are computed from each eye's gaze direction rather than the tracker's 3D gaze point, because video eye trackers do not measure vergence, and thus viewing distance, reliably {70}, {71}; in COLET, the estimated distance had a median of about 113 mm against a true 800 mm. Eye speed is computed from the change in direction of the averaged gaze vector {72}. Samples with only one valid eye are treated as gaps {30}.
## FIXATION AND SACCADE DETECTION
Fixations and saccades are detected with I-VT {31} using the settings of the COLET authors {18}. Eye velocity is computed from the 240 Hz gaze directions with a five-point smoothed difference {73}, and velocities above 1000°/s are rejected as artifacts {74}. Samples faster than 45°/s are saccades, and fixations shorter than 55 ms are discarded; these values match human-coded saccades and fixations in Andersson et al. {43} and lie within the range evaluated by Trabulsi et al. {75}.

Because detection depends on the algorithm and its settings {43}, {44}, a sanity check is applied to each activity: the median fixation duration must lie between 150 and 400 ms, and the ratio of saccades to fixations between 0.8 and 1.25. If either fails in any activity, all fixation and saccade features are dropped.
## FEATURE EXTRACTION
Each cleaned signal of a whole activity is summarized with 17 statistics: minimum, maximum, range, mean, median, standard deviation, variance, coefficient of variation, interquartile range, the 5th, 10th, 25th, 75th, 90th and 95th percentiles, skewness and kurtosis (Table 5). The COLET authors also used higher-order statistics {18}. Blink, fixation and saccade rates are added as single features. This gives 173 features. The statistics of a signal are left blank when a recording has fewer than 20 values of it.

**Table 5.** Eye signals and features.

| Signal | Unit | Features kept | Support |
|---|---|---|---|
| Pupil diameter | mm | 16 | {5}, {18}, {35} |
| Gaze angle, horizontal and vertical | degrees | 16 each | {16}, {41} |
| Gaze speed | °/s | 15 | {38} |
| Fixation duration | ms | 15 | {5}, {40} |
| Saccade amplitude, peak velocity, mean velocity | degrees, °/s | 15 each | {37}, {38} |
| Saccade duration | ms | 14 | {37} |
| Blink, fixation and saccade rates | per minute or second | 1 each | {5}, {36}, {40} |
| Removed before modeling | | 33 | |

Thirty-three features were removed before any model was trained. Blink-duration statistics (17) were blank in 79 of 88 recordings (fewer than 20 blinks); 8 of the 9 recordings with values were high load, so whether a value exists would reveal the label {15}. Variances (9) are the squares of the standard deviations and carry the same information, ranges of six signals whose minimum is fixed near zero copy the maximum {76}, and the minimum saccade duration was stuck at one sample. This leaves 140 features; none had missing values. Statistics of gaze location (for example, the lowest vertical gaze angle) describe where people looked and may also reflect where the stimuli appeared or how the headset sat; they are kept and reported with this caveat. Room lighting was controlled, and the COLET authors found pupil size following image brightness in only 2 of 47 participants {18}, so no further luminance correction is applied.
## MODEL DEVELOPMENT
Two models are trained (Table 6). Before each model, features are ranked by how well each one alone separates the two classes in the training participants (|ROC-AUC − 0.5|), and the top N are kept {guyon}. N is tuned together with the model settings from {5, 10, 20, 40, 80, 140}. Logistic Regression uses standardization fitted on the training fold and an elastic-net penalty {8}, {50}. XGBoost uses shallow regularized trees, since each fold trains on only 86 samples {9}, {51}. Both models get the same number of candidate settings and the same folds, and tuning is nested to avoid selection bias {33}.

**Table 6.** Model settings.

| Model | Settings searched |
|---|---|
| Both | Number of features N ∈ {5, 10, 20, 40, 80, 140} |
| Logistic Regression | Elastic-net penalty; C log-uniform 10⁻³–10²; L1 ratio 0–1 |
| XGBoost | Maximum depth 1–3; 50–300 trees; learning rate 0.05–0.1; row and column subsampling 0.7–1.0; L2 penalty 0.01–10 and L1 penalty 0.001–1 (log-uniform) |
## VALIDATION PROCEDURE
Models are evaluated with leave-one-participant-out validation over the 44 participants: in each fold, both recordings of one participant form the test set and the other 43 the training set {14}, {53}. Feature ranking, scaling and tuning are fitted on the training participants only {15}. Settings are tuned by random search with 30 candidates per model, scored by ROC-AUC in a five-fold inner cross-validation grouped by participant {33}, {54}, {55}.
## PERFORMANCE METRICS AND STATISTICAL ANALYSIS
The primary metric is ROC-AUC on the pooled held-out predictions {78}, with a 95% confidence interval from a bootstrap that resamples participants (2000 resamples) {28}. Accuracy, precision, recall and F1-score (per class and macro-averaged) at a 0.5 threshold {78}, the Brier score {79}, {80}, a majority-class baseline {16} and the number of participants whose A4 recording is ranked above their A1 recording are also reported.

The models are compared by the difference in pooled ROC-AUC with a paired participant-bootstrap confidence interval. A Wilcoxon signed-rank test on each participant's margin (the predicted probability for A4 minus that for A1) is a supporting test {81}. A grouped permutation test, which swaps each participant's two labels at random 200 times, checks performance against chance {82}; it reruns the whole pipeline, feature selection included, with default settings (N = 20), so the smallest possible p-value is 1/201 ≈ 0.005.

Three further analyses use the same procedure: the models with N fixed at each candidate value, to show how performance changes with the number of features; Logistic Regression on blink rate alone, because A4 involves speaking, which raises the blink rate {23}; and both models without blink rate.
## FEATURE IMPORTANCE ANALYSIS
SHAP values are computed for both models on the held-out participant of each fold, using TreeSHAP for XGBoost and linear SHAP for Logistic Regression {19}, {20}. A feature not selected in a fold counts as zero there. Each feature's share of the total mean absolute SHAP value is reported. The selection count, the number of folds in which a feature was among the top N, shows how stable the selection was {83}. Importance is read as what each model relied on, not as a cause of cognitive load {62}.
## IMPLEMENTATION
The pipeline was implemented in Python 3.11 with NumPy 2.2.4, pandas 2.3.1, SciPy 1.15.2, scikit-learn 1.7.1, XGBoost 3.2.0 and SHAP 0.51.0. All thresholds are held in one configuration file, and all random processes use a fixed seed, so repeated runs give identical results. The pipeline was tested on synthetic data with a known structure before it was applied to COLET.
## ETHICAL CONSIDERATIONS
This study is a secondary analysis of the public, de-identified COLET dataset (CC BY 4.0), whose collection was approved by the Ethics Committee of the Foundation for Research and Technology – Hellas (110/12-02-2021) {18}. No new data were collected, and no attempt is made to identify participants.
## DESIGN CHOICES
Some settings have no published source for their exact value and are the researchers' own choices: the 35% exclusion threshold, the 1 s gap rule for valid time, the 0.8–1.25 sanity-check tolerance, using each eye's gaze direction, treating one-eye samples as missing, merging blinks before applying the 50–500 ms range, the pupil filter, the 240 Hz grid, the five-point velocity filter, the minimum of 20 values per signal and the candidate values of N. A blink or a gap longer than 75 ms inside a fixation splits it into two.

# CHAPTER 4|RESULTS AND DISCUSSION
This chapter presents the manipulation check, data quality, the event-detection check, the features, the classification results, the number of features, the blink-rate checks and feature importance, followed by a discussion.
## MANIPULATION CHECK
Rated workload differed across the four activities (Table 7, Figure 5; Friedman χ²(3) = 82.72, p < 0.001, n = 43). The test excludes P28, because the owner of the recordings shared by P28 and P31 is unknown. All 43 participants rated A4 higher than A1, with a median difference of 31.7 points, so the labels reflect a difference in reported workload, although the conditions also differ in speaking and time pressure.

**Table 7.** NASA-RTLX ratings of the 43 analyzed participants with their own recordings.

| Activity | Condition | Mean NASA-RTLX (SD) | Median |
|---|---|---|---|
| A1 | Single task, no time pressure | 18.3 (11.8) | 16.7 |
| A2 | Single task, time pressure | 27.8 (13.8) | 29.2 |
| A3 | Counting aloud, no time pressure | 45.0 (15.9) | 44.2 |
| A4 | Counting aloud and time pressure | 51.8 (17.8) | 52.5 |

![NASA-RTLX ratings per activity.](@docx:figure5)
## DATA QUALITY AND EVENT DETECTION
Three recordings exceeded the 35% invalid-sample limit: P06 A3 (57.4%), P06 A4 (38.8%) and P17 A4 (48.6%). With P06, P17 and the duplicate P31 removed, 44 participants and 88 samples remained (Table 8). Invalid samples were about eight times more frequent in the counting activities, consistent with more blinking and tracking loss while talking.

**Table 8.** Data quality per activity (retained recordings).

| Activity | Invalid samples (median) | Samples with both eyes (median) | Recordings with a pupil-model refit |
|---|---|---|---|
| A1 | 0.9% | 99.5% | 8 |
| A2 | 0.7% | 99.5% | 11 |
| A3 | 6.9% | 98.6% | 20 |
| A4 | 6.9% | 98.7% | 18 |

The event-detection sanity check passed in every activity (Table 9), so all fixation and saccade features were kept.

**Table 9.** Event-detection sanity check per activity.

| Activity | Recordings | Median fixation duration (ms) | Saccades per fixation | Passed |
|---|---|---|---|---|
| A1 | 44 | 212.5 | 0.98 | Yes |
| A2 | 44 | 212.5 | 0.99 | Yes |
| A3 | 44 | 235.4 | 0.93 | Yes |
| A4 | 44 | 225.0 | 0.93 | Yes |
## EYE-TRACKING FEATURES BY CONDITION
Table 10 gives the medians of the most important features (see Feature Importance) and of mean pupil diameter. Under high load, participants blinked far more, made fewer saccades, reached a lower minimum vertical gaze angle and spread their gaze more widely from left to right, and their pupils were larger.

**Table 10.** Median feature values in A1 and A4 (44 participants).

| Feature | A1 (low) | A4 (high) | Higher in A4 | Unit |
|---|---|---|---|---|
| Blink rate | 1.7 | 11.7 | 42 of 44 | per minute |
| Saccade rate | 3.36 | 2.85 | 6 of 44 | per second |
| Vertical gaze angle, minimum | −14.5 | −23.2 | 10 of 44 | degrees |
| Horizontal gaze angle, range | 14.9 | 21.8 | 33 of 44 | degrees |
| Horizontal gaze angle, kurtosis | −0.97 | −0.51 | 32 of 44 | – |
| Pupil diameter, mean | 3.53 | 3.73 | 37 of 44 | mm |
## CLASSIFICATION RESULTS
Both models separated low from high load in participants not used in training (Table 11, Figure 6). Both reached a pooled ROC-AUC of 0.862, and both accuracies were well above the majority-class baseline (0.500). Logistic Regression had the higher accuracy (0.830 against 0.739), precision and F1-score; it found more of the low-load recordings (recall 0.932) than the high-load ones (0.727). The permutation test placed both pipelines above chance (p = 0.005, the smallest possible; default-setting ROC-AUC 0.838 and 0.830).

**Table 11.** Classification results under leave-one-participant-out validation (88 samples).

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC (95% CI) | 0.862 (0.794–0.933) | 0.862 (0.795–0.919) |
| Accuracy | 0.830 | 0.739 |
| Precision, macro (low / high) | 0.844 (0.774 / 0.914) | 0.740 (0.723 / 0.756) |
| Recall, low / high | 0.932 / 0.727 | 0.773 / 0.705 |
| F1-score, macro (low / high) | 0.828 (0.845 / 0.810) | 0.738 (0.747 / 0.729) |
| Brier score | 0.172 | 0.167 |
| A4 ranked above A1 | 42 of 44 | 40 of 44 |
| Permutation test p (default settings, N = 20) | 0.005 | 0.005 |
| Features selected (median N) | 5 | 10 |

![ROC curves of the pooled held-out predictions.](outputs/final/paper_figures/fig6_roc.png)

Logistic Regression chose N = 5 in 42 of 44 folds and a strong penalty (median C = 0.07, L1 ratio 0.82). XGBoost chose N between 5 and 140 (5 in 17 folds, 80 in 16) and mostly single-split trees (median depth 1, 141 trees).
## MODEL COMPARISON
The pooled ROC-AUC difference between the models was 0.000 (95% CI −0.054 to 0.048), so no difference in ranking was detected. The per-participant margins were wider for XGBoost (median 0.40 against 0.20; wider for 36 of 44 participants; Wilcoxon p < 0.001). Margins depend on each model's probability scale, and the strongly regularized Logistic Regression keeps its probabilities closer to 0.5, so this shows that XGBoost was more confident, not that it separated the conditions better. At the 0.5 threshold, Logistic Regression classified more recordings correctly (73 of 88 against 65 of 88). No equivalence margin was set, so the results do not show that the two models are equal; the interval does, however, make a difference larger than about 0.05 unlikely.
## NUMBER OF FEATURES
With N fixed at each candidate value, ROC-AUC stayed between 0.84 and 0.88 from 5 to 140 features, and every confidence interval overlapped (Table 12, Figure 7). Logistic Regression gave almost the same results from N = 40 upward (ROC-AUC 0.876). XGBoost scored highest at N = 5 (0.884), but this value was chosen after seeing the test results and is not the main result. A few features carried almost all of the information: the 17 statistics of one signal are often near-copies (160 feature pairs had a correlation above 0.9), and the typical feature alone reached a ROC-AUC of only about 0.6.

**Table 12.** Results with the number of features fixed (88 samples).

| Model | N | ROC-AUC (95% CI) | Accuracy | Precision, macro | Recall, low / high | F1, macro | Brier |
|---|---|---|---|---|---|---|---|
| LR | 5 | 0.855 (0.786–0.927) | 0.807 | 0.827 | 0.932 / 0.682 | 0.804 | 0.195 |
| LR | 10 | 0.855 (0.784–0.928) | 0.795 | 0.812 | 0.909 / 0.682 | 0.793 | 0.219 |
| LR | 20 | 0.868 (0.799–0.939) | 0.807 | 0.836 | 0.955 / 0.659 | 0.803 | 0.218 |
| LR | 40 | 0.876 (0.810–0.943) | 0.807 | 0.836 | 0.955 / 0.659 | 0.803 | 0.227 |
| LR | 80 | 0.876 (0.810–0.943) | 0.807 | 0.836 | 0.955 / 0.659 | 0.803 | 0.226 |
| LR | 140 | 0.876 (0.810–0.943) | 0.807 | 0.836 | 0.955 / 0.659 | 0.803 | 0.226 |
| XGBoost | 5 | 0.884 (0.821–0.940) | 0.818 | 0.819 | 0.795 / 0.841 | 0.818 | 0.137 |
| XGBoost | 10 | 0.861 (0.798–0.921) | 0.761 | 0.761 | 0.750 / 0.773 | 0.761 | 0.152 |
| XGBoost | 20 | 0.849 (0.785–0.916) | 0.773 | 0.773 | 0.795 / 0.750 | 0.773 | 0.157 |
| XGBoost | 40 | 0.864 (0.796–0.928) | 0.750 | 0.752 | 0.795 / 0.705 | 0.749 | 0.152 |
| XGBoost | 80 | 0.853 (0.773–0.928) | 0.807 | 0.808 | 0.841 / 0.773 | 0.807 | 0.151 |
| XGBoost | 140 | 0.838 (0.759–0.917) | 0.773 | 0.773 | 0.795 / 0.750 | 0.773 | 0.160 |

![ROC-AUC with the number of features fixed.](outputs/final/paper_figures/fig7_by_n.png)
## BLINK RATE ALONE AND WITHOUT BLINK RATE
Blink rate alone matched the full models in ROC-AUC (0.884; Table 13). Without blink rate, both models dropped but stayed above chance (0.727 and 0.792, p = 0.005), so the other eye features also carry information.

**Table 13.** Blink rate alone and both models without blink rate (88 samples).

| Metric | Blink rate alone (LR) | LR without blink rate | XGBoost without blink rate |
|---|---|---|---|
| ROC-AUC (95% CI) | 0.884 (0.822–0.942) | 0.727 (0.629–0.825) | 0.792 (0.714–0.865) |
| Accuracy | 0.807 | 0.716 | 0.727 |
| Precision, macro (low / high) | 0.815 (0.765 / 0.865) | 0.717 (0.702 / 0.732) | 0.728 (0.717 / 0.738) |
| Recall, low / high | 0.886 / 0.727 | 0.750 / 0.682 | 0.750 / 0.705 |
| F1-score, macro (low / high) | 0.806 (0.821 / 0.790) | 0.716 (0.725 / 0.706) | 0.727 (0.733 / 0.721) |
| Brier score | 0.147 | 0.227 | 0.197 |
| A4 ranked above A1 | 42 of 44 | 35 of 44 | 39 of 44 |
| Permutation test p | 0.005 | 0.005 | 0.005 |
| Features selected (median N) | 1 | 60 | 80 |
## FEATURE IMPORTANCE
Five features were selected in all 44 folds by both models: blink rate, saccade rate, the minimum vertical gaze angle, and the range and kurtosis of the horizontal gaze angle. By SHAP (Figure 8), blink rate and the minimum vertical gaze angle accounted for 87% of Logistic Regression's attributions and 52% of XGBoost's. Logistic Regression used only these five features: it kept five in 42 folds, and in the other two its penalty set the rest to zero. XGBoost spread its attributions more widely: the top five gave 72.5% and 121 features received some share. Pupil features contributed little (at most 2.4%, pupil range in XGBoost). Three of the five leading features are gaze-angle statistics (minimum, range and kurtosis) and carry the caveat stated in Chapter 3.

![Share of mean absolute SHAP value per feature (top 10).](outputs/final/paper_figures/fig8_shap.png)
## DISCUSSION
Both models reached a ROC-AUC of 0.862 on participants not used in training. This is similar to other participant-independent eye-tracking studies, such as Božak et al. (0.848 with leave-one-subject-out validation) {16} and Nerella et al. (0.76 for Logistic Regression and 0.85 for XGBoost) {12}. The COLET authors reported an accuracy of 0.85 for Logistic Regression on the same contrast {18}; their split was not participant-independent and their features were selected on all data, so the results are not directly comparable.

XGBoost did not outperform Logistic Regression: their ROC-AUC was identical, and Logistic Regression was more accurate at the 0.5 threshold. This agrees with Shao et al., who found no significant difference with eye-only features {13}, and with the clinical-prediction review {26}, and differs from Rolon-Merette et al. {11} and Nerella et al. {12}, where XGBoost was ahead. With 88 samples and one dominant feature, the extra flexibility of boosted trees had little to exploit; XGBoost chose single-split trees.

The strongest signal was blink rate, which alone matched the full models. Blink rate rose from 1.7 to 11.7 per minute between A1 and A4, but it followed speaking rather than time pressure: the median was 0.0 in A2 (silent, time pressure) and 11.7 in A3 (speaking, no time pressure). Because speaking raises blinking {23}, part of the result may reflect talking rather than mental load. The gaze-location features add a second caveat: the minimum vertical gaze angle was about 9° lower in A4, which may come from the stimuli or the headset rather than load. Without blink rate the models still reached 0.73–0.79, so the other eye features carry information, but the data cannot separate load from speech.

Saccade rate fell under high load, whereas the COLET authors reported an increase {18}. Such differences may come from the gaze-to-angle conversion, event detection and speaking during A4.

# CHAPTER 5|CONCLUSION AND RECOMMENDATION
## SUMMARY OF FINDINGS
This study compared Logistic Regression and XGBoost for detecting cognitive load from eye-tracking features on COLET, under leave-one-participant-out validation. The main findings are:

1.	Each cleaned eye signal was summarized with 17 statistics, giving 173 features; 33 were removed as copies or unusable, leaving 140 features for 44 participants and 88 recordings. The event-detection check passed in every activity.

2.	Both models reached a ROC-AUC of 0.862 (95% CI 0.794–0.933 for Logistic Regression and 0.795–0.919 for XGBoost). Logistic Regression had an accuracy of 0.830, precision 0.844 and F1-score 0.828; XGBoost had 0.739, 0.740 and 0.738. Both were above chance (p = 0.005) and the majority-class baseline (0.500).

3.	No difference in ROC-AUC was detected between the models (difference 0.000, 95% CI −0.054 to 0.048). XGBoost gave more confident probabilities, but Logistic Regression classified more recordings correctly at the 0.5 threshold.

4.	Performance barely changed with the number of features: ROC-AUC stayed between 0.84 and 0.88 from 5 to 140 features. Logistic Regression selected five features in 42 of 44 folds.

5.	Both models relied mainly on blink rate and the minimum vertical gaze angle (87% and 52% of SHAP attributions), and five features were selected in every fold. Blink rate alone matched the full models (ROC-AUC 0.884); without it, the models reached 0.727 and 0.792.
## CONCLUSIONS
Eye-tracking features can detect COLET's high-load task condition in people not used in training, at a level similar to other participant-independent studies. XGBoost did not outperform the simpler Logistic Regression, which reached the same ROC-AUC with five features and higher accuracy at the 0.5 threshold. A handful of features carried almost all of the information, led by blink rate. Because blink rate followed speaking, the results show that eye features detect a dual-task condition that includes speech, not cognitive load separated from speech.
## LIMITATIONS
•	The high-load condition includes speaking, which raises the blink rate {23}; blink rate was 0.0 per minute in the silent A2 and 11.7 in the spoken A3, so the results mix mental load with speech.

•	Three of the five leading features are gaze-angle statistics, which may reflect where the stimuli appeared or how the headset sat.

•	The sample is small (44 participants, 88 samples) and comes from a laboratory task with a chin rest, which limits generalization.

•	No equivalence margin was set, so the comparison shows only that no difference was detected.

•	The 45°/s velocity threshold was adopted from COLET and not validated for this tracker, and blinks were taken from the tracker's own detector.

•	No stimulus-luminance correction was applied, and the tracker refitted its pupil model more often in the counting activities.

•	The three excluded recordings came from counting activities, and two participant IDs (P28 and P31) hold the same recordings, whose owner cannot be determined.

•	Only the two extreme activities were classified, so finer load levels were not tested.
## RECOMMENDATIONS
For future research, the following are recommended:

1.	Repeat the comparison on a dataset whose high-load condition does not involve speaking, to separate mental load from speech.

2.	Classify A1 versus A2, the only COLET contrast without a difference in speaking, to see how much signal remains without speech.

3.	Test whether the gaze-location features reflect load or the stimulus layout, for example by comparing them with where the puzzle targets appeared.

4.	Extend the work to more load levels and to shorter time windows for real-time detection.

For practitioners, blink rate should not be used as a workload index on its own: speaking raises it {23}, and visual tasks can lower it {36}.

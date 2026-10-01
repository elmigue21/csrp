# Comparative Analysis of Logistic Regression and XGBoost for Eye-Tracking-Based Cognitive Load Detection with Feature Importance Analysis

> Draft status: technical first draft. University formatting, adviser-approved
> research questions, and final IEEE reference details still need review.

## Abstract

Cognitive load detection estimates a user's mental workload from observable
signals. Eye-tracking features are one non-invasive source of behavioral
evidence, but workload ratings and eye-movement patterns vary strongly between
participants. This study compares Logistic Regression and XGBoost for binary
cognitive load detection using eye-tracking features from the GAZELOAD dataset,
which contains 26 participants, five tasks per participant, and 130 complete
recordings. The published 250 ms windows were aggregated into 30-second epochs,
producing 1,087 epochs and 37 derived features. Two label schemes, absolute
thresholding and within-participant thresholding, were crossed with two feature
schemes, raw and within-participant standardized features, and evaluated under
leave-one-participant-out validation. Performance was measured with ROC-AUC,
accuracy, balanced accuracy, precision, recall, and F1-score. Feature importance
was examined through standardized Logistic Regression coefficients and XGBoost
feature attribution methods. XGBoost reached a higher ROC-AUC than Logistic
Regression in every configuration tested, although no configuration reached
significance in the participant-level paired ROC-AUC tests. The best analytical
configuration combined within-participant labeling with within-participant
feature standardization, where XGBoost reached a pooled ROC-AUC of 0.682 and
Logistic Regression reached 0.621. Preprocessing decisions, particularly
participant-relative labeling and normalization, mattered more than the choice
of model. XGBoost is therefore the stronger classifier on this dataset, but the
evidence should be read with caution given participant variability, lighting
influence, and the analytical nature of the participant-relative pipeline.

Keywords: cognitive load, eye tracking, Logistic Regression, XGBoost, feature
importance, ROC-AUC, leave-one-participant-out validation

## Chapter 1: Introduction

### 1.1 Background of the Study

Cognitive load is the amount of mental effort a task demands. In
human-computer interaction, human-robot collaboration, education, driving, and
safety-critical work, an estimate of cognitive load lets a system adapt to the
user's current state. When a user is overloaded, performance can decline, errors
can increase, and situational awareness can weaken [3]. Automatic cognitive load
detection is studied for that reason in affective computing, human factors, and
intelligent systems research.

Eye movements change with attention, visual search, task difficulty, and mental
effort, which makes eye-tracking one way to estimate cognitive load. Fixation
behavior, saccade behavior, blink activity, gaze dispersion, and gaze transition
measures can all be extracted from a recording, and a wearable device can
capture them without interrupting the task. Eye-tracking data are also noisy and
participant-dependent. Two people may have different baseline gaze patterns
while performing the same task.

Machine learning can classify cognitive load from these features. This study
uses two models. Logistic Regression is a linear classifier that works as an
interpretable baseline. XGBoost is a gradient-boosted tree model that can
capture nonlinear relationships and feature interactions [6]. Comparing the two
shows whether a more complex nonlinear method buys any real performance over a
simpler interpretable one.

The data come from the GAZELOAD dataset [8], which contains eye-tracking
recordings from 26 participants performing five task conditions. The work is
analytical rather than deployment-ready. Its goal is to compare model behavior
and feature contribution under a strict participant-level validation protocol.

### 1.2 Statement of the Problem

Eye-tracking features can be related to cognitive load, but it is unclear
whether a simple linear classifier or a nonlinear boosted-tree classifier
detects that load more effectively in participant-independent evaluation.
Participant-specific rating behavior and participant-specific eye-tracking
baselines complicate the question. A fixed workload threshold may not represent
the same mental state for every participant, and raw eye-tracking features may
carry stable individual differences unrelated to workload.

This study compares Logistic Regression and XGBoost under identical
preprocessing, labeling, and validation conditions. It also asks whether
participant-relative preprocessing improves classification, and which features
contribute most to the predictions.

### 1.3 Objectives of the Study

The general objective of this study is to compare Logistic Regression and
XGBoost for cognitive load detection using eye-tracking features.

The specific objectives are:

1. To prepare eye-tracking features from the GAZELOAD dataset for binary
   cognitive load classification.
2. To evaluate Logistic Regression and XGBoost under leave-one-participant-out
   validation.
3. To compare model performance using ROC-AUC, accuracy, balanced accuracy,
   precision, recall, and F1-score.
4. To examine the effect of label definition, feature standardization, and
   lighting-feature inclusion through experimental configurations.
5. To analyze feature importance using Logistic Regression coefficients and
   XGBoost attribution methods.

### 1.4 Research Questions

This study is guided by the following research questions:

1. How does Logistic Regression perform compared with XGBoost for binary
   cognitive load detection using eye-tracking features?
2. Which experimental configuration produces the strongest classification
   performance under leave-one-participant-out validation?
3. How do within-participant labeling and within-participant feature
   standardization affect model performance?
4. Which eye-tracking and contextual features contribute most to the predictions
   of Logistic Regression and XGBoost?
5. Is the observed difference between Logistic Regression and XGBoost consistent
   across participants?

### 1.5 Scope and Limitations

The study is limited to the GAZELOAD dataset, which contains 26 participants and
five task recordings per participant, for a total of 130 complete recordings.

Only the features available in the released eye-metrics files are used. Pupil
diameter and pupil dilation are excluded because the released dataset used by the
codebase does not contain them. The analysis therefore rests on fixation,
saccade, blink, gaze-direction, gaze-position, gaze-transition, dispersion, and
illuminance-related features.

The main analytical configuration uses within-participant labeling and
within-participant feature standardization. That setup suits an analysis of
relative workload patterns, but it does not amount to a deployment-ready model.
A deployed system would need calibration or prior recordings from a new user
before it could compute participant-specific feature statistics.

### 1.6 Significance of the Study

The study offers a controlled comparison of an interpretable linear classifier
and a nonlinear ensemble classifier for eye-tracking-based cognitive load
detection. It also shows how preprocessing choices move model performance, which
matters because weak validation or participant-insensitive labeling can produce
misleading conclusions. The feature importance analysis adds a further
contribution by identifying which features the models actually use when
detecting cognitive load.

## Chapter 2: Review of Related Literature

### 2.1 Cognitive Load and Mental Workload

Cognitive load describes the mental demand placed on a person while performing a
task. Foundational work on attention and effort treats mental effort as a
limited cognitive resource [1], while cognitive load theory explains how
excessive demands on working memory can impair learning and task performance
[2]. In applied settings, mental workload can be measured through self-report,
task performance, physiological signals, or behavioral signals. Instruments such
as NASA-TLX show that subjective workload assessment is an established method in
human factors research [16].

This study does not use NASA-TLX as its target variable. It uses the GAZELOAD
dataset's self-reported mental load rating on a 1-10 scale [8]. That label is
still subjective, so participants may interpret the scale differently.
Differences in rating behavior matter here because they affect how binary
high-load and low-load labels are constructed [9].

Some studies handle this variation with participant-relative or subject-wise
transformations. These approaches do not assume that the same numeric rating
means the same thing to every participant. They ask instead whether a task sits
higher or lower relative to that participant's own experience, which supports
within-participant labeling in analytical workload studies where the research
question concerns relative cognitive load rather than an absolute clinical or
operational threshold.

### 2.2 Eye-Tracking Features for Cognitive Load Detection

Eye-tracking appears in cognitive load research because visual behavior is
linked to attention and task demand. Common features include fixation count,
fixation duration, saccade count, saccade amplitude, saccade velocity, blink
rate, gaze dispersion, and gaze transition patterns [4], [12], [13], [18].
Researchers extract them over time windows and feed them to machine-learning
models.

Eye-tracking features are also sensitive to individual differences, recording
conditions, lighting, task context, and data quality, which is why
participant-level evaluation matters. If data from one participant appear in
both the training and test sets, a classifier may learn participant identity
rather than general workload-related behavior. Subject-independent or
participant-held-out evaluation is the more appropriate design when the goal is
to estimate performance on unseen users [10], [13].

Related public datasets and studies, including COLET and other eye-tracking
workload classification work, support the broader feasibility of detecting
cognitive workload from eye-movement features [17], [18]. They serve here as
related literature only. The present methodology uses GAZELOAD.

### 2.3 Logistic Regression for Classification

Logistic Regression is a supervised learning algorithm for binary
classification. It models the probability of class membership as a logistic
function of a weighted linear combination of features. Its main advantage is
interpretability: the coefficients show the direction and relative contribution
of standardized features, which is what makes the model a natural baseline in
comparative studies [5].

Its limitation is the assumption of a linear decision boundary, unless
interactions or nonlinear transformations are added explicitly. If cognitive
load is associated with nonlinear changes in eye behavior, Logistic Regression
will underfit the relationship.

### 2.4 XGBoost for Classification

XGBoost is a gradient-boosted decision-tree algorithm. It builds an ensemble of
trees in sequence, each tree correcting the errors of the ones before it. The
model can represent nonlinear relationships, feature interactions, and threshold
effects, and it handles missing values through learned default directions in
tree splits [6].

Those properties suit eye-tracking data, where the relationship between features
and workload may well be nonlinear. XGBoost is also less directly interpretable
than Logistic Regression, so feature importance methods are needed to explain
which features drive its predictions [7], [12], [19].

### 2.5 Within-Participant Labeling

Within-participant labeling defines high and low workload relative to each
participant's own rating distribution. In this study, a task is labeled high
load if its rating sits above the participant's median rating and low load if it
falls below the median. Ratings equal to the median are dropped because they are
ambiguous on that participant's own scale.

The approach is defensible because subjective workload ratings are not always
directly comparable across participants. Related mental-effort classification
work has used subject-wise median split procedures to separate low and high
perceived effort [11]. In this dataset, the within-participant label also agrees
more closely with objective task difficulty than the absolute threshold does,
based on the existing project results.

### 2.6 Within-Participant Feature Standardization

Within-participant feature standardization turns each feature into a
participant-relative value. A standardized value says how far a participant's
current feature value sits from their own mean, measured in units of their own
standard deviation, which removes between-participant baseline differences and
leaves the within-person deviations.

That matters for eye-tracking-based workload detection because participants
differ in baseline gaze behavior. Subject-wise normalization has been used in
workload-related modeling to reduce inter-subject variability and support
participant-held-out evaluation [10]. The transformation used here is
label-free: it reads only eye-tracking feature values, never workload labels.

### 2.7 Feature Importance Analysis

For Logistic Regression, standardized coefficients show whether a feature
raises or lowers the predicted probability of high cognitive load. For XGBoost,
importance can be read from tree-based gain and from SHAP values. SHAP values
are useful because they estimate each feature's contribution to individual
predictions and can then be summarized across samples [7].

Feature importance carries extra weight in this study because performance
figures alone do not reveal whether the classifier is using eye behavior,
lighting, artifacts, or some other contextual signal. Prior eye-tracking
workload studies have likewise used feature-importance analysis to interpret the
contribution of gaze-derived variables [12], [19].

### 2.8 Classification Metrics

ROC-AUC measures a classifier's ability to rank positive samples above negative
samples across thresholds, which makes it useful for comparing model
discrimination without depending on one fixed decision threshold [14]. Accuracy,
balanced accuracy, precision, recall, and F1-score give complementary views of
thresholded classification performance. Because this thesis compares model
families, ROC-AUC is the primary metric and the others are reported as secondary
measures.

## Chapter 3: Methodology

### 3.1 Research Design

This study uses a quantitative computational experimental design. Logistic
Regression and XGBoost are trained on identical eye-tracking feature sets and
evaluated under the same participant-level validation protocol. The comparison
runs across multiple preprocessing configurations, and feature importance is
analyzed for interpretability.

### 3.2 Dataset

The study uses GAZELOAD, an eye-tracking dataset collected during industrial
human-robot collaboration tasks [8]. It contains 26 participants with five task
recordings each, a complete grid of 130 recordings. The released sampling unit
is a non-overlapping 250 ms window, and the codebase reports 135,499 total
windows.

Each recording carries one self-reported mental load rating on a 1 to 10 scale,
which becomes the basis for the binary low-load and high-load labels.

### 3.3 Data Preparation

Participant identity is taken from the recording filename rather than the
`Participant_ID` column, because the project documentation notes that the
`Participant_ID` field is corrupted in some files. Task identifiers and
timestamps are dropped from the final feature matrix. Task identity in
particular can encode the experimental condition directly instead of the
participant's eye behavior.

The original 250 ms windows are aggregated into 30-second epochs. Epoch
boundaries are computed within each recording, and partial epochs covering less
than 80% of the window are removed, leaving 1,087 epochs. Prior eye-tracking
workload research has shown that aggregation choices can influence
classification performance, which supports the use of epoch-level feature
summaries rather than raw short windows alone [13].

For each epoch, the mean and standard deviation of the base eye-tracking
features are computed. Additional derived features include blink rate, the
fraction of missing saccade features, the fraction of missing fixation
dispersion index values, the fraction of nonzero gaze transition entropy values,
and the fraction of windows with zero fixations. With lighting features
included, the final feature set contains 37 features.

### 3.4 Labeling Schemes

Two binary labeling schemes are evaluated.

The first is an absolute threshold. Ratings of 4 or above are labeled high load
and ratings below 4 are labeled low load, which retains all 1,087 epochs.

The second is a within-participant threshold. Ratings above a participant's own
median are labeled high load, ratings below the median are labeled low load, and
ratings equal to the median are excluded, leaving 768 epochs. The scheme exists
because subjective workload ratings differ between participants.

### 3.5 Feature Schemes

Two feature schemes are evaluated.

The raw feature scheme uses the epoch-level feature values directly.

The within-participant standardization scheme transforms each feature within
each participant, expressing every value as a deviation from that participant's
mean divided by that participant's standard deviation. The result emphasizes
participant-relative changes in eye behavior.

### 3.6 Experimental Configurations

Eight configurations are evaluated:

| Config | Label Scheme | Feature Scheme | Lighting Features |
|---|---|---|---|
| A | Absolute threshold | Raw | Included |
| B | Absolute threshold | Raw | Removed |
| C | Absolute threshold | Within-participant standardized | Included |
| C_no_lux | Absolute threshold | Within-participant standardized | Removed |
| D | Within-participant threshold | Raw | Included |
| D_no_lux | Within-participant threshold | Raw | Removed |
| E | Within-participant threshold | Within-participant standardized | Included |
| F | Within-participant threshold | Within-participant standardized | Removed |

Config E is the main analytical pipeline because it combines the
participant-relative label with the participant-relative feature
representation. The rest serve as ablation comparisons.

### 3.7 Models

Two models are compared.

Logistic Regression runs as a pipeline with median imputation, missing-value
indicators, feature scaling, and Logistic Regression classification. Median
imputation is necessary because Logistic Regression cannot represent missing
values directly.

XGBoost runs as a binary logistic classifier with histogram-based tree
construction. Missing values are passed straight to the model, which learns
default directions for missing-value splits.

Hyperparameters are tuned by grid search inside the training folds only.

### 3.8 Evaluation Protocol

The study uses leave-one-participant-out validation. In each outer fold, every
epoch from one participant is held out for testing while the remaining 25
participants supply the training data, giving 26 outer folds. Hyperparameter
tuning uses stratified group cross-validation inside the training participants
only. The design follows the subject-independent evaluation logic of workload
classification research, where performance is estimated on participants the
model never saw during training [10], [13].

No participant can appear in both training and testing within the same fold.
That is stricter than random row-level splitting and gives a better picture of
performance on unseen participants.

### 3.9 Metrics and Statistical Comparison

The primary metric is pooled ROC-AUC across all held-out predictions. Secondary
metrics are accuracy, balanced accuracy, precision, recall, and F1-score.
Per-participant metrics are reported as well, to show how much performance
varies from person to person. ROC-AUC leads because it evaluates discrimination
across thresholds instead of depending on a single operating point [14].

The two models are compared with the paired Wilcoxon signed-rank test over
participant-level scores wherever the metric is defined. That choice follows
machine-learning guidance recommending nonparametric paired tests such as the
Wilcoxon signed-rank test when comparing classifiers over matched evaluation
units [15].

### 3.10 Feature Importance Methods

For Logistic Regression, feature importance is analyzed through standardized
coefficients across folds. For XGBoost, it is analyzed through gain importance
and out-of-fold SHAP values. Out-of-fold explanations keep the feature
attribution tied to held-out participant predictions.

## Chapter 4: Results and Discussion

### 4.1 Dataset Summary

The dataset contains 130 complete recordings from 26 participants, five tasks
each. The codebase reports 135,499 original 250 ms windows and 1,087 retained
30-second epochs. None of the 130 recordings showed timing discontinuities, and
each contained exactly one workload rating.

Only one participant, P12, had a rating range below three points, so most
participants varied their ratings across tasks even though participant-specific
rating behavior still shaped the labels.

### 4.2 Label Validity

The within-participant label agreed more closely with objective task difficulty
than the absolute threshold. The absolute threshold reached a Spearman rho of
0.534 with task index, against 0.735 for the within-participant label. The
within-participant label is therefore a reasonable analytical representation of
relative workload.

### 4.3 Overall Model Performance

XGBoost reached a higher pooled ROC-AUC than Logistic Regression in every
configuration. Under Config E, the main analytical pipeline, Logistic Regression
reached 0.621 and XGBoost reached 0.682. XGBoost was also more accurate under
Config E, at 0.629 against 0.570.

| Config | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| A | Logistic Regression | 0.488 | 0.515 | 0.561 | 0.537 | 0.479 |
| A | XGBoost | 0.526 | 0.550 | 0.580 | 0.565 | 0.530 |
| B | Logistic Regression | 0.467 | 0.498 | 0.580 | 0.536 | 0.446 |
| B | XGBoost | 0.510 | 0.535 | 0.564 | 0.549 | 0.501 |
| C | Logistic Regression | 0.519 | 0.539 | 0.635 | 0.583 | 0.514 |
| C | XGBoost | 0.534 | 0.558 | 0.585 | 0.571 | 0.550 |
| C_no_lux | Logistic Regression | 0.506 | 0.527 | 0.667 | 0.589 | 0.499 |
| C_no_lux | XGBoost | 0.522 | 0.544 | 0.613 | 0.576 | 0.528 |
| D | Logistic Regression | 0.535 | 0.559 | 0.602 | 0.580 | 0.565 |
| D | XGBoost | 0.579 | 0.604 | 0.611 | 0.608 | 0.628 |
| D_no_lux | Logistic Regression | 0.506 | 0.532 | 0.616 | 0.571 | 0.488 |
| D_no_lux | XGBoost | 0.527 | 0.550 | 0.614 | 0.580 | 0.521 |
| E | Logistic Regression | 0.570 | 0.586 | 0.660 | 0.621 | 0.621 |
| E | XGBoost | 0.629 | 0.644 | 0.677 | 0.660 | 0.682 |
| F | Logistic Regression | 0.569 | 0.577 | 0.716 | 0.639 | 0.575 |
| F | XGBoost | 0.565 | 0.586 | 0.626 | 0.605 | 0.603 |

XGBoost stays ahead on ROC-AUC throughout, but the margin is modest. Changing
the label and feature schemes moved performance further than changing the model
did.

### 4.4 Effect of Labeling and Standardization

Switching from the absolute threshold to the within-participant label improved
both models. Logistic Regression rose from 0.479 ROC-AUC in Config A to 0.565 in
Config D, and XGBoost from 0.530 to 0.628.

Combining within-participant labeling with within-participant feature
standardization produced the strongest performance of all. Config E lifted
Logistic Regression to 0.621 ROC-AUC and XGBoost to 0.682, which places
participant-relative preprocessing at the center of what worked on this dataset.

The full factorial design also isolates the effect of removing lighting
features, which lowered ROC-AUC in every matched comparison. The largest drops
came under the within-participant raw-feature condition: moving from Config D to
Config D_no_lux cost 0.078 ROC-AUC for Logistic Regression and 0.107 for
XGBoost. Under the main analytical pipeline, moving from Config E to Config F
cost 0.047 for Logistic Regression and 0.080 for XGBoost.

### 4.5 Logistic Regression vs XGBoost Statistical Comparison

The Wilcoxon signed-rank tests put XGBoost consistently higher in pooled
ROC-AUC, but the participant-level evidence was not always statistically
significant.

| Config | Mean ROC-AUC Difference, XGB - LR | p-value |
|---|---:|---:|
| A | 0.042 | 0.223 |
| B | 0.035 | 0.360 |
| C | 0.051 | 0.075 |
| C_no_lux | -0.005 | 0.777 |
| D | 0.032 | 0.465 |
| D_no_lux | 0.002 | 0.808 |
| E | 0.044 | 0.181 |
| F | 0.008 | 0.877 |

No configuration reached p < 0.05 for the participant-level ROC-AUC difference.
The defensible reading is that XGBoost achieved a higher pooled ROC-AUC in every
configuration while the participant-level evidence remains too weak to claim
statistically significant superiority in ROC-AUC. Config E did show a
statistically significant XGBoost advantage in accuracy (p = 0.031), which on
its own does not establish general model superiority across configurations.

### 4.6 Feature Importance Findings

Illuminance features were highly influential in both models. Under Config E, the
XGBoost SHAP analysis ranked illuminance standard deviation first and
illuminance mean second, and the Logistic Regression coefficients for the same
configuration put illuminance standard deviation at the top by absolute
magnitude.

Illuminance is an environmental feature rather than a direct eye-movement
feature, so part of the predictive signal may be coming from task context or
lighting conditions. Config F, which removes the lighting features, is therefore
the stricter eye-tracking-only comparison. There Logistic Regression reached
0.575 ROC-AUC and XGBoost reached 0.603.

### 4.7 Discussion

XGBoost generally outperformed Logistic Regression for eye-tracking-based
cognitive load detection, which answers the main research question. The
improvement was modest, though, and not consistently significant across
participants. XGBoost is the choice for predictive performance, while Logistic
Regression remains a useful baseline for being simpler and more interpretable.

Preprocessing choices proved decisive. The strongest performance appeared only
after both the cognitive load labels and the features had been expressed
relative to each participant, which fits the view that individual differences
weigh heavily on eye-tracking-based cognitive load detection.

The feature importance findings show why interpretation cannot be skipped. When
the highest-ranked features are lighting-related, the model may be reading
environmental context alongside eye behavior. That does not invalidate the
model, but it changes how the result should be described. Config E is the
strongest analytical pipeline, and Config F is the stricter eye-tracking-only
version of it.

## Chapter 5: Conclusion and Recommendations

### 5.1 Summary of Findings

This study compared Logistic Regression and XGBoost for cognitive load detection
using eye-tracking features from the GAZELOAD dataset, evaluating both under
leave-one-participant-out validation across the completed experimental
configurations.

The main findings are:

1. XGBoost achieved higher pooled ROC-AUC than Logistic Regression in all eight
   configurations.
2. Config E, which used within-participant labeling and within-participant
   feature standardization with lighting features included, produced the best
   performance.
3. Under Config E, Logistic Regression achieved 0.621 pooled ROC-AUC, while
   XGBoost achieved 0.682 pooled ROC-AUC.
4. The participant-level statistical tests showed that XGBoost's advantage was
   consistent in pooled ROC-AUC but not statistically significant in
   participant-level ROC-AUC.
5. Participant-relative preprocessing had a larger effect than the model choice.
6. Feature importance analysis showed that illuminance features contributed
   strongly, so lighting influence must be discussed as a limitation.

### 5.2 Conclusions

On pooled ROC-AUC, XGBoost is the stronger model for this dataset. Its capacity
for nonlinear relationships and feature interactions is the likely reason it
outperformed Logistic Regression. The gap at the participant level was modest,
however, and Logistic Regression held its value as an interpretable baseline.

Participant-relative preprocessing matters for cognitive load detection from
eye-tracking features. Within-participant labeling and within-participant
feature standardization both improved performance, which points to individual
differences in workload ratings and eye behavior as something a study of this
kind has to account for.

Feature importance analysis is also necessary for interpreting cognitive load
classifiers. The strong role of illuminance features shows that high performance
may partly reflect environmental or task-context signals rather than eye
movement alone.

### 5.3 Recommendations

Future studies should test these findings against larger participant samples and
additional task conditions. Artifact filtering deserves attention as well,
especially for implausible saccade and gaze-vector values. Participant 07, whom
the project documentation identifies as having poor gaze validity, should be
handled in a pre-registered data quality analysis.

Further experiments should compare epoch durations of 10, 30, and 60 seconds to
see whether the temporal aggregation window affects performance. Epoch duration
belongs to a separate experimental dimension, though, and should not be folded
into the same Config E result.

Deployment-oriented work will need an explicit calibration procedure for
computing participant-specific feature baselines. The model in this thesis is
analytical, so within-participant standardization is acceptable as long as the
limitation is stated clearly.

### 5.4 Final Statement

XGBoost gives better pooled predictive performance than Logistic Regression for
eye-tracking-based cognitive load detection on the GAZELOAD dataset, especially
under participant-relative preprocessing. Model choice is only one part of the
problem, though. Label definition, feature standardization, lighting-feature
inclusion, validation design, and feature interpretation carry equal weight in
producing defensible conclusions.

## Preliminary IEEE-Style References

> These entries need final formatting against the exact sources used by the
> adviser-approved RRL. Entries marked "metadata to verify" need final lookup
> before submission.

[1] D. Kahneman, Attention and Effort. Englewood Cliffs, NJ, USA:
Prentice-Hall, 1973.

[2] F. G. W. C. Paas and J. J. G. van Merrienboer, "Cognitive-load theory:
Methods to manage working memory load in the learning of complex tasks,"
Current Directions in Psychological Science, vol. 29, no. 4, pp. 394-398, 2020.

[3] R. Parasuraman and D. H. Manzey, "Complacency and bias in human use of
automation," Human Factors, vol. 52, no. 3, pp. 381-410, 2010.

[4] A. T. Duchowski, Eye Tracking Methodology: Theory and Practice, 3rd ed.
Cham, Switzerland: Springer, 2017, doi: 10.1007/978-3-319-57883-5.

[5] C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA:
Springer, 2006.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in
Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016,
pp. 785-794.

[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model
predictions," in Advances in Neural Information Processing Systems, vol. 30,
2017.

[8] B. Karbouj, B. E. Gaaloul, and J. Kruger, "GAZELOAD: A multimodal
eye-tracking dataset for mental workload in industrial human-robot
collaboration," arXiv preprint, arXiv:2601.21829, Jan. 2026. Metadata to
verify.

[9] A. C. Marinescu, S. Sharples, A. C. Ritchie, T. S. Lopez, A. McDowell, H.
Morvan, and L. M. Naismith, "Exploring the relationship between mental
workload, variation in performance, and physiological parameters," Human
Factors, 2018. Metadata to verify. Available:
https://pubmed.ncbi.nlm.nih.gov/28965433/

[10] I. Albuquerque, J. Monteiro, O. Rosanne, and T. H. Falk, "Estimating distribution shifts for predicting cross-subject generalization in electroencephalography-based mental workload assessment," Frontiers in Artificial Intelligence, vol. 5, Oct. 2022, doi: 10.3389/frai.2022.992732.

[11] S. Gado, K. Lingelbach, M. Wirzberger, and M. Vukelić, "Decoding Mental Effort in a Quasi-Realistic Scenario: A Feasibility Study on Multimodal Data Fusion and Classification," Sensors, vol. 23, no. 14, p. 6546, July 2023, doi: 10.3390/s23146546.

[12] M. Kaczorowska, M. Plechawska-Wojcik, and M. Tokovarov, "Interpretable
machine learning models for three-way classification of cognitive workload
levels for eye-tracking features," Brain Sciences, vol. 11, no. 2, p. 210,
Feb. 2021, doi: 10.3390/brainsci11020210.

[13] M. Kaczorowska, P. Karczmarek, M. Plechawska-Wojcik, and M. Tokovarov,
"On the improvement of eye tracking-based cognitive workload estimation using
aggregation functions," Sensors, vol. 21, no. 13, p. 4542, July 2021,
doi: 10.3390/s21134542.

[14] T. Fawcett, "An introduction to ROC analysis," Pattern Recognition
Letters, vol. 27, no. 8, pp. 861-874, June 2006,
doi: 10.1016/j.patrec.2005.10.010.

[15] J. Demsar, "Statistical comparisons of classifiers over multiple data
sets," Journal of Machine Learning Research, vol. 7, pp. 1-30, 2006.

[16] S. G. Hart and L. E. Staveland, "Development of NASA-TLX (Task Load
Index): Results of empirical and theoretical research," in Human Mental
Workload, P. A. Hancock and N. Meshkati, Eds. Amsterdam, The Netherlands:
North-Holland, 1988, pp. 139-183, doi: 10.1016/S0166-4115(08)62386-9.

[17] E. Ktistakis, V. Skaramagkas, D. Manousos, N. S. Tachos, E. E.
Tripoliti, D. I. Fotiadis, and M. Tsiknakis, "COLET: A dataset for COgnitive
workLoad estimation based on eye-tracking," Computer Methods and Programs in
Biomedicine, vol. 224, p. 106989, 2022, doi: 10.1016/j.cmpb.2022.106989.

[18] L. Rizzo, M. Dondio, and L. Longo, "A machine learning approach for
detecting cognitive interference based on eye-tracking data," Frontiers in
Human Neuroscience, vol. 16, 2022, doi: 10.3389/fnhum.2022.806330. Metadata
to verify.

[19] M. Trigka, E. Dritsas, and P. Mylonas, "Eye-based cognitive overload
prediction in human-machine interaction via machine learning," in Proc. 21st
Int. Conf. Web Information Systems and Technologies (WEBIST), 2025,
pp. 565-572, doi: 10.5220/0013782800003985.

# Comparative Analysis of Logistic Regression and XGBoost for Eye-Tracking-Based Cognitive Load Detection with Feature Importance Analysis

> Draft status: technical first draft. University formatting, adviser-approved
> research questions, and final IEEE reference details still need review.

## Abstract

Cognitive load detection aims to estimate a user's mental workload from
observable signals. Eye-tracking features offer a non-invasive source of
behavioral evidence, but workload ratings and eye-movement patterns vary strongly
between participants. This study compares Logistic Regression and XGBoost for
binary cognitive load detection using eye-tracking features from the GAZELOAD
dataset. The dataset contains 26 participants, five tasks per participant, and
130 complete recordings. Published 250 ms windows were aggregated into 30-second
epochs, producing 1,087 epochs and 37 derived features. Two label schemes,
absolute thresholding and within-participant thresholding, and two feature
schemes, raw and within-participant standardized features, were evaluated under
leave-one-participant-out validation. Model performance was measured using
ROC-AUC, accuracy, balanced accuracy, precision, recall, and F1-score. Feature
importance was examined using standardized Logistic Regression coefficients and
XGBoost feature attribution methods. Results show that XGBoost achieved higher
ROC-AUC than Logistic Regression in all tested configurations, although the
participant-level paired tests showed limited statistical significance. The best
analytical configuration used within-participant labeling and within-participant
feature standardization, where XGBoost reached a pooled ROC-AUC of 0.682 and
Logistic Regression reached 0.621. The findings indicate that preprocessing
decisions, especially participant-relative labeling and normalization, had a
larger effect than the model choice itself. The study concludes that XGBoost is
the stronger classifier in this dataset, but the evidence should be interpreted
with caution because of participant variability, lighting influence, and the
analytical nature of the participant-relative pipeline.

Keywords: cognitive load, eye tracking, Logistic Regression, XGBoost, feature
importance, ROC-AUC, leave-one-participant-out validation

## Chapter 1: Introduction

### 1.1 Background of the Study

Cognitive load refers to the amount of mental effort required to perform a task.
In human-computer interaction, human-robot collaboration, education, driving,
and safety-critical work, estimating cognitive load can help systems adapt to
the user's current state. When a user is overloaded, performance may decline,
errors may increase, and situational awareness may weaken [3]. For this reason,
automatic cognitive load detection has become an important topic in affective
computing, human factors, and intelligent systems research.

Eye-tracking is one possible approach for estimating cognitive load because eye
movements change with attention, visual search, task difficulty, and mental
effort. Features such as fixation behavior, saccade behavior, blink activity,
gaze dispersion, and gaze transition measures can be extracted from eye-tracking
recordings. These features are attractive because they can be captured
non-invasively, especially with wearable devices. However, eye-tracking data are
also noisy and participant-dependent. Two people may have different baseline
gaze patterns even when performing the same task.

Machine learning can be used to classify cognitive load from eye-tracking
features. This study focuses on two models: Logistic Regression and XGBoost.
Logistic Regression is a linear classifier that is useful as an interpretable
baseline. XGBoost is a gradient-boosted tree model that can capture nonlinear
relationships and feature interactions [6]. Comparing these two models allows the
study to examine whether a more complex nonlinear method provides meaningful
performance gains over a simpler interpretable model.

The study uses the GAZELOAD dataset [8], which contains eye-tracking recordings
from 26 participants performing five task conditions. The study is analytical
rather than deployment-ready: the goal is to compare model behavior and feature
contribution under a rigorous participant-level validation protocol.

### 1.2 Statement of the Problem

Although eye-tracking features can be related to cognitive load, it is unclear
whether a simple linear classifier or a nonlinear boosted-tree classifier is more
effective for detecting cognitive load in participant-independent evaluation.
The problem is complicated by participant-specific rating behavior and
participant-specific eye-tracking baselines. A fixed workload threshold may not
represent the same mental state for every participant, and raw eye-tracking
features may contain stable individual differences unrelated to workload.

This study addresses the problem by comparing Logistic Regression and XGBoost
under identical preprocessing, labeling, and validation conditions. It also
examines whether participant-relative preprocessing improves classification and
which features contribute most to model predictions.

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

The study is limited to the GAZELOAD dataset. The dataset contains 26
participants and five task recordings per participant, for a total of 130
complete recordings.

The study uses features available in the released eye-metrics files. Pupil
diameter and pupil dilation are not included because they are not available in
the released dataset used by the codebase. The study therefore focuses on
fixation, saccade, blink, gaze-direction, gaze-position, gaze-transition,
dispersion, and illuminance-related features.

The main analytical configuration uses within-participant labeling and
within-participant feature standardization. This is appropriate for analyzing
relative workload patterns, but it should not be interpreted as a fully
deployment-ready model. A deployed system would require calibration or prior
recordings from a new user to compute participant-specific feature statistics.

### 1.6 Significance of the Study

This study is significant because it provides a controlled comparison of an
interpretable linear classifier and a nonlinear ensemble classifier for
eye-tracking-based cognitive load detection. It also shows how preprocessing
choices affect model performance, which is important because poor validation or
participant-insensitive labeling can lead to misleading conclusions. The feature
importance analysis further contributes by identifying which features the models
use when detecting cognitive load.

## Chapter 2: Review of Related Literature

### 2.1 Cognitive Load and Mental Workload

Cognitive load describes the mental demand placed on a person while performing a
task. Foundational work on attention and effort treats mental effort as a
limited cognitive resource [1], while cognitive load theory explains how
excessive demands on working memory can impair learning and task performance
[2]. In applied settings, mental workload can be measured through self-report,
task performance, physiological signals, or behavioral signals. Established
instruments such as NASA-TLX show that subjective workload assessment is a
recognized method in human factors research [16].

The present study does not use NASA-TLX as its target variable. Instead, it uses
the GAZELOAD dataset's self-reported mental load rating on a 1-10 scale [8].
This label is still subjective, so participant interpretation of the scale may
vary. Such variation is important for this study because differences in rating
behavior can affect how binary high-load and low-load labels are constructed
[9].

Because subjective ratings can vary between people, some studies use
participant-relative or subject-wise transformations. These approaches do not
assume that the same numeric rating has identical meaning for every participant.
Instead, they examine whether a task is higher or lower relative to that
participant's own experience. This supports the use of within-participant
labeling in analytical workload studies when the research question concerns
relative cognitive load rather than an absolute clinical or operational
threshold.

### 2.2 Eye-Tracking Features for Cognitive Load Detection

Eye-tracking has been used in cognitive load research because visual behavior is
linked to attention and task demand. Common features include fixation count,
fixation duration, saccade count, saccade amplitude, saccade velocity, blink
rate, gaze dispersion, and gaze transition patterns [4], [12], [13], [18].
These features can be extracted over time windows and used as inputs to
machine-learning models.

However, eye-tracking features are sensitive to individual differences,
recording conditions, lighting, task context, and data quality. This makes participant-level
evaluation important. If data from the same participant are placed in both
training and test sets, a classifier may learn participant identity rather than
general workload-related behavior. Therefore, subject-independent or
participant-held-out evaluation is more appropriate when the goal is to estimate
performance on unseen users [10], [13].

Related public datasets and studies, such as COLET and other eye-tracking
workload classification work, also support the broader feasibility of detecting
cognitive workload from eye-movement features [17], [18]. These studies are used
as related literature only; the present methodology uses GAZELOAD.

### 2.3 Logistic Regression for Classification

Logistic Regression is a supervised learning algorithm for binary
classification. It models the probability of class membership as a logistic
function of a weighted linear combination of features. Its main advantage is
interpretability: coefficients can be examined to understand the direction and
relative contribution of standardized features. This makes Logistic Regression
appropriate as a baseline model in comparative studies [5].

The limitation of Logistic Regression is that it assumes a linear decision
boundary unless interactions or nonlinear transformations are explicitly added.
If cognitive load is associated with nonlinear changes in eye behavior,
Logistic Regression may underfit the relationship.

### 2.4 XGBoost for Classification

XGBoost is a gradient-boosted decision-tree algorithm. It builds an ensemble of
trees sequentially, with each tree improving on the errors of the previous
trees. XGBoost can model nonlinear relationships, feature interactions, and
threshold effects. It can also handle missing values through learned default
directions in tree splits [6].

These properties make XGBoost suitable for eye-tracking data, where relationships
between features and workload may be nonlinear. However, XGBoost is less directly
interpretable than Logistic Regression, so feature importance methods are needed
to explain which features contribute to predictions [7], [12], [19].

### 2.5 Within-Participant Labeling

Within-participant labeling defines high and low workload relative to each
participant's own rating distribution. In this study, a task is labeled high
load if its rating is above the participant's median rating and low load if it
is below the median. Ratings equal to the median are excluded because they are
ambiguous relative to that participant's own scale.

This approach is defensible because subjective workload ratings are not always
directly comparable across participants. Related mental-effort classification
work has used subject-wise median split procedures when separating low and high
perceived effort [11]. In the current dataset, within-participant labeling also
has stronger agreement with objective task difficulty than the absolute
threshold, based on the existing project results.

### 2.6 Within-Participant Feature Standardization

Within-participant feature standardization transforms each feature into a
participant-relative value. A standardized value indicates how far that
participant's current feature value is from their own mean, measured in units of
their own standard deviation. This reduces between-participant baseline
differences and emphasizes within-person deviations.

This is relevant for eye-tracking-based workload detection because participants
may differ in baseline gaze behavior. Subject-wise normalization has been used
in workload-related modeling to reduce inter-subject variability and support
participant-held-out evaluation [10]. In this study, the transformation is
label-free because it uses only eye-tracking feature values, not workload labels.

### 2.7 Feature Importance Analysis

Feature importance analysis helps explain model behavior. For Logistic
Regression, standardized coefficients can show whether a feature increases or
decreases the predicted probability of high cognitive load. For XGBoost,
importance can be examined using tree-based gain and SHAP values. SHAP values
are useful because they estimate each feature's contribution to individual
predictions and can be summarized across samples [7].

Feature importance is especially important in this study because model
performance alone does not show whether the classifier is using eye behavior,
lighting, artifacts, or other contextual signals. Prior eye-tracking workload
studies have also used feature-importance analysis to interpret the contribution
of gaze-derived variables [12], [19].

### 2.8 Classification Metrics

ROC-AUC measures a classifier's ability to rank positive samples above negative
samples across thresholds. It is useful for comparing model discrimination
without depending on one fixed decision threshold [14]. Accuracy, balanced accuracy,
precision, recall, and F1-score provide complementary views of thresholded
classification performance. Since the thesis compares model families, ROC-AUC is
used as the primary metric, while the other metrics are reported as secondary
measures.

## Chapter 3: Methodology

### 3.1 Research Design

This study uses a quantitative computational experimental design. Logistic
Regression and XGBoost are trained on identical eye-tracking feature sets and
evaluated under the same participant-level validation protocol. The study
compares the models across multiple preprocessing configurations and analyzes
feature importance for interpretability.

### 3.2 Dataset

The study uses the GAZELOAD dataset, an eye-tracking dataset collected during
industrial human-robot collaboration tasks [8]. The dataset contains 26
participants, with five task recordings per participant. This forms a complete
grid of 130 recordings. The released sampling unit is a non-overlapping 250 ms
window, and the codebase reports 135,499 total windows.

Each recording includes one self-reported mental load rating on a 1 to 10 scale.
This rating is used to construct binary low-load and high-load labels.

### 3.3 Data Preparation

Participant identity is extracted from the recording filename instead of the
`Participant_ID` column because the project documentation notes that the
`Participant_ID` field is corrupted in some files. Task identifiers and
timestamps are excluded from the final feature matrix. Task identity is excluded
because it can directly encode the experimental condition rather than the
participant's eye behavior.

The original 250 ms windows are aggregated into 30-second epochs. Epoch
boundaries are computed within each recording, and partial epochs with less than
80% coverage are removed. This produces 1,087 retained epochs. Prior
eye-tracking workload research has shown that aggregation choices can influence
classification performance, supporting the use of epoch-level feature summaries
rather than relying only on raw short windows [13].

For each epoch, the mean and standard deviation of the base eye-tracking
features are computed. Additional derived features include blink rate, the
fraction of missing saccade features, the fraction of missing fixation
dispersion index values, the fraction of nonzero gaze transition entropy values,
and the fraction of windows with zero fixations. The final feature set contains
37 features when lighting features are included.

### 3.4 Labeling Schemes

Two binary labeling schemes are evaluated.

The first scheme is an absolute threshold. Ratings greater than or equal to 4
are labeled high load, and ratings below 4 are labeled low load. This retains
all 1,087 epochs.

The second scheme is a within-participant threshold. Ratings above a
participant's own median rating are labeled high load, ratings below the median
are labeled low load, and ratings equal to the median are excluded. This retains
768 epochs. This scheme is used because subjective workload ratings differ
between participants.

### 3.5 Feature Schemes

Two feature schemes are evaluated.

The raw feature scheme uses the epoch-level feature values directly.

The within-participant standardization scheme transforms each feature within
each participant. Each feature value is expressed as a deviation from that
participant's mean divided by that participant's standard deviation. This
emphasizes participant-relative changes in eye behavior.

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

Config E is treated as the main analytical pipeline because it combines the
participant-relative label and participant-relative feature representation. The
other configurations are used as ablation comparisons.

### 3.7 Models

Two models are compared.

Logistic Regression is implemented as a pipeline with median imputation,
missing-value indicators, feature scaling, and Logistic Regression
classification. Median imputation is required because Logistic Regression cannot
directly represent missing values.

XGBoost is implemented as a binary logistic classifier using histogram-based
tree construction. Missing values are passed directly to XGBoost because the
model can learn default directions for missing-value splits.

Hyperparameters are tuned using grid search inside the training folds only.

### 3.8 Evaluation Protocol

The study uses leave-one-participant-out validation. In each outer fold, all
epochs from one participant are held out for testing, and the remaining 25
participants are used for training. This produces 26 outer folds. Hyperparameter
tuning is performed using stratified group cross-validation inside the training
participants only. This follows the subject-independent evaluation logic used in
workload classification research, where performance is estimated on participants
not seen during model training [10], [13].

This evaluation design prevents a participant from appearing in both training
and testing within the same fold. It is stricter than random row-level splitting
and better reflects performance on unseen participants.

### 3.9 Metrics and Statistical Comparison

The primary metric is pooled ROC-AUC across all held-out predictions. Secondary
metrics include accuracy, balanced accuracy, precision, recall, and F1-score.
Per-participant metrics are also reported to show variability across
participants. ROC-AUC is used because it evaluates discrimination across
thresholds rather than depending on a single operating point [14].

To compare Logistic Regression and XGBoost, the study uses the paired Wilcoxon
signed-rank test over participant-level scores where the metric is defined. This
choice is consistent with machine-learning guidance recommending nonparametric
paired tests such as the Wilcoxon signed-rank test when comparing classifiers
over matched evaluation units [15].

### 3.10 Feature Importance Methods

For Logistic Regression, feature importance is analyzed using standardized
coefficients across folds. For XGBoost, feature importance is analyzed using
gain importance and out-of-fold SHAP values. Out-of-fold explanations are used
so that feature attribution corresponds to held-out participant predictions.

## Chapter 4: Results and Discussion

### 4.1 Dataset Summary

The dataset contains 130 complete recordings from 26 participants, with five
tasks per participant. The codebase reports 135,499 original 250 ms windows and
1,087 retained 30-second epochs. There were no timing discontinuities in the 130
recordings. Each recording contained exactly one workload rating.

Only one participant, P12, had a rating range below three points. This indicates
that most participants showed some rating variation across tasks, although
participant-specific rating behavior remained important.

### 4.2 Label Validity

The within-participant label showed stronger agreement with objective task
difficulty than the absolute threshold. The absolute threshold had Spearman rho
of 0.534 with task index, while the within-participant label had Spearman rho of
0.735. This supports the use of the within-participant label as an analytical
representation of relative workload.

### 4.3 Overall Model Performance

Across all configurations, XGBoost achieved higher pooled ROC-AUC than Logistic
Regression. Under Config E, the main analytical pipeline, Logistic Regression
achieved a pooled ROC-AUC of 0.621, while XGBoost achieved 0.682. XGBoost also
had higher accuracy under Config E, with 0.629 compared with 0.570 for Logistic
Regression.

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

These results show that XGBoost is consistently ahead in ROC-AUC, but the size
of the advantage is modest. The difference between the models is smaller than
the difference caused by changing the label and feature schemes.

### 4.4 Effect of Labeling and Standardization

Changing from the absolute threshold to the within-participant label improved
both models. For Logistic Regression, ROC-AUC increased from 0.479 in Config A
to 0.565 in Config D. For XGBoost, ROC-AUC increased from 0.530 to 0.628.

Applying both within-participant labeling and within-participant feature
standardization produced the strongest performance. Config E improved Logistic
Regression to 0.621 ROC-AUC and XGBoost to 0.682 ROC-AUC. This indicates that
participant-relative preprocessing was important for this dataset.

The full factorial design also clarifies the effect of removing lighting
features. Removing lighting reduced ROC-AUC in all matched comparisons. The
largest reductions appeared under the within-participant raw-feature condition:
Config D to Config D_no_lux decreased ROC-AUC by 0.078 for Logistic Regression
and 0.107 for XGBoost. Under the main analytical pipeline, Config E to Config F
decreased ROC-AUC by 0.047 for Logistic Regression and 0.080 for XGBoost.

### 4.5 Logistic Regression vs XGBoost Statistical Comparison

The Wilcoxon signed-rank tests show that XGBoost was consistently higher in
pooled ROC-AUC, but the participant-level evidence was not always statistically
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
Therefore, the most defensible interpretation is that XGBoost achieved higher
pooled ROC-AUC in every configuration, but the participant-level evidence is not
strong enough to claim statistically significant superiority in ROC-AUC. For
accuracy, Config E showed a statistically significant XGBoost advantage
(p = 0.031), but this does not by itself establish a general model superiority
claim across all configurations.

### 4.6 Feature Importance Findings

The existing results show that illuminance features were highly influential in
the models. In Config E, XGBoost SHAP analysis ranked illuminance standard
deviation first and illuminance mean second. The same configuration's Logistic
Regression coefficients also ranked illuminance standard deviation as the
largest coefficient by absolute magnitude.

This result is important because illuminance is an environmental feature rather
than a direct eye-movement feature. Its importance suggests that some predictive
signal may come from task context or lighting conditions. For this reason,
Config F, which removes lighting features, should be discussed as the more
strict eye-tracking-only comparison. In Config F, Logistic Regression achieved
0.575 ROC-AUC and XGBoost achieved 0.603 ROC-AUC.

### 4.7 Discussion

The results answer the main research question by showing that XGBoost generally
outperformed Logistic Regression for eye-tracking-based cognitive load
detection. However, the improvement was modest and not consistently
statistically significant across participants. This means that XGBoost may be
preferred for predictive performance, but Logistic Regression remains a useful
baseline because it is simpler and more interpretable.

The results also show that preprocessing choices were critical. The strongest
performance occurred only after cognitive load labels and features were
expressed relative to each participant. This supports the idea that cognitive
load detection from eye-tracking features is strongly affected by individual
differences.

The feature importance findings further show that interpretation is necessary.
If the highest-ranked features are lighting-related, then the model may be using
environmental context as well as eye behavior. This does not invalidate the
model, but it changes how the result should be described. Config E is the
strongest analytical pipeline, while Config F is the stricter eye-tracking-only
version of that pipeline.

## Chapter 5: Conclusion and Recommendations

### 5.1 Summary of Findings

This study compared Logistic Regression and XGBoost for cognitive load
detection using eye-tracking features from the GAZELOAD dataset. The models were
evaluated under leave-one-participant-out validation across the completed
experimental configurations.

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

Based on the results, XGBoost is the stronger model for this dataset in terms of
pooled ROC-AUC. Its ability to model nonlinear relationships and feature
interactions likely helped it outperform Logistic Regression. However, the
difference between the two models was modest at the participant level, and
Logistic Regression remained valuable as an interpretable baseline.

The study also concludes that participant-relative preprocessing is important
for cognitive load detection using eye-tracking features. Within-participant
labeling and within-participant feature standardization improved performance,
suggesting that individual differences in workload ratings and eye behavior
must be considered.

Finally, the study concludes that feature importance analysis is necessary for
interpreting cognitive load classifiers. The strong role of illuminance features
shows that high performance may partly reflect environmental or task-context
signals rather than eye movement alone.

### 5.3 Recommendations

Future studies should validate the findings with larger participant samples and
additional task conditions. Future work should also investigate artifact
filtering, especially for implausible saccade and gaze-vector values.
Participant 07, which the project documentation identifies as having poor gaze
validity, should be considered in a pre-registered data quality analysis.

Further experiments should compare multiple epoch durations, such as 10, 30, and
60 seconds, to determine whether the temporal aggregation window affects
performance. However, changing epoch duration should be treated as a separate
experimental dimension rather than as the same Config E result.

For deployment-oriented work, future systems should design an explicit
calibration procedure for computing participant-specific feature baselines. For
the current thesis, the model is analytical, so within-participant
standardization is acceptable as long as this limitation is clearly stated.

### 5.4 Final Statement

This thesis shows that XGBoost provides better pooled predictive performance
than Logistic Regression for eye-tracking-based cognitive load detection in the
GAZELOAD dataset, especially under participant-relative preprocessing. However,
the study also shows that model choice is only one part of the problem. Label
definition, feature standardization, lighting-feature inclusion, validation
design, and feature interpretation are equally important for producing
defensible conclusions.

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

[10] I. Albuquerque, J. Monteiro, O. Rosanne, and T. H. Falk, “Estimating distribution shifts for predicting cross-subject generalization in electroencephalography-based mental workload assessment,” Frontiers in Artificial Intelligence, vol. 5, Oct. 2022, doi: 10.3389/frai.2022.992732.

[11] S. Gado, K. Lingelbach, M. Wirzberger, and M. Vukelić, “Decoding Mental Effort in a Quasi-Realistic Scenario: A Feasibility Study on Multimodal Data Fusion and Classification,” Sensors, vol. 23, no. 14, p. 6546, July 2023, doi: 10.3390/s23146546.

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

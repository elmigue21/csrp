# Concept Justification for the Comparative Model Study

This draft supports a thesis focused on the comparative analysis of Logistic
Regression and XGBoost for cognitive load detection using eye-tracking features.
It explains why the current experiment uses within-participant labeling,
within-participant feature standardization, leave-one-participant-out evaluation,
and feature importance analysis.

## 1. Main Comparative Focus: Logistic Regression vs XGBoost

The central comparison should remain between Logistic Regression and XGBoost.
Logistic Regression is useful as an interpretable linear baseline: it tests
whether cognitive load can be separated by a weighted linear combination of the
eye-tracking features. XGBoost is useful as a stronger nonlinear comparator:
it can model feature interactions, nonlinear thresholds, and missing-value
patterns that a linear model cannot represent directly.

This comparison is appropriate for a comparative-analysis thesis because both
models are trained and evaluated under the same data preparation, feature set,
label scheme, and participant-level validation protocol. Any performance
difference is therefore attributable to the model family rather than to a
different split or feature set.

The codebase implements this directly in `src/models.py`, where the two
classifiers under comparison are Logistic Regression and XGBoost. The result
tables then report both models for every configuration, allowing the thesis to
compare model behavior under identical experimental conditions.

Literature on cognitive workload classification also commonly compares multiple
machine-learning classifiers, including interpretable and noninterpretable
models, when evaluating eye-tracking features. For example, a review and
experiment on interpretable cognitive workload classification reports model
comparison as part of the methodological design and highlights the value of
interpretability alongside predictive performance.

Reference:
https://pmc.ncbi.nlm.nih.gov/articles/PMC7914927/

## 2. Why Within-Participant Labeling Is Defensible

The dataset uses self-reported mental load ratings from 1 to 10. These ratings
are subjective: two participants may experience or use the same number
differently. A universal threshold, such as rating >= 4, assumes that a rating of
4 means the same cognitive state for every participant. In this dataset, that
assumption is weak because participant identity explains a large share of the
rating variance, and some participants use only a narrow part of the rating
scale.

The within-participant label addresses this by defining high load relative to
each participant's own ratings. In the current implementation, a recording is
classified as high load when its rating is above that participant's median
rating, low load when it is below the median, and excluded when it is exactly at
the median. This turns the question from "is this rating universally high?" into
"was this task high for this participant?"

This is especially appropriate for an analytical thesis model because the goal
is to compare whether Logistic Regression and XGBoost can detect cognitive load
patterns from eye-tracking features after reducing participant-specific rating
bias. The repo's methodology notes that the within-participant label has
stronger agreement with objective task difficulty than the absolute threshold:
Spearman rho is 0.735 for the within-participant label compared with 0.534 for
the absolute threshold.

There is RRL support for this idea. Related mental-effort classification work
has used subject-wise median splits to separate low and high subjectively
perceived mental effort. This supports the use of participant-relative
thresholds when the target is subjective experience rather than an externally
fixed task label.

References:
https://www.mdpi.com/1424-8220/23/14/6546
https://pubmed.ncbi.nlm.nih.gov/28965433/

## 3. Why Within-Participant Feature Standardization Is Defensible

Eye-tracking features often contain stable person-specific differences. For
example, participants may differ in typical fixation behavior, gaze dispersion,
blink tendency, missing-gaze rate, or how their gaze is captured by the device.
If these between-person baselines are left uncorrected, a model may learn to
recognize participants rather than cognitive load.

Within-participant feature standardization reduces this problem by expressing
each feature as a deviation from that participant's own mean, scaled by that
participant's own standard deviation. In practical terms, the feature no longer
means "this participant has a high raw fixation value"; it means "this
participant is above or below their own usual fixation value." This better fits
the physiological interpretation of workload as a change from a person's
baseline state.

The codebase treats this as a label-free transformation. The standardization
uses only eye-tracking feature values from the same participant and does not use
the workload labels. This is important because it avoids directly leaking label
information into the feature matrix. The methodology also notes the limitation:
in a deployed system, this would require calibration data from each new user.
For the current thesis, however, the model is analytical rather than
deployment-ready, so the calibration requirement can be reported as a limitation
rather than a blocker.

RRL support exists for subject-wise normalization in workload modeling. Studies
on mental workload generalization discuss normalizing features with respect to
subject-specific baseline statistics to reduce inter-subject variation and
highlight changes associated with workload. This supports the conceptual basis
of using participant-relative feature values.

References:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9576998/
https://pmc.ncbi.nlm.nih.gov/articles/PMC8272248/

## 4. Why Leave-One-Participant-Out Evaluation Is Needed

The correct evaluation question is whether a model trained on some participants
can generalize to an unseen participant. Random row-level splitting would place
epochs from the same participant, and often the same recording, into both train
and test sets. That would overestimate performance because the model could
benefit from participant-specific patterns rather than general workload-related
patterns.

The current codebase uses leave-one-participant-out evaluation. Each outer fold
holds out one participant completely, trains on the remaining participants, and
tests only on the held-out participant. Hyperparameters are selected inside the
training participants only. This design is stricter and more thesis-defensible
than random cross-validation because no participant contributes samples to both
training and testing in the same fold.

This protocol is also consistent with the subject-independent evaluation
approach commonly discussed in cognitive workload and eye-tracking
classification literature.

Reference:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8272248/

## 5. Why Config E Is the Main Analytical Pipeline

Config E combines the two participant-relative choices:

- within-participant label;
- within-participant feature standardization;
- lighting features included.

In the current results, Config E gives the strongest overall performance for
both models, and XGBoost under Config E has the highest pooled ROC-AUC among the
reported configurations. This makes Config E the most defensible primary
pipeline for the main Logistic Regression vs XGBoost comparison.

The other configurations should still be reported as ablation comparisons:

- A vs D tests the effect of within-participant labeling.
- A vs C tests the effect of within-participant feature standardization under
  the absolute label.
- D vs E tests the effect of within-participant feature standardization under
  the within-participant label.
- A vs B, C vs C_no_lux, D vs D_no_lux, and E vs F test the effect of removing
  lighting features under each label/feature combination.
- Logistic Regression vs XGBoost within each config tests the thesis's main
  model-comparison question.

This structure lets the thesis say more than "Config E was selected because it
performed best." It shows how preprocessing choices affect both classifiers, then
uses the best-supported analytical pipeline for deeper model comparison.

## 6. Metric Justification

ROC-AUC should be the primary metric when the goal is to compare classifier
discrimination independent of a fixed decision threshold. This is useful here
because the thesis compares model families, not a finalized deployment decision
threshold.

However, ROC-AUC should not be reported alone. The thesis should also report
accuracy, balanced accuracy, precision, recall, and F1-score. Balanced accuracy
is important because the classes are not perfectly balanced, and F1-score helps
describe the precision-recall tradeoff after applying the default classification
threshold.

The recommended reporting structure is:

- primary metric: pooled ROC-AUC across all held-out participant folds;
- secondary metrics: accuracy, balanced accuracy, precision, recall, and F1;
- stability analysis: per-participant metrics summarized as mean and standard
  deviation;
- model comparison test: paired Wilcoxon signed-rank test over participant-level
  scores where the metric is defined.

## 7. Limitation Statement for Thesis Use

The within-participant label and within-participant feature standardization are
appropriate for an analytical thesis model, but they should be described
carefully.

The within-participant label does not claim to detect a universal high-workload
threshold. It detects whether a task is high or low relative to a participant's
own reported workload distribution.

The within-participant feature standardization does not represent a fully
deployment-ready system by itself. It assumes that participant-specific feature
statistics are available, which would require calibration or prior recordings
from a new user. Since the current thesis is analytical, this is acceptable as
long as the limitation is explicitly stated.

## 8. Suggested Thesis Framing

A defensible framing is:

> This study compares Logistic Regression and XGBoost for binary cognitive load
> detection using eye-tracking features under leave-one-participant-out
> evaluation. Because subjective workload ratings and eye-tracking baselines vary
> across participants, the main analytical pipeline uses within-participant
> workload labeling and within-participant feature standardization. Additional
> configurations are evaluated as ablation analyses to examine the effect of
> label definition, feature standardization, and lighting-feature inclusion.

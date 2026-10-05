# COLET leakage audit — 2026-10-05

Scope: current `src/` training, feature preparation, evaluation and importance code,
the Colab entry point, and saved COLET tables/caches. No model code or results were
changed. This audit cannot certify every possible source of leakage: the raw
recordings and conversion script are not present locally, and historical decisions
cannot be verified from code alone.

## 1. High priority: duplicated participants cross evaluation boundaries

In `outputs/colet/tables/04_feature_table.csv`, P28 and P31 have exactly identical
values in every column except participant ID for A1, A2, A3 and A4. Their rows begin
at CSV lines 110 and 122, respectively. Both saved Parquet feature caches contain
the same duplication. Their per-person standardized feature vectors also match.

`src/evaluate.py:28` groups by participant ID, so holding out P28 leaves P31 in
training, and holding out P31 leaves P28 in training. These folds therefore have
exact copies of their test feature vectors in the training set. Participant ID
separation alone does not prevent this contamination. Inner grouped validation
can similarly separate these copies. Reconstructing the configured splits found
exact normalized train/test duplicates in outer folds 28 and 31, and exact
train/validation duplicates in 86 of the 225 inner splits.

The cause could be upstream duplication, conversion, or a cache/data error; it is
not established without raw recordings. It does not establish that the entire
0.99 AUC is explained by duplication, nor does it justify changing a participant
ID or deleting records without checking their provenance.

Next: compare P28/P31 raw signal hashes and original dataset entries, determine
whether they represent the same recordings, and rerun evaluation with verified
independent recordings (or group duplicate identities together). Recompute all
metrics, intervals, comparisons and importance. Removing their predictions from
the current CSV is insufficient because they participated in other folds' training.

## 2. Confirmed test-recording access through normalization

`src/run_colet.py:62` standardizes all retained activities within each participant
before selecting A1/A4 or running cross-validation. `src/normalize.py:25` computes
each person's mean and SD using that person's own recordings.

A diagnostic that adds 10 mm to one participant's A4 pupil mean changes the
standardized pupil mean for all four of that participant's activities. Other
participants are unaffected. This is test-input access, not direct target-label
access or pooling test statistics into other participants' baselines.

For an offline protocol that explicitly permits the complete unlabeled test batch,
this is a transductive/calibration assumption. For prediction on a new recording
using only previously collected information, it is unavailable information and
constitutes leakage relative to that intended deployment. Moving the same
four-activity calculation inside the LOPO loop would not remove the issue.

Next: evaluate a separately collected calibration baseline or a training-only
transformation, and report that result separately from the current offline protocol.

## 3. Confirmed dataset-wide feature-set decision

`src/run_colet.py:54` computes event sanity checks on all analysis participants;
line 57 selects 10 features or the five-feature fallback from those checks before
cross-validation. `src/checks.py:25` aggregates event statistics by activity.
Thus held-out data influence the feature-set decision. This is an unsupervised
QC decision using known conditions, rather than selection maximizing held-out AUC.
Its effect on performance has not been quantified.

For a strict evaluation of the full learned procedure, make the adaptive choice
using training participants at each relevant validation level, or establish the
feature set using independent development data before evaluation. Rechecking
redundancy is report-only in the current code, but the documented original feature
selection used exploratory correlations and condition-specific missingness;
that historical development uses the same dataset and deserves disclosure.

## 4. Other concerns, distinct from direct leakage

- A4 includes speaking and time pressure; A1 does not. Blink or pupil differences
  can identify condition artifacts. This is confounding and limits a pure
  cognitive-load claim even with technically sound splits.
- Whole-recording interpolation, central differences and zero-phase pupil filtering
  use later samples within the same recording. Appropriate for completed-activity
  classification, they do not establish causal real-time prediction performance.
- `src/dataset.py:41` keys the cache by settings and feature code, but not data-root
  identity or source-file contents. Reusing a cache directory for changed input
  data can silently load another dataset's features. No evidence establishes that
  this caused the P28/P31 issue.
- Saved run metadata records library versions and settings but not source commit or
  input hashes. Exact provenance of the historical run cannot be certified.
- Shuffled-label tests use untuned models (`src/run_colet.py:82`), whereas headline
  metrics use tuned models. They test a different procedure and do not rule out
  test-input normalization, duplicated recordings, or confounding.
- Participant bootstrap intervals resample fixed out-of-fold predictions without
  refitting. They omit model-fitting variability and do not correct contamination.
- The ceiling rule reads held-out P1 AUC to choose which comparison to emphasize.
  Both analyses are run regardless; a documented pre-stated interpretation rule
  is not training leakage, but reporting both avoids selective presentation.
- Permutation importance and SHAP do not feed back into fitting in the inspected
  code. Permuting features across all rows disrupts participant pairing, which
  limits importance interpretation rather than inflating baseline predictions.

## Safeguards verified

- Reconstructed all 45 outer and 225 inner splits: participant IDs are disjoint
  across training/test and training/validation boundaries.
- Randomized search receives only outer-training rows and their groups.
- LR imputation and scaling are inside the estimator pipeline, including inner
  validation. XGBoost handles NaN directly.
- Inputs are explicit feature lists: participant ID, activity, NASA-RTLX, refit
  counts and QC columns are not model inputs.
- Labels are assigned from activity; NASA-RTLX is used only for a separate check.
- Feature extraction and quality exclusion otherwise operate on individual
  recordings with fixed settings, not on pooled fitted population statistics.
- Saved P1/P2 predictions have unique participant/activity keys, correct labels
  and finite probabilities. Main data have 90 rows, 45 per class, and no missing
  values in the ten selected raw features. Unique keys do not imply independent
  recordings, as P28/P31 demonstrate.

## Verification limits

Diagnostics ran successfully using `.venv/bin/python`. The targeted existing
pytest suite could not run because that environment has no `pytest` installed.
No full training rerun was performed. Raw signal identity, conversion integrity,
external validity, and the amount of metric inflation remain unresolved.

Conclusion: the implementation has nested participant-grouped evaluation and
regularization, but the saved data cannot currently be described as clean of
leakage. Duplicate feature copies across participant IDs are the first issue to
resolve, followed by evaluation aligned with the intended calibration protocol.

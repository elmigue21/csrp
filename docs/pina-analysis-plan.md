# Pina 2025 feasibility run: analysis plan (written 2026-11-13, BEFORE any eye data was loaded)

Data: Zenodo 18669059 (CC BY 4.0). HMDgaze.zip (36 users, Varjo XR-4, ~200 Hz), trial_data.zip (condition timing).
Known before writing this (from trial files only, no eye data): each condition is one continuous 4-minute
segment; trials (one per object found) last ~1 s, and low load has more trials than high load (task
performance), so trials are NOT the unit of analysis.

1. Contrast (main): 90-degree search area, low load (TrialType 0) vs high load (TrialType 1).
   Secondary, reported separately, never pooled with or chosen over the main: 360-degree (2 vs 3).
2. Unit: each 4-min segment cut into 8 non-overlapping 30 s windows from the first trial start.
   One row = one window. Rows per person = 16 (8 low + 8 high).
3. Signals mapped onto the unchanged src code (recording_signals -> flat_features, DROPPED applied):
   - gaze: per-eye forward vectors -> gaze_normal0/1; confidence = 1 if that sample's gaze status is VALID
     and both eyes valid, else 0.
   - pupil: per-eye pupil diameter (mm), method "3d", confidence as above.
   - blinks: runs where both eyes are invalid, then the existing rule (merge <100 ms, keep 50-500 ms).
4. Exclusions: window with invalid fraction > 0.35 dropped (existing rule). Sanity check on fixation median
   (150-400 ms); if it fails, the planned fallback drops gaze_speed/fixation/saccade features.
5. Features: the same 17 statistics per signal (flatten.py), no per-person standardization.
6. Models: MODELS_TOPN unchanged (in-fold Top-N, LR elastic net / XGBoost, 30-candidate random search,
   5-fold grouped inner CV), leave-one-participant-out.
7. Reported: every metric (ROC-AUC + participant-bootstrap 95% CI, accuracy, balanced accuracy, precision,
   recall per class, macro F1, Brier, people with high above low).
8. Pass tests (decided now):
   a. Above chance: AUC 95% CI lower bound > 0.50 AND label-permutation p < 0.05 (labels swapped per
      person, i.e. which segment is "high"; selection inside; hyperparameters fixed at the most-chosen values).
   b. Features work together: model minus in-fold one-feature baseline (TopN n=1 -> LR) has 95% CI > 0.
   c. Importance spread (descriptive): SHAP share of top-1 / top-3 / top-5 features, number of signal
      families with SHAP share >= 5%, selection counts, and the model refit without the top feature's family.
9. Whatever the outcome, all of the above are reported. No other contrasts, windows or feature sets are tried.

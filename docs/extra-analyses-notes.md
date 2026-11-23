# Extra robustness analyses (mock-defense follow-up)

Run 2026-11-07 on the cached COLET features (44 participants, the same retained set as the main
run). These are **notes only**: nothing here is in the paper yet, and no official table was
changed. All numbers come from the runs logged in `outputs/colet/tables/extra/log_*.txt`.

**How to read the numbers.** ROC-AUC 0.5 = coin flip, 1.0 = perfect. The [low, high] range is
the 95% confidence interval from resampling participants (2000 times). "XGB−LR" is XGBoost's
AUC minus LR's AUC; if its interval contains 0, the two models are not shown to differ.

Unless stated otherwise every analysis uses the main pipeline unchanged: nested LOPO
(leave one participant out), random search with 30 candidates per model inside 5 participant-
grouped inner folds, `RANDOM_STATE = 0`, 2000 participant-bootstrap resamples, the models in
`src/models.py`. P1 = 10 features, P2 = pupil only.

Background values recomputed on the 44 retained participants (medians per activity):

| | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| Blink rate (/min) | 1.7 | 0.0 | 11.7 | 11.7 |
| Invalid gaze samples | 0.9% | 0.7% | 6.9% | 6.9% |
| Valid time (s) | 34.8 | 29.4 | 60.9 | 48.4 |

Note: the main z-score (`normalize.per_participant_zscore`) uses pandas `groupby(...).transform("std")`,
which is the **sample SD (ddof = 1)**. The new scripts use the same.

## Reviewer decisions (2026-11-07)

- §1: A2-baseline centering goes into Chapter 4 (post hoc). A2+A3 z-score stays in these notes only.
- §2, §3: both go into Chapter 4 as a speech × time-pressure table and are cited in Chapter 5 and Limitations.
- §5: blink alone, blink + pupil and untuned models go into Chapter 4 as baselines.
- §4: windows go into Appendix B and a spoken answer.
- §6: not in the paper; spoken answer only, labelled post hoc and margin-sensitive.
- §7: one hedged sentence in Limitations.

Sunday one-liner: "Our models detect COLET's task condition, which includes speaking. Using only a separate
baseline activity (A2), AUC is still 0.90–0.92, close to the 0.85–0.88 without standardization. With speech
held constant (A3 vs A4), time-pressure detection is weak (0.60–0.66), so this dataset cannot separate mental
load from speech, and we say so."

Keep two blink numbers apart: single-feature AUC (no model; `researcher-notes.md`) is 0.895 raw / 0.968 z;
LR trained on blink alone in LOPO (§5) is 0.883 raw / 0.963 z.

## Summary

| # | Analysis | What it tests | Key result (ROC-AUC, LR / XGB) | For the thesis |
|---|---|---|---|---|
| 1 | Standardize with A2+A3 only, or A2-baseline centering | Is 0.99 partly because the test person's own A1/A4 set their "normal"? | P1 0.99/0.98 → **0.84/0.90** (A2+A3) and **0.90/0.92** (A2-baseline centering). P2 0.83/0.81 → 0.72/0.74 and 0.76/0.74 | **Hurts** (part of 0.99 is within-person ranking), but the signal stays well above chance |
| 2 | A3 vs A4 (both speaking, differ by time pressure) | Can the features see load when speech is held constant? | z: P1 0.62/0.60, P2 0.61/0.66; raw: ≈ chance | **Hurts**: time pressure while speaking is barely detectable |
| 3 | A1 vs A2 (both silent, differ by time pressure) | Is there a non-speech load signal? | z: P1 0.75/0.77, P2 0.75/0.68; raw: 0.50–0.64 | **Helps a little**: a speech-free signal exists, but it is much weaker than A1 vs A4 |
| 4 | 10 s windows | Does the result depend on one row per activity? | z: P1 0.93/0.90, P2 0.83/0.83; raw: P1 0.82/0.81, P2 0.63/0.62 | **Neutral / helps**: same pattern at window level; LR beats XGB slightly in P1 |
| 5 | Simple baselines | Do 10 features and tuning add anything? | Blink rate alone (LR) 0.96; blink + pupil mean 0.97/0.97; untuned P1 0.99/0.98 | **Hurts**: one feature (blink rate) gives almost all of P1; tuning adds nothing |
| 6 | Equivalence margin ±0.05 AUC | Can we say "LR and XGB are equivalent"? | P1 90% CI [−0.027, 0.000] → equivalent; P2 [−0.073, +0.036] → not | **Neutral**: P1 equivalence is real but sits at the ceiling; P2 is inconclusive |
| 7 | Activity order (inferred) | Fatigue / practice confound? | Order varied between people; fixation rate rises with position; predicted probabilities unrelated to position | **Neutral / helps**: order does not explain the model's output |

---

## 1. Standardizing without the test person's A1 and A4

**Question.** In the main pipeline each person is z-scored over all four activities, so the
held-out person's own A1 and A4 help set their "normal". Does 0.99 survive if A1 and A4 play no
part in the standardization?

**Method.**
- *A2+A3 z-score:* each person's mean and SD (ddof = 1) are computed from their A2 and A3 only and
  applied to their A1 and A4. Keeping A1/A4 out of the reference follows the leak-free calibration
  idea in Tognotti et al. 2026 (NM5, doi:10.3389/fdgth.2026.1827279); using A2+A3 as the reference
  is our own choice (A2 is silent and A3 is spoken, so the reference is balanced for speech).
- *A2-baseline centering:* A1 − A2 and A4 − A2 (subtractive baseline; this is centering, not a z-score, because one recording has no SD). Subtractive baseline correction is
  supported for pupil size by Mathôt et al. 2018 (NM7, doi:10.3758/s13428-017-1007-2); applying it to
  the other nine features is our own choice. One recording has no SD, so we do not divide: LR
  already rescales each feature in-fold with `StandardScaler` (fitted on training participants only)
  and XGBoost does not depend on scale, so this is the same as "difference scaled by the training-fold SD".
- No participant lacked A2 or A3, so **0 were dropped** (n = 44, 88 samples).

**Result.**

| Variant | Set | LR AUC [95% CI] | XGB AUC [95% CI] | Bal. acc. LR / XGB | XGB−LR [95% CI] |
|---|---|---|---|---|---|
| Main (all four, official) | P1 | 0.992 [0.979, 1.000] | 0.980 [0.957, 0.997] | 0.966 / 0.920 | −0.012 [−0.030, +0.002] |
| A2+A3 z-score | P1 | 0.839 [0.760, 0.915] | 0.898 [0.830, 0.956] | 0.818 / 0.841 | +0.058 [−0.008, +0.124] |
| A2-baseline centering | P1 | 0.903 [0.843, 0.958] | 0.921 [0.859, 0.975] | 0.841 / 0.841 | +0.018 [−0.051, +0.091] |
| No standardization (official optional) | P1 | 0.853 [0.783, 0.923] | 0.881 [0.822, 0.941] | 0.739 / 0.818 | — |
| Main (all four, official) | P2 | 0.831 [0.735, 0.912] | 0.812 [0.697, 0.912] | 0.807 / 0.773 | −0.019 [−0.086, +0.047] |
| A2+A3 z-score | P2 | 0.724 [0.647, 0.806] | 0.735 [0.646, 0.819] | 0.705 / 0.682 | +0.010 [−0.054, +0.072] |
| A2-baseline centering | P2 | 0.763 [0.677, 0.851] | 0.736 [0.633, 0.834] | 0.727 / 0.750 | −0.027 [−0.090, +0.040] |
| No standardization (official optional) | P2 | 0.611 [0.573, 0.657] | 0.649 [0.565, 0.732] | 0.568 / 0.568 | — |

**Interpretation.** When A1 and A4 are kept out of the person's "normal", P1 drops from about 0.99
to 0.84–0.92 and P2 from about 0.82 to 0.72–0.76. So a sizeable part of the 0.99 comes from the
z-score seeing the test person's own A1 and A4 (the panel's "within-person ranking" concern is
partly right). This matches the direction reported by Tognotti et al. 2026 (normalization statistics
that overlap the test data inflate results). The signal is still clearly above chance. A2-baseline centering is comparable to or modestly
above no standardization (0.90/0.92 vs 0.85/0.88), but its CIs contain the raw estimates, so it is not shown to be better. Caution: with only two recordings the A2+A3 SD is very
noisy (some standardized values reached |z| > 100), which likely hurts LR more than XGBoost and
explains why XGB is ahead in that variant; the A2-baseline centering variant avoids this problem. A2 is itself a time-pressure activity, not a neutral rest baseline.

**Suggested sentence.** "When each participant's reference was computed only from the two
activities not used for classification (A2 and A3), or by subtracting A2 alone (A2-baseline centering), ROC-AUC fell from 0.99 to
0.84–0.92 (all features) and from 0.83 to 0.72–0.76 (pupil only), indicating that part of the main
result reflects the transductive standardization, although detection remained well above chance."

## 2. A3 vs A4: both speaking, differ only by time pressure

**Question.** If speech is held constant, can the features still tell the conditions apart?

**Method.** Labels A3 = 0, A4 = 1. Same nested LOPO. Two versions: the main per-person z-score over
all four activities ("z"), and unstandardized ("raw"). The pair choice is our own choice (the COLET
paper, EM7, reports other pairs but not this one). n = 44, 88 samples.

**Result.**

| Version | Set | LR AUC [95% CI] | XGB AUC [95% CI] | Bal. acc. LR / XGB | XGB−LR [95% CI] |
|---|---|---|---|---|---|
| z | P1 | 0.624 [0.487, 0.752] | 0.602 [0.469, 0.736] | 0.580 / 0.557 | −0.022 [−0.129, +0.090] |
| z | P2 | 0.610 [0.474, 0.737] | 0.661 [0.538, 0.773] | 0.545 / 0.648 | +0.051 [−0.048, +0.148] |
| raw | P1 | 0.534 [0.443, 0.627] | 0.442 [0.354, 0.530] | 0.545 / 0.500 | −0.092 [−0.199, +0.009] |
| raw | P2 | 0.488 [0.407, 0.568] | 0.528 [0.430, 0.627] | 0.500 / 0.534 | +0.040 [−0.049, +0.135] |

**Interpretation.** With speech held constant, only the z-scored pupil model with XGBoost has a
confidence interval above 0.5 (0.66), and even that is weak; everything else is at or near chance.
Time pressure on top of counting aloud is barely visible in these features. This supports the
panel's speech concern: most of the A1 vs A4 separation comes from the speaking (multitask) part,
not from time pressure. (The COLET authors themselves describe time pressure as a weak manipulation, EM7.)

**Suggested sentence.** "Among activities that both involved counting aloud (A3 vs A4), the
models were close to chance (ROC-AUC 0.60–0.66 standardized, 0.44–0.53 raw), so the added time
pressure was barely detectable once speech was held constant."

## 3. A1 vs A2: both silent, differ only by time pressure

**Method.** Labels A1 = 0, A2 = 1, otherwise as in §2. n = 44, 88 samples.

**Result.**

| Version | Set | LR AUC [95% CI] | XGB AUC [95% CI] | Bal. acc. LR / XGB | XGB−LR [95% CI] |
|---|---|---|---|---|---|
| z | P1 | 0.752 [0.612, 0.869] | 0.765 [0.635, 0.874] | 0.693 / 0.739 | +0.013 [−0.064, +0.086] |
| z | P2 | 0.753 [0.608, 0.876] | 0.680 [0.549, 0.806] | 0.750 / 0.693 | −0.072 [−0.137, +0.000] |
| raw | P1 | 0.641 [0.560, 0.720] | 0.576 [0.487, 0.670] | 0.557 / 0.580 | −0.065 [−0.154, +0.016] |
| raw | P2 | 0.564 [0.514, 0.623] | 0.495 [0.434, 0.560] | 0.545 / 0.557 | −0.069 [−0.148, −0.001] |

**Interpretation.** Without any speech, time pressure is detectable after per-person
standardization (about 0.75, intervals above 0.5), including from pupil alone. So there is a real,
speech-free load signal, but it is much weaker than A1 vs A4. Caveats: the z-score here has the same
transductive issue as §1, and A2 is also shorter than A1 (29 vs 35 s median valid time). Without
standardization the signal is weak (0.50–0.64). In raw P2, LR is slightly ahead of XGBoost (CI just
below 0).

**Suggested sentence.** "Between the two silent activities (A1 vs A2), standardized features
separated the time-pressure condition at ROC-AUC ≈ 0.75 for both feature sets, showing a smaller
but speech-free workload signal."

## 4. Fixed-length windows

**Question.** Does the result depend on using one row per whole activity?

**Method.** Each retained recording was cut into non-overlapping **10 s** windows by gaze timestamp;
the last shorter piece was dropped. 10 s windows follow Kaczorowska et al. 2022 (EM18,
doi:10.3390/brainsci12050542); longer windows (30 s, Rahman 2021, EM8) are impossible because many
activities last under 40 s. The gaze table was sliced and `features.recording_features` was run
unchanged on each slice (it trims pupil and blinks to the slice). The recording exclusion rule
(invalid > 35%) was applied per window (our own choice). Per-person z-score over each person's
windows from all four activities (feasible, done as in the main pipeline), then the same nested
LOPO on A1 vs A4 windows, grouped by participant so no person is in both training and test.
Full settings (N_ITER = 30) were used; extraction took 13 s.

Windows: A1 153 (0 excluded), A2 113 (0), A3 285 (9), A4 241 (11). Model input: **153 A1 and 230 A4
windows** from all 44 participants (median 3 A1 and 4 A4 windows per person).

**Result.**

| Version | Set | LR AUC [95% CI] | XGB AUC [95% CI] | Bal. acc. LR / XGB | XGB−LR [95% CI] |
|---|---|---|---|---|---|
| z | P1 | 0.928 [0.893, 0.958] | 0.901 [0.851, 0.943] | 0.855 / 0.844 | −0.027 [−0.049, −0.009] |
| z | P2 | 0.829 [0.738, 0.906] | 0.825 [0.740, 0.899] | 0.750 / 0.782 | −0.005 [−0.029, +0.023] |
| raw | P1 | 0.819 [0.765, 0.874] | 0.806 [0.752, 0.862] | 0.761 / 0.741 | −0.014 [−0.054, +0.023] |
| raw | P2 | 0.626 [0.573, 0.693] | 0.621 [0.570, 0.671] | 0.513 / 0.570 | −0.005 [−0.054, +0.037] |

**Interpretation.** The pattern holds at window level: high with standardization, lower without,
pupil-only weaker, LR never worse than XGBoost. P1 drops from 0.99 to about 0.90–0.93, as expected
because a 10 s window holds only 0–2 blinks (blink rates need multi-second windows, Bentivoglio 1997,
ET5). Here LR is slightly but clearly better than XGBoost (interval entirely below 0). The z-score is
still transductive (it uses the test person's own windows).

**Suggested sentence.** "With 10 s windows (153 low-load and 230 high-load windows, participant-grouped
LOPO), ROC-AUC was 0.93 (LR) and 0.90 (XGBoost) for all features and 0.83 for pupil only, so the
findings do not depend on the whole-activity unit of analysis."

## 5. Simple baselines

**Method.** Same LOPO, A1 vs A4, n = 44, 88 samples, z and raw. (a) each feature alone with tuned LR;
(b) blink rate + pupil mean, tuned LR and XGBoost; (c) default hyperparameters (`run_lopo(tune=False)`),
P1 and P2. Choosing these baselines is our own choice.

**Result (a): single features, LR, AUC [95% CI].**

| Feature | z | raw |
|---|---|---|
| blink_rate | **0.963** [0.911, 0.997] | **0.883** [0.821, 0.940] |
| pupil_mean | 0.857 [0.761, 0.939] | 0.610 [0.570, 0.659] |
| saccade_rate | 0.851 [0.745, 0.942] | 0.750 [0.675, 0.831] |
| fixation_rate | 0.759 [0.633, 0.864] | 0.686 [0.614, 0.762] |
| fixation_duration | 0.721 [0.584, 0.840] | 0.675 [0.598, 0.754] |
| saccade_amplitude | 0.706 [0.574, 0.826] | 0.672 [0.588, 0.759] |
| pupil_sd | 0.695 [0.558, 0.811] | 0.635 [0.546, 0.719] |
| saccade_peak_velocity | 0.643 [0.503, 0.780] | 0.640 [0.558, 0.721] |
| gaze_sd_y | 0.627 [0.493, 0.770] | 0.677 [0.572, 0.782] |
| gaze_sd_x | 0.356 [0.271, 0.432] | 0.513 [0.426, 0.603] |

**Result (b) and (c).**

| Model set | Version | LR AUC [95% CI] | XGB AUC [95% CI] | Bal. acc. LR / XGB | XGB−LR [95% CI] |
|---|---|---|---|---|---|
| blink + pupil mean (tuned) | z | 0.973 [0.936, 1.000] | 0.970 [0.932, 0.995] | 0.943 / 0.943 | −0.004 [−0.028, +0.022] |
| blink + pupil mean (tuned) | raw | 0.883 [0.815, 0.945] | 0.854 [0.786, 0.919] | 0.773 / 0.784 | −0.029 [−0.063, +0.005] |
| P1 untuned | z | 0.991 [0.975, 1.000] | 0.977 [0.950, 0.997] | 0.943 / 0.920 | −0.014 [−0.040, +0.006] |
| P1 untuned | raw | 0.867 [0.805, 0.926] | 0.864 [0.797, 0.927] | 0.795 / 0.795 | −0.004 [−0.052, +0.042] |
| P2 untuned | z | 0.852 [0.754, 0.934] | 0.819 [0.707, 0.912] | 0.807 / 0.761 | −0.033 [−0.092, +0.021] |
| P2 untuned | raw | 0.603 [0.563, 0.652] | 0.625 [0.524, 0.727] | 0.557 / 0.580 | +0.022 [−0.090, +0.124] |

**Interpretation.** Blink rate alone reaches 0.96 (standardized) and 0.88 (raw); blink rate plus
pupil mean reaches 0.97. The full 10-feature model (0.99) adds very little, and tuning adds nothing
(untuned P1 0.991/0.977 vs tuned 0.992/0.980; untuned P2 reproduces the known 0.852/0.819). Since
blink rate mostly tracks speaking (A1 1.7, A2 0.0, A3 11.7, A4 11.7 /min), this strengthens the speech
concern. Pupil mean alone (0.86 z) is about as good as the P2 model. The gaze_sd_x result (0.36, below
chance) is most likely an artifact of leave-one-out with a feature that carries no signal (each held-out
person shifts the training mean the opposite way); it should not be read as a reversed effect (our
interpretation).

**Suggested sentence.** "Blink rate alone gave ROC-AUC 0.96 under the same LOPO protocol, and default
hyperparameters matched the tuned models, so the high P1 performance rests mainly on a single feature
that is strongly affected by speaking."

## 6. Equivalence margin for LR vs XGBoost

**Question.** The paper says neither model is better. Can we go further and say they are *equivalent*?

**Method.** Equivalence needs a pre-stated margin: the largest AUC difference we would call
unimportant. The idea of a "region of practical equivalence" comes from Benavoli et al. 2017 (M11,
JMLR 18(77)); the **value ±0.05 AUC is our own choice** (no project source gives an AUC margin), with
±0.02 as a strict alternative. Equivalence is claimed if the 90% CI of XGB−LR lies entirely inside the
margin (the usual two-one-sided-tests logic at α = 0.05; our own choice of rule). The CI is the same
participant bootstrap as the official run (same seed; the 95% limits reproduce `06_comparison.csv`
exactly). No new modelling.

**Result.**

| Set | XGB−LR | 95% CI (official) | 90% CI | Within ±0.05? | Within ±0.02? |
|---|---|---|---|---|---|
| P1 | −0.012 | [−0.030, +0.002] | [−0.027, 0.000] | **Yes** | No |
| P2 | −0.019 | [−0.086, +0.047] | [−0.073, +0.036] | No | No |

**Interpretation.** For all features, LR and XGBoost are equivalent within ±0.05 AUC, but this is
partly forced by the ceiling (both are near 1.0, so they cannot differ by much). For pupil only, the
interval is too wide: the data cannot tell "equivalent" from "one is up to 0.07 better". This supports
the panel's point that n = 88 cannot settle whether XGBoost's flexibility helps.

**Suggested sentence.** "Using a pre-stated equivalence margin of ±0.05 AUC, LR and XGBoost were
equivalent for all features (90% CI of the difference −0.027 to 0.000), although both were near the
ceiling; for pupil-only features the interval (−0.073 to +0.036) was too wide to conclude equivalence."

## 7. Activity order and fatigue

**Question.** If people always did A4 last, fatigue or practice could look like load.

**Method.** `annotation.csv` has **no order column**. COLET states the order was random (EM7, as
summarised in `RESEARCH-BRIEF.md`) but does not release it. We **inferred** the order from the first gaze
timestamp of each recording: Pupil Capture timestamps run on one clock per session, so the earlier
recording came first (our own inference). Gaps between consecutive starts: minimum 39 s, median 263 s,
maximum 7410 s (P18, an outlier); there were no ties. Tests (our own choice): (i) per-person z-scored
features minus the activity average across people (to remove the condition effect), Spearman ρ with
position 1–4 per person, then a Wilcoxon signed-rank test of the 44 ρ values against 0, with Holm
correction over 10 features; (ii) Spearman between position and the official P1 out-of-fold probability
within A1 and within A4, and a Mann-Whitney test of the per-person margin p(A4) − p(A1) between people
who did A4 after A1 and people who did A4 before A1.

**Result.** Order varied: position counts per activity (positions 1/2/3/4) were A1 8/6/17/13,
A2 17/7/8/12, A3 10/19/6/9, A4 9/12/13/10. **A4 came after A1 for only 18 of 44 people.**

| Feature drift with position | median ρ | Wilcoxon p | Holm p |
|---|---|---|---|
| fixation_rate | +0.4 | 0.002 | 0.024 |
| gaze_sd_x | −0.4 | 0.006 | 0.053 |
| pupil_sd | −0.2 | 0.018 | 0.144 |
| fixation_duration | −0.3 | 0.026 | 0.184 |
| others (6) | — | ≥ 0.055 | ≥ 0.33 |

| Model output vs position | statistic | p |
|---|---|---|
| LR prob. within A1 / within A4 (Spearman) | −0.03 / −0.09 | 0.84 / 0.55 |
| XGB prob. within A1 / within A4 (Spearman) | +0.03 / −0.20 | 0.82 / 0.19 |
| LR margin, A4 after vs before A1 (Mann-Whitney U) | 230 (medians 0.90 vs 0.94) | 0.93 |
| XGB margin, A4 after vs before A1 (Mann-Whitney U) | 254 (medians 0.91 vs 0.87) | 0.64 |

**Interpretation.** A4 was not systematically last, so fatigue cannot explain A1 vs A4. Fixation rate
rises slightly with time on task (a practice or fatigue drift; the only feature surviving Holm
correction), but the models' predicted probabilities show no link to position. Caveat: the order is
inferred from timestamps, not given by the dataset authors.

**Suggested sentence.** "Activity order, inferred from recording timestamps, varied across participants
(A4 followed A1 for 18 of 44), and neither model's predicted probability was related to order position
(all p ≥ 0.19)."

---

## What could not be done / caveats

- **Order (§7)** is inferred, not released by COLET; P28's order is used for the shared P28/P31 recording.
- **§1 A2+A3 SD** comes from just two recordings and is very noisy; A2-baseline centering is the cleaner check.
- **§§2–4** still use the transductive all-four-activities z-score where marked "z" (as asked); only §1 removes it.
- **§6** margin values are our own choice; no project source gives an AUC equivalence margin.
- No permutation tests, SHAP or importance were run for the extra analyses (not requested; time).
- Multiple comparisons across analyses were not corrected except in §7.

## How to reproduce

Run from `docs/colet-eda/` with the project's `python` (cached features are used; ~6–13 min each for
the modelling scripts on 12 workers). Results go to `outputs/colet/tables/extra/`.

| § | Command | Outputs |
|---|---|---|
| 1 | `python extra_1_baseline_standardization.py` | `1_baseline_standardization_*.csv` |
| 2, 3 | `python extra_2_3_other_contrasts.py` | `2_3_other_contrasts_*.csv` |
| 4 | `python extra_4_windows.py 10` | `4_window_features_10s.csv`, `4_window_counts_10s.csv`, `4_windows_10s_*.csv` |
| 5 | `python extra_5_simple_baselines.py` | `5_simple_baselines_*.csv` |
| 6 | `python extra_6_equivalence.py` | `6_equivalence.csv` |
| 7 | `python extra_7_activity_order.py` | `7_*.csv` |

Shared helper: `docs/colet-eda/extra_common.py`. Console logs: `outputs/colet/tables/extra/log_*.txt`.

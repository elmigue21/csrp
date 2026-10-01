# Re-planned Methodology (proposal for team review)

**Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

- **Status:** proposal only. Nothing has been re-run and no code has been changed.
- **Aim:** every step either follows what the closest literature does, or is flagged as a
  stated design choice.
- **Source IDs** (e.g. `EM5`, `LB4`) refer to entries in `rrl-master-list.md`.

Last updated: 2026-10-01

---

## 0. What changes, at a glance

| Step | Current pipeline | Proposed | Why (key sources) |
|---|---|---|---|
| **Labels** | Self-report, absolute ≥4 or per-person median split | **Designed task difficulty: low (sessions 1–2) vs high (session 5)**. Self-report becomes the manipulation check and a secondary analysis. | 8/12 twins use condition labels; no ML twin median-splits; with the same eye features, condition labels F1 0.69 vs self-report 0.35 (LB4). Also EM5, LB5–LB8 |
| **Normalization** | Per-person z-score over all 5 sessions (transductive) | **Primary: per-person calibration on a low-load session, kept out of the test set.** Transductive z-score and no normalization become sensitivity analyses. | Leak-free (EV2, NM5, LB13); transductive inflates results by 3–13 points (NM5). Transductive is still reported, because it is common practice (EM5, LB7, NM4). |
| **Lux** | Included in the main configuration | **Excluded from the primary model**; with-lux reported as a confound analysis | Shortcut learning (CF7–CF9); lux tracks task |
| **Features** | 37, including duplicates and absolute gaze position | **Literature-backed core set + exploratory set**. Absolute gaze position and duplicates dropped; correlation pruning inside folds. | R3 feature audit (§2 of the list); EM5 excludes gaze coordinates and prunes at \|r\|>0.80; FI13 |
| **Window** | 30 s | **Keep 30 s**, with a pre-stated 10 s / 60 s sensitivity check | EM8, EM9, HR1 (HRC, 30 s); counter-evidence WN3; EM5 does not tune the window |
| **Validation** | LOPO + inner grouped CV | **Keep**; give both models an equal tuning budget | EV3, EV4, EV7, EV8, EM16, EM17 |
| **Metrics** | Pooled ROC-AUC headline | **ROC-AUC + balanced accuracy + macro-F1 + per-class recall + majority baseline + calibration (Brier)**; pooled and per-participant | EM5, M13, M14, EV10 |
| **Statistics** | Wilcoxon | **Wilcoxon + effect size (rank-biserial, bootstrap CI)**; optional Bayesian equivalence test; chance-level permutation test | EV12, M9–M11, FI11 |
| **Importance** | LR coefficients, XGB gain, SHAP | **LR standardized coefficients + XGB TreeSHAP + permutation importance for both**; fold stability; ranking agreement; gain only in an appendix | FI3–FI14; EM5 cross-checks SHAP with univariate tests |
| **Configurations** | 8 configs (A–F plus two no-lux) chosen partly after results | **One pre-specified primary analysis + named sensitivity analyses** | Removes the "chose E after seeing results" critique |

---

## 1. Dataset

**GAZELOAD** (DS0): 26 participants and 5 sessions each, on Meta Aria Gen 1 in an industrial
HRC **lab testbed**.

- **How to describe it:** an "ecologically realistic lab HRC setting", *not* field data. The
  wearable-vs-screen argument uses DS1–DS3.
- **Limitations to state:**
  - it is a preprint;
  - there is no independent validation of Aria gaze accuracy (DS5 is by Meta authors);
  - there is no pupil data.
- **Contribution statement:** "to our knowledge, the first modelling study on GAZELOAD" (R4
  search found no other analyses).
- **Precedents for pupil-free models:** EM17 and EM18, and Božak's fixation-only ablation
  (EM5), which came within about 3 points of the full model.

**Data-quality rules (stated before analysis, justified by our own EDA):**

- Mask physiologically implausible saccades (amplitude beyond about 90°) and invalid gaze
  vectors (`EyeGaze_z < 0`).
- Report Participant 07 (72–97% invalid gaze) as an exclusion, **or** keep them and run a
  sensitivity check without them. The team must choose one before running.
- Drop the task, participant and timestamp identifiers from the features (EV1, EV2).

## 2. Labels (primary change)

**Primary target: designed difficulty, binary low vs high.**

- **Classes:** sessions 1–2 = low, session 5 = high; the medium sessions (3–4) are **left out**
  of the primary analysis.
  - Dropping the middle level follows Hogervorst (LB5).
  - Low-vs-high binarization follows ADABase (LB7) and Božak (EM5).
- **Supporting sources:** LB4 (head-to-head result), LB5–LB8, LB12, and HRC practice (HR1–HR4).

**Self-report ratings (1–10) are used two ways:**

1. **Manipulation check:**
   - A Friedman test of ratings across low / medium / high.
   - The within-person Spearman correlation between rating and designed level.
   - Supporting sources: LB6, LB8, LB12, EM5.
2. **Secondary analysis:** self-report labels as a sensitivity check (per-person median split,
   LB1). Gado found these at chance with eye features (LB4), and LB10/LB11 explain why.
   Report it; do not headline it.

**Optional secondary target:** 3-class low / medium / high (LB7, LB9).

**Consequences to handle:**

- **Classes are imbalanced 2:1** in the primary target. Use balanced accuracy and AUC, and
  report the majority baseline.
- **Possible time-on-task confound:** if session order was fixed (low → medium → high),
  difficulty is confounded with time-on-task and fatigue. **Check the GAZELOAD protocol for
  session order.** If it was fixed, state it as a limitation; no label choice removes it.
- **The within-participant label (D8) and the absolute ≥4 label (D7) drop out** of the primary
  design. Their weak-support problems go with them.

## 3. Per-person normalization (primary change)

**Primary: calibration normalization.**

- **Fit:** for each participant, compute each feature's mean and SD from **one low-load session
  only** (the calibration session).
- **Transform:** z-score (or subtract) all of that participant's other sessions.
- **Keep it out of the test set:** the calibration session is **never used as test data** for
  that participant (NM5, EV2 L1.2, LB13).
- **Framing:** "a short low-load onboarding session", which is deployment-realistic (NM5).
- **Subtraction vs z-score:** NM5 found them equivalent.
- **If the SD from one session is unstable:** use subtraction only, or pool the SD across
  training participants.

**Conflict to resolve (team decision):** calibration uses a low session, and the primary
target needs low sessions for testing.

| Option | Calibration | Test classes per person | Trade-off |
|---|---|---|---|
| **3a (recommended)** | Session 1 | Session 2 (low) vs Session 5 (high), balanced 1:1 | Leak-free and balanced; fewer test epochs per person |
| 3b | Session 1 | Session 2 + medium sessions vs Session 5 | More data, but mixes in medium |
| 3c | None (transductive z-score on all sessions) | Sessions 1–2 vs 5 | Matches Božak/ADABase practice and gives more data; open to the leakage critique |

**Sensitivity analyses to report:**

- **(i) Transductive per-person z-score:** the common practice (EM5, LB7, NM4). Label it as
  transductive and treat it as an **upper bound**.
- **(ii) No normalization.**
- **The gap between them** quantifies how much the choice of normalization matters (NM5, NM9).

**Honest caveats:**

- NM4 (Albuquerque) found baseline normalization **did not beat** the transductive z-score,
  so expect lower scores. Those are the honest estimate.
- The direct evidence for leak-free baselines comes from physiological signals (NM5) and EEG
  (NM4), not gaze-only eye tracking. State this gap.

## 4. Temporal aggregation

**Keep 30 s epochs:** consecutive 250 ms windows are grouped, epochs are cut by elapsed time,
and epochs with less than 80% coverage are discarded.

- **Precedent:** EM8 (30 s best of 15/30/60), EM9, and HR1 (HRC, 30 s intervals).
- **Why aggregate at all:** rates need multi-second windows (ET5). GTE is degenerate at 250 ms.
- **Counter-evidence to acknowledge:** WN3 and EM9 found longer windows sometimes better.
- **State the value as a design choice**, with a **10 s / 60 s sensitivity check**. It is not
  tuned on test results (EM5 practice).

## 5. Features

The grades come from the R3 feature audit in `rrl-master-list.md` §2.

**Core set (literature-backed; used in the primary model):**

| Feature (per epoch) | Construct | Grade |
|---|---|---|
| Blink rate | Blink rate | STRONG |
| SD of gaze direction x/y; SD of scene gaze position x/y | Gaze dispersion / concentration | STRONG |
| Mean saccade amplitude (+ SD) | Saccadic extent | ADEQUATE |
| Fixation rate (fixations/s; currently "fixation_count_mean") | Fixation rate | ADEQUATE |
| Saccade rate (keep **one** of `saccade_count` / `SaccRate`, which are exact duplicates) | Saccade rate | ADEQUATE− |

**Exploratory set (reported separately or as a second feature set):**

- Mean saccade velocity (it largely re-encodes amplitude; ET9–ET11).
- GTE (epoch level; ET17, ET18, EM24).
- FDI (ET19).
- Proportion of windows without a fixation or saccade (a data-quality / rate complement).
- **Optional:** stationary gaze entropy computed per epoch from gaze position, the best HRC
  measure (ET6/HR1).

**Dropped:**

- Mean gaze direction x/y/z and mean gaze position x/y. No workload construct, and a risk of
  learning layout or identity; EM5 excludes absolute coordinates.
- `fdi_missing_frac`, which is identical to `fixation_zero_frac`.
- The duplicate saccade-rate column.
- **Lux:** removed from features and used only in the confound analysis (§9).

**Inside each training fold:** prune correlated features at |r| > 0.80 (EM5, FI13), or group
them for importance (FI10).

**Open data item:** GAZELOAD does not define GTE, FDI or blink detection. Read
`01_Metadata/metrics_extraction.py` in the dataset download and state the definitions.

## 6. Models

| Model | Setup | Sources |
|---|---|---|
| Logistic Regression | Median imputation + missing-indicators + standardization (fold-fitted) + **L2 or elastic-net** penalty, strength tuned | M1, M8, MD1, MD2 |
| XGBoost | Native NaN handling; shallow and regularized (max_depth 2–4, low learning rate, subsample/colsample < 1, λ/α tuned) | M2, M6, MD1 |

- **Equal tuning budget and identical folds** for both models (EV7, M-section note).
- **Secondary check:** an untuned default XGBoost (M7).
- **Expected result, framed in advance:** LR ≈ XGBoost (M3, M4), with M5 as the counterpoint.

## 7. Validation

- **Outer loop:** leave-one-participant-out, 26 folds (EV3, EV4, EM16, EM17). This is the
  strictest scheme used by the twins.
- **Inner loop:** grouped K-fold by participant, for tuning only (EV7, EV8, EV9).
- **All preprocessing is fitted inside the training fold:** imputation, scaling, correlation
  pruning (EV2, LB13).
- **Acknowledge** that LOPO has high variance (EV9).
- **Optional robustness check:** repeated grouped 80/20 splits.

## 8. Metrics and statistics

**Metrics:**

- ROC-AUC (EV11), balanced accuracy, macro-F1, per-class recall, and a **majority-class
  baseline**, following EM5.
- **Calibration:** Brier score and a reliability plot (M13, M14).
- Reported **pooled** and **per participant** (median, IQR, bootstrap CI over participants)
  (EV10, EV13).

**Model comparison:**

- Paired **Wilcoxon signed-rank** over per-participant scores (EV12).
- **Effect size:** rank-biserial correlation plus a participant-bootstrap 95% CI of ΔAUC.
- **Optional:** a Bayesian signed-rank test with a region of practical equivalence of ±0.02 AUC
  (M11). This is what allows the conclusion "practically equivalent".
- **Chance-level test:** grouped label-permutation test for each model (FI11).
- **Limitation:** the Wilcoxon test was designed for independent datasets, and the folds share
  training data (M9, M10).

## 9. Feature importance (title component)

Importance is computed inside each LOPO fold on held-out data, then aggregated.

1. **LR:** standardized coefficients (FI12), reported as the mean ± SD across folds and the
   proportion of folds sharing the sign.
2. **XGBoost:** mean |TreeSHAP| on held-out participants (FI3). Gain goes in an appendix only
   (FI6).
3. **Both models on one scale:** held-out **permutation importance** (the AUC drop, ≥30
   repeats) (FI4, FI5).
4. **Agreement between models:** Kendall τ and top-k overlap. Disagreement is expected (FI5).
5. **Stability:** how often each feature appears in the top-k across the 26 folds, plus error
   bars (FI9, FI14).
6. **Correlated features:** grouped importance for correlated clusters (FI10, FI7, FI8), plus a
   drop-column check on the top features (FI8).
7. **Univariate cross-check:** a Friedman test or mixed model per feature across difficulty
   levels (EM5 practice).
8. **Interpretation rule:** importance means model reliance, not a physiological cause (FI9).

**Lux confound analysis:**

- Re-run the primary model **with** lux added.
- Report the gain in AUC and lux's importance rank as evidence of a contextual shortcut
  (CF7–CF9, CF1, CF2).
- Also show how well lux alone predicts the label.
- This keeps the current lighting finding as a stated result.

## 10. Pre-specified analysis plan

| # | Analysis | Role |
|---|---|---|
| P1 | Condition labels (low vs high) · calibration normalization · core features · no lux · 30 s · LR vs XGB | **Primary** |
| S1 | P1 + exploratory features | Sensitivity |
| S2 | P1 + lux | Confound analysis |
| S3 | P1 with transductive normalization | Sensitivity (upper bound) |
| S4 | P1 with no normalization | Sensitivity |
| S5 | P1 with self-report per-person median labels | Secondary (label comparison, cf. LB4) |
| S6 | P1 at 10 s and 60 s | Window sensitivity |
| S7 | 3-class low / medium / high | Secondary |
| S8 | P1 without Participant 07 (if P07 is kept in P1) | Data-quality sensitivity |

**Rule:** P1 is the headline result regardless of which analysis scores highest.

## 11. Remaining design choices with no citation (state them plainly)

- 30 s as the specific epoch length (precedented, not proven optimal).
- The 80% epoch-coverage threshold.
- Which low session is used for calibration.
- The |r| > 0.80 pruning threshold (taken from EM5).

## 12. Team decisions needed before re-running

1. Approve condition labels as the primary target, with self-report as the check and secondary
   analysis.
2. Choose the normalization option: **3a** (recommended), 3b or 3c.
3. Decide on Participant 07: exclude, or keep with S8.
4. Confirm the GAZELOAD session order (was it fixed?).
5. Approve the core / exploratory feature split.
6. Get the adviser's OK: the results chapter will change substantially.

Only after these are settled do we change `src/` and re-run.

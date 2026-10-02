# Methodology — COLET

**Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

**Status:** working plan (clean version, 2026-10-01). It reflects decisions C1–C3 in
`rrl-decision-log.md`. **All methodology decisions are made (2026-10-01);** only adviser sign-off (C5) is pending. The layered earlier
version is archived at `archive/methodology-colet-v1-layered.md`.

**Related files:**
- `rrl-colet-list.md`: full citations for every source ID used here (e.g. `EM7`, `PP1`).
- `researcher-notes.md`: caveats and justifications the team must be able to explain.
- `review-issues-status.md`: status of the independent review.

---

## 0. Scope rule (decided 2026-10-01)

Only steps required to produce **valid P1 and P2 results** are included:
- **P1:** LR vs XGBoost, A1 vs A4, 10 features.
- **P2:** pupil-only.

Additional analyses suggested by the literature or reviewers are recorded as **optional,
limitations, or future work** (researcher-notes N19, N23).

**Deliberate exclusions:** everything considered and intentionally left out is listed, with
reasons, in **§17 Out of scope**. These are scope decisions, not omissions.

## 1. Study summary

**What we do:**
- We test whether Logistic Regression and XGBoost can tell **low** from **high** cognitive
  load using only eye-tracking features from the COLET dataset.
- Labels come from the experimental design: A1 = low, A4 = high. NASA-RTLX serves as the
  manipulation check.
- Models are evaluated with **leave-one-participant-out** (LOPO) validation and nested,
  participant-grouped tuning.
- **Feature importance** is analysed for both models and compared.
- **Optional (not a stated contribution):** quantifying how much COLET's published accuracies
  depend on its random split (§16).

**Research gap:** among the studies identified in our literature search (search scope:
`rrl-master-list.md` §§3, 4b, 9b; protocol in §12):
- none applied XGBoost or SHAP to COLET;
- none combined an LR-vs-XGBoost comparison with LOPO and SHAP for both models;
- on COLET, the one person-independent evaluation found (Fenoglio 2023a: 4-fold user-grouped
  CV, overlapping windows, RF/neural networks) used neither LR, XGBoost nor any feature
  importance; the other COLET studies read in full used random or trial-level splits.

**Full-text checks done (V2-6, 2026-10-01):**
- **Fenoglio 2023a** (MUM '23, full paper; V): unseen test users, so the earlier claim "all
  COLET studies used random splits" was **reworded** above. No LR, XGBoost or SHAP. Its label
  text says "high (A1 and A2) and low (A3 and A4)", which contradicts its own activity
  descriptions; treat it as a typo and don't cite the labels.
- **Fenoglio 2023b** (MUM '23, 3-page short paper): full text not available; abstract only
  (unsupervised personalisation). Cite only for what the abstract says.
- **Rolon-Merette 2026** (V): own n-back data, not COLET; no feature importance or SHAP for any
  model; "LOPO" was 10 randomly chosen held-out participants, not all 89. Claims 1–2 hold.

**Claims are phrased relative to the studies identified,** not as absolute statements
(review M8). The search protocol is listed in §12.

## 2. Dataset

| Item | Value | Source |
|---|---|---|
| Dataset | COLET (Ktistakis *et al.*, *Comput. Methods Programs Biomed.* 224:106989, 2022), peer-reviewed | EM7 |
| Access | Zenodo record 7766785, v3 (MD5 verified), data licence CC BY 4.0; article CC BY-NC-ND | — |
| Participants | 56 recruited, 9 excluded by the authors (7 vision criteria, 2 poor recordings). **47** released: 26 F / 21 M, age 32 ± 8 | EM7 |
| Ethics | Approved by the FORTH Ethics Committee (110/12-02-2021). Our study is a secondary analysis of public, de-identified data. | EM7 |
| Setting | Lab. Chin and head rest; 24" LCD, 1280×720, 80 cm away. Controlled photopic light (400 lx screen off / 450 lx blank screen). | EM7 |
| Tracker | Pupil Labs Pupil Core, binocular, **gaze stream ≈ 242 Hz** (binocular, measured); **pupil ≈ 121 Hz per eye** (each sample appears twice, as 2d and 3d rows), accuracy 0.60° (reported) | EM7; P0 |
| Task | Visual-search "CAPTCHA" puzzles, 5 images per activity (drawn from 21). Activities in random order, with 2-minute breaks. | EM7 |
| Design | 2 × 2: time pressure (spoken instruction only) × secondary task (counting backwards **aloud** from 1000 by 4) | EM7 |
| Released data | Raw Pupil Core exports per participant × activity: `gaze`, `pupil`, `blinks`, `annotation` (NASA-RTLX), plus `subject_info`. **No fixation/saccade events; no rest baseline.** | Zenodo README |
| Converted copy | `data/colet/parquet/` (564 files); script `data/colet/convert_colet.py`; not in git | researcher-notes N12 |

**The four activities** (medians from our inventory):

| Activity | Condition | Median duration | NASA-RTLX mean | Role |
|---|---|---|---|---|
| A1 | Single task, no time pressure | 37 s | 19.4 | **Low load** |
| A2 | Single task, time pressure | 29 s | 28.7 | Manipulation check only |
| A3 | Multitask, no time pressure | 62 s | 45.0 | Manipulation check only |
| A4 | Multitask, time pressure | 51 s | 51.9 | **High load** |

**Measured data facts:**
- Durations range from 11 to 141 s.
- Invalid gaze samples (confidence < 0.8), median: 0.9% in A1 and 7.3% in A4. Higher in A4
  for **41/45** participants (Wilcoxon p = 4.8×10⁻⁸).
- Blink rate, median: 2/min in A1 vs 15/min in A4.
- All 47/47 participants rated A4 above A1.
- Details: `colet-eda/README.md`; overview page `colet-eda/dataset-overview.html`.

**Limitations of the dataset:**
- It is a screen-based lab task with a chin rest (DS1, DS3).
- The secondary task is spoken (researcher-notes N3).
- Time pressure was weak, by the authors' own judgment (N4).

## 3. Labels and unit of analysis

- **Labels (C2, decided):**
  - **A1 = low, A4 = high** for every participant (condition labels).
  - A2 and A3 are excluded from the primary analysis.
  - Justification: researcher-notes **N0b**, with sources EM7, EM5, LB4–LB9, EM14, CL4, CL6,
    LB10, LB11.
- **Unit (C1, decided):**
  - **One sample per whole activity.** Features are computed over each full recording.
  - There are no windows, because activities are too short (N5). Same unit as the COLET
    paper (EM7).
- **Manipulation check:**
  - Friedman test of NASA-RTLX across A1–A4.
  - Per-person A4 > A1 check (47/47, median +31.7).
  - Sources: EM7, LB6, LB12.
- **Main analysis size after exclusion (C3):** 45 participants × 2 = **90 samples**, balanced
  1:1.

## 4. Preprocessing (raw data → clean signals)

All thresholds were chosen after a **label-blind** data-quality inspection and fixed before any model was run. Source IDs are in `rrl-colet-list.md` §F.

| Step | What we do | Support |
|---|---|---|
| P0. Inventory | Duration, sampling rate measured from timestamps, gaps, eyes available | PP7 |
| P1. Time alignment | **Trim pupil and blink data to each recording's gaze time range.** P18 A3/A4 contain stray pupil samples about 2 hours later. | Data check (review D1) |
| P2. Sample validity | A gaze/pupil sample is invalid if confidence < **0.8** | PP7, PP8, PP5, PP6 |
| P3. Recording exclusion (C3) | Exclude a recording if > **35%** of its gaze samples are invalid. A participant missing A1 or A4 is dropped from the main analysis. Removes P06 A3, P06 A4 and P17 A4, so P06 and P17 drop out. | Nenna 2023 (HR4) |
| P4. Valid time | Rates use **valid recording time**, excluding gaps over 1 s (P18 A3: 12 s; P18 A4: 3.7 s; P16 A4: 1.4 s). Event rates (fixations, saccades) are per second of **time with a valid gaze direction**; blink rate is per minute of valid recording time (researcher-notes N26) | Data check (review D2) |
| P5. Blinks | Use COLET's blink events. Keep 50–500 ms; merge blinks < 100 ms apart; treat blink periods as missing for gaze and pupil. | PP10, PP6, CF2 |
| P6. Gaze gaps | Linearly interpolate gaze gaps < 75 ms; leave longer gaps missing | PP7, PP11 |
| P7. Pupil | `diameter_3d` (mm) from the `3d c++` rows. Range 1.5–9 mm; MAD speed-outlier removal; average both eyes; interpolate gaps ≤ 250 ms; 4 Hz low-pass. **Count 3D-model refits per recording** (reported in P14; V2-3). | PP11, PP9, NM7, CF2 |
| P8. Gaze to degrees | **Per-eye gaze direction** (`gaze_normal0/1`; D-M1 revised by R4-1): average the two eyes; samples with only one valid eye are treated as missing (P6 gap rule; decided 2026-10-02, researcher-notes N25); see §5 | PP17, PP18 (vergence depth unreliable); PP19 (vector-angle velocity); data checks (R4-1, N25) |
| P9. Event detection | See §5 | PP1–PP3, PP12, PP13 |
| P10. Features | §6, computed per whole activity, as **rates** (never counts or duration). Fixation and saccade rates: per second of time with a valid gaze direction; blink rate: per minute of valid recording time (N26) | N5 |
| P11. Quality report | Valid-sample % per recording, reported by condition. It is **not** a feature. | PP7, LB13 |
| P12. Luminance check | Brightness of the 21 stimulus images by condition (**optional**, C8; decided 2026-10-02). COLET's own check: 2 of 47 participants correlated. | CF2, CF3, EM7 |
| P13. Fold-aware steps | Per-person normalization (§7), scaling and imputation, all fitted inside training folds | EV2, LB13 |
| P14. Retention table | Samples and recordings kept at each step, by participant and condition | LB13 |

## 5. Event detection (fixations and saccades)

**Why:** COLET contains only raw gaze positions, so fixations and saccades must be detected
(researcher-notes N1).

**Recipe** (the COLET paper's cited method):
1. Convert gaze to degrees of visual angle.
2. Compute velocity with a five-point smoothed central difference of the unit gaze vectors on a
   uniform 240 Hz grid (**our own choice**; reference check C7 found Duchowski's 5-tap filter is a
   {1,2,3,2,1} smoother, not this differentiator).
3. Classify with **I-VT at 45°/s**, following Salvucci & Goldberg 2000 (PP1). The threshold
   follows COLET, which took it from the human-coded data of Andersson 2017 (PP3; C7: 45.4°/s was
   the coders' minimum peak saccade velocity there, not a general recommendation).
4. Keep fixations of **≥ 55 ms** (COLET and Andersson 2017; Trabulsi 2021, PP13, evaluated
   50–75 ms, so 55 ms lies within its range but is not its recommendation).
5. Reject velocities > 1000°/s (PP8).
6. Sensitivity check: I-DT, 1.0°, 100 ms (PP1, PP12). Optional.
7. **Sanity check (V2-2, decided; tolerances V3-3; per activity R4-2):** computed **for each
   activity (A1–A4) separately**, label-blind. Both must hold in every activity:
   - median fixation duration **150–400 ms**;
   - saccade:fixation count ratio **0.8–1.25**.
   - Pass: use the fixation and saccade features.
   - Fail (either condition, in any activity): P1 runs on the 5 non-event features (pupil
     mean, pupil SD, blink rate, gaze spread x/y).
   - The 0.8–1.25 tolerance is our own choice; no RRL source sets it (§13).
   - Checking per activity is the same kind of quality control as P11 (invalid % by
     condition); it never looks at how well a feature separates A1 from A4.

**Coordinate frame (review M1), checked in the data:**
- `norm_pos` is relative to the **world-camera image**, not the screen.
  - Gaze occupies only about 0.4–0.6 of the image width.
  - Mapping `norm_pos` onto the tracker's 3D gaze direction gives an implied camera field of
    view of about **94° × 53°**, consistent with a wide-angle world camera.
- So converting with the **screen** size (as the earlier plan said) would be **wrong**.
- **The 3D gaze point is also unreliable (round-4 review R4-1):**
  - `gaze_point_3d` depends on a binocular vergence-depth estimate. In COLET it is
    implausible: median depth about **113 mm** against a true viewing distance of **800 mm**,
    and points **behind the camera** (z ≤ 0) in 39 of 188 recordings, which flips the
    direction by about 180° (P37: gaze spread about 72–102°).
  - `norm_pos` is the projection of that same 3D point, so it inherits the error and is not an
    independent cross-check.
  - Reviewer's figures (`review-issues-status-4.md`), reproduced by our own label-blind check
    (researcher-notes N25).
- **Decided (D-M1, revised 2026-10-01 after R4-1):** compute angles from each eye's own gaze
  direction, **`gaze_normal0/1`**, which does not depend on gaze depth. Angle between
  consecutive directions = how far the eye moved. Average the two eyes.
  - **Decided 2026-10-02 (N25):** the two eyes' directions converge by about 25° (expected about
    4–5° at 80 cm), so a single eye is biased by about 10°. Switching to one eye when the other
    drops out would create fake jumps. So samples with only one valid eye are treated as
    missing and handled by the P6 gap rule (median 0.8% of samples per recording, max 19%).
  - The `norm_pos` cross-check and its ±15% agreement rule are **removed** (scope shrinks).
  - Validation is the sanity check (step 7). Main-sequence plausibility (ET10) and COLET's
    Table 4 (about 14° median saccade; conversion undisclosed; gaze spans only about 20°
    horizontally) are compared descriptively only.

**Limitation (final review):** blinks (about 15/min in A4 versus about 2 in A1) and gaps
longer than 75 ms split fixations, which can raise A4's fixation rate and lower its mean
fixation duration by roughly 5-8% (estimate, not measured); there is no blink padding, so
eyelid-edge samples may create more pseudo-saccades in A4. Stated as a limitation, not a
method change (N26).

**First-pass caution:** the exploratory hand-rolled detector's fixation and saccade numbers
must not be used (N1).

## 6. Features

**Decided (C4):** the 10 features marked Core below. The reasons for leaving features out are in
researcher-notes N21. Features were judged on quality and literature support only, never on how
well they separate A1 from A4.

| Feature (whole activity) | Status | Support |
|---|---|---|
| Pupil diameter mean, SD | Core (matches COLET Table 4) | CL7, CF5, EM5 |
| Blink rate (/min of valid time) | Core (matches COLET) | CL7, ET12, ET22 |
| Blink duration | **Dropped (D-M3 decided):** missing whenever there are no blinks (13/45 A1, 0/45 A4), so missingness would reveal the label | EM9 (Hogervorst), EV2 |
| Fixation rate, fixation duration | Core, after the event detection is validated | CL7, ET21 |
| Saccade rate, amplitude | Core, after validation | CL7, ET8, ET20 |
| Saccade peak velocity | **Core (D-N2 decided).** Limitation: the ≈ 242 Hz gaze stream may blunt peak speeds. | EM7, ET9, ET10, CL7 |
| Gaze dispersion x/y | Core; Božak 2026 "fixation dispersion". SD of gaze angle from `gaze_normal0/1` (§5). The 102° artifact came from the 3D gaze point (R4-1), not from fast samples, so the velocity rejection alone did not remove it (V3-4 superseded) | ET14–ET16, EM5; PP8 (rejection) |
| Mean saccade velocity | Dropped (duplicate of peak velocity, ρ = 0.95) | FI13 |
| Saccade duration | Dropped (tied to amplitude, ρ = 0.83) | FI13 |
| Stationary gaze entropy | Exploratory only | ET17, PP14 |

**Rules:**
- No absolute gaze-position features (EM5).
- The feature set is **fixed in advance**, not pruned inside each fold. Every fold then uses
  the same features, which keeps importance comparable across folds (review N1).

**D-M3 (decided 2026-10-01): blink duration dropped.** Background:
- Blink duration is missing in **13 of 45 A1 recordings and 0 of 45 A4 recordings**
  (no blinks).
- A missing-value indicator, or XGBoost's own missing-value handling, would therefore learn
  "missing = low load" directly.
- **Decision:** drop the feature and keep blink rate, following Hogervorst 2014 (which discarded segments with undefined blink duration).

## 7. Per-person normalization

- **Method:**
  - Z-score each feature using that person's own mean and SD over all 4 activities.
  - Label-free; COLET has no baseline recording.
  - Precedent: EM5, LB7, LB1, NM4.
- **Disclosure:**
  - It uses unlabelled data from the test person (transductive).
  - It can raise scores (NM5).
  - It exploits the balanced design (review M4).
- **Decided (D-M4):** per-person standardization is the method. Bias and options are in
  researcher-notes N7, and it is stated as a limitation.
- **No-normalization run:** optional (S4); whether to report it is decided with the scope (D-N5).

## 8. Models

| Model | Configuration | Support |
|---|---|---|
| Logistic Regression | Standardization fitted in-fold; **elastic-net penalty** (D-N6), strength and L1/L2 mix tuned in-fold | M1, M8, X5 (Kaczorowska) |
| XGBoost | Shallow and regularized: **max_depth 1–3, 50–300 trees** (V2-9), learning_rate 0.05–0.1, subsample/colsample 0.7–1.0, λ/α tuned | M2, X14 (Walocha: depth 2), X16, EV7 |

- **Equal tuning budget** and identical folds for both models (EV7).
- **Missing values:** median imputation for LR; native handling for XGBoost (MD1). With blink duration dropped (D-M3), little missingness remains; it is reported per feature.
- **Expected outcome:** LR ≈ XGBoost is a likely and legitimate result (M3, M4, X12; review
  N7). The study tests whether XGBoost adds anything over an interpretable baseline.

## 9. Validation

- **Outer loop:** LOPO over 45 participants (EV3, EV4, EM5, EM16, EM17).
- **Inner loop:** participant-grouped CV for tuning only (EV7, EV8, X14).
- **All preprocessing is fitted inside training folds** (EV2).
- **COLET-protocol replication:** see §16 (optional analysis).

## 10. Metrics, statistics, importance

**Metrics:**
- **Primary:** pooled out-of-fold ROC-AUC with a **participant-level cluster bootstrap CI**
  (review N8; EV10).
- **Also reported:**
  - balanced accuracy and macro-F1;
  - per-class recall;
  - majority baseline (EM5);
  - Brier score (M14);
  - share of participants whose A4 is ranked above their A1.

**Model comparison:**
- **Main:** the difference in pooled AUC with a cluster-bootstrap CI.
- **Paired Wilcoxon** on per-participant scores as a supporting test (EV12). It is an
  adaptation; see M9, M10.
- **Optional:** Bayesian equivalence test (M11).
- **Chance check:** grouped permutation test against chance (FI11). It uses each model's
  **default hyperparameters** (not tuned), so the null and observed runs are treated alike; its
  p-value tests the untuned pipeline against chance. The headline AUC is the nested one (N26);
  `perm_observed_auc` (untuned) is not the headline AUC. With 200 permutations the p-value
  cannot go below 1/(200+1) ≈ 0.005.

**Importance:**
- LR standardized elastic-net coefficients (FI12, X5).
- **SHAP for both models** (D-M8): TreeSHAP for XGBoost (FI3), linear SHAP for LR (FI1), on held-out data. LinearExplainer uses the interventional (feature-independence) convention. Rankings are compared on the same scale.
- **Permutation importance for both models, on pooled out-of-fold predictions** (FI4, FI5;
  review N4).
- Agreement between the two models' rankings (Kendall τ, **with a bootstrap CI or permutation p-value**, as 10 features give a wide interval) and fold stability (FI14). Fold stability is shown by LR sign consistency and coefficient SD, and by XGBoost `folds_used` (per-fold SHAP under LOPO covers only two rows).
- Gain importance in the appendix only (FI6).
- Interpretation: importance means model reliance, not cause (FI9).

## 11. Analysis plan

**Decided:** P1 and P2 are required (D-N5, V2-1); everything else is optional (researcher-notes N19).

| # | Analysis | Role |
|---|---|---|
| P1 | A1 vs A4 · whole activity · core features · per-person normalization · LOPO · LR vs XGBoost | **Primary** |
| **P2** | **Pupil-only model:** A1 vs A4 with pupil mean + pupil SD only; otherwise identical to P1 | **Required (V2-1):** talking-robustness check |
| S1 | Multitask vs single (A1+A2 vs A3+A4) | Optional (kept as an option per D-M5; not required per D-N5) |
| S2 | NASA-RTLX labels using **COLET's bins**: low (0–29) vs high (50–100), medium dropped. **Comparable with COLET's C1/C3 result (GNB 0.88) only under COLET's own protocol** (random split; §16), not under LOPO. Pooled metrics only, since some participants have one class. | Comparability (review R1) |
| S3 | COLET-protocol replication (§16) | **Optional:** inclusion in the paper decided later |
| S4 | No normalization | Optional (D-M4 chose per-person standardization) |
| S5 | Pupil-free model | Luminance check |
| Optional | No-blink model; I-DT detection; confidence 0.6; A1 vs A2 | Robustness |

**Rule:** P1 is the headline result whatever scores highest.

**Ceiling rule (V2-4, pre-stated):** if both models exceed 0.95 AUC in P1, the model
comparison is interpreted mainly from P2. In that case, the feature-importance analysis is
still reported from P1, noting that top-ranked blink and saccade features may partly reflect
talking (researcher-notes N17, N22) (V3-2). Sensitivity results are descriptive.

**P2 limitation (V3-1, decided):** *"The pupil-only model (P2) is not fully independent of the
spoken secondary task: Pupil Core's 3D eye-model refits, which can shift pupil-size estimates,
were more frequent in the multitask activities (researcher-notes N22)."*

## 12. Literature search protocol (for the gap claims)

- **Databases and services:** OpenAlex, Semantic Scholar, Crossref, Europe PMC/PubMed;
  publisher sites (MDPI, Frontiers, IEEE, ACM, Springer, Elsevier).
- **Also used:** citation tracking of COLET (DOI 10.1016/j.cmpb.2022.106989) and GAZELOAD.
- **Keywords:** eye tracking / gaze / pupil × cognitive load / mental workload ×
  classification / machine learning, with logistic regression, XGBoost / gradient boosting,
  leave-one-subject-out / cross-participant, and SHAP / feature importance.
- **Searches run:** September–October 2026.
- **Inclusion:** peer-reviewed; existence confirmed on a publisher or DOI page.
- **Results:** recorded in `rrl-master-list.md` §§3, 4b and 9b, and `rrl-colet-list.md`.

## 13. Design choices without a source for the exact value

- The 35% recording-exclusion threshold. Nenna 2023 used 35% for trials; we apply it to
  recordings.
- The 1 s gap rule for valid time.
- The 50–500 ms blink range (our choice around the typical ~200 ms blink; C7: Steinhauer 2022
  gives no range and notes blinks can exceed 0.5 s).
- Using the extremes A1/A4 as the binary labels (precedented, but a choice).
- The gaze-to-degrees input, `gaze_normal0/1` (D-M1 revised, R4-1): no peer-reviewed study
  names this field. The switch is justified by our data check (N25), supported by evidence that
  vergence depth is unreliable (PP17, PP18) and by the vector-angle velocity method (PP19).
- The sanity-check tolerance: saccade:fixation ratio 0.8–1.25 (V3-3).
- Implementation choices with no published value (set when the pipeline was written):
  - pupil resampling grid of 120 Hz;
  - 2nd-order Butterworth for the 4 Hz pupil low-pass;
  - both-eyes rule for pupil averaging (a sample needs both eyes);
  - blinks merged before the 50–500 ms filter;
  - sanity statistic = median of the recording-level medians;
  - blink gaps ≤ 250 ms are interpolated in pupil only (gaze keeps blinks missing); short finite pupil runs (≤ 30 samples at 120 Hz) are dropped; the MAD is floored.

## 14. Decisions (summary)

All methodology decisions are recorded in `rrl-decision-log.md`:
- C1–C4;
- D-M1–D-M5, D-M7, D-M8;
- D-N2, D-N5, D-N6;
- round-2 to round-4 review decisions (V2-x, V3-1–V3-3, R4-1, R4-2).

**Round 4:** the reviewer proposes freezing the methodology once R4-1, R4-2 and V3-1 are applied
(`review-issues-status-4.md` §5); the next review would be of the pipeline and results, not the
plan. **Frozen by the team on 2026-10-02.** Further changes only if the data shows a Critical
problem when the pipeline runs.

Still pending: **C5, adviser sign-off** (on hold). Also pinned: C7 reference checks. C8 luminance check: optional (2026-10-02).

## 15. Code status

- **Written, not run on real data:** the COLET pipeline in `src/`:
  - `config.py` (settings), `data.py` (loading), `dataset.py` (rows and labels), `preprocess.py`
    (pupil and gaze cleaning), `events.py` (I-VT fixations/saccades), `features.py`, `checks.py`
    (sanity and NASA-RTLX checks);
  - `labels.py`, `normalize.py`, `models.py`, `evaluate.py`, `importance.py`;
  - `run_colet.py` (entry point: `python src/run_colet.py`).
- **Colab:** `notebooks/colet_pipeline.ipynb` with `requirements-colab.txt` (pinned versions).
- **Exploration:** `docs/colet-eda/eda_colet.py` (first pass only).

## Appendix A. RRL support matrix (regraded 2026-10-01)

**Grade definitions:**
- **STRONG** = 2+ claim-matched sources, at least one **read in full** and at least one from
  eye-tracking workload research.
- **ADEQUATE** = support exists but is indirect (another modality), rests on one source, or
  rests mainly on abstract/metadata-level reading.
- **DESIGN CHOICE** = no source sets the exact value (§13).

Rows marked ↓ were regraded from STRONG after the review (R5).

| # | Decision | Key sources | Grade |
|---|---|---|---|
| 1 | Compare LR vs XGBoost | X1, X2, X12, X13 (V); M3, M4, M5 | STRONG |
| 2 | COLET as dataset | EM7 (V); 5 reuse studies | STRONG |
| 3 | Eye-tracking-only features | CL7 (V), EM5 (V), EM17, EM18 | STRONG |
| 4 | Condition labels A1 = low, A4 = high | EM7 (V), LB4 (V), LB5 (V), LB7 (V), EM5 (V), EM14 (V) | STRONG |
| 5 | Drop the middle conditions | LB5 (V), EM14 (V), LB1 (V), EM7 (V) | STRONG |
| 6 | NASA-RTLX as manipulation check | EM7 (V), LB6 (V), LB8 (V); CL8, CL9 (M) for RTLX | STRONG |
| 7 | Whole-activity unit | EM7 (V), EM2 (V), EM14 (V) | STRONG |
| 8 | Event detection: I-VT 45°/s primary, I-DT sensitivity | EM7 (V), PP1 (V), PP3 (V), PP13 (V) | STRONG |
| 8a | Gaze-to-degrees conversion (`gaze_normal0/1`, R4-1) | PP17 (A), PP18 (A), PP19 (V); data checks (N13, N25) | DESIGN CHOICE, supported (no study names the field) |
| 8b | Confidence ≥ 0.8, gaps, blinks | PP7 (V), PP6 (V), PP10 (V), CF2 (V); PP8 (partial) | STRONG |
| 8c | Pupil preprocessing | PP11 (V), PP9 (V), NM7 (A), CF2 (V) | STRONG |
| 9 | Recording exclusion > 35% | HR4 (V) | ADEQUATE (one precedent, applied to recordings) |
| 10 | Pupil features | CL7 (V), CF5 (V), EM5 (V) | STRONG |
| 11 | Blink features | CL7 (V), ET12 (A), ET22 (A) | STRONG |
| 12 ↓ | Gaze dispersion | ET14 (A), ET15 (A), ET16 (A) | ADEQUATE |
| 13 | Fixation/saccade features | CL7 (V), ET8–ET10 (A), ET20 (A), ET21 (A) | ADEQUATE |
| 14 | Fixed feature set; no absolute gaze position | EM5 (V); FI7 (V), FI13 (A) | STRONG |
| 15 | Per-person normalization (no-normalization run optional, S4) | EM5 (V), LB7 (V), NM4 (V), NM5 (V) | STRONG |
| 16 | Missing-value handling | MD1 (V), MD2 (A) | ADEQUATE (see D-M3) |
| 17 | LR penalty; XGB shallow and regularized | X2 (V), X14 (V), M2 (V), M8 (A) | STRONG |
| 18 | Equal budget, nested grouped tuning | X14 (V), EV7 (V), EV8 (V) | STRONG |
| 19 | LOPO outer loop | EV3 (V), EV4 (V), EV9 (V), EM5 (V), EM17 (V) | STRONG |
| 20 | Preprocessing fitted inside folds | EV2 (V) | ADEQUATE |
| 21 | COLET-protocol replication + ablation | EM7 (V), EV3 (V), EV2 (V) | STRONG |
| 22 | Pooled AUC + cluster bootstrap; majority baseline | EV10 (A), EV11 (M), EM5 (V) | ADEQUATE |
| 23 | Calibration (Brier) | M13 (V), M14 (A) | ADEQUATE |
| 24 ↓ | Wilcoxon over participants | EV12 (M), M9 (A), M10 (M) | ADEQUATE |
| 25 | LR coefficients as importance | X5 (V), X13 (V), FI12 (A) | STRONG |
| 26 | TreeSHAP for XGB | FI3 (V), X7 (V) | STRONG |
| 27 ↓ | Permutation importance | FI4 (A), FI5 (A), FI11 (A) | ADEQUATE |
| 28 | Gain only in the appendix | FI3 (V), FI6 (V) | STRONG |
| 29 | Ranking agreement, stability | FI9 (V), FI14 (A), X5 (V) | ADEQUATE |

**Evidence codes:** V = full text read; A = abstract only; M = metadata only.

**To upgrade the ADEQUATE rows:** read the full texts of the A/M sources. This is part of the
pinned reference checks (C7).

## 16. Optional analysis: where COLET's 85% came from (D-M7)

**Status:** approved to run as an **optional** analysis. **Whether it appears in the paper is
decided later.**

**Purpose:**
- COLET reported LR 0.85 on A1 vs A4 (k-NN 0.86) using a 20% hold-out and 5-fold CV that are not
  described as participant-independent (C7: the paper never says "random"), feature selection on
  all the data, and scaling before the split.
- Our leave-one-participant-out result will likely be lower.
- This analysis shows whether the gap comes from **the testing method** rather than the
  models.

**Plan (same 90 samples, same features, same LR and XGBoost):**

| | Run 1: COLET's method | Run 2: our method (= main result P1) |
|---|---|---|
| Split | Random 80/20; the same person can be in training and test | Leave-one-participant-out |
| Feature selection | ANOVA on all data, test included | Fixed feature set in advance |
| Scaling | Before splitting | Inside training folds only |

**Further option, the ladder:** fix one choice at a time:
1. all of COLET's choices;
2. + LOPO split;
3. + feature selection on training data only;
4. + in-fold scaling (= our method).

**How to read it:**

| Result | Meaning |
|---|---|
| Run 1 ≈ 0.85, Run 2 lower | The gap is due to leakage in the testing method |
| Both similar and low | Our features differ from COLET's; discuss |
| Both similar and high | Performance holds for new people |

**Support:** Saeb 2017 (EV3); Kapoor & Narayanan 2023 (EV2); COLET paper (EM7).

## 17. Out of scope (concluded)

Everything below was considered and deliberately left out, under the scope rule (§0). Each can
be named as a limitation or future work.

| Item | Why it is out | Recorded in |
|---|---|---|
| **Cross-dataset generalization** (training on one dataset, testing on another, e.g. COLET + GAZELOAD) | Rejected by the panel; the revised title has no cross-dataset component | decision log (scope) |
| **GAZELOAD dataset** | Replaced by COLET: preprint, no pupil, no raw gaze, no reuse | decision log; `archive/` |
| **ADABase dataset** | On hold (needs a signed licence); email draft ready | C6 |
| **Models other than LR and XGBoost** (SVM, random forest, deep learning) | The title compares LR vs XGBoost only | title |
| **Real-time or deployed detection; designing a calibration procedure** | Offline, analytical study; per-person standardization assumes calibration and is stated as a limitation | N7 |
| **Non-eye signals** (EEG, heart rate, skin conductance) | COLET is eye-tracking only, and the title says "eye tracking features" | title |
| **Time windows** (30 s, 10 s) | Activities are 11–141 s, so one sample per whole activity (C1) | C1, N5 |
| **Trial-level (per-puzzle) analysis** | COLET's authors: trials too short; NASA-RTLX is per activity only | N5 |
| **Self-report labels as the main target** | Condition labels A1 = low / A4 = high are used (C2); NASA bins only as optional S2 | C2, N0b |
| **A2 and A3 in the main analysis** | Middle conditions left out ("drop the middle") | C2, N0b |
| **Statistically controlling for talking** (beyond P2) | Follow the RRL (D-M2); P2 (pupil-only) is the one required check | D-M2, V2-1, N3 |
| **Rest-baseline calibration** | COLET has no rest recording | N7 |
| **Features:** blink duration, mean saccade velocity, saccade duration, COLET's variation/skewness/kurtosis | Missingness leaks the label; duplicates; too many features for 90 samples | C4, N21 |
| **Gaze entropy in the main model** | Mixed evidence; exploratory only | C4, N21 |
| **Per-segment pupil handling for 3D-model refits** | Speed filter + refit count + limitation instead | V2-3, N23 |
| **A second gaze-conversion method as a cross-check** and a **formal main-sequence slope test** | Lean sanity check instead (fixation 150–400 ms; saccade:fixation 0.8–1.25, per activity). `gaze_normal0/1` is now the *primary* input (R4-1), not an extra | V2-2, V3-3, R4-1, N23 |
| **Refit-free P2 run** (P2 on participants with no refits) | Limitation sentence instead (§11) | V3-1, N24 |
| **Deep or large XGBoost settings** | Depth 1–3, 50–300 trees for 88 training samples | V2-9, N23 |
| **Stimulus-luminance correction** | COLET's own check (2/47 correlated) is cited; our own brightness check is optional (C8, decided 2026-10-02) | N10, C8 |
| **Required COLET replication** ("showing COLET's inflation" as a contribution) | Kept optional (§16); removed from stated contributions | V2-5, D-M7, N23 |
| **Required no-standardization run** | Kept optional; answer with Tognotti 2026 and N7 if asked | V2-8, D-M4, N23 |
| **Optional analyses** S1 (multitask), S2 (NASA bins), S3 (replication), S4 (no standardization), S5 (pupil-free), and robustness runs | Not required for P1/P2; decide later | D-N5, N19 |

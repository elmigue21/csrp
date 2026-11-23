# CSRP Thesis — Complete Research Brief (paste into any AI agent)

> **For the agent reading this:** you have no access to our files or code. Everything you need
> is below. Treat it as the current, authoritative state of our thesis as of **2026-11-06**.
> If something isn't covered here, say so rather than guessing. Rules for working with us are
> in §13. Full IEEE references are in Part 2 (pasted separately, if provided).

---

## 1. The project in one screen

- **Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
  DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS
  (a reviewer recommends retitling around "workload conditions" — R5-30, awaiting adviser).
- **Team:** Caacbay, Magkasi, Uaje, Ventura (BS Computer Science).
- **Question:** using only eye-tracking features, can Logistic Regression (LR) and XGBoost tell
  low from high cognitive load, does XGBoost beat the interpretable LR baseline, and which
  features do the models rely on?
- **Dataset:** COLET (Ktistakis *et al.*, *Comput. Methods Programs Biomed.* 224:106989, 2022;
  Zenodo 7766785, CC BY 4.0).
- **Labels:** A1 (single task, no time pressure) = **low**; A4 (counting backwards aloud +
  time pressure) = **high**. One sample per whole activity.
- **Final sample:** **44 participants × 2 = 88 samples**, balanced.
- **Validation:** leave-one-participant-out (LOPO), nested participant-grouped tuning.
- **Headline result (Route A, per-person standardized):** both models near-perfect
  (LR AUC 0.992, XGBoost 0.980); **no significant difference** between them. Pupil-only model
  (P2): LR 0.831, XGBoost 0.812, again no significant difference.
- **One-line conclusion:** XGBoost brought no clear benefit over LR; LR was equal or better.
  Top features for both models: blink rate, pupil mean, fixation rate.

## 2. Timeline / status

| Date | What happened |
|---|---|
| Earlier | Original plan used GAZELOAD (+ COLET cross-dataset). Panel rejected cross-dataset; GAZELOAD dropped (arXiv preprint, no pupil, no raw gaze). |
| 2026-10-01 | Switched to COLET. All methodology decisions made (C1–C4, D-M*, D-N*). 4 independent review rounds done. |
| 2026-10-02 | Methodology **frozen**. Pipeline (`src/`) + Colab notebook written and **run**. Optional runs: no-blink (N29), no-standardization "Route B" (N30). Reference check C7 done (N28). Mock-defense review → adviser questions Q1, Q2. |
| 2026-10-05 | Leakage audit: found P28 and P31 have identical features. |
| 2026-10-15 | Review round 5. Confirmed P28/P31 share one eye recording in COLET v3 itself. Kept once → **44 participants**; everything rerun (N31). Draft **v4** of Chapters 1–5 has new numbers + approved wording as tracked changes. |
| 2026-11-03 | **Adviser approved Route A** (per-person standardization) as the main analysis. |
| Now | Presentation deck + live demo script exist. Open: whether to report Route B as secondary; title change; several optional robustness analyses; reference verification. |

## 3. Dataset (COLET)

- 56 recruited, 9 excluded by authors → **47 released** (26 F / 21 M, age 32 ± 8). Ethics:
  FORTH Ethics Committee 110/12-02-2021. Ours is secondary analysis of public de-identified data.
- Lab setting, chin + head rest, 24" LCD 1280×720 at 80 cm, controlled lighting.
- Tracker: Pupil Labs Pupil Core, binocular. Gaze stream ≈ 242 Hz (measured); pupil ≈ 121 Hz
  per eye (each sample appears twice as `2d c++` and `3d c++` rows).
- Task: visual-search "CAPTCHA" puzzles, 5 images per activity, random activity order,
  2-min breaks. 2×2 design: time pressure (spoken instruction only) × secondary task
  (counting backwards **aloud** from 1000 by 4).
- Released: raw gaze, pupil, blinks per participant × activity; NASA-RTLX; demographics.
  **No fixation/saccade events, no rest baseline.**

| Activity | Condition | Median duration | NASA-RTLX mean | Role |
|---|---|---|---|---|
| A1 | Single task, no time pressure | 37 s | 19.4 | **Low** |
| A2 | Single task, time pressure | 29 s | 28.7 | Standardization only |
| A3 | Multitask, no time pressure | 62 s | 45.0 | Standardization only |
| A4 | Multitask, time pressure | 51 s | 51.9 | **High** |

- Durations 11–141 s. Invalid gaze (confidence < 0.8) median 0.9% in A1 vs 7.3% in A4 (higher
  in A4 for 41/45, Wilcoxon p = 4.8×10⁻⁸). Blink rate median 2/min A1 vs 15/min A4.
- **Data errors found by us:**
  - P18 A3/A4 contain stray pupil samples ~2 h later → trimmed to gaze time range.
  - `gaze_point_3d` is broken (median depth 113 mm vs true 800 mm; points behind camera in 39/188
    recordings) → not used. `norm_pos` is in world-camera coordinates (not screen) and inherits
    the same error → not used.
  - **P28 and P31: all 12 raw files byte-identical**, but different demographics (age 23 vs 26)
    and NASA-RTLX. Two people share one recording in COLET v3. Kept once (as P28); both excluded
    from the RTLX check.

## 4. Method (frozen 2026-10-02)

**Scope rule:** only what is needed for valid **P1** (10 features, LR vs XGBoost, A1 vs A4) and
**P2** (pupil-only) results. Everything else is optional, a limitation, or future work.

### 4.1 Exclusion
- Sample invalid if confidence < 0.8. Recording excluded if > 35% of gaze samples invalid
  (Nenna 2023 used 35% for trials; we apply to recordings).
- Removed: P06 A3 (57%), P17 A4 (49%), P06 A4 (39%) → P06, P17 dropped (missing A1 or A4).
- Plus P31 duplicate → **44 participants**. Note: all excluded recordings are counting activities (R5-5).

### 4.2 Preprocessing (P0–P14)
1. Inventory: measured sampling rate, gaps, eyes available.
2. Trim pupil/blinks to gaze time range (P18 fix).
3. Validity: confidence ≥ 0.8.
4. Valid time: rates use time excluding gaps > 1 s. Fixation/saccade rates per second of valid
   gaze-direction time; blink rate per minute of valid recording time.
5. Blinks: COLET's blink events; keep 50–500 ms (our choice); merge < 100 ms apart; blink periods
   = missing for gaze and pupil.
6. Gaze gaps < 75 ms linearly interpolated; longer left missing.
7. Pupil: `diameter_3d` (mm) from 3d rows; 1.5–9 mm range; MAD speed-outlier removal (Kret &
   Sjak-Shie 2019); average both eyes (both required); interpolate gaps ≤ 250 ms; 4 Hz low-pass
   (2nd-order Butterworth, our choice), 120 Hz grid. Count 3D-model refits per recording.
8. **Gaze → angles:** per-eye gaze direction `gaze_normal0/1`, averaged across eyes. One-eye
   samples treated as missing (eyes converge ~25° vs expected 4–5°, single eye biased ~10°).
   No study names this field — design choice backed by data checks and vergence-unreliability
   papers (Hooge 2019, Velisar & Shanidze 2024; vector-angle velocity: Kothari 2020).
9. **Event detection:** velocity = 5-point smoothed central difference of unit vectors on a
   240 Hz grid (our choice); **I-VT 45°/s** (Salvucci & Goldberg 2000; threshold from COLET via
   Andersson 2017); min fixation **55 ms**; reject > 1000°/s.
10. **Sanity check (per activity, label-blind):** median fixation 150–400 ms AND
    saccade:fixation ratio 0.8–1.25 in every activity, else fall back to 5 non-event features.
    **Result: passed** (fixation medians 212–235 ms; ratios 0.93–0.99).
11. Features computed per whole activity as rates.
12. Quality report (not a feature); retention table.

### 4.3 The 10 features (fixed in advance, never selected by label separation)
pupil mean, pupil SD, blink rate, fixation rate, fixation duration, saccade rate, saccade
amplitude, saccade peak velocity, gaze spread x, gaze spread y.

**Dropped:** blink duration (missing in 13/45 A1 vs 0/45 A4 → missingness reveals label);
mean saccade velocity (ρ = 0.99 with peak velocity); saccade duration (ρ = 0.80 with amplitude);
COLET's variance/skew/kurtosis stats (too many features for 88 samples). Gaze entropy:
exploratory only. No absolute gaze position. Missingness in final features: **zero**.

### 4.4 Per-person standardization (D-M4; adviser-approved = "Route A")
- Z-score each feature with that person's own mean/SD over **all 4 activities** (A1–A4).
  Label-free. COLET has no rest baseline.
- **Disclosed bias:** it uses the test person's own unlabelled recordings (transductive;
  assumes a calibration session); it exploits the balanced design. Tognotti 2026 reports
  +3–13 points from this kind of normalization. Precedents: Božak 2026, Oppelt 2023, Gado 2023,
  Albuquerque 2022.

### 4.5 Models
- **LR:** elastic net (Zou & Hastie 2005), C and L1 ratio tuned in-fold; median imputation +
  scaling inside the pipeline.
- **XGBoost:** max_depth 1–3, 50–300 trees, learning rate 0.05–0.1, subsample/colsample
  0.7–1.0, λ/α tuned (Walocha 2025: depth 2 most chosen).
- Same folds, **equal number of candidate settings** (30 random draws each) — wording per
  R5-21; not claimed as an "equal budget".
- Expected outcome stated in advance: LR ≈ XGBoost is a legitimate result.

### 4.6 Validation & metrics
- Outer: LOPO over 44 participants. Inner: participant-grouped CV for tuning only.
- Primary metric: **pooled out-of-fold ROC-AUC** with participant-level cluster bootstrap 95% CI
  (2000 resamples). Also: balanced accuracy, macro-F1, per-class recall, majority baseline
  (0.5), Brier score, share of people whose A4 is ranked above their A1.
- Model comparison: AUC difference (XGB − LR) with cluster-bootstrap CI; paired Wilcoxon on
  per-participant margins as supporting test.
- Shuffled-label check: grouped permutation test (200 permutations, untuned default
  hyperparameters; min p ≈ 0.005).
- **Ceiling rule (pre-stated):** if both models > 0.95 AUC in P1, interpret the model
  comparison mainly from P2; feature importance still from P1. **It triggered.**

### 4.7 Feature importance
- LR standardized elastic-net coefficients.
- **SHAP for both models** (TreeSHAP for XGBoost, linear SHAP for LR, on held-out data).
- Permutation importance on pooled out-of-fold predictions.
- Ranking agreement: Kendall τ with bootstrap CI, plus top-3 overlap.
- Gain importance in appendix only. Importance = model reliance, not causation.

### 4.8 Software
Python 3.11.9, numpy 2.2.4, pandas 2.3.1, scipy 1.15.2, scikit-learn 1.7.1, xgboost 3.2.0,
shap 0.51.0. Runtime ~270 s. Run metadata records git commit and SHA-256 of `src/` and every
input file. Colab notebook clones repo from GitHub, data on Google Drive.

## 5. Results (current: 44 participants, rerun 2026-10-15)

### 5.1 Manipulation check
NASA-RTLX (n = 43): Friedman χ²(3) = 82.72, p = 8×10⁻¹⁸. **All 43 rated A4 above A1**
(median +31.7 points).

### 5.2 Main results — Route A (standardized, approved main analysis)

| Analysis | Model | ROC-AUC (95% CI) | Bal. acc. | Macro-F1 | Brier | A4 > A1 | Shuffle p |
|---|---|---|---|---|---|---|---|
| **P1** (10 feat.) | LR | **0.992** (0.979–1.000) | 0.966 | 0.966 | 0.032 | 44/44 | 0.005 |
| **P1** | XGBoost | **0.980** (0.957–0.997) | 0.920 | 0.920 | 0.060 | 44/44 | 0.005 |
| **P2** (pupil only) | LR | **0.831** (0.735–0.912) | 0.807 | 0.807 | 0.176 | 37/44 | 0.005 |
| **P2** | XGBoost | **0.812** (0.697–0.912) | 0.773 | 0.772 | 0.175 | 37/44 | 0.005 |

Majority baseline balanced accuracy = 0.50.

**Model comparison (XGB − LR):**

| | AUC diff (95% CI) | Per-person margins | Wilcoxon p |
|---|---|---|---|
| P1 | −0.012 (−0.030 to 0.002) | XGB larger in 15/44 | 0.112 |
| P2 | −0.019 (−0.086 to 0.047) | XGB larger in 29/44 (medians 0.474 vs 0.214) | 0.007 |

→ No significant pooled-AUC difference. In P2, pooled AUC slightly favors LR while
within-person margins favor XGBoost — report both, don't switch metrics.

**Tuned hyperparameters were unstable** across folds: P1 LR median C 4.97; XGBoost chose depth
2 in 20/44 folds, depth 1 in 16; P2 XGBoost used depth-1 stumps in 37/44 (so P2 cannot test
XGBoost's interaction advantage). Hence the wording "LR at least matches a lightly tuned XGBoost".

### 5.3 Feature importance (P1, mean |SHAP|, rank)

| Rank | LR | XGBoost |
|---|---|---|
| 1 | blink rate (5.07) | blink rate (2.55) |
| 2 | pupil mean (3.38) | pupil mean (1.20) |
| 3 | fixation rate (1.91) | fixation rate (0.66) |
| 4 | gaze SD x | saccade rate |
| 5 | gaze SD y | gaze SD x |
| 6 | saccade rate | fixation duration |
| 7 | saccade peak velocity | gaze SD y |
| 8 | saccade amplitude | saccade amplitude |
| 9 | pupil SD | saccade peak velocity |
| 10 | fixation duration | pupil SD |

Kendall τ = **0.69** (0.60–0.73); top-3 overlap **3/3**. P2: pupil mean ≫ pupil SD for both.

**Single-feature AUCs (no model):** blink rate 0.97 standardized (0.89 raw); pupil mean 0.87
(0.62 raw). 42/44 blinked more in A4.

### 5.4 Optional robustness analyses (all rerun at 44)

| ROC-AUC (95% CI) | LR | XGBoost | XGB − LR (95% CI) |
|---|---|---|---|
| P1 **without blink rate** (9 feat., N29) | 0.929 (0.863–0.980) | 0.886 (0.803–0.956) | **−0.043 (−0.084 to −0.007)** ← LR better, only CI excluding 0 |
| **Route B** P1 unstandardized (N30) | 0.853 (0.783–0.923) | 0.881 (0.822–0.941) | +0.028 (−0.034 to 0.094) |
| **Route B** P2 unstandardized | 0.611 (0.573–0.657) | 0.649 (0.565–0.732) | +0.038 (−0.060 to 0.129) |

- No-blink: LR margins larger in 30/44 (p = 0.003). Without blinks, LR leaned on fixation
  rate/duration and pupil; XGBoost on pupil, saccade rate, fixation rate.
- Route B: within-person ranking still high (A4 > A1 for 42/44 both models), so the drop is
  between-person variation. Raw pupil-only is weak → the pupil signal is **relative to each
  person's own baseline**. Route B balanced accuracy P1: 0.739 / 0.818. Route B Kendall τ 0.24.
- **Direction flips with protocol:** point estimates favor LR with standardization, XGBoost
  without — neither significant.

### 5.5 Leakage checks
- **Shuffled labels** (40 repeats, untuned): within-person shuffle LR 0.49 / XGB 0.53; fully
  random 0.46 / 0.53 → chance. Rules out *label* leakage in the untuned procedure only — not a
  "no leakage anywhere" certificate (R5-24: section renamed "Shuffled-label check").
- Verified: all outer/inner splits participant-disjoint; tuning sees only outer-training rows;
  imputation/scaling in-pipeline; participant ID, activity, RTLX, refits, QC not model inputs.
- **Known, disclosed:** (1) standardization uses test person's unlabelled recordings;
  (2) the sanity check / feature-set decision used all participants (unsupervised QC, disclosed
  as limitation, R5-4); (3) bootstrap CIs resample fixed predictions (ignore refit variance);
  (4) whole-recording filters use future samples (fine offline, not real-time).

## 6. Why P1 is near-perfect (defense answer)

1. **Talking:** A4 includes counting aloud, which raises blinking ~7× → blink rate alone gives
   AUC 0.97. P2 (pupil-only) shows signal beyond blinks.
2. **Per-person standardization** removes individual differences (pupil 0.62 → 0.87).
3. **Whole-activity features** and extreme conditions (A1 vs A4).

> Panel line: "Near-perfect discrimination reflects a large condition effect, largely from the
> spoken secondary task, and per-person standardization. A shuffled-label control fell to chance,
> ruling out label leakage. The pupil-only model shows a load signal beyond blink-rate input,
> though speech-related effects cannot be ruled out and the signal depends on calibration."

**COLET comparison:** COLET reported LR 0.85 (k-NN 0.86) for A1 vs A4 with a random 80/20 split
+ 5-fold CV not described as participant-independent, feature selection on all data, scaling
before split. We expected lower under LOPO but got higher — due to standardization and
whole-activity features, not participant overlap. (COLET's GNB 0.88 is RTLX classes C1 vs C3,
not A1 vs A4. COLET RTLX = overall score, not Mental Demand.)

## 7. Limitations (as stated / to state)

- A1 vs A4 bundles workload with **speech, counting, time pressure, duration and tracking
  quality** → frame target as "workload condition" classification (R5-9, required).
- P2 is not fully talking-free: 3D pupil-model refits were more frequent in multitask activities.
- Per-person standardization assumes calibration; uses test person's unlabelled data.
- Time pressure was weak (COLET authors' own judgment). Lab, chin-rest, screen task.
- Small sample (44 people). Many thresholds are our own choices (35%, 1 s, 50–500 ms blinks,
  0.8–1.25 ratio, velocity filter, gaze_normal).
- Blinks/gaps > 75 ms can split fixations; fragments < 55 ms are discarded, so fixation counts
  can move **either** direction (R5-38). No blink padding.
- Saccade peak velocity at ~242 Hz may be blunted.
- Recording length differs (A4 longer) — may affect how stable estimates are (R5-41).
- Permutation importance shuffles across participants → model reliance, not unique/causal
  contribution. SHAP explainers differ between models → rankings descriptive.
- "No evidence of a difference" — never "equivalent" (no equivalence margin).
- Offline whole-activity analysis; does not establish real-time or cold-start performance.

**Approved "What this study can and cannot claim" paragraph (in v4):**
> This study evaluates whether eye-tracking features distinguish COLET's A1 and A4 task
> conditions under participant-disjoint training. The within-person protocol uses each
> participant's recordings from all four activities for standardization. Results do not
> establish real-time performance, cold-start performance under that protocol, or
> speech-independent detection of cognitive load. Model rankings differed in direction across
> the standardized and raw-feature analyses, and neither protocol established a clear AUC
> advantage for either model.

## 8. Research gap (phrased relative to studies we found)

Searched OpenAlex, Semantic Scholar, Crossref, Europe PMC/PubMed, publisher sites; citation
tracking of COLET; Sept–Oct 2026. Among studies identified:
- none applied XGBoost or SHAP to COLET;
- none combined LR-vs-XGBoost with LOPO and SHAP for both models;
- the one person-independent COLET evaluation (Fenoglio 2023a: 4-fold user-grouped CV,
  overlapping windows, RF/neural nets) used no LR, XGBoost or feature importance; the other
  COLET studies read in full (e.g. Fuhl 2023, COLET itself) used random or trial-level splits.
- Closest twins: Rolon-Merette 2026 (LR vs XGB, own n-back data, 10 held-out people, no
  importance); Nerella 2026 (LR vs XGB, AR); Božak 2026 (feature-based ML + SHAP, eye tracking).

## 9. Decision register (all ✅ unless noted)

| ID | Decision |
|---|---|
| C1 | Unit = whole activity (activities too short for windows) |
| C2 | Labels A1 = low, A4 = high; A2/A3 dropped from classification ("drop the middle": Hogervorst 2014, Wu 2020; COLET) |
| C2b / D-M2 | Talking confound: follow RRL (COLET, Pluchino 2023) — keep blinks, no special control, P2 as check, limitation sentence |
| C3 | > 35% invalid → exclude recording; missing A1/A4 → drop person |
| C4 | The 10 features (§4.3) |
| D-M1 | Gaze angles from `gaze_normal0/1` (revised after round-4 review R4-1); one-eye = missing |
| D-M3 | Drop blink duration (missingness leaks label) |
| D-M4 | Per-person standardization — **adviser approved 2026-11-03** |
| D-M5 | Keep A1 vs A4 (P1) and single vs multitask (S1, optional) |
| D-M7 | COLET-protocol replication: optional, paper inclusion TBD |
| D-M8 | SHAP for both models |
| D-N2 | Keep saccade peak velocity (with sampling-rate limitation) |
| D-N5 | Only P1 (+ P2 via V2-1) required; rest optional |
| D-N6 | LR = elastic net |
| V2-1 | P2 pupil-only run required (talking robustness) |
| V2-2/V3-3/R4-2 | Event sanity check per activity, tolerances above |
| V2-3 / V3-1 | Count pupil refits; limitation only |
| V2-4 / V3-2 | Ceiling rule; importance still from P1 |
| V2-9 | XGB depth 1–3, 50–300 trees |
| SCOPE | Only what P1/P2 need |
| C5 | Adviser sign-off on COLET switch — ⏸ on hold (Route A approval implies working acceptance) |
| C6 | ADABase access email — ⏸ on hold |
| C7 | Reference checks — done for method sources (N28); Tao *et al.* + 2026 refs still to verify (R5-36) |
| C8 | Stimulus-luminance check — optional (cite COLET's own: 2/47 correlated) |

**Reference-check corrections (C7) to remember:** Duchowski's 5-tap filter is a smoother → our
velocity filter is our own choice. Andersson 2017's 45.4°/s was coders' minimum peak saccade
velocity → credited as COLET's source. Trabulsi 2021 evaluated 50–75 ms (not 55). Steinhauer 2022
gives no blink range → 50–500 ms ours. Tao 2019: eye measures most *widely used*, not most often
significant. Bentivoglio 1997 supports only "talking raises blink rate". Hogervorst 2014 does not
support the blink-duration missingness argument (use Kapoor & Narayanan 2023). Kaczorowska 2021
used repeated random splits. Permutation-vs-chance test: cite Ojala & Garriga 2010.

## 10. Open items

**Waiting on adviser / team decision:**
- Report Route B (unstandardized) as a secondary analysis? (Q2 Option 1, team recommends yes;
  also answers Q1.) Reviewer R5-2 calls side-by-side reporting **required**.
- R5-30: retitle around "LR vs XGBoost on workload conditions" (adviser).
- R5-9: reframe target as "workload condition" throughout (required framing; v4 defines it once
  but doesn't rename throughout — R5-33).

**Recommended analyses not yet run (scope-check marked most as "out" → limitation/future work):**
- R5-3: standardize on A2+A3 only (closest thing to a neutral baseline).
- R5-10: A1 vs A2 (time-pressure sensitivity), A3 vs A4.
- R5-11: quality-only baseline (% invalid, refits, duration).
- R5-12: dose-response — apply A1/A4 model to A2/A3, compare with RTLX order.
- R5-17: blink-only baseline model (N27's single-feature AUC 0.97 cited instead).
- R5-21: larger XGBoost search (~100 draws + min_child_weight) — borderline.
- Optional: R5-5 looser exclusion, R5-6 threshold sensitivity table, R5-14 fixed-window (~11 s),
  R5-8 cache key, COLET-protocol replication (D-M7), S1 multitask, S2 RTLX bins.

**Admin:** commit pending doc updates (R5-37); verify Tao *et al.* and 2026 references (R5-36);
possibly confirm P28/P31 with a second .mat reader (h5py); move `convert_colet.py` into `src/`.

**Thesis writing:** current draft = `CSRP-Chapter-1-to-5-draft-COLET-v4.docx` (tracked changes,
author "Claude"). Chapters 4–5 carry the 44-participant numbers.

## 11. Plain-language glossary (for explaining to teammates/panel)

- **ROC-AUC:** chance that a random high-load sample gets a higher score than a random low-load
  one. 0.5 = coin flip, 1.0 = perfect.
- **LOPO:** train on 43 people, test on the 44th, repeat for everyone → tests on "new" people.
- **Nested tuning:** settings are chosen using only the training people, never the test person.
- **Per-person z-score:** "higher than this person's usual" instead of "high in absolute terms".
- **Cluster bootstrap CI:** resample *people* (not rows) 2000 times to get an uncertainty range.
- **Shuffled-label test:** scramble labels; if score stays high, something is leaking. Ours → 0.5.
- **SHAP:** how much each feature pushed each prediction up or down; averaged |SHAP| = reliance.
- **Kendall τ:** how similar two rankings are (1 = identical).
- **Ceiling rule:** if both models are ~perfect, the test can't separate them → look at the
  harder P2 test instead.
- **I-VT:** a sample moving faster than 45°/s is a saccade; slower = fixation.

## 12. Likely panel questions → short answers

| Question | Answer |
|---|---|
| Isn't 0.99 too good / leakage? | Shuffled labels → 0.5. High score comes from talking-driven blinks and standardization (§6). Disclosed. Route B gives 0.85–0.88 without standardization. |
| Isn't it just detecting talking? | Partly, possibly. P2 pupil-only 0.81–0.83, no-blink 0.89–0.93. We frame it as workload *condition* classification, and state speech can't be ruled out. |
| Why not NASA-RTLX labels? | Condition labels are the design; RTLX is the manipulation check (43/43 A4 > A1). RTLX bins optional (S2). |
| Why drop A2/A3? | "Drop the middle" precedent; time pressure was weak; A2/A3 still used in standardization. |
| Which model is better? | Neither significantly. LR equal or better; only the no-blink run shows a significant difference, favoring LR. Direction flips with protocol. |
| Why XGBoost so shallow? | 88 samples; depth 1–3 per Walocha 2025; tuned settings unstable → "lightly tuned XGBoost". |
| Why per-person standardization? | Pre-specified, used by closest studies, adviser-approved; assumes calibration (limitation). |
| Your result beats COLET's 0.85 — how? | Standardization + whole-activity features, not overlap. COLET used non-participant-independent splits. |
| P28/P31? | Dataset error in COLET v3 (byte-identical files, different people). Kept once, rerun; conclusions unchanged qualitatively. |
| Why gaze_normal not gaze_point_3d? | 3D point's depth is broken (113 mm vs 800 mm, points behind camera). |

## 13. How to work with us (rules for the agent)

- **RRL quality bar:** cite only peer-reviewed sources whose existence is confirmed (publisher/
  DOI/PubMed), with the claim matched to the text. Prefer eye-tracking sources, 2+ for risky
  decisions. If support is weak, **change the method, don't pad citations.** Say clearly when
  something is our own choice. Never invent references, page numbers or statistics.
- **Scope rule:** keep it small. Test every suggestion with "is it needed for a valid P1/P2
  result?" Otherwise → optional, limitation, or future work.
- **Stay label-blind:** never choose features or thresholds by how well they separate A1 from A4.
- **Don't change frozen methodology** without flagging it as a change needing adviser approval.
- **Never call the models "equivalent"**; say "no evidence of a difference".
- **Explain simply,** with examples and tables; expect "explain simpler" and "what did the RRL do?"
- Numbers in this brief are the source of truth (44-participant rerun). Older 45-participant
  numbers (e.g. P1 0.990/0.983, P2 0.851/0.804) are **superseded**.
- We use Conventional Commit messages; the team commits, not the agent.

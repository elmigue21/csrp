# RRL and Methodology Decision Log

Working log for justifying the dataset and processing pipeline with solid
literature. Anyone picking this up (human or agent) should read this file first,
then update it rather than starting over.

- **Goal:** every dataset and processing decision in Chapter 3 is backed by solid
  RRL or, where none exists, explicitly justified by our own data.
- **Fallback:** if a decision cannot be supported, change the method rather than
  pad it with weak citations (see §5).
- **Rule:** no Chapter 2/3 prose is written until the decisions in §3 are settled.

Last updated: 2026-10-01

**Revised title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE
LOAD DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

**Dataset decision (2026-10-01): switch to COLET for now.**
- **Reason:** the easier option compared with ADABase. Open Zenodo download, no request needed.
- **ADABase is pinned (on hold):** strongest design, but it needs a signed EULA. The email draft
  is ready.
- **GAZELOAD** was the adviser-approved dataset, so **the switch still needs adviser sign-off.**
- `methodology-replan.md` was written for GAZELOAD. The COLET version is
  **`methodology-colet.md`**, which supersedes it.

## COLET decision register

| ID | Decision | Status | Choice / options | Support |
|---|---|---|---|---|
| C1 | Unit of analysis | ✅ **Decided 2026-10-01** | **Whole activity:** one feature row per participant × activity. Short windows remain a possible sensitivity check. | COLET paper (EM7); Kaczorowska 2021; Wu 2020; Shafiei 2025. Activities last only 11–141 s (inventory). |
| C2 | Primary label | ✅ **Decided 2026-10-01** | **Two levels: low = A1 (single task, no time pressure) vs high = A4 (multitask + time pressure).** A2 and A3 are left out of the primary analysis. | COLET paper (A1 vs A4: LR 0.85); Hogervorst 2014, Gado 2023, Wu 2020 (extremes only, middle dropped); 8/12 twins use condition labels |
| C2b | Talking (counting aloud) and blink confound | ✅ **Decided 2026-10-01** | **Follow the RRL:** blinks are kept as workload features, with no dedicated speech control. This matches the COLET paper and Pluchino 2023, neither of which separated talking from workload. | COLET paper (EM7); Pluchino 2023 (HR2); Recarte 2008 (ET12). Bentivoglio 1997 (ET5) documents that talking raises blink rate. |
| C3 | Recording exclusion | ✅ **Decided 2026-10-01** | (a) Exclude recordings with **> 35%** gaze samples below confidence 0.8. (b) A participant missing A1 or A4 is dropped from A1 vs A4 entirely. (c) Invalid sample = confidence < 0.8. (d) Blink events outside 50–500 ms are ignored. **Result:** 185/188 recordings kept (P06 A3, P06 A4, P17 A4 removed); A1 vs A4 = **45 participants, 90 samples** (96% kept). | Nenna 2023; Faraji 2023; Hausamann 2020; Steinhauer 2022; Hershman 2018 |
| D-M3 | Blink-duration missingness | ✅ **Decided 2026-10-01** | **Drop the blink-duration feature;** keep blink rate. Blink duration is missing in 13/45 A1 vs 0/45 A4 recordings (no blinks), so "missing" would reveal the label. | Hogervorst 2014 (discarded segments with undefined blink duration); Kapoor & Narayanan 2023 (illegitimate-feature leakage) |
| D-M5 | Keep both A1 vs A4 (P1) and single vs multitask (S1) | ✅ **Decided 2026-10-01** | **Keep both.** P1 = low vs high headline (COLET's own A1/A4 task); S1 = the effective factor, using all data. | COLET paper reports both; Hogervorst 2014 |
| D-N2 | Saccade peak velocity | ✅ **Decided 2026-10-01** | **Keep it as a core feature,** with a limitation sentence on the ≈ 242 Hz gaze sampling rate (no RRL source sets a minimum either way). | COLET paper used it on the same data; Di Stasi 2010, 2011; Tao 2019 (60%) |
| D-M1 | Gaze-to-angle conversion | ✅ **Revised 2026-10-01 (R4-1)** | **Per-eye gaze directions** (`gaze_normal0/1`), averaged over the two eyes; angles between consecutive directions. Replaces `gaze_point_3d`, whose vergence depth is implausible (median 113 mm vs 800 mm; points behind the camera in 39/188 recordings). The `norm_pos` cross-check and ±15% rule are removed (not independent). One-eye samples treated as missing (**decided 2026-10-02**; N25). RRL: no study addresses it; follows Faraji 2023 (unreliable samples = gaps). Main sequence and COLET Table 4 descriptive only. | Data checks (reviewer R4-1; ours N25). Hooge 2019 (PP17) and Velisar & Shanidze 2024 (PP18): vergence depth unreliable; Kothari 2020 (PP19): velocity from unit-vector angles. Design choice (§13): no study names `gaze_normal`. Hausamann 2020 is **not** support (uses the 3D point). Ehinger 2019: accuracy only |
| D-M2 | Talking confound (revisited) | ✅ **Decided 2026-10-01** | **Follow the RRL** (C2b reaffirmed): no special control; one limitation sentence. Data-loss figures (0.9% vs 7.3%) are documented in researcher-notes N15. | COLET paper; Pluchino 2023 |
| D-M7 | COLET-protocol replication | ✅ **Approved as OPTIONAL 2026-10-01** | Run it: (1) COLET's method (random 80/20, feature selection on all data, scaling before the split) vs (2) our LOPO pipeline, same features and models. A 4-step ladder is a further option. **Whether it goes in the paper is decided later.** | Saeb 2017; Kapoor & Narayanan 2023; COLET paper |
| D-M4 | Per-person standardization | ✅ **Decided 2026-10-01** | **Use per-person standardization** (z-score over each person's 4 activities, label-free). The bias (test-person hint; reliance on the balanced design) is disclosed as a limitation. Options and RRL in researcher-notes N7. | Božak 2026; Oppelt 2023; Gado 2023; Albuquerque 2022; caveat Tognotti 2026, Kapoor & Narayanan 2023 |
| D-N5 | Analysis scope | ✅ **Decided 2026-10-01** | **Only P1 is required** (A1 vs A4, LR vs XGBoost, LOPO, feature importance). All other analyses are **optional**, listed in researcher-notes N19; which ones go in the paper is decided later. | Božak 2026 (small focused set); Demirezen 2024 (pre-specify) |
| D-N6 | LR penalty | ✅ **Decided 2026-10-01** | **Elastic net,** with its strength and L1/L2 mix tuned inside the training folds | Kaczorowska 2021, 2022 (elastic-net LR weights as importance); Zou & Hastie 2005 |
| D-M8 | SHAP | ✅ **Decided 2026-10-01** | **SHAP for both models:** TreeSHAP for XGBoost, linear SHAP for LR, so importance is on the same scale. LR weights are still reported. No RRL study did SHAP for both; justified by SHAP being model-agnostic (Lundberg & Lee 2017). | Lundberg & Lee 2017; Lundberg 2020; Božak 2026 (SHAP); Kaczorowska (weights) |
| C4 | Feature set | ✅ **Decided 2026-10-01** | **10 features:** pupil mean, pupil SD, blink rate, fixation rate, fixation duration, saccade rate, saccade amplitude, saccade peak velocity, gaze spread x/y. Dropped: blink duration (D-M3), mean saccade velocity, saccade duration, COLET's higher-order statistics. Gaze entropy exploratory. Justification: researcher-notes N21. | COLET; Tao 2019; Božak 2026; Hogervorst 2014; Kapoor & Narayanan 2023; Dormann 2013 |
| V2-1 | Talking-robustness check (round-2 review) | ✅ **Decided 2026-10-01** | **Required extra run: pupil-only model** (pupil mean + pupil SD), LR vs XGBoost, same LOPO pipeline as P1. P1 keeps all 10 features. Pupil is the most established workload signal and the least affected by talking. | Rolon-Merette 2026, Appel 2018, Stolte 2020 (pupil-only models); Tao 2019 (pupil 79%); evidence in researcher-notes N22. Karunathilake 2025 dropped as support (WEAK; V3-5). |
| SCOPE | Scope rule | ✅ **Decided 2026-10-01** | Only steps required to produce valid P1/P2 results are included. Other analyses suggested by the literature or reviewers are recorded as optional, limitations or future work. | Title: LR vs XGBoost, cognitive load, eye-tracking features, feature importance |
| V2-3 | Pupil 3D-model refits | ✅ **Decided (lean)** | Keep the Kret & Sjak-Shie speed filter (P7); **count refits per recording** in the retention table; one limitation sentence. No per-segment handling. | Kret & Sjak-Shie 2019; data check N22 |
| V2-2 | Fixation/saccade detection check | ✅ **Decided (lean)** | One sanity check after detection, **per activity (R4-2)**: median fixation 150–400 ms and saccade:fixation count ratio 0.8–1.25 (tolerance per V3-3). Pass in every activity → use the features. Fail → drop the saccade/fixation features (5-feature fallback). | Komogortsev 2010; Andersson 2017; COLET (273 ms); Božak 2026 (subset) |
| V3-3 | Sanity-check tolerances (round-3 review) | ✅ **Decided 2026-10-01** | Saccade:fixation ratio **0.8–1.25**. Any failure → 5-feature fallback. (The ±15% agreement rule was removed by R4-1.) | **Our own choice** (no RRL source sets this value); reviewer's suggested value |
| V3-4 | Gaze-spread cleaning (round-3 review) | ⤴ **Superseded by R4-1** | The velocity rejection did not remove the 102° artifact (102.1° before and after). The artifact came from the 3D gaze point; with `gaze_normal0/1` max spread is 9.4° (N25). | Data checks (R4-1, N25) |
| R4-1 | Gaze input (round-4 review, Critical) | ✅ **Decided 2026-10-01** | See D-M1 (revised). Scope shrinks: one input changed, one check removed. | Reviewer's data check, reproduced (N25) |
| R4-2 | Sanity check per activity (round-4 review) | ✅ **Decided 2026-10-01** | Computed for A1–A4 separately; a failure in any activity → 5-feature fallback. Label-blind QC like P11. | — (wording; talking noise is concentrated in A3/A4, N22) |
| V3-1 | Refits vs P2 (round 3, carried to round 4) | ✅ **Decided 2026-10-01** | **Limitation sentence only** (`methodology-colet.md` §11; N23). The refit-free P2 run is not done (it would add an analysis); future work. | Data check N22 |
| V3-2 | Importance under the ceiling rule (round-3 review) | ✅ **Decided 2026-10-01** | If the ceiling rule applies, feature importance is still reported from P1, noting that top blink/saccade features may partly reflect talking (N17, N22). No new analysis. | — (wording; importance from P1 is already planned) |
| V2-9 | XGBoost search range | ✅ **Decided** | max_depth 1–3, 50–300 trees (equal tuning budget with LR) | Walocha 2025 (depth 2 most chosen); Shafiei 2025; Cawley & Talbot 2010 |
| V2-4 | Ceiling interpretation | ✅ **Decided** | Pre-stated: "If both models exceed 0.95 AUC in P1, the model comparison is interpreted mainly from P2." | COLET (0.98 on A2/A4); Kaczorowska 2021 (LR = RF at the top); Benavoli 2017 |
| C5 | Adviser sign-off for the COLET switch | ⏸ On hold | — | — |
| C6 | ADABase access request | ⏸ On hold (email draft ready) | — | — |
| C7 | Reference checks: 5 titles, page/issue numbers, full texts of abstract-only sources | 📌 Pinned | — | — |
| C8 | Stimulus-luminance check (P12) | ✅ **Optional (decided 2026-10-02)** | COLET controlled lighting and ran its own check (2/47 correlated); cited instead | CF2, CF3, EM7 |

**Scope confirmed by the team:**
- The COLET + GAZELOAD cross-dataset evaluation was **rejected** (not approved).
- The study is GAZELOAD-only.

---

## 1. Source documents and their state

| Document | What it is | State |
|---|---|---|
| `docs/CSRP-Chapter 1 to 3.pdf` | Adviser-submitted Ch 1–3 (May 2026) | **Describes a different study**: COLET + GAZELOAD cross-dataset, pupil dilation, label harmonization. Section D (synthesis) is empty. |
| `CSRP Chapter 1 to 5 (draft).pdf` (Downloads) | Initial full draft, 6 configs | RRL identical to `thesis-draft-chapters-1-5.md`. Results come from the **Aug 4 run** and are stale. |
| `docs/thesis-draft-chapters-1-5.md` | Current markdown draft, 8 configs | RRL matches the method, but is thin on the key decisions. |
| `docs/methodology.md` | Method reference | Calls Config A the "headline", which conflicts with the draft (E is main). |
| `docs/results.md` | Current results (Sep 6 run, 8 configs) | Source of truth for numbers. |
| `docs/thesis-notes.md` | Defense notes | Admits that the 30 s epoch has no citation. Cites Tao et al., which is missing from the reference list. |

### Run reproducibility issue

- **Same code and seed, different results.** The Aug 4 run (Windows, `outputs/run_log.txt`)
  and the Sep 6 run (Linux, `run_metadata.json`) give different numbers for configs A–F.
- **It changes a conclusion.** Config C's Wilcoxon p goes from **0.042 to 0.075**, so the
  draft's only "significant" ROC-AUC result disappears.
- **Likely cause:** XGBoost nondeterminism across platform, threads or library versions.
- **Action:** report only the Sep 6 numbers, and state the machine and library versions.
  The LR–XGB gap is about the size of run-to-run noise, so present it that way.

---

## 2. Quality bar for any RRL source

A source is accepted only if **all** of these hold:

1. It is in a peer-reviewed journal or major conference. No preprints, except the GAZELOAD
   paper itself, and no unindexed or predatory venues.
2. Its existence is confirmed on the publisher or DOI page, not a search snippet.
3. The claim is matched to the text, with a quoted sentence showing it says what we cite
   it for.
4. It is from the same domain where possible: eye-tracking, then physiological workload,
   then general ML (only for method-level claims).
5. High-risk decisions have **at least two independent** sources.

Anything that fails is either dropped or labelled "design choice, justified by our data".

Strength grades used below: **STRONG** / **ADEQUATE** / **WEAK** / **NONE**.

---

## 3. Decision register

| ID | Decision | Current support | Grade | Defense risk | Status |
|---|---|---|---|---|---|
| D1 | Compare LR (interpretable baseline) vs XGBoost | Bishop; Chen & Guestrin; E2, E4; E1 as contrast only (§8 T5) | STRONG | Low | Keep; demote Trigka |
| D2 | GAZELOAD only; cross-dataset dropped | Cross-dataset was **rejected by the panel/adviser**, so the drop is resolved. Choice of a wearable HRC dataset: DS1–DS9 (§8 T6) | ADEQUATE | Low | Resolved; call it a lab testbed, not field data |
| D3 | No pupil features (not in public release) | Dataset limitation | n/a | Medium | Keep; state as limitation |
| D4 | Drop `tasks` column (task alone gives AUC 0.93) | LK1, LK2 (L2); own EDA | STRONG | Low | Done |
| D5 | 30 s epochs, ≥80% coverage | W30a–W30e (§8 T3). Precedent exists but the literature disagrees; 30 s value is a design choice; [13] invalid here | ADEQUATE | Medium | Consider 10/30/60 s sensitivity run |
| D6 | Mean + SD per feature, plus proportion features | W30a (per-window mean/SD/max); R1 (feature set) | ADEQUATE | Low | Keep |
| D7 | Absolute label: rating ≥ 4 | Critique of pooled thresholds is STRONG (LB2, LB3, LB5); the value 4 is uncited | NONE for the value | Medium | Frame A as naive baseline; justify 4 by class balance (53%) |
| D8 | Within-participant median label, ties dropped | LB1 (exact precedent), LB2, LB4, E6; own ρ 0.735 vs 0.534. Tie exclusion uncited | ADEQUATE | Medium | T1 done; state tie exclusion as a design choice |
| D9 | Per-participant feature z-scoring | N1–N5, V1–V3 (§8). Strong EEG ablation; eye-tracking precedent without an ablation. **Leakage-type critique possible (LK2 L1.2)** | ADEQUATE | **Medium–High** | Check N1's Methods; consider the calibration fallback |
| D10 | Lux included vs removed | GAZELOAD; L1–L5 (pupil-specific); S1–S4 shortcut learning (§8 T4) | ADEQUATE | Medium | T4 done; frame via shortcut learning; supports F as fair comparison |
| D11 | LR: median impute + missing indicators; XGB: native NaN | MD1–MD3; Chen & Guestrin | STRONG | Low | Done |
| D12 | Leave-one-participant-out + inner grouped tuning | LK3–LK6, NT1–NT3; N1, E2, E4, E7, E9 | STRONG | Low | Done. Note NT3's variance caveat |
| D13 | Pooled ROC-AUC as primary metric (+ per-participant spread) | Fawcett 2006; SN1, SN2 | STRONG (metric) / ADEQUATE (pooling) | Low | Keep; report both pooled and per-participant |
| D14 | Wilcoxon signed-rank over participants | Demšar 2006 | STRONG | Low | Keep |
| D15 | LR coefficients, XGB gain, out-of-fold SHAP | Lundberg & Lee; Kaczorowska 2021a; Trigka 2025 | STRONG | Low | Keep |
| D16 | Config E as main result | None (chosen after results) | NONE | **High** | Needs team decision |

---

## 4. Reference audit

### 4.1 Current draft reference list (19 entries)

| # | Source | Used for | Status |
|---|---|---|---|
| 1 | Kahneman 1973 | Effort as limited resource | OK (background) |
| 2 | Paas & van Merriënboer 2020 | Cognitive load theory | OK |
| 3 | Parasuraman & Manzey 2010 | "Overload causes errors" | **Miscited**: the paper is about automation complacency. Replace with W1 (Young et al. 2015). |
| 4 | Duchowski 2017 | Eye-tracking features | OK |
| 5 | Bishop 2006 | LR baseline | OK |
| 6 | Chen & Guestrin 2016 | XGBoost, missing values | STRONG |
| 7 | Lundberg & Lee 2017 | SHAP | STRONG |
| 8 | Karbouj et al. 2026 (GAZELOAD) | Dataset | **Verified**: arXiv preprint only; add Mendeley DOI 10.17632/9smd7nbtwc.1 (§8 T4) |
| 9 | Marinescu et al. 2018 | Rating behaviour affects labels | **Wrong title** (that title is a 2016 IFAC paper). Correct: LB3, *Hum. Factors* 60(1):31–56. It never binarises, so rephrase the claim (§8 T1). |
| 10 | Albuquerque et al. 2022 | Leave-one-participant-out; subject-wise normalization | EEG only; **claim unverified** |
| 11 | Gado et al. 2023 | Subject-wise median split | **Verified**: exact subject-wise median split, LOSO. It gives no rationale and no tie handling. |
| 12 | Kaczorowska et al. 2021a (Brain Sci.) | Features, importance | OK. Note: its features include DSST task-performance features (not eye-only); 3-class task. |
| 13 | Kaczorowska et al. 2021b (Sensors) | Aggregation; participant-held-out evaluation | **Confirmed miscited for aggregation**: it aggregates classifier outputs. Valid only as a participant-disjoint hold-out (not LOSO). |
| 14 | Fawcett 2006 | ROC-AUC | STRONG |
| 15 | Demšar 2006 | Wilcoxon | STRONG |
| 16 | Hart & Staveland 1988 | NASA-TLX | OK (not our label) |
| 17 | Ktistakis et al. 2022 (COLET) | Feasibility | OK |
| 18 | Rizzo et al. 2022 | Features, feasibility | **Wrong authors.** Real: A. Rizzo, S. Ermini, D. Zanca, D. Bernabini, A. Rossi. It is a Stroop interference task, not general workload. |
| 19 | Trigka et al. 2025 (WEBIST) | Closest precedent | **Weak**: N=9, labels derived from gaze features (circular), no cross-subject validation, no SHAP. Use as contrast only. |

### 4.2 Problems in the adviser-submitted PDF references

These must not survive into any future version.

| PDF ref | Problem |
|---|---|
| [5] Reddy et al., IEEE Access 2020 | **Not found**; possibly nonexistent |
| [7] Cortes & Vapnik 1995 (SVM) | Irrelevant to the claim |
| [12] Kaczorowska, Rodrigues, Longo | Wrong authors/venue. Real: Kaczorowska, Plechawska-Wójcik, Tokovarov, *Brain Sciences* 2021 |
| [13] He et al., BSPC 2025 | **Not found**; carries the "cross-task harder than cross-subject" claim |
| [14] Karunathilake et al., Sensors 2025 | Wrong title/co-authors/venue. Real: HFES Annual Meeting 2025, doi 10.1177/10711813251357928 |
| [16] Rolon-Merette et al., Front. Neuroergonomics 2026 | Wrong venue/co-authors. Real: FLAIRS 2026 proceedings |
| [17] Rizzo et al., IEEE SMC 2022 | Wrong venue. Real: *Frontiers in Human Neuroscience* 2022 |
| [21] Wahn et al., PLOS ONE 2016 | Wrong title. Real: "Pupil Sizes Scale with Attentional Load and Task Experience in a Multiple Object Tracking Task" |
| In-text | "Beatty (1982) [1]" points to Kahneman; "Zhang et al. (2021) [8]" points to UN SDGs |

---

## 5. Fallback options if a decision cannot be supported

| Decision | Option if literature is weak |
|---|---|
| D5 30 s epoch | Run a 10 / 30 / 60 s sensitivity analysis, so the data justifies the choice |
| D7 ≥ 4 cutoff | Justify as the class-balance point, or replace with task-difficulty labels (common in the literature) |
| D8 median label | Per-participant z-scored rating, or regression / ranking formulation |
| D9 per-participant z | Baseline calibration using only each person's first/easiest task (deployable; baseline-correction literature) |
| D10 lux | Make Config F (no lux) primary; report lux as a finding |
| D16 Config E | Pre-specify the primary configuration; report the rest as ablations |
| D2 / D3 | Bigger pivot: add COLET (has pupil; matches approved Ch 1–3) as a secondary dataset |

Anything that needs a re-run is real work. Choose it only where §3 stays WEAK/NONE after
research.

---

## 6. Open questions for the team

1. **Main claim:** (a) "XGBoost beats LR", (b) "preprocessing matters more than model
   choice under honest validation", or (c) both, with (b) leading? Current lean: (c).
2. **Primary configuration:** A (pre-planned), E (best), or F (eye-tracking only)?
   Current lean: A as baseline, E and F reported together, F as the fair model comparison.
3. **Lux:** limitation, or a reported finding on environmental confounds?
4. **≥ 4 cutoff:** is there any source, or was it chosen for class balance?
5. **30 s epoch:** cite and call it a design choice, or run the sensitivity check?
6. **Chapter 2 structure:** rewrite, or keep the PDF's A/B/C/D structure with C changed
   from cross-dataset to participant variability and validation?
7. ~~**Adviser:** has dropping cross-dataset been approved?~~ **Resolved:** cross-dataset
   was rejected; revised title adopted.
8. **Pupil:** the old Ch 1–3 lists pupil dilation as a feature. Confirm the adviser knows
   GAZELOAD's public release has no pupil data.
9. **Feature importance is now in the title.** Should the lux finding (the #1 SHAP
   feature) and the E vs F comparison be the centrepiece of the importance chapter?

---

## 7. Research tracks

Each track must return verified sources that meet §2, graded STRONG / ADEQUATE / WEAK.

| Track | Covers | Status |
|---|---|---|
| T1 Participant-relative labeling | D7, D8; verify Gado [11], Marinescu [9]; Paas 1992 single-item scale | **Done**: D8 → ADEQUATE; [9] wrong title; value 4 and tie exclusion remain design choices |
| T2 Subject-wise normalization | D9; verify Albuquerque [10] | **Done**: D9 now ADEQUATE; [10] verified |
| T3 Window length / aggregation | D5, D6; verify Kaczorowska [13], COLET windows, Tao et al. review | **Done**: D5 → ADEQUATE (precedent); [13] miscited; Tao verified; COLET full text still needed |
| T4 Lighting confound / shortcut learning | D10; verify GAZELOAD metadata | **Done**: D10 → ADEQUATE; GAZELOAD verified (16M/10F, preprint) |
| T5 Empirical studies for synthesis | D1, D12; synthesis table; replacement for [3] | **Done**: 9 studies; gap confirmed; [18] and [19] corrected |
| T6 Dataset choice and leakage | D2, D4, D11, D12; Kaufman 2012, Kapoor & Narayanan 2023, Saeb 2017, Cawley & Talbot 2010 | **Done**: D4, D11, D12 → STRONG; D2 choice ADEQUATE; new D9 leakage risk |

Results will be recorded in §8 as they arrive.

### Round 2: switch to the most RRL-supported method

- **Why:** after round 1, most support for our current pipeline is *indirect*.
  - Normalization and labeling evidence comes from EEG/ECG; lighting evidence is
    pupil-only.
  - No study uses our exact combination.
  - Our count-per-window features do not match the literature's per-event features.
- **Team decision (2026-10-01):** where the method is weakly supported, change it to what
  the literature actually does.

**What GAZELOAD allows us to change:**
- Only the 250 ms metrics, lux, event log and ratings are released. No raw gaze, so there
  are no per-event fixation durations.
- Designed difficulty is available: sessions 1–2 low, 3–4 medium, 5 high.

| Track | Covers | Status |
|---|---|---|
| R1 Closest methodological twins | Full pipelines of the most similar studies; "consensus pipeline"; Božak full text | In progress |
| R2 Label source + baseline calibration | Condition labels vs self-report; per-person normalization that does not use test data | In progress |
| R3 Validity of our available features | Which count/rate/entropy/dispersion features are real workload indicators | In progress |
| R4 GAZELOAD users + HRC twins | Any other work on GAZELOAD; industrial/HRC eye-tracking workload pipelines | In progress |

---

## 8. Verified sources (accepted)

Format: ID supported · IEEE citation with DOI · venue · what it shows · quote ·
VERIFIED (full text read) / ABSTRACT-ONLY · grade.

### T2: Subject-wise normalization (D9) and between-person variability

**Track verdict:**
- **D9 moves from WEAK to ADEQUATE.** One strong EEG ablation shows per-subject z-scoring
  improves leave-one-subject-out results, a second EEG study corroborates it, there is
  eye-tracking precedent for per-person normalization, and three papers show oculomotor
  baselines differ stably between people.
- **The gap:** no peer-reviewed *eye-tracking* paper with a normalization ablation was
  found. Chapter 2 must say "evidence from EEG workload research" and not imply
  eye-tracking proof.

**Normalization improves cross-subject performance**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| N1 | I. Albuquerque, J. Monteiro, O. Rosanne, T. H. Falk, "Estimating distribution shifts for predicting cross-subject generalization in electroencephalography-based mental workload assessment," *Front. Artif. Intell.*, vol. 5, 2022, doi:10.3389/frai.2022.992732 (PMC9576998) | EEG, N=18, LOSO. Compares no normalization vs per-subject z-score vs baselines. Z-scoring "improved mental workload assessment on unseen subjects". | VERIFIED | STRONG (EEG) |
| N2 | J. Fdez, N. Guttenberg, O. Witkowski, A. Pasquali, "Cross-subject EEG-based emotion recognition through neural networks with stratified normalization," *Front. Neurosci.*, vol. 15, 626277, 2021, doi:10.3389/fnins.2021.626277 | EEG emotion, N=15, leave-one-participant-out. Per-participant normalization significantly beat batch normalization. | VERIFIED | ADEQUATE (emotion, EEG) |

**Eye-tracking precedent for per-person normalization**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| N3 | T. Appel, C. Scharinger, P. Gerjets, E. Kasneci, "Cross-subject workload classification using pupil-related measures," *Proc. ACM ETRA '18*, Art. 4, 2018, doi:10.1145/3204493.3204531 | Cross-subject eye/pupil workload using "normalized features". Exact normalization method not confirmed. | ABSTRACT-ONLY | ADEQUATE; check PDF before describing the method |
| N4 | J. Wei *et al.*, "Cognitive load inference using physiological markers in virtual reality," *Proc. IEEE VR*, 2025, doi:10.1109/VR59515.2025.00098 | VR eye/pupil + PPG, N=738, participant-disjoint test. Per-individual normalization "mitigates individual variations". No ablation. | VERIFIED | ADEQUATE |
| N5 | M. Laut, E. Dorschky, R. Richer, N. Rohleder, B. M. Eskofier, "Classifying mental stress from eye tracking data...," *Sci. Rep.*, vol. 16, 2026, doi:10.1038/s41598-026-58429-7 | Pupil baseline correction (not a z-score), nested LOSO; "inter-individual differences remain a relevant factor". | Near-verbatim | ADEQUATE (stress) |

**Stable between-person differences in eye movements (motivation for D9)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| V1 | G. Bargary *et al.*, "Individual differences in human eye movements: An oculomotor signature?" *Vision Res.*, vol. 141, pp. 157–169, 2017, doi:10.1016/j.visres.2017.03.001 | N>1000; each person has a stable "personal oculomotor signature". | ABSTRACT-ONLY | STRONG |
| V2 | W. Poynter, M. Barber, J. Inman, C. Wiggins, "Individuals exhibit idiosyncratic eye-movement behavior profiles across tasks," *Vision Res.*, vol. 89, pp. 32–38, 2013, doi:10.1016/j.visres.2013.07.002 | Fixation and saccade profiles are idiosyncratic and stable across tasks. | ABSTRACT-ONLY | STRONG |
| V3 | M. S. Castelhano, J. M. Henderson, "Stable individual differences across images in human saccadic eye movements," *Can. J. Exp. Psychol.*, vol. 62, no. 1, pp. 1–14, 2008, doi:10.1037/1196-1961.62.1.1 | Eye movements differ across people but are consistent within a person. | ABSTRACT-ONLY | STRONG |

**Notes and exclusions**

- **Ref [13] (Kaczorowska 2021b, PMC8272248):**
  - Normalization is generic, not per-subject, so **do not cite it for D9**.
  - Validation is a participant-disjoint hold-out (test set = 6 participants), not LOSO.
    It is usable for D12 as eye-tracking precedent for participant-held-out evaluation.
- **PMC9576998** in the thesis notes is the Albuquerque paper itself, not a separate source.
- **Before quoting N1 verbatim**, check the exact wording and whether it states that the
  held-out subject's own statistics are used (same label-free, transductive setup as ours).
- **Excluded (preprints):** CLARE (arXiv 2404.17098), MambaGaze (arXiv 2605.22775),
  Rossi et al. (arXiv 2002.11192).
- **Weaker optional sources:**
  - Ademe & Jain, EMBC 2025, doi:10.1109/embc58623.2025.11252924: within-subject 99.9% vs
    LOSO 63.1%. Motivates the need; uses no normalization.
  - Mathôt *et al.*, *Behav. Res. Methods* 2018: pupil trial baseline only.

### T5: Empirical eye-tracking workload studies (synthesis; D1, D12, D15)

**Track verdict:**
- **The gap is real and citable:** no study combines LR + XGBoost + full LOPO + ROC-AUC +
  importance comparison.
- **Weak spots in how precedents are used:**
  - Trigka (the "closest precedent") is weak evidence: N=9, circular labels, no
    cross-subject validation. Demote it to a contrast case.
  - The best LR-vs-XGBoost precedent under participant hold-out is Rolon-Merette 2026
    (LR near chance, XGBoost >75%).

**Synthesis table**

| Ref | Study | Venue | N | Models | Validation | Result | Importance | Evidence |
|---|---|---|---|---|---|---|---|---|
| E1 | Trigka, Dritsas, Mylonas 2025 | WEBIST (SCITEPRESS conf.) | 9 | LR, NB, SVM, XGB, MLP | Stratified 80/20; **not subject-independent** | XGB AUC 0.956 vs LR 0.894 | None (no SHAP) | VERIFIED |
| E2 | Kaczorowska, Plechawska-Wójcik, Tokovarov 2021 | *Brain Sci.* (PubMed) | 29 | 8 incl. LR, RF, SVM | Participant-disjoint hold-out (~6 test) | 3-class F1 up to 0.97 | LR coefficients for selection | VERIFIED |
| E3 | Rizzo, Ermini, Zanca, Bernabini, Rossi 2022 | *Front. Hum. Neurosci.* (PubMed) | 64 | RF, LR, ANN, SVM | 5-fold CV; subject separation not stated | AUC >0.8 best cases | None | VERIFIED |
| E4 | Rolon-Merette *et al.* 2026 | FLAIRS-39 (conf.) | 89 | SVM, XGB, LR, CNN, Transformer | LOPO over 10 random held-out participants | Inter-subject LR ~52%, XGB >75%, Transformer 85% | None | VERIFIED |
| E5 | Karunathilake *et al.* 2025 | Proc. HFES (conf.) | 20 | RF, XGB, LSTM | Not in abstract | LSTM best | Not in abstract | ABSTRACT-ONLY |
| E6 | Nasri *et al.* 2024 | IEEE ISMAR-Adjunct (short) | 19 | MLP, RF | Not described | MLP acc 0.84 | None | VERIFIED |
| E7 | Božak *et al.* 2026 | *AI* (MDPI) | 54 | Feature-based ML | **LOSO and LOGO** | Load vs rest 81.4% LOSO | **SHAP**; subject-normalised features strong | ABSTRACT-ONLY |
| E8 | Ktistakis *et al.* 2022 (COLET) | *CMPB* (PubMed) | 47 | "well-known classifiers" | Not in abstract | Up to 88% | Not in abstract | ABSTRACT-ONLY |
| E9 | Appel *et al.* 2018 (= N3) | ACM ETRA | n/r | Similarity-weighted ensemble | Cross-subject | 70.4% real-time, 76.8% offline | Not in abstract | ABSTRACT-ONLY |

**IEEE citations for E1–E9 and the [3] replacement**

- **E1:** M. Trigka, E. Dritsas, P. Mylonas, WEBIST 2025, doi:10.5220/0013782800003985.
  Pages 567–574 were inferred from the PDF; confirm on the publisher page.
- **E2:** doi:10.3390/brainsci11020210.
- **E3:** A. Rizzo, S. Ermini, D. Zanca, D. Bernabini, A. Rossi, "A machine learning approach
  for detecting cognitive interference based on eye-tracking data," *Front. Hum. Neurosci.*,
  vol. 16, 806330, 2022, doi:10.3389/fnhum.2022.806330.
- **E4:** T. Rolon-Merette, G. Hardy Joseph, A. J. Karran, C. Belanger, C. K. Coursaris,
  S. Sénécal, P.-M. Léger, "Towards a cross-participant cognitive load classification using
  eye tracking and deep learning," *Proc. FLAIRS*, vol. 39, no. 1, 2026,
  doi:10.32473/flairs.39.1.141863.
- **E5:** S. Karunathilake, N. A. Choudhury, A. Deep, P. Saravanan, *Proc. HFES Annu. Meet.*,
  vol. 69, no. 1, pp. 1680–1686, 2025, doi:10.1177/10711813251357928.
- **E6:** M. Nasri, M. Kosa, L. Chukoskie, M. Moghaddam, C. Harteveld, ISMAR-Adjunct 2024,
  pp. 51–54, doi:10.1109/ISMAR-Adjunct64951.2024.00022.
- **E7:** T. Božak, S. Goyal, M. Langheinrich, M. Gjoreski, G. Slapničar, "Evaluating
  feature-based machine-learning models with post hoc explainability for eye-tracking-based
  task type and workload inference," *AI*, vol. 7, no. 8, 325, 2026, doi:10.3390/ai7080325.
- **E8:** doi:10.1016/j.cmpb.2022.106989.
- **E9:** doi:10.1145/3204493.3204531.
- **W1 (replaces [3]):** M. S. Young, K. A. Brookhuis, C. D. Wickens, P. A. Hancock, "State
  of science: Mental workload in ergonomics," *Ergonomics*, vol. 58, no. 1, pp. 1–17, 2015,
  doi:10.1080/00140139.2014.956151.
  - ABSTRACT-ONLY; review-level.
  - Supports "workload affects performance". It does not quantify errors, so word it softly.

**Cross-links to other decisions**

- **D8:** Nasri 2024 (E6) split NASA-TLX mental demand into high/low **per participant**.
  That is a second precedent for participant-relative labels, alongside Gado. It is a short
  paper, so treat it as supporting only.
- **D9:** Rizzo 2022 (E3) uses subject-wise normalisation. Božak 2026 (E7) found
  subject-normalised features among the strongest by SHAP (abstract only).
- **D12:** E4, E7, E9 and E2 all hold out participants and report drops in cross-subject
  performance. E7 is the best LOSO precedent.

**Venue strength**

- **Can carry claims:** E2, E3, E8 (journals, PubMed); E9 (ETRA).
- **Pair with a stronger source:** E4 (FLAIRS); E7 (MDPI *AI*, abstract only).
- **Supporting only:** E1, E5, E6.

**Gap statement (for the synthesis section):**
- Most studies use within-subject or unstated splits (E1, E3, E6, E8).
- Participant-held-out studies show large drops (E4, E7, E9).
- The LR-vs-boosting comparisons are within-subject (E1) or use only a few held-out
  participants without AUC (E4).
- Importance analysis is rare (E2 used it only for feature selection; E7).
- None combines LR + XGBoost + full LOPO + ROC-AUC + an importance comparison.

**Before stating as fact:** check the full texts of E5, E7, E8 and E9. COLET's window length
and validation are pending in T3.

### T3: Window length and aggregation (D5, D6)

**Track verdict:**
- **No source shows 30 s is optimal, and the literature does not agree.**
  - Two eye-tracking studies used 30 s; one found it best.
  - Two studies found longer windows (60–120 s) better.
- **D5 goes from WEAK to ADEQUATE for precedent only.** The specific value stays a stated
  design choice. **Recommended:** a 10 / 30 / 60 s sensitivity analysis on GAZELOAD
  (fallback in §5).
- **Ref [13] is miscited in draft §3.3.** Its "aggregation functions" combine classifier
  outputs, not time windows.

**Sources**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| W30a | H. Rahman, M. U. Ahmed, S. Barua, P. Funk, S. Begum, "Vision-based driver's cognitive load classification considering eye movement using machine learning and deep learning," *Sensors*, vol. 21, no. 23, 8019, 2021, doi:10.3390/s21238019 | 33 drivers; 15/30/60 s windows; "F1-score and Accuracy are better for 30 s". Per-window statistics (mean/SD/max) of fixation and saccade features. Validation not subject-independent. | VERIFIED | STRONG as precedent for 30 s and for D6 |
| W30b | M. A. Hogervorst, A.-M. Brouwer, J. B. F. van Erp, "Combining and comparing EEG, peripheral physiology and eye-related measures for the assessment of mental workload," *Front. Neurosci.*, vol. 8, 322, 2014, doi:10.3389/fnins.2014.00322 | n-back, N=14. 30 s segments about 5–8% worse than 120 s. Some 30 s segments contained no blinks. | VERIFIED | STRONG (precedent + caveat) |
| W30c | T. Chihara, J. Sakamoto, "Effect of time length of eye movement data analysis on the accuracy of mental workload estimation during automobile driving," *Proc. IEA 2021*, LNNS, Springer, pp. 593–599, doi:10.1007/978-3-030-74608-7_72 | 30–150 s windows; 30 s had "significantly lower AUC"; recommends 60–120 s. | ABSTRACT-ONLY | ADEQUATE (counter-evidence) |
| W30d | J. Tervonen, K. Pettersson, J. Mäntyjärvi, "Ultra-short window length and feature importance analysis for cognitive load detection from wearable sensors," *Electronics*, vol. 10, no. 5, 613, 2021, doi:10.3390/electronics10050613 | Physiological (not eye), XGBoost, LOSO. Shorter windows mostly worse; 25–30 s near best of 5–30 s. | VERIFIED | ADEQUATE |
| W30e | A. R. Bentivoglio *et al.*, "Analysis of blink rate patterns in normal subjects," *Mov. Disord.*, vol. 12, no. 6, pp. 1028–1034, 1997, doi:10.1002/mds.870120629 | Blink rate 4.5–26/min, so a rate is undefined in 250 ms. That implication is our reasoning, not the authors' claim. | ABSTRACT-ONLY | ADEQUATE (supports aggregating at all) |
| R1 | D. Tao, H. Tan, H. Wang, X. Zhang, X. Qu, T. Zhang, "A systematic review of physiological measures of mental workload," *Int. J. Environ. Res. Public Health*, vol. 16, no. 15, 2716, 2019, doi:10.3390/ijerph16152716 | 91 studies; 13 eye measures; significant results for blink rate 71%, pupil 79%, fixation duration 73%. **Add to reference list.** No window guidance. | VERIFIED | STRONG (feature set) |

**Corrections and cautions**

- **[13] Kaczorowska 2021b:**
  - Features were computed over whole task parts (90/90/180 s), not windows.
  - Do not cite it for temporal aggregation.
  - It is still valid for D12 as a participant-disjoint hold-out (6 test participants).
    Do not call it LOSO.
- **COLET:**
  - Window length and validation are **unverified**; only the abstract was read because
    ScienceDirect blocked access.
  - Open the publisher PDF manually before citing it for either.
- **Suggested Chapter 3 wording:** "consistent with prior eye-tracking workload studies
  using 30 s windows [W30a, W30b], while acknowledging that longer windows can perform
  better [W30b, W30c]; the value was chosen to balance rate-feature stability against the
  number of epochs."

### T4: Lighting confound, shortcut learning, GAZELOAD facts (D10, dataset)

**Track verdict:**
- **D10 goes from NONE to ADEQUATE / STRONG.** Treating lux as a confound and ablating it
  is well supported.
- **How to frame it:** the direct luminance literature is about **pupil** (strong, 5+
  sources). We have no pupil data, so the argument that fits us is **shortcut learning**:
  lux is a contextual variable correlated with task, and task with the label (strong,
  4 sources, all imaging).
- **Gaps:** no strong source shows luminance affects blink or gaze metrics, and nothing
  applies shortcut learning to eye-tracking data. The thesis must make that link itself
  and say so.

**GAZELOAD facts (verified from the arXiv v1 full text)**

- **Citation:** B. Karbouj, B. E. Gaaloul, J. Krüger, "GAZELOAD: A multimodal eye-tracking
  dataset for mental workload in industrial human–robot collaboration," arXiv:2601.21829
  [cs.RO], Jan. 2026, doi:10.48550/arXiv.2601.21829.
  - Data: Mendeley Data, doi:10.17632/9smd7nbtwc.1, CC BY 4.0.
  - **Preprint only.** No journal version was found.
- **Participants:** "Twenty-six participants ... (16 male, 10 female; mean age 23.3 years,
  SD 2.6, range 20-34)". This **resolves the thesis-notes §2.1 question**: 16/10 is correct.
- **Hardware:** Meta ARIA Gen 1; UR5 and Franka Emika Panda robots.
- **Tasks:** three workload-graded blocks mapped to five sessions.
  - Low: sessions 1 and 2.
  - Medium: sessions 3 and 4, with pre-simulated faults.
  - High: session 5, both benches, overlapping faults.
- **Rating:** "a self-rating of perceived mental workload on a 1–10 Likert scale". No anchors
  are given.
- **Lux:**
  - Recorded by a VEML7700 sensor at 1 Hz, interpolated to 4 Hz, "to account for
    contextual influences on cognitive load".
  - The abstract mentions "controlled manipulations of task difficulty **and ambient
    conditions**". The Methods never say whether ambient light was crossed with task or
    confounded with it.
  - The authors themselves list lighting's influence on eye markers as a use case.
- **Other:** no baseline ML results in the paper; stated limitations are a single lab, a
  narrow age range, and no EEG/ECG.

**Luminance confounds ocular workload measures (pupil-specific)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| L1 | M. Lohani, B. R. Payne, D. L. Strayer, "A review of psychophysiological measures to assess cognitive states in real-world driving," *Front. Hum. Neurosci.*, vol. 13, 57, 2019, doi:10.3389/fnhum.2019.00057 | In the field, luminance co-varying with conditions is "a critical confounding factor". | VERIFIED | STRONG |
| L2 | S. R. Steinhauer, M. M. Bradley, G. J. Siegle, K. A. Roecklein, A. Dix, "Publication guidelines and recommendations for pupillary measurement in psychophysiological studies," *Psychophysiology*, vol. 59, no. 4, e14035, 2022, doi:10.1111/psyp.14035 | The light reflex can exceed the psychological effect; measure and report luminance. | VERIFIED | STRONG |
| L3 | B. Pfleging, D. K. Fekety, A. Schmidt, A. L. Kun, "A model relating pupil diameter to mental workload and lighting conditions," *Proc. CHI '16*, pp. 5776–5788, 2016, doi:10.1145/2858036.2858117 | Eye-tracking systems must account for lighting. | ABSTRACT-ONLY | STRONG |
| L4 | M. Eckert, T. Robotham, E. A. P. Habets, O. S. Rummukainen, "Pupillary light reflex correction for robust pupillometry in virtual reality," *Proc. ACM Comput. Graph. Interact. Tech.*, vol. 5, no. 2, 2022, doi:10.1145/3530798 | Head-worn trackers: light masks cognitive effects unless corrected. | ABSTRACT-ONLY | ADEQUATE |
| L5 | S. Mathôt, "Pupillometry: Psychology, physiology, and function," *J. Cognition*, vol. 1, no. 1, 16, 2018, doi:10.5334/joc.18 | The same signal is driven by both light and effort. | VERIFIED | STRONG |

**Contextual confounds in wearable ML**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| C1 | P. Schmidt, A. Reiss, R. Dürichen, K. Van Laerhoven, "Wearable-based affect recognition—A review," *Sensors*, vol. 19, no. 19, 4079, 2019, doi:10.3390/s19194079 | Environmental variables confound wearable signals; lab accuracy exceeds field accuracy. | VERIFIED | ADEQUATE (affect) |

**Shortcut learning and confounders**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| S1 | R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nat. Mach. Intell.*, vol. 2, no. 11, pp. 665–673, 2020, doi:10.1038/s42256-020-00257-z | Definition of shortcut learning. It is about deep nets, so apply it as a concept only. | VERIFIED | STRONG |
| S2 | J. R. Zech *et al.*, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs," *PLoS Med.*, vol. 15, no. 11, e1002683, 2018, doi:10.1371/journal.pmed.1002683 | A contextual variable alone gave AUC 0.861. Mirrors lux → task → label. | VERIFIED | STRONG |
| S3 | M. A. Badgeley *et al.*, "Deep learning predicts hip fracture using confounding patient and healthcare variables," *npj Digit. Med.*, vol. 2, 31, 2019, doi:10.1038/s41746-019-0105-1 | Balancing the confounder dropped AUC to 0.52. Analogous to our lux ablation. | ABSTRACT-ONLY | STRONG |
| S4 | A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nat. Mach. Intell.*, vol. 3, no. 7, pp. 610–619, 2021, doi:10.1038/s42256-021-00338-7 | Explainability (SHAP-type) reveals shortcut reliance. | ABSTRACT-ONLY (preprint abstract); check the journal text before quoting | ADEQUATE |

### T6: Dataset choice, leakage, tuning, missing data, small-N (D2, D4, D11, D12, D13)

**Track verdict:**
- **Leakage, nested tuning and missing-data handling are STRONG.**
- **Dataset choice is ADEQUATE:**
  - GAZELOAD is a *lab* mock-up of industrial HRC, not field data. Do not call it
    "real-world".
  - It is not peer-reviewed.
  - No peer-reviewed independent validation of Aria gaze accuracy exists.
- **LOSO is justified as removing subject leakage, not as the best estimator.**
  Varoquaux 2017 notes its high variance.

**Dataset choice (D2)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| DS1 | T. Foulsham, E. Walker, A. Kingstone, "The where, what and when of gaze allocation in the lab and the natural environment," *Vision Res.*, vol. 51, no. 17, pp. 1920–1931, 2011, doi:10.1016/j.visres.2011.07.002 | Real-world gaze differs from lab gaze on the same scenes. | ABSTRACT-ONLY | STRONG |
| DS2 | X. Fu *et al.*, "Implementing mobile eye tracking in psychological research: A practical guide," *Behav. Res. Methods*, vol. 56, no. 8, pp. 8269–8288, 2024, doi:10.3758/s13428-024-02473-6 | Screen-based setups are limited for real interaction. | ABSTRACT-ONLY | STRONG |
| DS3 | N. V. Valtakari *et al.*, "Eye tracking in human interaction: Possibilities and limitations," *Behav. Res. Methods*, vol. 53, no. 4, pp. 1592–1608, 2021, doi:10.3758/s13428-020-01517-x | Wearables suit tasks with movement but give less accurate data. Use it for the advantage and the limitation. | VERIFIED | STRONG |
| DS4 | F. N. Biondi *et al.*, "Distracted worker: Using pupil size and blink rate to detect cognitive load during manufacturing tasks," *Appl. Ergon.*, vol. 106, 103867, 2023, doi:10.1016/j.apergo.2022.103867 | Ocular workload measurement in manufacturing assembly. | ABSTRACT-ONLY | STRONG |
| DS5 | S. Upasani, D. Srinivasan, Q. Zhu, J. Du, A. Leonessa, "Eye-tracking in physical human–robot interaction: Mental workload and performance prediction," *Hum. Factors*, vol. 66, no. 8, pp. 2104–2119, 2024, doi:10.1177/00187208231204704 | Eye tracking for workload in HRC; stationary gaze entropy among the most reliable measures (relevant to our GTE features). | ABSTRACT-ONLY | STRONG (HRC) |
| DS6 | D. C. Niehorster *et al.*, "The impact of slippage on the data quality of head-worn eye trackers," *Behav. Res. Methods*, vol. 52, pp. 1140–1160, 2020, doi:10.3758/s13428-019-01307-0 | Head-worn trackers in research, with data-quality caveats. | ABSTRACT-ONLY | STRONG |
| DS7 | Y. Mansour *et al.*, "Enabling eye tracking for crowd-sourced data collection with Project Aria," *IEEE Access*, vol. 13, pp. 114736–114745, 2025, doi:10.1109/ACCESS.2025.3583623 | The only peer-reviewed Aria gaze paper. Meta authors, so not independent. | ABSTRACT-ONLY | ADEQUATE |
| DS8 | M. D. Wilkinson *et al.*, "The FAIR Guiding Principles for scientific data management and stewardship," *Sci. Data*, vol. 3, 160018, 2016, doi:10.1038/sdata.2016.18 | Public reusable data supports reproducibility. | ABSTRACT-ONLY | STRONG |
| DS9 | J. Pineau *et al.*, "Improving reproducibility in machine learning research," *JMLR*, vol. 22, no. 164, pp. 1–20, 2021 | Same code + data gives reproducibility. | ABSTRACT-ONLY | STRONG |

**Leakage and split design (D4, D12)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| LK1 | S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM TKDD*, vol. 6, no. 4, 15, 2012, doi:10.1145/2382577.2382579 | Definition of leakage. Justifies dropping `tasks`. | ABSTRACT-ONLY | STRONG |
| LK2 | S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, 100804, 2023, doi:10.1016/j.patter.2023.100804 | Taxonomy of leakage: L1.2 preprocessing on train + test, L2 illegitimate features, L3.2 non-independence. **See the D9 risk below.** | VERIFIED | STRONG |
| LK3 | S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, K. P. Kording, "The need to approximate the use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, pp. 1–9, 2017, doi:10.1093/gigascience/gix019 | Record-wise CV "massively overestimates" accuracy; 62-paper review. | VERIFIED | STRONG |
| LK4 | M. A. Little, G. Varoquaux, S. Saeb *et al.*, "Using and understanding cross-validation strategies. Perspectives on Saeb et al.," *GigaScience*, vol. 6, no. 5, pp. 1–6, 2017, doi:10.1093/gigascience/gix020 | The split must match the use case (here: new workers). | VERIFIED | STRONG |
| LK5 | G. Brookshire *et al.*, "Data leakage in deep learning studies of translational EEG," *Front. Neurosci.*, vol. 18, 1373515, 2024, doi:10.3389/fnins.2024.1373515 | Segment-based holdout strongly overestimates performance. Analogous to our 250 ms windows. | ABSTRACT-ONLY | STRONG |
| LK6 | D. R. Roberts *et al.*, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881 | Grouped/blocked CV for correlated data. | ABSTRACT-ONLY | STRONG |

**Nested / grouped tuning (D12)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| NT1 | G. C. Cawley, N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," *JMLR*, vol. 11, pp. 2079–2107, 2010 | Tuning must happen within each fold. | VERIFIED | STRONG |
| NT2 | S. Varma, R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinformatics*, vol. 7, 91, 2006, doi:10.1186/1471-2105-7-91 | Nested CV is almost unbiased. | VERIFIED | STRONG |
| NT3 | G. Varoquaux *et al.*, "Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines," *NeuroImage*, vol. 145, pp. 166–179, 2017, doi:10.1016/j.neuroimage.2016.10.038 | Nested CV that leaves out subjects. **Caveat:** prefers repeated 20% splits over leave-one-out because of variance. | VERIFIED (author copy) | STRONG |

**Missing data (D11)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| MD1 | A. Perez-Lebel, G. Varoquaux, M. Le Morvan, J. Josse, J.-B. Poline, "Benchmarking missing-values approaches for predictive models on health databases," *GigaScience*, vol. 11, giac013, 2022, doi:10.1093/gigascience/giac013 | Adding an indicator matters when data are missing not at random. Native NaN handling in boosted trees does best. Matches our LR vs XGB design exactly. | VERIFIED | STRONG |
| MD2 | M. Van Ness, T. M. Bosschieter, R. Halpin-Gregorio, M. Udell, "The missing indicator method: From low to high dimensions," *Proc. ACM KDD '23*, pp. 5004–5015, 2023, doi:10.1145/3580305.3599911 | Missing indicators help with informative missingness and do not hurt linear models. | ABSTRACT-ONLY | STRONG |
| MD3 | J. Josse, J. M. Chen, N. Prost, G. Varoquaux, E. Scornet, "On the consistency of supervised learning with missing values," *Stat. Papers*, vol. 65, no. 9, pp. 5447–5479, 2024, doi:10.1007/s00362-024-01550-4 | Theory for constant imputation. | ABSTRACT-ONLY | STRONG |

**Small-N and reporting (D13)**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| SN1 | G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018, doi:10.1016/j.neuroimage.2017.06.061 | Small N gives large error bars. Supports reporting the per-subject spread. | ABSTRACT-ONLY | STRONG |
| SN2 | G. Forman, M. Scholz, "Apples-to-apples in cross-validation studies: Pitfalls in classifier performance measurement," *ACM SIGKDD Explor.*, vol. 12, no. 1, pp. 49–57, 2010, doi:10.1145/1882471.1882479 | Pooled vs per-fold AUC give different results. | ABSTRACT-ONLY (approximate) | ADEQUATE |

**Could not be supported**

- Peer-reviewed status of GAZELOAD (preprint).
- Independent validation of Aria gaze accuracy (Engel et al. is arXiv-only; DS7 is by Meta).
- "Wearable workload classifiers are more valid". Present this as ecological-validity
  reasoning only.
- A dedicated source for pooled vs per-subject reporting.
- LOSO as the optimal scheme.

**⚠ New risk for D9, raised by LK2 (L1.2, "preprocessing on train + test"):**
- Per-participant z-scoring of the *held-out* participant uses all of that person's epochs,
  across all five tasks, before prediction.
- It is label-free, but a reviewer can call it test-set preprocessing (transductive). It
  also presumes future data from the person are available.
- **Albuquerque (N1) did the same.** That is a defense to check in its Methods.
- **Fallback:** calibration-based normalization using only each person's first/easiest
  session (§5), so no future test data is used.

### T1: Participant-relative labeling (D7, D8)

**Track verdict:**
- **D8 goes from WEAK to ADEQUATE.** Precedents:
  - Gado 2023 uses an exact subject-wise median split with LOSO.
  - Xu 2026 shows per-person-centred labels beat pooled raw labels under LOSO with XGBoost.
  - Nasri 2024 (T5) is a supporting per-participant split.
  - The per-subject principle is supported in affective computing.
- **The critique of absolute thresholds (a) is STRONG:** Hart & Staveland, Marinescu, Xu.
- **Still NONE:**
  - the specific cut at 4;
  - excluding median-equal ratings (Gado does not say how it handled ties).
  - Both must be stated as design choices.
- **No eye-tracking-only paper uses a per-participant median split.** Present D8 as
  justified by these sources, not as standard practice.

**Sources**

| Ref | Citation | What it shows | Evidence | Grade |
|---|---|---|---|---|
| LB1 | S. Gado, K. Lingelbach, M. Wirzberger, M. Vukelić, "Decoding mental effort in a quasi-realistic scenario...," *Sensors*, vol. 23, no. 14, 6546, 2023, doi:10.3390/s23146546 | N=18, eye + physiology + fNIRS, LOSO. "we performed a subject-wise median split" of NASA-TLX effort. Mean threshold 3.8 (SD 3.2): personal medians differ. No stated reason for the split; tie handling not described. | VERIFIED | STRONG (precedent) |
| LB2 | R. Xu, S. Cao, M. Barnett-Cowan, E. Irving, E. Niechwiej-Szwedo, S. Kearns, "Consumer-grade wearable sensors for classifying pilot workload and stress during real flight training: A leave-one-subject-out validation study," *Sensors*, vol. 26, no. 12, 3627, 2026, doi:10.3390/s26123627 | 35 pilots, single-item rating per segment (close to GAZELOAD's design), ECG/EDA, LOSO, XGBoost. "Raw-score labels conflate between-pilot scale-use differences with genuine within-pilot state variation". Per-person residual labels beat a raw median split (F1 0.598 vs 0.523). | VERIFIED | STRONG |
| LB3 | A. C. Marinescu, S. Sharples, A. C. Ritchie, T. Sánchez López, M. McDowell, H. P. Morvan, "Physiological parameter response to variation of mental workload," *Hum. Factors*, vol. 60, no. 1, pp. 31–56, 2018, doi:10.1177/0018720817733101 | N=10. Ratings have "limited absolute validity" but robust "relative validity"; "strong interparticipant differences". Builds no binary labels. | VERIFIED | STRONG (critique of a) |
| LB4 | H. P. Martínez, G. N. Yannakakis, J. Hallam, "Don't classify ratings of affect; rank them!," *IEEE Trans. Affect. Comput.*, vol. 5, no. 3, pp. 314–326, 2014, doi:10.1109/TAFFC.2014.2352268 | Different subjective scales across users are "safely bypassed" by per-subject relative labels. About affect, and favours ranking. | VERIFIED | ADEQUATE |
| LB5 | S. G. Hart, L. E. Staveland, "Development of NASA-TLX...," in *Human Mental Workload*, pp. 139–183, 1988, doi:10.1016/S0166-4115(08)62386-9 (existing [16]) | "high between-subject variability"; workload definitions differ between subjects. | VERIFIED | STRONG (critique of a) |
| LB6 | H. Baumgartner, J.-B. E. M. Steenkamp, "Response styles in marketing research: A cross-national investigation," *J. Mark. Res.*, vol. 38, no. 2, pp. 143–156, 2001, doi:10.1509/jmkr.38.2.143.18840 | Response styles contaminate ratings. General survey methodology only. | ABSTRACT-ONLY | Secondary only |
| LB7 | F. G. W. C. Paas, "Training strategies for attaining transfer of problem-solving skill in statistics: A cognitive-load approach," *J. Educ. Psychol.*, vol. 84, no. 4, pp. 429–434, 1992, doi:10.1037/0022-0663.84.4.429 | The single-item mental effort scale; its details are confirmed only by secondary sources. GAZELOAD's 1–10 scale is "Paas-style", not the Paas scale. | Metadata only | ADEQUATE (background) |

**Corrections**

- **Ref [9] has the wrong title.** "Exploring the relationship between mental workload,
  variation in performance, and physiological parameters" is a *different* 2016 IFAC
  conference paper (doi:10.1016/j.ifacol.2016.10.618).
  - PubMed 28965433 is LB3: "Physiological parameter response to variation of mental
    workload", *Hum. Factors* 2018.
  - Use LB3.
  - Rephrase the claim to "limited absolute validity of rating numbers; stronger relative
    validity". The "affects binary label construction" claim belongs to LB2.
- **Gado:** cite it only for *using* a subject-wise median split with LOSO. Do not claim it
  justified the split or excluded median ties.

**Options raised for D7 / D8**

- **D7 (≥4):** do not look for a citation for the number.
  - Present Config A as the naive pooled-threshold baseline that the literature critiques
    (LB2, LB3, LB5).
  - Justify 4 by class balance: 53.0% positive (`results.md`). Confirm with the team that
    this was the original rationale.
- **D8 alternative:** Xu-style per-person-centred labels (rating minus the participant's
  mean) as a robustness check. Their version also removes the task mean, which would
  remove our signal. Use only the participant-centred part.

**Suggested framing for D10:** lux was retained because the dataset authors recorded it
to capture contextual influence [GAZELOAD]. Luminance is a known confound of ocular
workload measures in field settings [L1, L2, L5]. A contextual variable correlated with
the label can act as a shortcut [S1–S3], so a lux-free ablation (Config F) isolates the
oculomotor signal. That supports **F as the fair eye-tracking comparison**.

---

## 9. Action log

| Date | Action | Outcome |
|---|---|---|
| 2026-10-01 | Reviewed Ch 1–3 PDF, markdown draft, methodology, results, notes | Submitted RRL describes a different study; draft RRL is thin on D5, D7–D10, D16 |
| 2026-10-01 | Spot-checked PDF references online | 8 problems found (§4.2) |
| 2026-10-01 | Compared initial Ch 1–5 draft PDF against outputs | Draft uses stale Aug 4 run; Config C significance does not reproduce |
| 2026-10-01 | Set quality bar (§2) and started research tracks T1–T6 | Awaiting results |
| 2026-10-01 | T2 returned | D9 WEAK → ADEQUATE; Albuquerque verified (per-subject z-score + LOSO); [13] must not be cited for D9 |
| 2026-10-01 | T6 returned | D4/D11/D12 → STRONG; dataset choice ADEQUATE (lab testbed, preprint, no independent Aria validation); Saeb title corrected; D9 transductive-normalization risk flagged |
| 2026-10-01 | Dataset survey + COLET usage survey + LR/XGB anchor search | COLET, ADABase, CL-Drive, Pillai ranked; see `rrl-master-list.md` §4b, §9b |
| 2026-10-01 | Review fix pass | Resolved R1–R6, M6, N1, N3, N4, N8–N10, H1–H10, D1, D2 (see `review-issues-status.md`). `methodology-colet.md` rewritten cleanly (old version archived). M1 checked: `norm_pos` is world-camera coordinates. Researcher notes N0b, N13–N16 added. Remaining decisions: C4, M1, M2, M3, M4, M5, M7, M8, N2, N5, N6. |
| 2026-10-01 | Event detection noted for researchers | The first-pass EDA used a hand-rolled detector with unsourced details; its fixation/saccade numbers did not match COLET Table 4 and must not be used. Final pipeline: COLET's cited recipe (Duchowski FIR filter; Salvucci & Goldberg I-VT 45°/s; 55 ms), validated against Table 4. See `methodology-colet.md` §3c. |
| 2026-10-01 | Label check | Every A1 = low and every A4 = high (condition labels). All 47/47 participants rated A4 above A1 on RTLX (median +31.7). A2/A3 are used only in the manipulation check and optional extras. Normalization over all 4 activities proposed. |
| 2026-10-01 | Team: items on hold / pinned | **On hold:** adviser sign-off for COLET; ADABase email. **Pinned:** confirming 5 reference titles; checking pages/issues against DOIs; full-text checks of abstract-only sources; stimulus-luminance check. |
| 2026-10-01 | COLET downloaded, decoded and converted; P0 inventory run | `data/colet/parquet/` + `inventory.csv`. Activities last 11–141 s (median 29–62 s), so 30 s windows are infeasible. 240 Hz per eye (corrected 2026-10-02: ≈ 242 Hz gaze stream, ≈ 121 Hz pupil per eye). Blink confound confirmed (0–2 vs 13–15 blinks/min). Unit-of-analysis decision pending (`methodology-colet.md` §2c). |
| 2026-10-01 | Archived GAZELOAD-era files | Docs, outputs and old drafts moved to `archive/` (see `archive/README.md`). File paths cited earlier in this log now live under `archive/gazeload/` or `archive/old-drafts/`. |
| 2026-10-01 | COLET full PDF read (added to the repo by the team) | Confirmed setup and labels. Time pressure weak; secondary task is spoken (blink confound); durations not reported. Original A1/A4 LR = 0.85 under a random split with leakage. Plan updated (`methodology-colet.md` §2b). |
| 2026-10-01 | Team decision: go with COLET for now; ADABase on hold | Pending adviser approval; methodology to be adapted |
| 2026-10-01 | Team clarified scope | COLET + GAZELOAD cross-dataset was rejected; revised title adopted; D2 resolved |
| 2026-10-01 | All six tracks complete | Ready for team brainstorm on remaining weak points (§6) |
| 2026-10-01 | T1 returned | D8 WEAK → ADEQUATE (Gado exact precedent; Xu 2026 strongest); [9] wrong title; D7 value uncitable, so frame A as naive baseline |
| 2026-10-01 | T4 returned | D10 NONE → ADEQUATE via luminance (pupil) + shortcut-learning literature; GAZELOAD facts verified; sex split 16M/10F confirmed |
| 2026-10-01 | T3 returned | 30 s has precedent (Rahman 2021, Hogervorst 2014) but no optimality evidence; counter-evidence exists; [13] confirmed miscited for aggregation; Tao 2019 verified |
| 2026-10-01 | T5 returned | Synthesis table (9 studies); gap confirmed; D12 → ADEQUATE; [18] wrong authors; [19] demoted; [3] replacement found; Nasri = 2nd per-participant label precedent (D8) |
| 2026-10-01 | Round-3 review (`review-issues-status-3.md`): team approved the items that add no analyses | V3-2, V3-3, V3-4 decided (register above); V3-5: Karunathilake dropped from V2-1 support, Stolte (X9) added to `rrl-colet-list.md`, Hefron marked EEG in N7; V3-6 wording fixed. **Still open:** V3-1 (refit handling vs P2), V2-6, V2-13. Scope check of round 3 in researcher-notes N24. |
| 2026-10-01 | Round-4 review (`review-issues-status-4.md`): team approved R4-1, R4-2 and the V3-1 limitation sentence | D-M1 revised to `gaze_normal0/1`; ±15% cross-check removed; V3-4 superseded; sanity check per activity. Our label-blind check (N25, `colet-eda/check_r4_1.py`) reproduced the reviewer's numbers and found that each eye alone is biased about 10° inward, so one-eye samples are proposed as missing (pending team OK). RRL search for the per-eye input running. |
| 2026-10-02 | Team decisions | **Methodology frozen** (round-4 reviewer's proposal accepted): further changes only if the data shows a Critical problem. **Colab notebook required** (team will run on Colab; pinned versions, N11). Same day: **one-eye samples → missing approved** (N25); **C8 optional**. |
| 2026-10-01 | V2-6 full-text reads | Fenoglio 2023a (V): COLET with unseen test users (4-fold user-grouped CV), no LR/XGBoost/SHAP, so the gap claim "COLET studies used random splits" was reworded (`methodology-colet.md` §1). Rolon-Merette 2026 (V): no feature importance; LOPO only 10 runs. Fenoglio 2023b: abstract only. RRL for the gaze input added (PP17–PP19). |

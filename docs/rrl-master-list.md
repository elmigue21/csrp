# RRL Master List

**Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

This file lists every related-literature source found for the thesis, organized by the
parts of the title. It is the source pool for re-planning the methodology (see
`rrl-decision-log.md` for decisions and their status).

Last updated: 2026-10-01

## How to read this list

- **Every entry was confirmed** on a publisher, DOI, PubMed/PMC or proceedings page.
- **Evidence column:**
  - **VERIFIED** = full text read and the claim matched.
  - **ABSTRACT** = only the abstract read. Cite for what the abstract says; check the full
    text before describing its methods.
  - **META** = metadata confirmed only.
- **Grade:** STRONG / ADEQUATE / WEAK, judged for *our* use of the source.
- **Preprints** are allowed only where marked (the dataset itself).
- **Section 11** lists sources that must **not** be cited, or must be corrected.

---

## 1. Cognitive load: theory and measurement

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| CL1 | D. Kahneman, *Attention and Effort*. Prentice-Hall, 1973 | Mental effort as a limited resource | META | ADEQUATE (background) |
| CL2 | F. Paas, J. J. G. van Merriënboer, "Cognitive-load theory: Methods to manage working memory load in the learning of complex tasks," *Curr. Dir. Psychol. Sci.*, vol. 29, no. 4, pp. 394–398, 2020 | Cognitive load theory | META | ADEQUATE |
| CL3 | M. S. Young, K. A. Brookhuis, C. D. Wickens, P. A. Hancock, "State of science: Mental workload in ergonomics," *Ergonomics*, vol. 58, no. 1, pp. 1–17, 2015, doi:10.1080/00140139.2014.956151 | Workload affects performance (review level; do not claim quantified error rates) | ABSTRACT | STRONG |
| CL4 | S. G. Hart, L. E. Staveland, "Development of NASA-TLX...," in *Human Mental Workload*, pp. 139–183, 1988, doi:10.1016/S0166-4115(08)62386-9 | Subjective workload measurement; "high between-subject variability" in ratings | VERIFIED | STRONG |
| CL5 | F. G. W. C. Paas, "Training strategies for attaining transfer of problem-solving skill in statistics: A cognitive-load approach," *J. Educ. Psychol.*, vol. 84, no. 4, pp. 429–434, 1992, doi:10.1037/0022-0663.84.4.429 | Single-item mental-effort rating (GAZELOAD's 1–10 scale is "Paas-style") | META | ADEQUATE |
| CL6 | A. C. Marinescu *et al.*, "Physiological parameter response to variation of mental workload," *Hum. Factors*, vol. 60, no. 1, pp. 31–56, 2018, doi:10.1177/0018720817733101 | Ratings have "limited absolute validity" but robust relative validity; strong differences between participants | VERIFIED | STRONG |
| CL7 | D. Tao *et al.*, "A systematic review of physiological measures of mental workload," *Int. J. Environ. Res. Public Health*, vol. 16, no. 15, 2716, 2019, doi:10.3390/ijerph16152716 | 91 studies; 13 eye measures; blink rate significant in 71%, fixation duration 73%, pupil 79% | VERIFIED | STRONG |

## 2. Eye tracking and eye-tracking features for cognitive load

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| ET1 | A. T. Duchowski, *Eye Tracking Methodology*, 3rd ed. Springer, 2017, doi:10.1007/978-3-319-57883-5 | Eye-tracking fundamentals and metrics | META | STRONG (textbook) |
| ET2 | G. Bargary *et al.*, "Individual differences in human eye movements: An oculomotor signature?" *Vision Res.*, vol. 141, pp. 157–169, 2017, doi:10.1016/j.visres.2017.03.001 | Stable personal eye-movement "signature" (N>1000) | ABSTRACT | STRONG |
| ET3 | W. Poynter, M. Barber, J. Inman, C. Wiggins, "Individuals exhibit idiosyncratic eye-movement behavior profiles across tasks," *Vision Res.*, vol. 89, pp. 32–38, 2013, doi:10.1016/j.visres.2013.07.002 | Personal fixation and saccade profiles are stable across tasks | ABSTRACT | STRONG |
| ET4 | M. S. Castelhano, J. M. Henderson, "Stable individual differences across images in human saccadic eye movements," *Can. J. Exp. Psychol.*, vol. 62, no. 1, pp. 1–14, 2008, doi:10.1037/1196-1961.62.1.1 | Eye movements differ between people but are consistent within a person | ABSTRACT | STRONG |
| ET5 | A. R. Bentivoglio *et al.*, "Analysis of blink rate patterns in normal subjects," *Mov. Disord.*, vol. 12, no. 6, pp. 1028–1034, 1997, doi:10.1002/mds.870120629 | Blink rate is 4.5–26/min, so rates need multi-second windows | ABSTRACT | ADEQUATE |
| ET6 | S. Upasani, D. Srinivasan, Q. Zhu, J. Du, A. Leonessa, "Eye-tracking in physical human–robot interaction: Mental workload and performance prediction," *Hum. Factors*, vol. 66, no. 8, pp. 2104–2119, 2024, doi:10.1177/00187208231204704 | HRC workload; stationary gaze entropy among the most reliable measures | ABSTRACT | STRONG |
| ET7 | F. N. Biondi *et al.*, "Distracted worker: Using pupil size and blink rate to detect cognitive load during manufacturing tasks," *Appl. Ergon.*, vol. 106, 103867, 2023, doi:10.1016/j.apergo.2022.103867 | Ocular workload measures in manufacturing | ABSTRACT | STRONG |
| ET8 | J. G. May, R. S. Kennedy, M. C. Williams, W. P. Dunlap, J. R. Brannan, "Eye movement indices of mental workload," *Acta Psychol.*, vol. 75, pp. 75–89, 1990, doi:10.1016/0001-6918(90)90067-p | Saccadic extent decreases as workload increases | ABSTRACT | STRONG |
| ET9 | L. L. Di Stasi *et al.*, "Saccadic peak velocity sensitivity to variations in mental workload," *Aviat. Space Environ. Med.*, vol. 81, no. 4, pp. 413–417, 2010, doi:10.3357/asem.2579.2010 | Peak velocity decreases with workload | ABSTRACT | STRONG (for peak velocity) |
| ET10 | L. L. Di Stasi, A. Antolí, J. J. Cañas, "Main sequence: An index for detecting mental workload variation in complex tasks," *Appl. Ergon.*, vol. 42, no. 6, pp. 807–813, 2011, doi:10.1016/j.apergo.2011.01.003 | Peak velocity is sensitive to workload | ABSTRACT | ADEQUATE |
| ET11 | L. L. Di Stasi *et al.*, "Gaze entropy reflects surgical task load," *Surg. Endosc.*, vol. 30, pp. 5034–5043, 2016, doi:10.1007/s00464-016-4851-8 | Mobile tracker; gaze entropy and velocity **increase** with complexity (counter-direction) | ABSTRACT | ADEQUATE |
| ET12 | M. A. Recarte, E. Pérez, A. Conchillo, L. M. Nunes, "Mental workload and visual impairment: Differences between pupil, blink, and subjective rating," *Span. J. Psychol.*, vol. 11, pp. 374–385, 2008, doi:10.1017/s1138741600004406 | Cognitive load raises blink rate; visual demand inhibits it | ABSTRACT | STRONG |
| ET13 | J. A. Navia *et al.*, *Front. Psychol.*, vol. 16, 1644721, 2025, doi:10.3389/fpsyg.2025.1644721 | Blink frequency decreased under high traffic load | VERIFIED | ADEQUATE |
| ET14 | Y. Wang, B. Reimer, J. Dobres, B. Mehler, "The sensitivity of different methodologies for characterizing drivers' gaze concentration under increased cognitive demand," *Transp. Res. F*, vol. 26, pp. 227–237, 2014, doi:10.1016/j.trf.2014.08.003 | SD of horizontal gaze has the largest effect; scanning narrows under load | ABSTRACT | STRONG |
| ET15 | M. A. Recarte, L. M. Nunes, "Mental workload while driving: Effects on visual search, discrimination, and decision making," *J. Exp. Psychol. Appl.*, vol. 9, pp. 119–137, 2003, doi:10.1037/1076-898x.9.2.119 | Mental tasks concentrate gaze spatially | ABSTRACT | STRONG |
| ET16 | M. A. Recarte, L. M. Nunes, "Effects of verbal and spatial-imagery tasks on eye fixations while driving," *J. Exp. Psychol. Appl.*, vol. 6, pp. 31–43, 2000, doi:10.1037//1076-898x.6.1.31 | Visual field narrows under load | ABSTRACT | STRONG |
| ET17 | B. Shiferaw, L. Downey, D. Crewther, "A review of gaze entropy as a measure of visual scanning efficiency," *Neurosci. Biobehav. Rev.*, vol. 96, pp. 353–366, 2019, doi:10.1016/j.neubiorev.2018.12.007 | Gaze transition entropy as a construct; notes methodological limits | ABSTRACT | ADEQUATE |
| ET18 | C. Diaz-Piedra *et al.*, "The effects of flight complexity on gaze entropy...," *Appl. Ergon.*, vol. 77, pp. 92–99, 2019, doi:10.1016/j.apergo.2019.01.012 | Gaze entropy decreases with complexity | ABSTRACT | ADEQUATE |
| ET19 | P. Maggi, F. Di Nocera, "Sensitivity of the spatial distribution of fixations to variations in the type of task demand and its relation to visual entropy," *Front. Hum. Neurosci.*, vol. 15, 642535, 2021, doi:10.3389/fnhum.2021.642535 | Fixation spatial dispersion (nearest-neighbour index); direction depends on demand type | VERIFIED | WEAK–ADEQUATE |
| ET20 | S. Scannella *et al.*, "Assessment of ocular and physiological metrics to discriminate flight phases in real light aircraft," *Hum. Factors*, vol. 60, no. 7, pp. 922–935, 2018, doi:10.1177/0018720818787135 | Saccade rate the best discriminator in real flight | ABSTRACT | ADEQUATE |
| ET21 | J. C. Liu, K. A. Li, S. L. Yeh, S. Y. Chien, "Assessing perceptual load and cognitive load by fixation-related information of eye movements," *Sensors*, vol. 22, no. 3, 1187, 2022, doi:10.3390/s22031187 | More fixations under perceptual load, fewer under cognitive load | ABSTRACT | ADEQUATE |
| ET22 | K. F. Van Orden, W. Limbert, S. Makeig, T.-P. Jung, "Eye activity correlates of workload during a visuospatial memory task," *Hum. Factors*, vol. 43, pp. 111–121, 2001, doi:10.1518/001872001775992570 | Blink frequency, fixation frequency and pupil the most predictive | ABSTRACT | STRONG |

**Our GAZELOAD features against the literature (R3)**

"Direction" is the expected change under higher load. Only six of our features appear in
Tao's list of 13 (CL7). The four best-supported measures there (pupil, fixation duration,
blink duration, blink amplitude) are **not in the dataset**.

| Our feature | Literature construct | Sources | Direction | Grade | Recommendation |
|---|---|---|---|---|---|
| Blink rate (from blink flag) | Blink rate | CL7, ET12, ET13, ET22 | ↓ for visual load, ↑ for cognitive load | **STRONG** | Keep; state the direction depends on task type |
| Gaze direction SD (x/y), gaze position SD | Gaze dispersion / concentration | ET14, ET15, ET16, CL7 | ↓ (gaze concentrates) | **STRONG** (driving) | Keep; best spatial features. Aria gaze is eye-in-head. |
| Saccade amplitude | Saccadic amplitude/extent | ET8, CL7, ET22 | ↓ | **ADEQUATE** | Keep |
| Fixation count per window | Fixation rate | CL7, ET21, ET22 | Mixed (↑ perceptual, ↓ cognitive) | **ADEQUATE** | Keep; rename "fixation rate" |
| Saccade count / saccade rate | Saccade rate | CL7, ET20 | Mostly ↓ | **ADEQUATE−** | Keep one of the two (exact duplicates) |
| Saccade velocity | Saccadic peak velocity | ET9, ET10, ET11 | ↓ (counter-evidence ET11) | **WEAK** for our mean-velocity feature | Exploratory; it mostly re-encodes amplitude |
| GTE | Gaze transition entropy | ET17, ET18, ET11 | Inconsistent | **WEAK–ADEQUATE** | Exploratory; epoch-level only; undocumented in GAZELOAD |
| FDI | Fixation spatial dispersion (NNI) | ET19, CL7 | Depends on demand type | **WEAK** | Exploratory; undocumented in GAZELOAD |
| Mean gaze direction / position | None | — | — | **NONE** | Drop as load features (risk of learning layout or identity) |
| Proportion of empty windows | None (rate complement) | — | — | **NONE** | Data-quality / exploratory only |
| Lux | Confound, not an indicator | CF1–CF9 | — | **NONE** as a feature | Covariate or confound check only |
| *(not in data)* Stationary gaze entropy | SGE | ET6 (Upasani, HRC) | ↓ | ADEQUATE | Could be computed per epoch from gaze position x/y |

**Open data questions:**
- GAZELOAD's paper does not define GTE, FDI or blink detection. Check
  `01_Metadata/metrics_extraction.py` in the dataset download for the definitions.
- Its fixation/saccade detection is I-VT (30°/s threshold, 60 ms minimum fixation, 75 ms
  merge gap).

## 3. Empirical eye-tracking workload classification studies (closest precedents)

| ID | Citation | Pipeline notes | Evidence | Grade |
|---|---|---|---|---|
| EM1 | M. Kaczorowska, M. Plechawska-Wójcik, M. Tokovarov, "Interpretable machine learning models for three-way classification of cognitive workload levels for eye-tracking features," *Brain Sci.*, vol. 11, no. 2, 210, 2021, doi:10.3390/brainsci11020210 | N=29, DSST difficulty levels (condition labels), participant-disjoint hold-out, LR among 8 models, LR coefficients for importance. Features include DSST performance. | VERIFIED | STRONG |
| EM2 | M. Kaczorowska, P. Karczmarek, M. Plechawska-Wójcik, M. Tokovarov, "On the improvement of eye tracking-based cognitive workload estimation using aggregation functions," *Sensors*, vol. 21, no. 13, 4542, 2021, doi:10.3390/s21134542 | Participant-disjoint hold-out (6 test participants). "Aggregation" means combining classifier outputs, **not time windows**. | VERIFIED | ADEQUATE (validation only) |
| EM3 | A. Rizzo, S. Ermini, D. Zanca, D. Bernabini, A. Rossi, "A machine learning approach for detecting cognitive interference based on eye-tracking data," *Front. Hum. Neurosci.*, vol. 16, 806330, 2022, doi:10.3389/fnhum.2022.806330 | N=64, Stroop, fixation and saccade features, subject-wise normalization, RF/LR/ANN/SVM | VERIFIED | ADEQUATE |
| EM4 | T. Rolon-Merette *et al.*, "Towards a cross-participant cognitive load classification using eye tracking and deep learning," *Proc. FLAIRS*, vol. 39, no. 1, 2026, doi:10.32473/flairs.39.1.141863 | N=89, n-back (condition labels), LR vs XGBoost cross-participant: LR ~52%, XGB >75% | VERIFIED | ADEQUATE (pair with a stronger source) |
| EM5 | T. Božak, S. Goyal, M. Langheinrich, M. Gjoreski, G. Slapničar, "Evaluating feature-based machine-learning models with post hoc explainability for eye-tracking-based task type and workload inference," *AI*, vol. 7, no. 8, 325, 2026, doi:10.3390/ai7080325 | N=54, LOSO + leave-one-group-out, SHAP; subject-normalized features strong | ABSTRACT (R1 reading full text) | ADEQUATE |
| EM6 | T. Appel, C. Scharinger, P. Gerjets, E. Kasneci, "Cross-subject workload classification using pupil-related measures," *Proc. ACM ETRA*, 2018, doi:10.1145/3204493.3204531 | Cross-subject; "normalized features"; classifiers are "highly subject-dependent" | ABSTRACT | STRONG |
| EM7 | E. Ktistakis *et al.*, "COLET: A dataset for COgnitive workLoad estimation based on eye-tracking," *Comput. Methods Programs Biomed.*, vol. 224, 106989, 2022, doi:10.1016/j.cmpb.2022.106989 | N=47, puzzles, NASA-TLX labels, up to 88% | ABSTRACT | STRONG (feasibility) |
| EM8 | H. Rahman, M. U. Ahmed, S. Barua, P. Funk, S. Begum, "Vision-based driver's cognitive load classification considering eye movement using machine learning and deep learning," *Sensors*, vol. 21, no. 23, 8019, 2021, doi:10.3390/s21238019 | N=33 drivers; 15/30/60 s windows, 30 s best; per-window mean/SD/max features | VERIFIED | STRONG |
| EM9 | M. A. Hogervorst, A.-M. Brouwer, J. B. F. van Erp, "Combining and comparing EEG, peripheral physiology and eye-related measures for the assessment of mental workload," *Front. Neurosci.*, vol. 8, 322, 2014, doi:10.3389/fnins.2014.00322 | n-back (condition labels); 30 s vs 120 s segments | VERIFIED | STRONG |
| EM10 | M. Nasri, M. Kosa, L. Chukoskie, M. Moghaddam, C. Harteveld, "Exploring eye tracking to detect cognitive load in complex virtual reality training," *Proc. IEEE ISMAR-Adjunct*, pp. 51–54, 2024, doi:10.1109/ISMAR-Adjunct64951.2024.00022 | N=19; NASA-TLX split per participant; MLP/RF | VERIFIED | WEAK (short paper) |
| EM11 | S. Karunathilake, N. A. Choudhury, A. Deep, P. Saravanan, "Predicting cognitive workload in visuospatial tasks using pupillometry: A machine learning approach," *Proc. HFES Annu. Meet.*, vol. 69, no. 1, pp. 1680–1686, 2025, doi:10.1177/10711813251357928 | N=20, Lego task, RF/XGB/LSTM, K-means labels | ABSTRACT | WEAK |
| EM12 | M. Trigka, E. Dritsas, P. Mylonas, "Eye-based cognitive overload prediction in human-machine interaction via machine learning," *Proc. WEBIST*, 2025, doi:10.5220/0013782800003985 | N=9; labels derived from gaze features (circular); not subject-independent; no SHAP. XGB beat LR. | VERIFIED | WEAK (contrast case only) |
| EM13 | J. Wei *et al.*, "Cognitive load inference using physiological markers in virtual reality," *Proc. IEEE VR*, 2025, doi:10.1109/VR59515.2025.00098 | N=738; per-individual normalization; participant-disjoint test | VERIFIED | ADEQUATE |
| EM14 | C. Wu *et al.*, *Hum. Factors*, 2020, doi:10.1177/0018720819874544 | Robotic surgery; N=8; **Tobii Pro Glasses 2**; NASA-TLX self-report with top/bottom quartiles, middle dropped; gaze entropy; Naïve Bayes 84.7% | VERIFIED | ADEQUATE (small N, split unclear) |
| EM15 | S. B. Shafiei *et al.*, *Hum. Factors*, 2025, doi:10.1177/00187208241285513 | Surgery; N=26; Tobii glasses; SURG-TLX continuous; **per-participant z-score**; **XGBoost + SHAP**; random 20% split | VERIFIED | ADEQUATE (not subject-independent) |
| EM16 | He *et al.*, *Sensors*, vol. 25, 2377, 2025, doi:10.3390/s25082377 | Construction workers in a heat chamber; N=30; **Tobii Glasses 3**; condition labels; **LOSO**; eye-only AUC 0.844 | VERIFIED | STRONG |
| EM17 | Choi, Nam, *J. Eye Mov. Res.*, vol. 19, no. 3, 50, 2026, doi:10.3390/jemr19030050 | VR; **N=26**; condition labels; **no pupil**; **LOSO**; **XGBoost 78.9%** vs RF 73.8% | VERIFIED | STRONG |
| EM18 | M. Kaczorowska *et al.*, *Brain Sci.*, vol. 12, no. 5, 542, 2022, doi:10.3390/brainsci12050542 | DSST; N=30; condition labels; 10 s windows; **no pupil**; subject-independent 80:20 ×200; **LR** among models; LR-based ranking | VERIFIED | STRONG |
| EM19 | T. Appel *et al.*, "Cross-task and cross-participant classification of cognitive load in an emergency simulation game," *IEEE Trans. Affect. Comput.*, 2023, doi:10.1109/TAFFC.2021.3098237 | Condition labels; 4 s windows; baseline subtraction then **per-participant z-transform**; cross-participant + cross-task; self-report only correlated (r≈.40–.48) | VERIFIED | STRONG |
| EM20 | Wozniak, Zahabi, *Appl. Ergon.*, vol. 119, 104305, 2024 | Real police patrol; N=24; **Pupil Labs glasses**; composite labels (circular); 5-min windows; 2-min pupil baseline; random split | VERIFIED | WEAK (circular labels, record-level split) |
| EM21 | Aygun *et al.*, *Sensors*, vol. 22, 6834, 2022, doi:10.3390/s22186834 | Driving; N=43; **Pupil Core**; condition labels; pre-stimulus baseline; 80/20 split | VERIFIED | ADEQUATE |
| EM22 | Xu *et al.*, *Ergonomics*, 2026, doi:10.1080/00140139.2025.2511877 | Flight cadets; N=26; condition labels; RF best; SHAP | ABSTRACT | ADEQUATE |
| EM23 | Xiao *et al.*, *Sensors*, vol. 26, 4835, 2026, doi:10.3390/s26154835 | Tobii Glasses 2, N=30. The only **median split of NASA-TLX** found. Statistics only, not ML; no significant eye features. | VERIFIED | WEAK (not ML) |
| EM24 | Bakhchina *et al.*, *J. Eye Mov. Res.*, vol. 19, no. 1, 2026, doi:10.3390/jemr19010001 | 685 drivers; gaze transition entropy the best indicator (supports keeping GTE) | VERIFIED | ADEQUATE |

**EM5 Božak 2026: full pipeline (now VERIFIED full text). Our closest twin.**
- **Labels:** condition labels only. Self-reports were a *manipulation check*, "not used as supervised learning targets".
- **Windows:** 3 s, inherited from the dataset protocol and deliberately **not tuned**.
- **Features:**
  - absolute gaze coordinates excluded to prevent leakage;
  - |r|>0.80 correlation pruning, 488 features down to 58.
- **Normalization:** "centering window-level measures on each participant's own recording-level
  summary", plus min-max scaling fitted inside each fold. This is the **same participant-referenced
  approach as ours, in eye tracking**.
- **Validation:** LOSO primary, plus leave-one-task-out.
- **Models:** baselines, kNN, NB, SVM, RF, XGBoost, LightGBM.
- **Metrics:** accuracy, macro-F1, macro-AUC, with a **majority baseline**.
- **Results:** binary load vs rest 81.4% LOSO against a 76% baseline.
- **Pupil-free ablation:** fixation-only reached 78.3%, close to the full model. This supports a
  pupil-free pipeline.
- **Importance:** SHAP + gain + ANOVA/Friedman cross-check.
  - Top features: fixation dispersion, subject-relative pupil, fixation duration.

**Consensus pipeline across the 12 closest studies (R1)**

| Step | What most studies do | Our current choice | Match? |
|---|---|---|---|
| Label source | **Condition / designed difficulty: 8/12.** Self-report 3/12. | Self-report | ❌ Diverges |
| Binarization | Designed low vs high, middle dropped or paired. **No ML study used a median split.** | Per-person median split | ❌ Diverges |
| Self-report role | Manipulation check or secondary analysis | Primary target | ❌ |
| Window | 2.5–10 s most common; whole block; 5 min. Not tuned. | 30 s | ~ Within range (Rahman supports 30 s) |
| Pupil | 10/12 use it; pupil-free studies still work (EM17, EM18, Božak ablation) | None (not available) | ~ Limitation, but precedented |
| Per-person normalization | 6/12. Centering/z-scoring on the participant's own data (Božak, Appel, Shafiei) or a baseline | Per-person z-score on all sessions | ✅ Matches (participant-referenced) |
| Absolute gaze position | Excluded (Božak) | Included | ❌ Drop |
| Correlation pruning | \|r\|>0.80 (Božak) | Duplicates kept | ❌ Add |
| Validation | 6/12 subject-independent; 3 strict LOSO | LOPO | ✅ Strictest option |
| Models | RF 8/12, SVM 6/12, boosting 3/12, LR 2/12. **No LR-vs-XGBoost under LOPO.** | LR vs XGBoost | ✅ Gap we can claim |
| Metric | Accuracy primary 10/12; AUC 4/12; best practice adds a majority baseline | ROC-AUC primary | ~ Add balanced accuracy/macro-F1, per-class recall, baseline |
| Importance | SHAP 3/12; model-native 5/12; Božak cross-checks SHAP with ANOVA | SHAP + LR coefficients + gain | ✅ Add a univariate cross-check |

**Could not read:** Pillai *et al.*, *IEEE/ASME Trans. Mechatronics*, vol. 27, no. 4, 2022,
doi:10.1109/TMECH.2022.3175774 (gaze entropy + NNI, driving). Access was blocked.

**Other research using GAZELOAD (R4 Task A)**

- **No study has analyzed GAZELOAD** as of 2026-10-01.
- **What was searched, all with no modeling hits:**
  - arXiv full text, Semantic Scholar, OpenAlex and Crossref citations;
  - DataCite (0 citations), Google Scholar, GitHub, Kaggle, Hugging Face, Papers-with-Code;
  - the authors' other publications.
- **The only citation found** is a non-peer-reviewed SportRxiv/SSRN preprint (McDonald 2026)
  that mentions the dataset in passing.
- **The GAZELOAD paper itself has no ML, baselines or validation.** It is a data descriptor.
- **Warning:** a search-engine summary claiming GAZELOAD reports "84.7% accuracy" is **false**.
  Do not cite it.
- **For the thesis:** "To our knowledge, this is the first modelling study on GAZELOAD."

**HRC / manufacturing eye-tracking workload studies (R4 Task B)**

| ID | Citation | Pipeline notes | Evidence | Grade |
|---|---|---|---|---|
| HR1 | ET6 Upasani *et al.* 2024 | Physical HRI with a Baxter robot (VR display), N=18. Condition labels (difficulty) + NASA-TLX. **30-s intervals.** Baseline-subtracted pupil; entropies normalized. Logistic regression, no held-out validation. SGE and pupil most reliable. | VERIFIED (via the dissertation chapter) | STRONG |
| HR2 | P. Pluchino *et al.*, *Front. Robot. AI*, vol. 10, 1275572, 2023, doi:10.3389/frobt.2023.1275572 | UR10e cobot assembly, N=11 senior workers, **Pupil Labs glasses**. Condition labels (single vs dual task). Resting baseline. Blink rate rose from 14.9 to 23.3/min under dual task. Statistics only. | VERIFIED | ADEQUATE |
| HR3 | E. Gervasi, M. Capponi, L. Mastrogiacomo, F. Franceschini, *Prod. Eng.*, vol. 19, pp. 47–64, 2025, doi:10.1007/s11740-024-01294-y | 8-h UR3e cobot assembly, N=4–6, **Tobii Pro Glasses 3**. **Per-participant z-scores** to remove personal characteristics. Wilcoxon, no ML. | VERIFIED | ADEQUATE |
| HR4 | F. Nenna, V. Orso, D. Zanardi, L. Gamberini, *Virtual Reality*, vol. 27, pp. 553–571, 2023, doi:10.1007/s10055-022-00667-x | Physical vs VR cobot, N=21. Condition labels. Pupil with subtractive baseline. Mixed models, no ML. | VERIFIED | ADEQUATE |
| HR5 | M. Capponi, E. Gervasi, L. Mastrogiacomo, F. Franceschini, *Robot. Comput.-Integr. Manuf.*, vol. 89, 102789, 2024, doi:10.1016/j.rcim.2024.102789 | Cobot assembly complexity × modality; eye + EDA + HRV | ABSTRACT | ADEQUATE |
| HR6 | Mariscal *et al.*, *Int. J. Comput. Integr. Manuf.*, vol. 37, no. 7, pp. 900–919, 2024, doi:10.1080/0951192X.2023.2263428 | Cobot vs human partner, N=32, pupil; t-tests | ABSTRACT | WEAK |
| HR7 | ET7 Biondi *et al.* 2023 | Manufacturing, n-back condition labels; pupil and blink rate rose with demand | ABSTRACT | STRONG |
| HR8 | Bassi *et al.*, *JMIR*, 2025, doi:10.2196/75658 | Systematic review: only 5 of 46 cobot studies used eye tracking; none used ML on it | ABSTRACT | ADEQUATE (supports the gap) |

**What HRC studies typically do:**
- **Labels:** condition labels. NASA-TLX is collected once per block as a manipulation check.
- **Features:** computed per block or trial, or in windows of about 1.5 s up to **30 s**.
- **Between-person differences:** handled with baselines or per-participant z-scores.
- **Validation:** almost always statistics only. **No verified HRC eye-tracking study used held-out
  or LOPO ML**, so this is a genuine gap.

## 4. Logistic Regression, XGBoost, and comparing them

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| M1 | C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006 | Logistic Regression | META | STRONG (textbook) |
| M2 | T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. ACM SIGKDD*, pp. 785–794, 2016, doi:10.1145/2939672.2939785 | XGBoost, regularization, sparsity-aware splits | META | STRONG |
| M3 | E. Christodoulou *et al.*, "A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models," *J. Clin. Epidemiol.*, vol. 110, pp. 12–22, 2019, doi:10.1016/j.jclinepi.2019.02.004 | In low-bias comparisons, LR ≈ ML (Δlogit AUC 0.00); biased designs favor ML | ABSTRACT | STRONG |
| M4 | S. Nusinovici *et al.*, "Logistic regression was as good as machine learning for predicting major chronic diseases," *J. Clin. Epidemiol.*, vol. 122, pp. 56–69, 2020, doi:10.1016/j.jclinepi.2020.03.002 | LR vs GBM differences small and not significant | ABSTRACT | STRONG |
| M5 | R. Couronné, P. Probst, A.-L. Boulesteix, "Random forest versus logistic regression: A large-scale benchmark experiment," *BMC Bioinformatics*, vol. 19, 270, 2018, doi:10.1186/s12859-018-2264-5 | Counterpoint: trees beat LR on average, depending on the datasets chosen | ABSTRACT | STRONG |
| M6 | L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," *Proc. NeurIPS (Datasets & Benchmarks)*, pp. 507–520, 2022 | Tree ensembles are strong on tabular data (justifies choosing XGBoost) | ABSTRACT | STRONG |
| M7 | R. Shwartz-Ziv, A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84–90, 2022, doi:10.1016/j.inffus.2021.11.011 | XGBoost is the tabular default and needs little tuning | ABSTRACT | ADEQUATE |
| M8 | H. Zou, T. Hastie, "Regularization and variable selection via the elastic net," *J. R. Stat. Soc. B*, vol. 67, no. 2, pp. 301–320, 2005, doi:10.1111/j.1467-9868.2005.00503.x | Regularized LR; handles correlated predictors | ABSTRACT | STRONG |
| M9 | T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998, doi:10.1162/089976698300017197 | Naive paired t-tests on resampled splits inflate Type I error | ABSTRACT | STRONG |
| M10 | C. Nadeau, Y. Bengio, "Inference for the generalization error," *Mach. Learn.*, vol. 52, no. 3, pp. 239–281, 2003, doi:10.1023/A:1024068626366 | Corrected resampled t-test (overlapping training sets) | META | STRONG |
| M11 | A. Benavoli, G. Corani, J. Demšar, M. Zaffalon, "Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis," *JMLR*, vol. 18, no. 77, pp. 1–36, 2017 | Bayesian comparison with a region of practical equivalence, so a result can be called "practically equivalent" | ABSTRACT | STRONG |
| M12 | G. Corani, A. Benavoli, "A Bayesian approach for comparing cross-validated algorithms on multiple data sets," *Mach. Learn.*, vol. 100, pp. 285–304, 2015, doi:10.1007/s10994-015-5486-z | Bayesian correlated t-test | META | ADEQUATE |
| M13 | A. Niculescu-Mizil, R. Caruana, "Predicting good probabilities with supervised learning," *Proc. ICML*, pp. 625–632, 2005, doi:10.1145/1102351.1102430 | Boosted trees can be miscalibrated (AdaBoost-style; check, don't assume) | VERIFIED | STRONG |
| M14 | B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, E. W. Steyerberg, "Calibration: The Achilles heel of predictive analytics," *BMC Med.*, vol. 17, 230, 2019, doi:10.1186/s12916-019-1466-7 | Report calibration for LR and ML alike | ABSTRACT | STRONG |
| M15 | M. Saygin, M. Schoenmakers, M. J. Gevonden, E. J. C. de Geus, "Speech detection via respiratory inductance plethysmography, thoracic impedance, accelerometers, and gyroscopes: A machine learning-informed comparative study," *Psychophysiology*, vol. 62, 2025, doi:10.1111/psyp.70021 | Applied XGBoost vs LR on physiological data with nested CV | ABSTRACT | ADEQUATE |

**Note on EV12 (Demšar):** its Wilcoxon test was designed for independent *datasets*. Using
it across 26 folds whose training sets overlap is an adaptation (M9, M10). State it as a
limitation.

## 4b. Cognitive-load studies that used Logistic Regression and/or XGBoost

These are our methodological anchors. All are peer-reviewed. "Split" means how train and
test were separated.

**Eye-tracking features**

| ID | Study | N / task | Models | Split | LR vs XGB result | Importance | Evidence |
|---|---|---|---|---|---|---|---|
| X1 | EM4 Rolon-Merette *et al.*, FLAIRS 2026 | 89, n-back | LR, XGB, SVM, CNN, Transformer | LOPO (10 held-out participants) | LR **52%** vs XGB **>75%** (LR fed raw time series) | None | VERIFIED |
| X2 | G. Nerella, D. Gan, B. David-John, R. Alghofaili, "Gaze-based prediction of cognitive load in augmented reality," *Proc. ACM Hum.-Comput. Interact.* (ETRA), vol. 10, no. 3, 2026, doi:10.1145/3806040 | 31, AR search | LR baseline, RF, XGB; **both tuned with Optuna**. LR: SAGA, L1, C=0.38. XGB: 728 trees, depth 7. Per-participant normalization. | Participant-level hold-out | AUC LR **0.76** vs XGB **0.85** | XGB split frequency | VERIFIED |
| X3 | S. Chakraborty, P. Kiefer, M. Raubal, "Estimating perceived mental workload from eye-tracking data based on benign anisocoria," *IEEE Trans. Hum.-Mach. Syst.*, vol. 54, no. 5, pp. 499–507, 2024, doi:10.1109/THMS.2024.3432864 | 28, n-back (public Pillai dataset); 1–30 s windows | DT, LR, kNN, SVM, RF, GB, AdaBoost, XGB, LightGBM | 10-fold nested CV, "subject-independent" (grouping not explicit) | XGB 78–82%; LR not in the top 3. LR got worse with longer windows, boosting better. | Model-native | VERIFIED |
| X4 | EM12 Trigka 2025 | 9 | LR (L2, L-BFGS), XGB (100 trees, lr 0.1) | Random split | AUC LR 0.894 vs XGB 0.956 | None | VERIFIED (weak) |
| X5 | EM1 / EM2 / EM18 Kaczorowska 2021–2022 | 29–30, DSST | **Elastic-net LR**, RF, SVM, … (no XGB) | Participant-disjoint | LR F1 0.95–0.97 | **Elastic-net LR coefficients** | VERIFIED |
| X6 | EM17 Choi & Nam 2026 | 26, VR | XGB, RF, LSTM (no LR) | **LOSO** | XGB 78.9% (best) | None | VERIFIED |
| X7 | EM5 Božak 2026 | 54 | XGB, LightGBM, … (no LR); **library defaults** (tuning gave no gain) | **LOSO + LOGO** | LightGBM AUC 0.848; **majority baseline 76%** | SHAP + gain | VERIFIED |
| X8 | EM8 Rahman 2021 | 33 drivers, 30 s | LR, SVM, LDA, … | Random 5-fold | LR 0.92 (tied best) | None | VERIFIED |
| X9 | M. Stolte, B. Gollan, U. Ansorge, "Tracking visual search demands and memory load through pupil dilation," *J. Vis.*, vol. 20, no. 6, 21, 2020, doi:10.1167/jov.20.6.21 | 21 / 17, Pupil Labs | LR per participant | Within-participant | AUC 0.76 | None | VERIFIED |
| X10 | Skaramagkas *et al.*, *Proc. IEEE BIBE*, 2021, doi:10.1109/BIBE52308.2021.9635166 (COLET precursor) | 37 | 11 incl. LR and GB | Random 80/20 | RF 88% best; LR not reported | None | VERIFIED |
| X11 | FI16 Gao, Gao, Kasneci, ICMR 2025 | VR | LightGBM best | Unknown | 0.78 | SHAP | ABSTRACT |

**Eye tracking + other signals**

| ID | Study | N | Models | Split | LR vs XGB result | Importance | Evidence |
|---|---|---|---|---|---|---|---|
| X12 | X. Shao, X. Ma, F. Chen, X. Pan, "Multimodal machine learning framework for driver mental workload classification: A comparative and interpretable approach," *Appl. Sci.*, vol. 16, no. 7, 3581, 2026, doi:10.3390/app16073581 | 26, driving | LR, XGB, SVM, RF, … (grid search) | Random 80/20 | **Eye-only: LR 76.8% / AUC 0.866 vs XGB 78.5% / 0.86. "No significant differences."** | KernelSHAP (SVM) | VERIFIED |
| X13 | W. Chen *et al.*, "Machine learning models to predict individual cognitive load in collaborative learning: Combining fNIRS and eye-tracking data," *Mach. Learn. Knowl. Extr.*, vol. 7, no. 2, 51, 2025, doi:10.3390/make7020051 | 78 | LR, XGB, RF, … | Participant-wise split | Eye-only F1: LR 0.71 vs XGB 0.84 | Gini (trees); \|coef\| (LR) | VERIFIED |
| X14 | F. Walocha, A. Schrank, H. P. Nguyen, K. Ihme, "Multimodal assessment of mental workload during automated vehicle remote assistance," *Information*, vol. 16, no. 1, 64, 2025, doi:10.3390/info16010064 | 37 | XGB; random search in **repeated nested LOSO**. Most-chosen: max_depth 2, min_child_weight 5, gamma 1 | Nested LOSO | 3-class 57.7% (chance 33%) | Split frequency | VERIFIED |
| X15 | LB7 Oppelt (ADABase) 2023 | 30 released | XGB (TPE-tuned), SVM, kNN | Nested 10×10 CV (subject grouping not stated) | Eye-only AUC 0.86–0.89 | **XGB gain / weight / cover** | VERIFIED |
| X16 | EM15 Shafiei 2025 | 26 | XGB regression (grid ranges reported) | Random 80/20 | R² 0.81–0.83 | SHAP | VERIFIED |
| X17 | Ş. Harputlu Aksu, E. Çakıt, M. Dağdeviren, "Mental workload assessment using machine learning techniques based on EEG and eye tracking data," *Appl. Sci.*, vol. 14, no. 6, 2282, 2024, doi:10.3390/app14062282 | 15 | XGB, LightGBM, GBM, … | Random 5-fold | XGB 70.2%, LightGBM 72–77% | LightGBM ranking | VERIFIED |
| X18 | LB1 Gado 2023 | 18 | LR + others (randomized grid) | **LOSO** | LR on visual measures **below chance** (F1 0.20) | None | VERIFIED |
| X19 | Y. Que, Y. Zheng, J. H. Hsiao, X. Hu, "Using eye movements, electrodermal activities, and heart rates to predict different types of cognitive load during reading with background music," *Sci. Rep.*, vol. 15, 32635, 2025, doi:10.1038/s41598-025-03052-1 | 102 | Explanatory stepwise LR | None held out | Eye measures predicted all load types | LR coefficients | VERIFIED |
| X20 | Angkan *et al.* (CL-Drive), *IEEE T-ITS*, 2024 | 21 | XGB + others; **no LR**; no gaze-only result | LOSO | EEG+gaze XGB 66.7% | None | VERIFIED |

**Other physiological signals**

| ID | Study | Models | Split | Note | Evidence |
|---|---|---|---|---|---|
| X21 | WN4 Tervonen 2021 | XGB, Bayesian search (300 iterations) | Leave-two-subjects-out tuning, LOSO test | Hyperparameter search space reported | VERIFIED |
| X22 | LB2 Xu 2026 | XGB, LightGBM, linear SVC | LOSO + nested inner GroupKFold | SHAP on all data called "exploratory" | VERIFIED |
| X23 | F. Dell'Agnola, N. Momeni, A. Arza, D. Atienza, "Cognitive workload monitoring in virtual reality based rescue missions with drones," *HCII*, LNCS, pp. 397–409, 2020, doi:10.1007/978-3-030-49695-1_26 | XGB, LR, RF, …; XGB+SHAP feature elimination | Unseen test set | XGB 80.2% binary | ABSTRACT |
| X24 | M. Haseeb *et al.*, *Front. Robot. AI*, vol. 12, 1441801, 2025, doi:10.3389/frobt.2025.1441801 | Ridge multinomial LR chosen from 15 models (EEG) | Unknown | 84.6% | ABSTRACT |

**What these studies tell us**

- **XGB ≥ LR in all 7 head-to-heads.**
  - The gap is **large** under participant-held-out validation: X1, X2, X13.
  - It is **small or absent** with engineered features and random splits: X12 (no significant
    difference), X4.
  - LR can **collapse** cross-subject (X18).
  - With good subject-normalized features, LR reaches 0.95+ (X5).
  - Combined with M3/M4: expect XGB ≥ LR, but do not promise a large margin.
- **Always report a majority/chance baseline.** Božak's 81.4% was only 5 points above its 76%
  baseline (X7).
- **LR settings used in the field:**
  - L2 with L-BFGS (X4);
  - tuned L1/SAGA (X2);
  - **elastic net when coefficients are read as importance** (X5).
- **XGB settings used in the field:**
  - n_estimators 100–1000, max_depth **2–7** typical, learning rate about 0.1;
  - subsample and colsample 0.5–1.0, reg α/λ 0–1, min_child_weight about 5 (X2, X14, X16, X21).
- **Tuning:**
  - grid, random or Bayesian search **inside nested subject-grouped folds** (X14, X21, X22);
  - or fixed defaults declared in advance (X7).
- **Importance methods:**
  - XGB: SHAP (X7, X16, X22, X23), gain/weight/cover (X15), split frequency (X2, X14).
  - LR: elastic-net or standardized coefficients (X5, X13, X19).
- **Gaps we can claim:**
  - **No study computed SHAP for both LR and XGB.**
  - Only X5 compared importance rankings across two models.
  - SHAP computed out-of-fold within LOPO is rare (X22 warns that all-data SHAP is
    "exploratory").
- **Anchors for our methodology:**
  - **Primary:** X1, X2, X7, X14.
  - **Secondary:** X3, X12, X5, X15, X21.
  - **Weak or contrast only:** X4, X8, X17, X16, X10, X20.

## 5. Feature importance analysis

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| FI1 | S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, vol. 30, 2017 | SHAP | META | STRONG |
| FI2 | A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nat. Mach. Intell.*, vol. 3, no. 7, pp. 610–619, 2021, doi:10.1038/s42256-021-00338-7 | Explainability reveals shortcut reliance | ABSTRACT (preprint abstract; check the journal text) | ADEQUATE |
| FI3 | S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020, doi:10.1038/s42256-019-0138-9 | Exact TreeSHAP; global importance from local values; older path methods are inconsistent | VERIFIED | STRONG |
| FI4 | L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi:10.1023/A:1010933404324 | Origin of permutation importance | ABSTRACT | STRONG |
| FI5 | A. Fisher, C. Rudin, F. Dominici, "All models are wrong, but many are useful...," *JMLR*, vol. 20, no. 177, pp. 1–81, 2019 | Model reliance; LR and XGBoost can legitimately rank features differently | ABSTRACT | STRONG |
| FI6 | C. Strobl, A.-L. Boulesteix, A. Zeileis, T. Hothorn, "Bias in random forest variable importance measures: Illustrations, sources and a solution," *BMC Bioinformatics*, vol. 8, 25, 2007, doi:10.1186/1471-2105-8-25 | Impurity/gain-type importance is biased, so don't use gain as primary | VERIFIED | STRONG |
| FI7 | C. Strobl, A.-L. Boulesteix, T. Kneib, T. Augustin, A. Zeileis, "Conditional variable importance for random forests," *BMC Bioinformatics*, vol. 9, 307, 2008, doi:10.1186/1471-2105-9-307 | Correlated predictors inflate importance | VERIFIED | STRONG |
| FI8 | G. Hooker, L. Mentch, S. Zhou, "Unrestricted permutation forces extrapolation...," *Stat. Comput.*, vol. 31, 82, 2021, doi:10.1007/s11222-021-10057-z | Permutation over-credits correlated features; use drop-column/refit as a check | VERIFIED | STRONG |
| FI9 | C. Molnar *et al.*, "General pitfalls of model-agnostic interpretation methods for machine learning models," in *xxAI*, LNAI 13200, pp. 39–68, 2022, doi:10.1007/978-3-031-04083-2_4 | Interpret dependent features jointly; report uncertainty; importance is not causal | VERIFIED | ADEQUATE–STRONG |
| FI10 | K. Aas, M. Jullum, A. Løland, "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values," *Artif. Intell.*, vol. 298, 103502, 2021, doi:10.1016/j.artint.2021.103502 | SHAP assuming independent features can mislead; grouped Shapley values | ABSTRACT | STRONG |
| FI11 | A. Altmann, L. Toloşi, O. Sander, T. Lengauer, "Permutation importance: A corrected feature importance measure," *Bioinformatics*, vol. 26, no. 10, pp. 1340–1347, 2010, doi:10.1093/bioinformatics/btq134 | Null distribution and p-values for importance | ABSTRACT | STRONG |
| FI12 | A. Gelman, "Scaling regression inputs by dividing by two standard deviations," *Stat. Med.*, vol. 27, no. 15, pp. 2865–2873, 2008, doi:10.1002/sim.3107 | Standardized LR coefficients | ABSTRACT | STRONG |
| FI13 | C. F. Dormann *et al.*, "Collinearity: A review of methods to deal with it and a simulation study evaluating their performance," *Ecography*, vol. 36, no. 1, pp. 27–46, 2013, doi:10.1111/j.1600-0587.2012.07348.x | Collinearity misidentifies predictors; \|r\|>0.7 pre-filter | ABSTRACT | STRONG |
| FI14 | S. Nogueira, K. Sechidis, G. Brown, "On the stability of feature selection algorithms," *JMLR*, vol. 18, no. 174, pp. 1–54, 2018 | Stability of feature rankings across folds | ABSTRACT | STRONG |
| FI15 | J. N. Setu *et al.*, "Predicting and explaining cognitive load, attention, and working memory in virtual multitasking," *IEEE Trans. Vis. Comput. Graph.*, 2025, doi:10.1109/TVCG.2025.3549850 | SHAP used on eye/physiological workload models (not participant-independent) | VERIFIED | ADEQUATE |
| FI16 | H. Gao, Y. Gao, E. Kasneci, "An explainable machine learning approach for cognitive load detection in virtual reality using eye tracking data," *Proc. ACM ICMR*, pp. 340–348, 2025, doi:10.1145/3731715.3733275 | Tree boosting + SHAP on eye-tracking cognitive load (closest domain match) | META | ADEQUATE (read before citing specifics) |

**Gaps:**
- The direct proof that XGBoost gain importance is inconsistent is a preprint
  (Lundberg, Erion & Lee, arXiv:1802.03888). Rely on FI3 and FI6 instead.
- No strong journal paper combines SHAP with participant-independent validation for eye
  tracking. This is part of our research gap.

## 6. Dataset and wearable eye tracking

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| DS0 | B. Karbouj, B. E. Gaaloul, J. Krüger, "GAZELOAD: A multimodal eye-tracking dataset for mental workload in industrial human–robot collaboration," arXiv:2601.21829, 2026; data doi:10.17632/9smd7nbtwc.1 | The dataset. **Preprint.** 26 participants (16M/10F), Aria Gen 1, 5 sessions (2 low / 2 medium / 1 high), 1–10 Likert, lux sensor | VERIFIED | Required (preprint) |
| DS1 | T. Foulsham, E. Walker, A. Kingstone, "The where, what and when of gaze allocation in the lab and the natural environment," *Vision Res.*, vol. 51, no. 17, pp. 1920–1931, 2011, doi:10.1016/j.visres.2011.07.002 | Lab gaze differs from real-world gaze | ABSTRACT | STRONG |
| DS2 | X. Fu *et al.*, "Implementing mobile eye tracking in psychological research: A practical guide," *Behav. Res. Methods*, vol. 56, no. 8, pp. 8269–8288, 2024, doi:10.3758/s13428-024-02473-6 | Limits of screen-based eye tracking | ABSTRACT | STRONG |
| DS3 | N. V. Valtakari *et al.*, "Eye tracking in human interaction: Possibilities and limitations," *Behav. Res. Methods*, vol. 53, no. 4, pp. 1592–1608, 2021, doi:10.3758/s13428-020-01517-x | Wearables suit movement but are less accurate | VERIFIED | STRONG |
| DS4 | D. C. Niehorster *et al.*, "The impact of slippage on the data quality of head-worn eye trackers," *Behav. Res. Methods*, vol. 52, pp. 1140–1160, 2020, doi:10.3758/s13428-019-01307-0 | Head-worn tracker data quality | ABSTRACT | STRONG |
| DS5 | Y. Mansour *et al.*, "Enabling eye tracking for crowd-sourced data collection with Project Aria," *IEEE Access*, vol. 13, pp. 114736–114745, 2025, doi:10.1109/ACCESS.2025.3583623 | Aria gaze (Meta authors, not independent) | ABSTRACT | ADEQUATE |
| DS6 | M. D. Wilkinson *et al.*, "The FAIR Guiding Principles for scientific data management and stewardship," *Sci. Data*, vol. 3, 160018, 2016, doi:10.1038/sdata.2016.18 | Public data and reproducibility | ABSTRACT | STRONG |
| DS7 | J. Pineau *et al.*, "Improving reproducibility in machine learning research," *JMLR*, vol. 22, no. 164, pp. 1–20, 2021 | ML reproducibility | ABSTRACT | STRONG |

**Gaze input (round-4 review R4-1, searched 2026-10-01).** Publisher pages blocked fetching for the A rows; abstracts read via Semantic Scholar or Europe PMC.

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| PP17 | I. T. C. Hooge, R. S. Hessels, M. Nyström, "Do pupil-based binocular video eye trackers reliably measure vergence?," *Vision Res.*, vol. 156, pp. 1–9, 2019, doi:10.1016/j.visres.2019.01.004 | Target at 77 cm: pupil-based trackers "not accurate enough" to determine vergence or distance to the binocular fixation point | ABSTRACT | STRONG (near-identical distance) |
| PP18 | A. Velisar, N. M. Shanidze, "Noise estimation for head-mounted 3D binocular eye tracking using Pupil Core eye-tracking goggles," *Behav. Res. Methods*, vol. 56, no. 1, pp. 53–79, 2024, doi:10.3758/s13428-023-02150-0 | Same device: "improper gaze point depth estimation" adds noise. A snippet says depth was "severely underestimated": **read the full text before citing that** | ABSTRACT | STRONG (same device) |
| PP19 | R. Kothari *et al.*, "Gaze-in-wild: A dataset for studying eye and head coordination in everyday activities," *Sci. Rep.*, vol. 10, 2539, 2020, doi:10.1038/s41598-020-59251-5 | Pupil Labs data: angular velocity from the angle between successive eye-in-head unit gaze vectors | VERIFIED (Europe PMC) | ADEQUATE (method precedent; does not name `gaze_normal`) |
| — | M. Lamb *et al.*, *Front. Virtual Real.* 3:864653, 2022, doi:10.3389/frvir.2022.864653; M. Weier *et al.*, *Proc. ETRA* 2018, doi:10.1145/3204493.3204547 | Vergence underestimates depth beyond about 1 m (VR headsets) | V / A | Backup only (other devices) |
| — | Hausamann 2020 (PP8) | **Not** support for per-eye vectors: it derives eye orientation from the 3D gaze point | VERIFIED | Do not cite for R4-1 |

## 7. Preprocessing: labels, normalization, windows, missing data

**Labels**

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| LB1 | S. Gado, K. Lingelbach, M. Wirzberger, M. Vukelić, "Decoding mental effort in a quasi-realistic scenario...," *Sensors*, vol. 23, no. 14, 6546, 2023, doi:10.3390/s23146546 | Subject-wise median split of NASA-TLX effort, LOSO. **Eye features were at chance with the subjective labels.** | VERIFIED | STRONG (precedent) |
| LB2 | R. Xu *et al.*, "Consumer-grade wearable sensors for classifying pilot workload and stress during real flight training: A leave-one-subject-out validation study," *Sensors*, vol. 26, no. 12, 3627, 2026, doi:10.3390/s26123627 | Raw ratings mix scale use with state; per-person labels beat pooled ones (ECG/EDA) | VERIFIED | STRONG |
| LB3 | H. P. Martínez, G. N. Yannakakis, J. Hallam, "Don't classify ratings of affect; rank them!," *IEEE Trans. Affect. Comput.*, vol. 5, no. 3, pp. 314–326, 2014, doi:10.1109/TAFFC.2014.2352268 | Per-subject relative labels bypass differences in scale use | VERIFIED | ADEQUATE |
| LB4 | LB1 Gado 2023, **head-to-head result** | Same eye features: **designed-load labels F1 0.69 (above chance); self-report median-split labels F1 0.35 (below chance)** | VERIFIED | **STRONG** (the only direct comparison) |
| LB5 | EM9 Hogervorst 2014 | Condition labels; classified 0-back vs 2-back, **middle level dropped**; subjective ratings as manipulation check | VERIFIED | STRONG |
| LB6 | I. Albuquerque *et al.*, "WAUC: A multi-modal database for mental workload assessment under physical activity," *Front. Neurosci.*, vol. 14, 549524, 2020, doi:10.3389/fnins.2020.549524 | Task difficulty is the ground truth; NASA-TLX is the manipulation check | VERIFIED | STRONG |
| LB7 | M. P. Oppelt *et al.*, "ADABase: A multimodal dataset for cognitive load estimation," *Sensors*, vol. 23, no. 1, 340, 2022, doi:10.3390/s23010340 | Eye tracking; n-back level labels; low vs high binary and a 3-class variant; subject-wise split; **subject-wise z-score over the whole phase** | VERIFIED | STRONG |
| LB8 | CF6 Schmidt 2019 | Recommends study conditions as labels and questionnaires "to verify" the states | VERIFIED | ADEQUATE |
| LB9 | L. Fridman, B. Reimer, B. Mehler, W. T. Freeman, "Cognitive load estimation in the wild," *Proc. CHI*, 2018, doi:10.1145/3173574.3174226 | Eye region, N=92; 3-class n-back condition labels | ABSTRACT | ADEQUATE |
| LB10 | G. Matthews, L. E. Reinerman-Jones, D. J. Barber, J. Abich, "The psychometrics of mental workload: Multiple measures are sensitive but divergent," *Hum. Factors*, 2015, doi:10.1177/0018720814539505 | Workload measures diverge; no general factor | ABSTRACT | ADEQUATE |
| LB11 | P. A. Hancock, G. Matthews, "Workload and performance: Associations, insensitivities, and dissociations," *Hum. Factors*, 2019, doi:10.1177/0018720818809590 | Subjective and physiological measures often dissociate | ABSTRACT | ADEQUATE |
| LB12 | A. Gorin *et al.*, *Sensors*, 2024, doi:10.3390/s24061759 | Review: studies manipulate difficulty; NASA-TLX the most common check | VERIFIED | ADEQUATE |
| LB13 | Demirezen, Taşkaya Temizel, Brouwer, *Front. Neuroergon.*, vol. 5, 1346794, 2024, doi:10.3389/fnrgo.2024.1346794 | Reporting checklist: state the label source; fit normalization on training data only (31% of studies report this) | VERIFIED | ADEQUATE |
| — | Counter-example: EM7 COLET uses NASA-TLX as the label | | ABSTRACT | — |

**Per-person normalization**

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| NM1 | I. Albuquerque, J. Monteiro, O. Rosanne, T. H. Falk, "Estimating distribution shifts for predicting cross-subject generalization in electroencephalography-based mental workload assessment," *Front. Artif. Intell.*, vol. 5, 2022, doi:10.3389/frai.2022.992732 | Per-subject z-score improves LOSO (EEG) | VERIFIED | STRONG (EEG) |
| NM2 | J. Fdez, N. Guttenberg, O. Witkowski, A. Pasquali, "Cross-subject EEG-based emotion recognition through neural networks with stratified normalization," *Front. Neurosci.*, vol. 15, 626277, 2021, doi:10.3389/fnins.2021.626277 | Per-participant normalization beats batch normalization (EEG emotion) | VERIFIED | ADEQUATE |
| NM3 | M. Laut *et al.*, "Classifying mental stress from eye tracking data...," *Sci. Rep.*, vol. 16, 2026, doi:10.1038/s41598-026-58429-7 | Pupil baseline correction, nested LOSO | Near-verbatim | ADEQUATE |
| NM4 | NM1 Albuquerque 2022, **exact methods** | Z-score used the subject's **own task data (transductive)**. Accuracy: no norm 0.649, z-score 0.708, rest baseline 0.637, physical baseline 0.600. **Baseline normalization did not help.** | VERIFIED | STRONG |
| NM5 | Tognotti, Otesteanu, Anceschi, Menon, *Front. Digit. Health*, vol. 8, 1827279, 2026, doi:10.3389/fdgth.2026.1827279 | N=52, physiological stress, subject-independent. Normalization stats overlapping the test data **inflated balanced accuracy by 3–13 points**. A task-clip baseline kept out of the test data works as well as a rest baseline. Even leak-free normalization beats none by 11–18 points. | VERIFIED | **STRONG** (most directly relevant) |
| NM6 | Hefron *et al.*, *Sensors*, vol. 18, 1339, 2018, doi:10.3390/s18051339 | Per-participant normalization within block (transductive), a common-practice example | VERIFIED | ADEQUATE |
| NM7 | S. Mathôt, J. Fabius, E. Van Heusden, S. Van der Stigchel, "Safe and sensible preprocessing and baseline correction of pupil-size data," *Behav. Res. Methods*, 2018, doi:10.3758/s13428-017-1007-2 | Subtractive baseline correction for pupil data | ABSTRACT | ADEQUATE (pupil only) |
| NM8 | Liu, Ayaz, Shewokis, *Front. Hum. Neurosci.*, vol. 11, 389, 2017, doi:10.3389/fnhum.2017.00389 | Per-subject z-score to reduce between-subject variation | VERIFIED | WEAK |
| NM9 | Apicella *et al.*, *Eng. Appl. Artif. Intell.*, 2023, doi:10.1016/j.engappai.2023.106205 | The normalization choice strongly affects cross-subject performance | ABSTRACT | WEAK–ADEQUATE |

**Normalization summary:**
- **Transductive per-person z-score** is common practice: Božak (EM5), Appel (EM19), Shafiei
  (EM15), Gado, ADABase (LB7), Hefron, Albuquerque, Gervasi (HR3).
- **But it can inflate results by 3–13 points** (NM5), and it is preprocessing-type leakage
  (EV2 L1.2).
- **A leak-free calibration baseline** is supported in principle (NM5, EV2, LB13), but has no
  gaze-feature ablation in eye tracking.

**Windows**

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| WN1 | EM8 (Rahman 2021) | 30 s best of 15/30/60 s | VERIFIED | STRONG |
| WN2 | EM9 (Hogervorst 2014) | 30 s used; 120 s better | VERIFIED | STRONG |
| WN3 | T. Chihara, J. Sakamoto, "Effect of time length of eye movement data analysis on the accuracy of mental workload estimation during automobile driving," *Proc. IEA*, LNNS, pp. 593–599, 2021, doi:10.1007/978-3-030-74608-7_72 | 30 s worst; 60–120 s recommended | ABSTRACT | ADEQUATE (counter-evidence) |
| WN4 | J. Tervonen, K. Pettersson, J. Mäntyjärvi, "Ultra-short window length and feature importance analysis for cognitive load detection from wearable sensors," *Electronics*, vol. 10, no. 5, 613, 2021, doi:10.3390/electronics10050613 | Shorter windows are worse; XGBoost, LOSO (physiological) | VERIFIED | ADEQUATE |

**Missing data**

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| MD1 | A. Perez-Lebel *et al.*, "Benchmarking missing-values approaches for predictive models on health databases," *GigaScience*, vol. 11, giac013, 2022, doi:10.1093/gigascience/giac013 | Missing indicator for informative missingness; native NaN handling in boosted trees does best | VERIFIED | STRONG |
| MD2 | M. Van Ness, T. M. Bosschieter, R. Halpin-Gregorio, M. Udell, "The missing indicator method: From low to high dimensions," *Proc. ACM KDD*, pp. 5004–5015, 2023, doi:10.1145/3580305.3599911 | Missing indicators help linear models | ABSTRACT | STRONG |
| MD3 | J. Josse *et al.*, "On the consistency of supervised learning with missing values," *Stat. Papers*, vol. 65, no. 9, pp. 5447–5479, 2024, doi:10.1007/s00362-024-01550-4 | Theory of imputation for prediction | ABSTRACT | STRONG |

## 8. Evaluation: leakage, participant-held-out validation, tuning, metrics, statistics

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| EV1 | S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM TKDD*, vol. 6, no. 4, 15, 2012, doi:10.1145/2382577.2382579 | Leakage definition; drop task identifiers | ABSTRACT | STRONG |
| EV2 | S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, 100804, 2023, doi:10.1016/j.patter.2023.100804 | Leakage taxonomy (L1.2 preprocessing, L2 illegitimate features, L3.2 non-independence) | VERIFIED | STRONG |
| EV3 | S. Saeb *et al.*, "The need to approximate the use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, 2017, doi:10.1093/gigascience/gix019 | Record-wise CV overestimates accuracy; use subject-wise | VERIFIED | STRONG |
| EV4 | M. A. Little, G. Varoquaux, S. Saeb *et al.*, "Using and understanding cross-validation strategies. Perspectives on Saeb et al.," *GigaScience*, vol. 6, no. 5, 2017, doi:10.1093/gigascience/gix020 | The split must match the use case | VERIFIED | STRONG |
| EV5 | G. Brookshire *et al.*, "Data leakage in deep learning studies of translational EEG," *Front. Neurosci.*, vol. 18, 1373515, 2024, doi:10.3389/fnins.2024.1373515 | Segment-level holdout inflates performance | ABSTRACT | STRONG |
| EV6 | D. R. Roberts *et al.*, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881 | Grouped/blocked CV | ABSTRACT | STRONG |
| EV7 | G. C. Cawley, N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," *JMLR*, vol. 11, pp. 2079–2107, 2010 | Tune inside each fold | VERIFIED | STRONG |
| EV8 | S. Varma, R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinformatics*, vol. 7, 91, 2006, doi:10.1186/1471-2105-7-91 | Nested CV is almost unbiased | VERIFIED | STRONG |
| EV9 | G. Varoquaux *et al.*, "Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines," *NeuroImage*, vol. 145, pp. 166–179, 2017, doi:10.1016/j.neuroimage.2016.10.038 | Leave subjects out; leave-one-out has high variance | VERIFIED | STRONG |
| EV10 | G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018, doi:10.1016/j.neuroimage.2017.06.061 | Small N gives large error bars; report the spread | ABSTRACT | STRONG |
| EV11 | T. Fawcett, "An introduction to ROC analysis," *Pattern Recognit. Lett.*, vol. 27, no. 8, pp. 861–874, 2006, doi:10.1016/j.patrec.2005.10.010 | ROC-AUC | META | STRONG |
| EV12 | J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *JMLR*, vol. 7, pp. 1–30, 2006 | Wilcoxon signed-rank for comparing classifiers | META | STRONG |
| EV13 | G. Forman, M. Scholz, "Apples-to-apples in cross-validation studies: Pitfalls in classifier performance measurement," *ACM SIGKDD Explor.*, vol. 12, no. 1, pp. 49–57, 2010, doi:10.1145/1882471.1882479 | Pooled vs per-fold AUC | ABSTRACT | ADEQUATE |

## 9. Confounds and shortcut learning (for the lighting finding)

| ID | Citation | Supports | Evidence | Grade |
|---|---|---|---|---|
| CF1 | M. Lohani, B. R. Payne, D. L. Strayer, "A review of psychophysiological measures to assess cognitive states in real-world driving," *Front. Hum. Neurosci.*, vol. 13, 57, 2019, doi:10.3389/fnhum.2019.00057 | Luminance is "a critical confounding factor" in the field (pupil) | VERIFIED | STRONG |
| CF2 | S. R. Steinhauer *et al.*, "Publication guidelines and recommendations for pupillary measurement in psychophysiological studies," *Psychophysiology*, vol. 59, no. 4, e14035, 2022, doi:10.1111/psyp.14035 | The light reflex can exceed the cognitive effect | VERIFIED | STRONG |
| CF3 | B. Pfleging, D. K. Fekety, A. Schmidt, A. L. Kun, "A model relating pupil diameter to mental workload and lighting conditions," *Proc. CHI*, pp. 5776–5788, 2016, doi:10.1145/2858036.2858117 | Lighting must be modeled with workload | ABSTRACT | STRONG |
| CF4 | M. Eckert, T. Robotham, E. A. P. Habets, O. S. Rummukainen, "Pupillary light reflex correction for robust pupillometry in virtual reality," *Proc. ACM Comput. Graph. Interact. Tech.*, vol. 5, no. 2, 2022, doi:10.1145/3530798 | Light masks cognitive effects on head-worn trackers | ABSTRACT | ADEQUATE |
| CF5 | S. Mathôt, "Pupillometry: Psychology, physiology, and function," *J. Cognition*, vol. 1, no. 1, 16, 2018, doi:10.5334/joc.18 | Pupil is driven by both light and effort | VERIFIED | STRONG |
| CF6 | P. Schmidt, A. Reiss, R. Dürichen, K. Van Laerhoven, "Wearable-based affect recognition—A review," *Sensors*, vol. 19, no. 19, 4079, 2019, doi:10.3390/s19194079 | Environmental confounds; lab accuracy exceeds field accuracy | VERIFIED | ADEQUATE |
| CF7 | R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nat. Mach. Intell.*, vol. 2, no. 11, pp. 665–673, 2020, doi:10.1038/s42256-020-00257-z | Shortcut learning (concept) | VERIFIED | STRONG |
| CF8 | J. R. Zech *et al.*, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs," *PLoS Med.*, vol. 15, no. 11, e1002683, 2018, doi:10.1371/journal.pmed.1002683 | A context variable alone gave AUC 0.861 | VERIFIED | STRONG |
| CF9 | M. A. Badgeley *et al.*, "Deep learning predicts hip fracture using confounding patient and healthcare variables," *npj Digit. Med.*, vol. 2, 31, 2019, doi:10.1038/s41746-019-0105-1 | Removing the confounder dropped AUC to chance | ABSTRACT | STRONG |

## 9b. Candidate datasets (survey, 2026-10-01)

Citation counts are from OpenAlex; a citation is not the same as a use. Access was checked
where the landing page could be reached.

| Dataset | Descriptor | Peer-reviewed | Access | N / setting | Eye data | Labels | Reuse | Fit |
|---|---|---|---|---|---|---|---|---|
| **GAZELOAD** (current) | DS0, arXiv 2601.21829 | ❌ Preprint | Open (CC BY 4.0) | 26 / lab HRC, wearable Aria | 250 ms aggregated metrics only; **no pupil, no raw gaze** | 5 designed sessions + 1–10 self-report | **None** | Novel; weakest precedent |
| **COLET** | EM7, *CMPB* 2022, doi:10.1016/j.cmpb.2022.106989 | ✅ | **Open, no request**: Zenodo record 7766785, CC BY 4.0, ~791 MB | 47 / lab screen, visual-search puzzles (2×2: multitasking × time pressure) | **Raw 240 Hz gaze, pupil, blinks** (Pupil Core; tracker spec from a secondary source) | NASA-TLX per activity + experimental conditions | 46 citations; ≥3 peer-reviewed uses | **Best fit for the title** (eye tracking only, binary precedent) |
| **CL-Drive** | Angkan *et al.*, *IEEE Trans. Intell. Transp. Syst.*, 2024, doi:10.1109/TITS.2023.3345846 | ✅ | Open: Borealis doi:10.5683/SP3/JJ2YZZ, CC BY-NC 4.0 | 21 / driving simulator, 9 complexity levels | Tobii Pro Glasses 2 (**wearable**) 50 Hz: fixations, saccades, pupil, blinks | Self-report every 10 s + designed levels; published LOSO baselines incl. XGBoost | 59 citations, mostly EEG uses | Very good; driving domain, small N |
| **ADABase** | LB7, *Sensors* 2023, doi:10.3390/s23010340 | ✅ | **Restricted**: signed EULA by email, academic only | 30 released / lab n-back + driving dual task | Tobii Pro Fusion 250 Hz: fixations, saccades, blinks, pupil | **Graded n-back levels** + NASA-TLX | 45 citations / 2 uses | Strongest design; EULA barrier |
| **CLARE** | Bhatti *et al.*, *IEEE Trans. Cogn. Dev. Syst.*, 2025, doi:10.1109/TCDS.2025.3555517 | ✅ | Likely open (GitHub → Borealis), untested | 24 / lab MATB-II | Tobii Pro Glasses 2 50 Hz: pupil, blinks, fixations, saccades | Self-report every 10 s | 12 citations, authors' group only | Backup to CL-Drive |
| Pillai *et al.* | *Data in Brief* 2020, doi:10.1016/j.dib.2020.106389 | ✅ (data paper) | Mendeley CC BY 4.0 (landing page not testable) | 28 / lab screen, n-back single vs dual | Gazepoint 60 Hz: raw pupil, gaze, fixations, blinks | n-back level + NASA-TLX | 19 citations, little ML reuse | Clean labels, low reuse |
| HP Omnicept VR | EM13, *Proc. IEEE VR* 2025 | ✅ (conference) | Subset N=100; terms unknown (page 403) | 100 / VR | Vive Pro Eye 120 Hz, pupil + gaze | 3 difficulty levels + single-item TLX | Unknown | Large N; access unverified |
| CLUES | Used by EM5 | Descriptor under review | **Not public** | 54 / lab | Tobii 60 Hz | Condition labels | — | Not usable yet |
| MOCAS, CLAS, WAUC, EEGEyeNet | — | — | — | — | No eye tracker, or no workload labels | — | — | Excluded |

### COLET in detail (full survey)

**Dataset facts** (compiled from PubMed, Zenodo/FORTH and the papers that used it; the
original full text was blocked):

- **Participants:** 47 kept of 56 recorded (26 women, 21 men, mean age 32).
- **Setting:** chin rest, 24-inch screen, Pupil Core at 240 Hz.
- **Design:** 2×2 (time pressure × multitasking), giving **4 activities per person and 188
  recordings**.
- **Labels:** mean NASA-TLX per activity (self-report).
  - Three classes: low <30, medium 30–49, high ≥50.
  - The original binary task was low+medium vs high.
- **Release:** raw gaze, pupil and blink streams (`.mat`, Pupil Labs exports). The authors'
  processed feature table is **not** included.
- **Original pipeline:**
  - features per whole activity, min-max scaling, **random 80/20 split (not
    subject-independent)**;
  - 8 classifiers including LR and gradient boosting;
  - 88% binary, 59% three-class.

**Peer-reviewed studies that analyzed COLET (5)**

| Citation | What they did | Validation | Result |
|---|---|---|---|
| W. Fuhl *et al.*, "A temporally quantized distribution of pupil diameters as a new feature for cognitive load classification," *Proc. ACM ETRA*, 2023, doi:10.1145/3588015.3590116 | New pupil feature; 3 TLX classes; 10–30 s windows | Random 80/20 | RF 68–71% (3-class) |
| Fenoglio *et al.*, "Federated learning for privacy-aware cognitive workload estimation," *Proc. MUM*, 2023, pp. 25–36, doi:10.1145/3626705.3627783 | Federated learning, COLET + ADABase | **Person-independent:** 4-fold user-grouped CV, unseen test users (§5.1); windows; GNB/RF/NN; no LR, XGBoost or SHAP | VERIFIED (author PDF, Zenodo 14577562; V2-6) |
| Fenoglio, Gjoreski, Langheinrich, "A federated unsupervised personalisation for cognitive workload estimation," *Proc. MUM*, 2023, pp. 526–528, doi:10.1145/3626705.3631796 | Federated personalization (3-page short paper) | Not verified: full text unavailable | ABSTRACT |
| S. Wibirama *et al.*, "Classification of cognitive load using deep learning based on eye movement indices," *IEEE Access*, vol. 13, 2025, doi:10.1109/ACCESS.2025.3613292 | Raw gaze x/y, heavily overlapping windows, 8 classic models + BiLSTM | 5-fold over windows (**likely leakage**) | BiLSTM 0.878 vs LR 0.510 (3-class) |
| Dell'Acqua *et al.*, "Inferring cognitive workload from symbolic gaze sequences using large language models," *IEEE Access*, 2025, doi:10.1109/ACCESS.2025.3646271 | LLM on gaze event sequences; RF baseline | Trial-level 70/15/15 (not subject-grouped) | LLM 70%, RF 66% |

**Preprints:**
- Hallam *et al.*, arXiv:2511.01060: windowed features, no protocol stated.
- Haag *et al.*, bioRxiv 2026: **subject-level 5-fold CV**; on COLET all models dropped, best
  **AUC 0.670**.

**Takeaways:**
- **No XGBoost, no SHAP, and no peer-reviewed subject-independent validation** on COLET.
- The high published numbers (88%, 0.878) come from within-subject or random splits.
- The one subject-level result (a preprint) is AUC 0.67, similar to our GAZELOAD results.
- COLET labels are **self-report** (NASA-TLX). Condition labels are possible from the 2×2
  design, but no study used them as the primary target.

## 10. Pending searches (round 2)

| Track | Question | Fills section |
|---|---|---|
| R1 | Closest methodological twins and their full pipelines; the "consensus pipeline" | 3 (**done**) |
| R2 | Condition labels vs self-report; normalization that does not use test data | 7 (**done**) |
| R3 | Workload validity of each of our available features | 2 (**done**) |
| R4 | Other work using GAZELOAD; HRC/manufacturing eye-tracking workload pipelines | 3, 6 (**done**) |
| R5 | Fair LR-vs-XGBoost comparison; feature-importance methodology and pitfalls | 4, 5 (**done**) |

## 11. Do not cite, or cite only with corrections

| Source | Problem |
|---|---|
| R. Parasuraman, D. H. Manzey 2010 (draft [3]) | About automation complacency; does not support "overload causes errors". Use CL3. |
| "Exploring the relationship between mental workload, variation in performance and physiological parameters" (draft [9] title) | That title is a 2016 IFAC paper (doi:10.1016/j.ifacol.2016.10.618). Use CL6. |
| Kaczorowska *et al.* 2021, *Sensors* (draft [13]) for temporal aggregation | Its "aggregation" combines classifier outputs. Use it only for validation (EM2). |
| "L. Rizzo, M. Dondio, L. Longo" (draft [18] authors) | Wrong authors. Use EM3. |
| Trigka *et al.* 2025 as "closest precedent" or for feature importance | Circular labels, N=9, no SHAP. Contrast case only. |
| P. R. K. Reddy *et al.*, IEEE Access 2020 (old PDF [5]) | Not found; possibly nonexistent |
| Z. He *et al.*, BSPC 2025 (old PDF [13]) | Not found; possibly nonexistent |
| C. Cortes, V. Vapnik 1995 (old PDF [7]) | Irrelevant to the claim |
| Old PDF [12], [14], [16], [17], [21] | Wrong authors, titles or venues (see `rrl-decision-log.md` §4.2) |
| Preprints: CLARE (arXiv 2404.17098), MambaGaze (arXiv 2605.22775), Engel *et al.* Project Aria (arXiv 2308.13561) | Not peer-reviewed. Project Aria may be cited only as a device description, labeled as a preprint. |

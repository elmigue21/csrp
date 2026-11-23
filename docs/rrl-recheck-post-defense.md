# Reference re-check after the defense

**Date:** 2026-11-11
**Checked:** all 86 references in the v4-6 draft, against the post-defense method
(`post-defense-data-processing.md`, `post-defense-modeling-plan.md`,
`thesis-changes-post-defense.md`). No other file was changed.

## Summary

Of the 86 references, **56 KEEP**, **11 REWORD** (still needed, but the sentence that cites
them changes), **9 OPTIONAL** (background that a stronger source already covers; cut them if
the paper must be shorter) and **10 DROP** (they only supported removed content: per-person
standardization, the pupil-only analysis P2, permutation importance, LR coefficients as
importance, and the gain appendix). Every reference is cited in the text, and every citation
has an entry. Most new claims are already covered by verified sources (for example, Kapoor &
Narayanan [15] names feature selection on the whole dataset as a type of leakage). Three
claims still **need a source**: univariate filter selection, the events-per-variable rule of
thumb, and the metric definitions (Fawcett [78] probably covers the last one; check it). After
the cuts, renumber the list in order of first citation (IEEE style).

- **How to explain it:** "We kept every source that still supports something we do. We
  removed the sources for steps we no longer do, and added sources only for the new steps."

**Wording rule:** where standardization is removed, say it neutrally ("removed after the
proposal defense"). Never say it inflated the results.

## 1. Every reference

K = KEEP, RW = REWORD, OPT = OPTIONAL, D = DROP. "Cited in" uses the v4-6 section names.

| Ref | Short citation | Cited in (old) | Supports | Verdict | Reason |
|---|---|---|---|---|---|
| 1 | Kahneman 1973 | Ch1 Context; Ch2 Cog. load | Mental effort is limited | OPT | Background; [2] and [3] already cover it. Metadata only |
| 2 | Paas & van Merriënboer 2020 | Ch1 Context; Ch2 | Cognitive-load theory | K | Defines the construct |
| 3 | Young et al. 2015 | Ch1 Context, Significance; Ch2 | Workload matters for performance and safety | K | Background |
| 4 | Hart & Staveland 1988 | Ch1; Ch2 | NASA-TLX | K | Manipulation check is unchanged |
| 5 | Tao et al. 2019 | Ch1; Ch2; Ch3 Table 5, Analysis plan (P2); Ch4 Discussion | Eye measures track workload | K | Core source. Remove only its P2 use ("pupil the most established") |
| 6 | Matthews et al. 2015 | Ch1; Ch2 | Workload measures disagree | OPT | Background; [34] covers the reason for condition labels |
| 7 | Bishop 2006 | Ch1; Ch2 | Logistic regression | K | Model definition |
| 8 | Kaczorowska et al. 2021 | Ch1; Ch2 LR, Importance, Synthesis; Ch3 Models, Importance, Analysis plan | Elastic-net LR for workload; LR coefficients as importance; top models tie (ceiling rule) | RW | Keep for elastic-net LR, Table 1 and the ceiling rule. Delete the Ch3 Importance sentence (coefficients are no longer an importance method) |
| 9 | Chen & Guestrin 2016 | Ch1; Ch2; Ch3 Models | XGBoost | K | Model definition |
| 10 | Grinsztajn et al. 2022 | Ch1; Ch2 | Trees do well on tabular data | K | Reason to test XGBoost |
| 11 | Rolon-Merette et al. 2026 | Ch1; Ch2; Table 1; Ch3 Standardization, Analysis plan (P2); Ch4 Discussion | LR vs XGBoost on unseen people; pupil size differs between people; pupil-only models | RW | Keep as an LR-vs-XGBoost study (also in the new comparison table). Delete its uses in Standardization and P2 |
| 12 | Nerella et al. 2026 | Ch1; Ch2 | LR vs XGBoost, AR workload | K | Now also in the new comparison (AUC LR 0.76 vs XGB 0.85) |
| 13 | Shao et al. 2026 | Ch1; Ch2; Synthesis; Ch4 Discussion | No AUC difference between LR and XGBoost | K | Matches the new tie (AUC difference 0.000) |
| 14 | Saeb et al. 2017 | Ch1; Ch2 Validation; Ch3 Validation | Test on new people | K | LOPO is unchanged |
| 15 | Kapoor & Narayanan 2023 | Ch1; Ch2 Validation (x2); Ch3 Features, Validation | Leakage; preprocessing fitted on all data; blink-duration missingness | RW | Keep, and cite it for the **new** rule: feature selection on the whole dataset is leakage (type L1.3). Delete the sentence on standardization using test data |
| 16 | Božak et al. 2026 | Ch1; Ch2 Validation, Importance; Table 1; Ch3 Detection fallback, Table 5, Features, Standardization, Metrics | Closest study: LOSO + SHAP; feature subsets; near-duplicates; majority baseline; per-person features | RW | Keep (closest study, SHAP, majority baseline, fallback, new comparison). Delete its Standardization use and the near-duplicate use (those features are now kept). New use: it computed many features (488) and then pruned them |
| 17 | Shafiei et al. 2025 | Ch1; Ch2; Table 1 | XGBoost + SHAP in workload | K | Precedent for SHAP on XGBoost |
| 18 | Ktistakis et al. 2022 (COLET) | Ch1–5, many places | The dataset; its settings; LR 0.85; ethics | RW | Essential. Change the Ch3 sentence "higher-order statistics … omitted": we now also use many statistics, as the COLET authors did. Delete the Ch4 post-hoc use (A3 vs A4, A1 vs A2) |
| 19 | Lundberg & Lee 2017 | Ch1 Purpose; Ch2; Ch3 Importance | SHAP | K | SHAP is the importance method |
| 20 | Lundberg et al. 2020 | Ch1 Purpose; Ch2; Ch3 Importance | TreeSHAP | K | SHAP for XGBoost |
| 21 | Breiman 2001 | Ch1 Purpose; Ch2; Ch3 Importance | Permutation importance (origin) | D | Permutation importance removed (MD4) |
| 22 | Fisher et al. 2019 | Ch1 Purpose, Framework; Ch2; Ch3 Importance | Permutation importance | D | Permutation importance removed (MD4). Ch1 Framework now cites SHAP [19] |
| 23 | Bentivoglio et al. 1997 | Ch1 Purpose, Scope; Ch2; Ch3 Analysis plan (P2); Ch4 P2, Discussion; Ch5 Limitations; App. B | Talking raises blink rate | RW | Key source for the talking limitation (blink rate alone does as well as the full model). Move it from the P2 reason (Ch1 L37, Ch3) to the Limitations and Discussion |
| 24 | UN 2015 (SDGs) | Ch1 Significance | SDG relevance | K | Keep if the school template asks for SDGs; otherwise optional |
| 25 | Wu et al. 2020 | Ch1 Significance; Ch3 Labels | Workload in surgery; compare only the extremes | K | A1 vs A4 is unchanged |
| 26 | Christodoulou et al. 2019 | Ch1; Ch2; Synthesis; Ch4 Discussion | ML often no better than LR | K | Fits the new tie |
| 27 | Foulsham et al. 2011 | Ch1 Scope | Lab results may not transfer | K | Limitation |
| 28 | Varoquaux 2018 | Ch1 Scope; Ch2; Ch3 Metrics | Small N gives wide error bars | K | Bootstrap CI is unchanged |
| 29 | Kret & Sjak-Shie 2019 | Ch3 Table 4 | Pupil cleaning | K | Cleaning C1–C10 unchanged |
| 30 | Faraji et al. 2023 | Ch1 Framework; Ch3 Table 4, gaze text | Confidence 0.8; gaps; quality report | K | Cleaning unchanged |
| 31 | Salvucci & Goldberg 2000 | Ch1; Ch2; Ch3 Detection | I-VT | K | Detection unchanged |
| 32 | Bargary et al. 2017 | Ch2 Validation; Ch3 Standardization; Ch4 Discussion | People have stable eye "signatures" (reason for standardization) | OPT | Its main use (the reason for standardization) is gone. Keep only if you add a limitation: "differences between people are not removed" |
| 33 | Cawley & Talbot 2010 | Ch1 Framework; Ch2; Ch3 Models, Validation | Tune inside the folds | K | Now also covers tuning N inside the folds |
| 34 | Hart 2006 | Ch2; Ch3 Labels (x2) | Raw TLX is common; ratings depend on scale use | K | Manipulation check unchanged. Our notes rate it metadata only (CL8): read it before final submission |
| 35 | Mathôt 2018 | Ch2; Ch3 Features (x2); Ch4 Discussion | Pupil responds to effort and light | K | Luminance limitation. Delete the P2 sentence in the Discussion |
| 36 | Recarte et al. 2008 | Ch2; Ch3 Table 5; Ch5 Recommendations | Blink rate rises with mental load and falls with visual load | K | Blink rate is now the top feature |
| 37 | May et al. 1990 | Ch2; Ch3 Table 5; Ch4 Discussion | Saccade amplitude under load | K | Saccade amplitude is a signal (S6) |
| 38 | Di Stasi et al. 2010 | Ch2; Ch3 Table 5; Ch4 Discussion | Saccade peak velocity under load | K | Peak velocity is a signal (S7) |
| 39 | Di Stasi et al. 2011 | Ch2 | Main sequence as a workload index | OPT | Repeats [38] |
| 40 | Liu et al. 2022 | Ch2; Ch3 Table 5 | Fixations and load | K | Fixation signals (S5, rate) |
| 41 | Wang et al. 2014 | Ch2; Ch3 Table 5; Ch4 Discussion | Gaze spread under load | K | Gaze spread statistics are among the top features |
| 42 | Recarte & Nunes 2003 | Ch2; Ch4 Discussion | Gaze concentrates under load | K | Needed for the Discussion: gaze range grew in A4 instead (14.9° to 21.8°) |
| 43 | Andersson et al. 2017 | Ch2; Ch3 Detection (x2) | Detection depends on the algorithm; 45°/s and 55 ms | K | Detection unchanged |
| 44 | Komogortsev et al. 2010 | Ch2; Ch3 Detection | Check detection quality | K | Sanity check unchanged |
| 45 | Skaramagkas et al. 2021 | Ch2 COLET | Earlier COLET conference paper | OPT | [18] is enough |
| 46 | Fuhl et al. 2023 | Ch2; Table 1; Synthesis | COLET, random split | K | Research gap |
| 47 | Wibirama et al. 2025 | Ch2; Table 1; Synthesis | COLET, window split | K | Research gap |
| 48 | Dell'Acqua et al. 2025 | Ch2; Table 1; Synthesis | COLET, recording split | K | Research gap |
| 49 | Fenoglio et al. 2023 | Ch2; Table 1; Synthesis | COLET on unseen users | K | Research gap |
| 50 | Zou & Hastie 2005 | Ch2; Ch3 Models | Elastic net; correlated features | K | More important now: 160 feature pairs have r > 0.9 |
| 51 | Walocha et al. 2025 | Ch2; Ch3 Models | XGBoost, nested LOSO, shallow trees | K | Model settings unchanged |
| 52 | Chakraborty et al. 2024 | Ch2 | LR vs boosting on eye data | OPT | [11], [12], [13] already cover the comparison |
| 53 | Little et al. 2017 | Ch2; Ch3 Validation | Split must match the use case | OPT | Repeats [14] |
| 54 | Varma & Simon 2006 | Ch2; Ch3 Validation | Nested CV | K | Also covers tuning N inside the folds |
| 55 | Varoquaux et al. 2017 | Ch3 Validation | Grouped inner CV | K | Inner folds grouped by participant |
| 56 | Oppelt et al. 2023 (ADABase) | Ch2 Validation; Ch3 Standardization | Studies standardize per person | D | Standardization removed |
| 57 | Gado et al. 2023 | Ch2 Validation; Ch3 Standardization | Studies standardize per person | D | Standardization removed (`rrl-by-topic.md` also flagged this support as doubtful) |
| 58 | Appel et al. 2023 | Ch2 Validation | Studies standardize per person | D | Standardization removed |
| 59 | Albuquerque et al. 2022 | Ch2 Validation | z-score best across subjects (EEG) | D | Standardization removed |
| 60 | Gelman 2008 | Ch2 Importance | LR coefficients on a common scale | D | Coefficients are no longer an importance method (MD4) |
| 61 | Strobl et al. 2007 | Ch2 Importance; Ch3 Importance | Built-in (gain) importance can be biased | D | Gain appendix removed. Optional: keep one sentence "SHAP, not built-in importance, because the latter can be biased" |
| 62 | Molnar et al. 2022 | Ch2; Ch3 Importance | Importance is not cause; read dependent features together | RW | Keep. Also cite it for the new caveat: near-copy statistics (mean, median, p75 …) share credit, so read SHAP per signal as well as per feature |
| 63 | Xu et al. 2026 | Ch2 Importance | SHAP on all data is only exploratory | K | Supports computing SHAP on the held-out person in each fold |
| 64 | Hogervorst et al. 2014 | Ch3 Labels | Compare only the extremes | K | Also the precedent for dropping blink duration (D1) |
| 65 | Kassner et al. 2014 | Ch3 Table 4 | Pupil Labs confidence | K | Cleaning unchanged |
| 66 | Nenna et al. 2023 | Ch3 Table 4 | 35% exclusion threshold | K | Cleaning unchanged |
| 67 | Hershman et al. 2018 | Ch3 Table 4 | Blink merging | K | Cleaning unchanged |
| 68 | Steinhauer et al. 2022 | Ch3 Table 4 (x2) | Blink length; pupil guidelines | K | Cleaning unchanged |
| 69 | Petersch & Dierkes 2022 | Ch3 Table 4 | Use 3D pupil size | K | Cleaning unchanged |
| 70 | Hooge et al. 2019 | Ch3 Table 4, gaze text | Vergence depth unreliable | K | Gaze input unchanged |
| 71 | Velisar & Shanidze 2024 | Ch3 Table 4, gaze text | Pupil Core depth errors | K | Gaze input unchanged |
| 72 | Kothari et al. 2020 | Ch3 Table 4, gaze text | Speed from gaze-vector angles | K | Gaze speed is now also a signal (S4) |
| 73 | Duchowski 2017 | Ch3 Detection | Velocity filtering | K | Detection unchanged |
| 74 | Hausamann et al. 2020 | Ch3 Detection | Reject speeds above 1000°/s | K | Detection unchanged |
| 75 | Trabulsi et al. 2021 | Ch3 Detection | 50–75 ms minimum fixation | K | Detection unchanged |
| 76 | Dormann et al. 2013 | Ch3 Features | Drop near-duplicate features | OPT | Near-duplicates are now kept. Exact copies (variance, range) need no source. [50] and [62] cover correlated features |
| 77 | Perez-Lebel et al. 2022 | Ch3 Models | Median imputation vs native missing values | RW | Keep if the imputation step stays in the LR pipeline; reword after re-checking blanks in the 140 features (currently 0) |
| 78 | Fawcett 2006 | Ch3 Metrics | ROC-AUC | K | Primary metric. May also cover precision/recall (see §2) |
| 79 | Niculescu-Mizil & Caruana 2005 | Ch3 Metrics | Probability quality (Brier) | OPT | [80] is enough for the Brier score |
| 80 | Van Calster et al. 2019 | Ch3 Metrics | Calibration; Brier | K | Brier is still reported |
| 81 | Demšar 2006 | Ch3 Metrics | Wilcoxon test | K | Wilcoxon on margins kept (p < 0.001) |
| 82 | Ojala & Garriga 2010 | Ch3 Metrics | Permutation test against chance | RW | Kept. Reword: the shuffled run repeats the whole pipeline, feature selection included, with default settings (N = 20) |
| 83 | Nogueira et al. 2018 | Ch3 Importance | Stability of weights across folds | RW | Move: cite it for the new **selection count** (how often a feature is picked across the 44 folds) |
| 84 | Pluchino et al. 2023 | Ch3 Analysis plan | Spoken dual task raised blinking | RW | Move from the P2 reason to the talking limitation, next to [23] |
| 85 | Appel et al. 2018 | Ch3 Analysis plan | Pupil-only models (P2) | D | P2 removed (MD3) |
| 86 | Stolte et al. 2020 | Ch3 Analysis plan | Pupil-only models (P2) | D | P2 removed (MD3) |

**DROP list:** [21] Breiman, [22] Fisher, [56] Oppelt (ADABase), [57] Gado, [58] Appel 2023,
[59] Albuquerque, [60] Gelman, [61] Strobl, [85] Appel 2018, [86] Stolte.

## 2. New claims that need a source

IDs refer to `rrl-master-list.md` (or `rrl-colet-list.md` for PP/EV14). "Verified (how)" is
what was checked on 2026-11-11 for this file.

| New claim | Covered by (ID) | Status |
|---|---|---|
| Feature selection must be fitted inside the training folds; selection on all data is leakage and makes results too good | [15] Kapoor & Narayanan (EV2): leakage type **L1.3, "Feature selection on training and test set"** | **Covered.** L1.3 wording checked in the arXiv version of the paper (ar5iv 2207.07048). Optional classic source: C. Ambroise and G. J. McLachlan, "Selection bias in gene extraction on the basis of microarray gene-expression data," *Proc. Natl. Acad. Sci. USA*, vol. 99, no. 10, pp. 6562–6566, 2002, doi:10.1073/pnas.102102699. Metadata verified (web search: PMC124442); full text not read |
| The number of features (N) is tuned inside the folds like any other setting | [33] Cawley & Talbot (EV7), [54] Varma & Simon (EV8): tune inside each fold (nested CV) | **Covered** |
| Univariate filter: rank each feature alone and keep the top N; fast and the same for both models, but ignores redundancy | None in the master list | **NEEDS SOURCE.** UNVERIFIED — check before citing: I. Guyon and A. Elisseeff, "An introduction to variable and feature selection," *J. Mach. Learn. Res.*, vol. 3, pp. 1157–1182, 2003 (metadata verified on the JMLR listing via search). Or Y. Saeys, I. Inza and P. Larrañaga, "A review of feature selection techniques in bioinformatics," *Bioinformatics*, vol. 23, no. 19, pp. 2507–2517, 2007, doi:10.1093/bioinformatics/btm344 (metadata verified via search; its Table 1 lists univariate filters as fast and classifier-independent but ignoring feature dependencies; check the table itself) |
| Using ROC-AUC (distance from 0.5) as the score of one feature | [78] Fawcett (EV11) defines AUC | Covered for AUC itself; the ranking rule is our choice (say so) |
| Summarizing each eye signal with many statistics (mean, SD, percentiles, skew, kurtosis …) | [18] COLET (EM7) used extra statistics per measure (variation, skewness, kurtosis; per `researcher-notes.md`); [16] Božak (EM5, full text verified) computed 488 features and pruned them | **Covered.** Check in the COLET paper which statistics it lists before naming them. Optional: Rahman et al. 2021 (EM8, verified: per-window mean/SD/max features), not in the old list |
| Gaze-position features kept, with a caveat that they may reflect screen layout or headset position | [16] Božak (EM5) **excluded** absolute gaze coordinates to prevent leakage | Covered as a contrast: say "unlike Božak et al., we kept them and flag them". Optional: Geirhos et al. 2020 (CF7, verified) for shortcut learning |
| Small sample vs many features; about 10 cases of the smaller class per feature (events per variable) | None in the master list. [28] Varoquaux covers only wide error bars | **NEEDS SOURCE.** UNVERIFIED — check before citing: P. Peduzzi, J. Concato, E. Kemper, T. R. Holford and A. R. Feinstein, "A simulation study of the number of events per variable in logistic regression analysis," *J. Clin. Epidemiol.*, vol. 49, no. 12, pp. 1373–1379, 1996 (title, journal, volume and pages verified via PubMed search result; **DOI not checked**). Caution: the rule is for unpenalized LR and has been questioned (e.g. van Smeden et al., *BMC Med. Res. Methodol.*, 2016 — UNVERIFIED). Use it only as a "rule of thumb" |
| Accuracy, precision, recall, F1 (macro and per class) definitions | [78] Fawcett 2006 probably covers them (its confusion-matrix figure lists precision, recall, accuracy and F-measure, from memory); our notes rate EV11 as metadata only | **NEEDS SOURCE (or check [78]).** If Fawcett does not cover it, UNVERIFIED candidate: M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manag.*, vol. 45, no. 4, pp. 427–437, 2009, doi:10.1016/j.ipm.2009.03.002 (metadata verified via search). Accuracy = balanced accuracy here is arithmetic (44 of each class): no source needed |
| SHAP for both models; TreeSHAP for XGBoost; linear SHAP for LR | [19] Lundberg & Lee (FI1), [20] Lundberg et al. (FI3) | **Covered** |
| SHAP computed on the held-out person in each fold; a feature not picked counts as 0 | [63] Xu et al. (LB2): SHAP on all data is only exploratory | Covered for out-of-fold SHAP; the zero rule is our choice (say so) |
| Selection count across folds as a measure of importance/stability | [83] Nogueira et al. (FI14, abstract) | **Covered** (move [83] here) |
| Near-copy statistics share SHAP credit, so one signal can look weaker than it is | [62] Molnar et al. (FI9, verified); optional Aas et al. 2021 (FI10, abstract) | **Covered** |
| Elastic net sets redundant features to zero (why LR is identical from N = 40) | [50] Zou & Hastie (M8) | **Covered** |
| Talking raises blink rate, so blink rate may partly measure talking | [23] Bentivoglio (ET5), [84] Pluchino (HR2, verified: spoken dual task raised blinking) | **Covered** |
| Blink-duration statistics dropped because their blanks follow the label (D1) | [15] Kapoor & Narayanan; [64] Hogervorst (EM9) | **Covered** |
| Chance test reruns the whole pipeline (selection included) with shuffled labels | [82] Ojala & Garriga (EV14) | **Covered** |
| Comparison with other unseen-person studies (modeling plan §6.2) | He 2025 (EM16), Choi & Nam 2026 (EM17), Chen 2025 (X13) are verified in the master list but **not in the old reference list**; Božak [16], Rolon-Merette [11], Nerella [12], Shao [13], COLET [18] already are | Add EM16, EM17, X13 only if the comparison table goes into Ch. 4. The master list has no full titles for EM16 and EM17: look them up first |
| Gaze speed (S4) as a workload signal | No direct source. Di Stasi et al. 2016 (ET11, abstract) reports gaze velocity rising with task complexity | Optional. Otherwise say "included for completeness; the importance analysis decides" |
| Minimum 20 values per statistic (D2); dropping exact copies (D8, D9) and stuck features (D6a); min/max/range depend on length (D3) | — | Our own data-quality rules: list them under Design Choices, no source needed |

## 3. Problems in the reference list

- **No missing or uncited entries.** All 86 entries are cited, and every in-text number has
  an entry (checked by script).
- **Order is not strict IEEE.** Numbers do not follow first appearance in places (for
  example [25] is cited before [24]; [33] and [34] before [32]; [30] and [31] before [29]).
  Renumber everything after the cuts.
- **Missing details:** [10] Grinsztajn has no pages (the master list gives pp. 507–520);
  [19] Lundberg & Lee has no pages. JMLR entries without DOIs ([22], [33], [81], [82], [83]) are
  fine.
- **Year mismatches between our own files:** `rrl-by-topic.md` says Walocha 2024 and Xu 2025;
  the reference list says 2025 and 2026 (the list matches the DOIs). ADABase [56] is 2023 in
  the thesis but 2022 in the master list (online 2022, issue 2023; moot once dropped).
- **Evidence level to fix before submission:** [34] Hart 2006 is metadata only (CL8, "read
  before citing specific claims"); [74] Hausamann only partly supports our use; [78] Fawcett is
  metadata only, and would now carry the metric definitions too.
- **No duplicates found.** Appel 2018 [85] and Appel 2023 [58] are different papers, as are
  Varoquaux 2017 [55] and 2018 [28].

## Not sure

- Fawcett 2006 covering precision/recall/F1 is from memory; check its confusion-matrix
  figure.
- Peduzzi 1996: DOI not checked.
- The exact statistics COLET computed come from `researcher-notes.md`, not from re-reading
  the COLET paper.
- The modeling plan tunes N from {5, 10, 20, 40, 80, 140}, not every value from 5 to 140;
  describe it that way in Ch. 3.

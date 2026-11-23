# RRL by topic (study sheet)

All 86 references in `CSRP-Chapter-1-to-5-draft-COLET-v4-6.md`, grouped by how the thesis uses
them. Each reference is listed once, under its main role. Numbers match the thesis reference list.

**Know these cold:** [5] Tao, [8] Kaczorowska, [15] Kapoor, [16] Božak, [18] COLET,
[23] Bentivoglio, [32] Bargary, [56] ADABase, [84] Pluchino.

---

## 1. Cognitive load: what it is and how it's measured
| Ref | Source | Used for |
|---|---|---|
| [1] | Kahneman 1973 | Attention and mental effort theory |
| [2] | Paas & van Merriënboer 2020 | Cognitive-load theory |
| [3] | Young et al. 2015 | Mental workload in ergonomics |
| [4] | Hart & Staveland 1988 | NASA-TLX, the original |
| [34] | Hart 2006 | NASA-TLX 20 years later, and the unweighted RTLX |
| [6] | Matthews et al. 2015 | Self-report measures can disagree |
| [24] | UN 2015 | Sustainable Development Goals (relevance) |

## 2. Eye signals that track workload
| Ref | Source | Used for |
|---|---|---|
| [5] | **Tao et al. 2019** | **Systematic review:** pupil, blinks and fixations track workload |
| [35] | Mathôt 2018 | Pupil reflects effort, and also light |
| [85] | Appel et al. 2018 | Pupil-only workload model (supports P2) |
| [86] | Stolte et al. 2020 | Pupil tracks visual search and memory load (supports P2) |
| [23] | Bentivoglio et al. 1997 | **Speaking raises blink rate** (the talking confound) |
| [36] | Recarte et al. 2008 | Mental load raises blinks, visual demand lowers them |
| [37] | May et al. 1990 | Saccades and workload |
| [38] | Di Stasi et al. 2010 | Saccade peak velocity and workload |
| [39] | Di Stasi et al. 2011 | Saccade "main sequence" and workload |
| [40] | Liu et al. 2022 | Fixations and cognitive load |
| [41] | Wang et al. 2014 | Gaze spread under load |
| [42] | Recarte & Nunes 2003 | Gaze concentrates under mental load |

## 3. COLET and studies that used it
| Ref | Source | Used for |
|---|---|---|
| [18] | **Ktistakis et al. 2022** | **The COLET dataset.** LR 0.85 on A1 vs A4. Time pressure is a weak manipulation. |
| [45] | Skaramagkas et al. 2021 | Earlier conference version of COLET |
| [46] | Fuhl et al. 2023 | COLET pupil feature, random split |
| [47] | Wibirama et al. 2025 | COLET deep learning, window-based split |
| [48] | Dell'Acqua et al. 2025 | COLET with LLMs, 70/15/15 split |
| [49] | Fenoglio et al. 2023 | COLET federated learning, **tested on unseen users** |

## 4. The models: LR and XGBoost
| Ref | Source | Used for |
|---|---|---|
| [7] | Bishop 2006 | Logistic regression (textbook) |
| [50] | Zou & Hastie 2005 | Elastic net, the LR penalty |
| [9] | Chen & Guestrin 2016 | XGBoost |
| [10] | Grinsztajn et al. 2022 | Tree models do well on tabular data |

## 5. Studies comparing LR, XGBoost and similar models
| Ref | Source | Used for |
|---|---|---|
| [8] | **Kaczorowska et al. 2021** | Eye-tracking workload, LR among 8 models, **top models tie** (ceiling rule), LR weights for importance |
| [11] | Rolon-Merette et al. 2026 | LR vs XGBoost on unseen people (XGBoost better, but LR got raw data) |
| [12] | Nerella et al. 2026 | LR vs XGBoost, AR workload |
| [13] | Shao et al. 2026 | LR vs XGBoost, driving: no AUC difference |
| [52] | Chakraborty et al. 2024 | LR vs boosting, eye tracking |
| [26] | Christodoulou et al. 2019 | Review: ML doesn't beat LR in fair comparisons |
| [17] | Shafiei et al. 2025 | XGBoost + SHAP, EEG + eye tracking, surgery |
| [51] | Walocha et al. 2024 | XGBoost with nested leave-one-person-out, shallow trees chosen |

## 6. Testing on new people, and leakage
| Ref | Source | Used for |
|---|---|---|
| [14] | Saeb et al. 2017 | Test on new people, not the same people |
| [53] | Little et al. 2017 | Response to Saeb on cross-validation strategy |
| [15] | **Kapoor & Narayanan 2023** | **Leakage** makes results look too good |
| [33] | Cawley & Talbot 2010 | Tuning outside the folds biases results → nested CV |
| [54] | Varma & Simon 2006 | Same point: tune inside the folds |
| [28] | Varoquaux 2018 | Small samples → wide error bars, so report uncertainty |
| [55] | Varoquaux et al. 2017 | Cross-validation guidelines |

## 7. Per-person standardization
| Ref | Source | Used for |
|---|---|---|
| [32] | **Bargary et al. 2017** | **Why:** each person has a stable eye "signature" |
| [16] | **Božak et al. 2026** | **Closest study:** per-person features, eye tracking, explainability |
| [56] | **Oppelt et al. 2023** (ADABase) | Workload dataset paper that standardizes per person |
| [57] | Gado et al. 2023 | Per-person handling ⚠ *check: your verified note covers a label split* |
| [58] | Appel et al. 2023 | Cross-participant workload ⚠ *your notes say it used a rest baseline* |
| [59] | Albuquerque et al. 2022 | Per-person z-score works best on unseen people (**EEG**) |

## 8. Feature importance and explainability
| Ref | Source | Used for |
|---|---|---|
| [19] | Lundberg & Lee 2017 | SHAP |
| [20] | Lundberg et al. 2020 | TreeSHAP, for XGBoost |
| [21] | Breiman 2001 | Permutation importance (origin) |
| [22] | Fisher et al. 2019 | Model-agnostic permutation importance |
| [60] | Gelman 2008 | Comparing LR coefficients on a common scale |
| [61] | Strobl et al. 2007 | Built-in tree importance can be biased |
| [62] | Molnar et al. 2022 | Importance ≠ cause, plus interpretation pitfalls |
| [63] | Xu et al. 2025 | SHAP on all data should be treated as exploratory |
| [83] | Nogueira et al. 2018 | Stability of features across folds |

## 9. Cleaning the eye data
| Ref | Source | Used for |
|---|---|---|
| [65] | Kassner et al. 2014 | Pupil Labs tracker |
| [30] | Faraji et al. 2023 | Confidence threshold 0.8 for valid samples |
| [66] | Nenna et al. 2023 | Exclude recordings with more than 35% bad data |
| [67] | Hershman et al. 2018 | Blink detection |
| [29] | Kret & Sjak-Shie 2019 | Pupil cleaning guidelines |
| [68] | Steinhauer et al. 2022 | Pupil measurement standards |
| [69] | Petersch & Dierkes 2022 | Pupil size depends on gaze angle |
| [70] | Hooge et al. 2019 | Binocular tracker reliability |
| [71] | Velisar & Shanidze 2024 | Pupil Core noise |
| [72] | Kothari et al. 2020 | Eye-and-head gaze direction |
| [74] | Hausamann et al. 2020 | Reject velocities above 1000°/s |
| [31] | **Salvucci & Goldberg 2000** | **I-VT** fixation and saccade detection |
| [43] | Andersson et al. 2017 | Detection results depend on the algorithm |
| [44] | Komogortsev et al. 2010 | Checking detection quality |
| [75] | Trabulsi et al. 2021 | Fixation filters on screens |
| [73] | Duchowski 2017 | Eye-tracking methods (textbook) |

## 10. Feature selection and missing data
| Ref | Source | Used for |
|---|---|---|
| [76] | Dormann et al. 2013 | Dropping near-duplicate features (collinearity) |
| [77] | Perez-Lebel et al. 2022 | Handling missing values |

## 11. Metrics and statistics
| Ref | Source | Used for |
|---|---|---|
| [78] | Fawcett 2006 | ROC analysis |
| [79] | Niculescu-Mizil & Caruana 2005 | Probability quality (Brier) |
| [80] | Van Calster et al. 2019 | Calibration |
| [81] | Demšar 2006 | Comparing classifiers (Wilcoxon) |
| [82] | Ojala & Garriga 2010 | Permutation test against chance |

## 12. Study design and scope
| Ref | Source | Used for |
|---|---|---|
| [64] | Hogervorst et al. 2014 | Compare only the extremes (A1 vs A4) |
| [25] | Wu et al. 2020 | Extremes only, plus workload in surgery |
| [84] | **Pluchino et al. 2023** | Spoken dual task, blinks rose, talking not controlled (**precedent**) |
| [27] | Foulsham et al. 2011 | Lab results may not transfer to real life |

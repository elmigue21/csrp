# RRL List for the COLET Methodology

**Title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

These are the sources used by `methodology-colet.md`, grouped by where they support the
method. All were confirmed on a publisher, DOI or PubMed page.

- **IDs** in brackets (e.g. `[EM7]`) point to the full entry, quote and evidence status in
  `rrl-master-list.md`.
- **Evidence:** V = full text read; A = abstract only (cite for what the abstract says; check
  the full text before describing methods); M = metadata only.
- **Before final submission:** check page numbers, issue numbers and author lists against each
  DOI. Entries marked "title to confirm" need their exact title looked up.

Last updated: 2026-10-01

---

## A. Cognitive load and its measurement (Ch. 1–2 background)

1. D. Kahneman, *Attention and Effort*. Englewood Cliffs, NJ, USA: Prentice-Hall, 1973. `[CL1]` M
2. F. Paas and J. J. G. van Merriënboer, "Cognitive-load theory: Methods to manage working
   memory load in the learning of complex tasks," *Curr. Dir. Psychol. Sci.*, vol. 29, no. 4,
   pp. 394–398, 2020. `[CL2]` M
3. M. S. Young, K. A. Brookhuis, C. D. Wickens, and P. A. Hancock, "State of science: Mental
   workload in ergonomics," *Ergonomics*, vol. 58, no. 1, pp. 1–17, 2015,
   doi:10.1080/00140139.2014.956151. `[CL3]` A
4. S. G. Hart and L. E. Staveland, "Development of NASA-TLX (Task Load Index): Results of
   empirical and theoretical research," in *Human Mental Workload*, P. A. Hancock and
   N. Meshkati, Eds. Amsterdam: North-Holland, 1988, pp. 139–183,
   doi:10.1016/S0166-4115(08)62386-9. `[CL4]` V
5. A. C. Marinescu *et al.*, "Physiological parameter response to variation of mental
   workload," *Hum. Factors*, vol. 60, no. 1, pp. 31–56, 2018,
   doi:10.1177/0018720817733101. `[CL6]` V
6. D. Tao *et al.*, "A systematic review of physiological measures of mental workload," *Int. J.
   Environ. Res. Public Health*, vol. 16, no. 15, 2716, 2019, doi:10.3390/ijerph16152716.
   `[CL7]` V
7. G. Matthews, L. E. Reinerman-Jones, D. J. Barber, and J. Abich, "The psychometrics of mental
   workload: Multiple measures are sensitive but divergent," *Hum. Factors*, 2015,
   doi:10.1177/0018720814539505. `[LB10]` A
8. P. A. Hancock and G. Matthews, "Workload and performance: Associations, insensitivities, and
   dissociations," *Hum. Factors*, 2019, doi:10.1177/0018720818809590. `[LB11]` A

## B. Eye-tracking features as workload indicators (Ch. 2; Ch. 3 features)

9. A. T. Duchowski, *Eye Tracking Methodology: Theory and Practice*, 3rd ed. Cham: Springer,
   2017, doi:10.1007/978-3-319-57883-5. `[ET1]` M
10. J. G. May, R. S. Kennedy, M. C. Williams, W. P. Dunlap, and J. R. Brannan, "Eye movement
    indices of mental workload," *Acta Psychol.*, vol. 75, pp. 75–89, 1990,
    doi:10.1016/0001-6918(90)90067-p. `[ET8]` A (saccade amplitude)
11. L. L. Di Stasi *et al.*, "Saccadic peak velocity sensitivity to variations in mental
    workload," *Aviat. Space Environ. Med.*, vol. 81, no. 4, pp. 413–417, 2010,
    doi:10.3357/asem.2579.2010. `[ET9]` A (peak velocity)
12. L. L. Di Stasi, A. Antolí, and J. J. Cañas, "Main sequence: An index for detecting mental
    workload variation in complex tasks," *Appl. Ergon.*, vol. 42, no. 6, pp. 807–813, 2011,
    doi:10.1016/j.apergo.2011.01.003. `[ET10]` A
13. M. A. Recarte, E. Pérez, A. Conchillo, and L. M. Nunes, "Mental workload and visual
    impairment: Differences between pupil, blink, and subjective rating," *Span. J. Psychol.*,
    vol. 11, pp. 374–385, 2008, doi:10.1017/s1138741600004406. `[ET12]` A (blink rate)
14. K. F. Van Orden, W. Limbert, S. Makeig, and T.-P. Jung, "Eye activity correlates of workload
    during a visuospatial memory task," *Hum. Factors*, vol. 43, pp. 111–121, 2001,
    doi:10.1518/001872001775992570. `[ET22]` A
15. J. C. Liu, K. A. Li, S. L. Yeh, and S. Y. Chien, "Assessing perceptual load and cognitive
    load by fixation-related information of eye movements," *Sensors*, vol. 22, no. 3, 1187,
    2022, doi:10.3390/s22031187. `[ET21]` A (fixation rate)
16. Y. Wang, B. Reimer, J. Dobres, and B. Mehler, "The sensitivity of different methodologies for
    characterizing drivers' gaze concentration under increased cognitive demand," *Transp. Res.
    F*, vol. 26, pp. 227–237, 2014, doi:10.1016/j.trf.2014.08.003. `[ET14]` A (gaze dispersion)
17. M. A. Recarte and L. M. Nunes, "Mental workload while driving: Effects on visual search,
    discrimination, and decision making," *J. Exp. Psychol. Appl.*, vol. 9, pp. 119–137, 2003,
    doi:10.1037/1076-898x.9.2.119. `[ET15]` A
18. B. Shiferaw, L. Downey, and D. Crewther, "A review of gaze entropy as a measure of visual
    scanning efficiency," *Neurosci. Biobehav. Rev.*, vol. 96, pp. 353–366, 2019,
    doi:10.1016/j.neubiorev.2018.12.007. `[ET17]` A (gaze entropy)
19. S. Upasani, D. Srinivasan, Q. Zhu, J. Du, and A. Leonessa, "Eye-tracking in physical
    human–robot interaction: Mental workload and performance prediction," *Hum. Factors*,
    vol. 66, no. 8, pp. 2104–2119, 2024, doi:10.1177/00187208231204704. `[ET6/HR1]` A
    (stationary gaze entropy; 30 s intervals)
20. S. Mathôt, "Pupillometry: Psychology, physiology, and function," *J. Cognition*, vol. 1,
    no. 1, 16, 2018, doi:10.5334/joc.18. `[CF5]` V (pupil: effort and light)

## C. Dataset: COLET and its prior use (Ch. 2 related work; Ch. 3 dataset)

21. E. Ktistakis *et al.*, "COLET: A dataset for COgnitive workLoad estimation based on
    eye-tracking," *Comput. Methods Programs Biomed.*, vol. 224, 106989, 2022,
    doi:10.1016/j.cmpb.2022.106989. Data: Zenodo record 7766785. `[EM7]` **V** (full PDF in the
    repo root)
22. V. Skaramagkas *et al.*, "Cognitive workload level estimation based on eye tracking: A
    machine learning approach," *Proc. IEEE BIBE*, 2021, doi:10.1109/BIBE52308.2021.9635166.
    `[X10]` V (COLET precursor)
23. W. Fuhl *et al.*, "A temporally quantized distribution of pupil diameters as a new feature
    for cognitive load classification," *Proc. ACM ETRA*, 2023, doi:10.1145/3588015.3590116.
    `[§9b]` V (used COLET; random split; 10–30 s windows)
24. D. Fenoglio, D. Josifovski, A. Gobbetti, M. Formo, H. Gjoreski, M. Gjoreski, and
    M. Langheinrich, "Federated learning for privacy-aware cognitive workload estimation," in
    *Proc. 22nd Int. Conf. Mobile Ubiquitous Multimedia (MUM)*, 2023, pp. 25–36,
    doi:10.1145/3626705.3627783. `[§9b]` V (COLET; 4-fold user-grouped CV, unseen test users;
    5–30 s overlapping windows; GNB/RF/neural nets; no LR, XGBoost or SHAP; V2-6)
25. D. Fenoglio, M. Gjoreski, and M. Langheinrich, "A federated unsupervised personalisation for
    cognitive workload estimation," in *Proc. MUM*, 2023, pp. 526–528,
    doi:10.1145/3626705.3631796. `[§9b]` A (short paper; full text not available)
26. S. Wibirama *et al.*, "Classification of cognitive load using deep learning based on eye
    movement indices," *IEEE Access*, vol. 13, 2025, doi:10.1109/ACCESS.2025.3613292. `[§9b]` V
    (used COLET; overlapping windows, likely leakage)
27. Dell'Acqua, Garofalo, La Rosa, and Villari, "Inferring cognitive workload from symbolic gaze
    sequences using large language models," *IEEE Access*, 2025,
    doi:10.1109/ACCESS.2025.3646271. `[§9b]` V (used COLET; trial-level split)
28. T. Foulsham, E. Walker, and A. Kingstone, "The where, what and when of gaze allocation in the
    lab and the natural environment," *Vision Res.*, vol. 51, no. 17, pp. 1920–1931, 2011,
    doi:10.1016/j.visres.2011.07.002. `[DS1]` A (screen-based lab limitation)
29. N. V. Valtakari *et al.*, "Eye tracking in human interaction: Possibilities and limitations,"
    *Behav. Res. Methods*, vol. 53, no. 4, pp. 1592–1608, 2021,
    doi:10.3758/s13428-020-01517-x. `[DS3]` V
30. M. D. Wilkinson *et al.*, "The FAIR Guiding Principles for scientific data management and
    stewardship," *Sci. Data*, vol. 3, 160018, 2016, doi:10.1038/sdata.2016.18. `[DS6]` A
    (public data)
31. J. Pineau *et al.*, "Improving reproducibility in machine learning research," *J. Mach.
    Learn. Res.*, vol. 22, no. 164, pp. 1–20, 2021. `[DS7]` A

## D. Closest methodological precedents (Ch. 2 synthesis table)

32. T. Rolon-Merette *et al.*, "Towards a cross-participant cognitive load classification using
    eye tracking and deep learning," *Proc. FLAIRS*, vol. 39, no. 1, 2026,
    doi:10.32473/flairs.39.1.141863. `[EM4/X1]` V (LR vs XGB on own n-back data; 10 held-out
    participants, not full LOPO; no feature importance; V2-6)
33. G. Nerella, D. Gan, B. David-John, and R. Alghofaili, "Gaze-based prediction of cognitive load
    in augmented reality," *Proc. ACM Hum.-Comput. Interact.* (ETRA), vol. 10, no. 3, 2026,
    doi:10.1145/3806040. `[X2]` V (LR vs XGB, tuned settings reported)
34. T. Božak, S. Goyal, M. Langheinrich, M. Gjoreski, and G. Slapničar, "Evaluating
    feature-based machine-learning models with post hoc explainability for eye-tracking-based
    task type and workload inference," *AI*, vol. 7, no. 8, 325, 2026, doi:10.3390/ai7080325.
    `[EM5/X7]` V (closest twin: condition labels, LOSO, SHAP, baseline)
35. S. Chakraborty, P. Kiefer, and M. Raubal, "Estimating perceived mental workload from
    eye-tracking data based on benign anisocoria," *IEEE Trans. Hum.-Mach. Syst.*, vol. 54,
    no. 5, pp. 499–507, 2024, doi:10.1109/THMS.2024.3432864. `[X3]` V (LR vs boosting,
    eye-only)
36. M. Kaczorowska, M. Plechawska-Wójcik, and M. Tokovarov, "Interpretable machine learning
    models for three-way classification of cognitive workload levels for eye-tracking features,"
    *Brain Sci.*, vol. 11, no. 2, 210, 2021, doi:10.3390/brainsci11020210. `[EM1]` V
    (elastic-net LR importance)
37. M. Kaczorowska, P. Karczmarek, M. Plechawska-Wójcik, and M. Tokovarov, "On the improvement of
    eye tracking-based cognitive workload estimation using aggregation functions," *Sensors*,
    vol. 21, no. 13, 4542, 2021, doi:10.3390/s21134542. `[EM2]` V (participant-disjoint
    validation only)
38. M. Kaczorowska, M. Plechawska-Wójcik, M. Tokovarov, and P. Krukow, "Automated classification
    of cognitive workload levels based on psychophysiological and behavioural variables of
    ex-Gaussian distributional features," *Brain Sci.*, vol. 12, no. 5, 542, 2022,
    doi:10.3390/brainsci12050542. `[EM18]` V (pupil-free)
39. Choi and Nam, *J. Eye Mov. Res.*, vol. 19, no. 3, 50, 2026, doi:10.3390/jemr19030050 (title
    to confirm). `[EM17]` V (XGB, LOSO, no pupil)
40. He *et al.*, *Sensors*, vol. 25, 2377, 2025, doi:10.3390/s25082377 (title to confirm).
    `[EM16]` V (condition labels, LOSO, wearable)
41. T. Appel *et al.*, "Cross-task and cross-participant classification of cognitive load in an
    emergency simulation game," *IEEE Trans. Affect. Comput.*, 2023,
    doi:10.1109/TAFFC.2021.3098237. `[EM19]` V
42. T. Appel, C. Scharinger, P. Gerjets, and E. Kasneci, "Cross-subject workload classification
    using pupil-related measures," *Proc. ACM ETRA*, 2018, doi:10.1145/3204493.3204531. `[EM6]` A
43. X. Shao, X. Ma, F. Chen, and X. Pan, "Multimodal machine learning framework for driver mental
    workload classification: A comparative and interpretable approach," *Appl. Sci.*, vol. 16,
    no. 7, 3581, 2026, doi:10.3390/app16073581. `[X12]` V (LR ≈ XGB on eye-only)
44. W. Chen *et al.*, "Machine learning models to predict individual cognitive load in
    collaborative learning: Combining fNIRS and eye-tracking data," *Mach. Learn. Knowl.
    Extr.*, vol. 7, no. 2, 51, 2025, doi:10.3390/make7020051. `[X13]` V
45. F. Walocha, A. Schrank, H. P. Nguyen, and K. Ihme, "Multimodal assessment of mental workload
    during automated vehicle remote assistance," *Information*, vol. 16, no. 1, 64, 2025,
    doi:10.3390/info16010064. `[X14]` V (XGB nested LOSO settings)
46. M. Trigka, E. Dritsas, and P. Mylonas, "Eye-based cognitive overload prediction in
    human-machine interaction via machine learning," *Proc. WEBIST*, 2025,
    doi:10.5220/0013782800003985. `[EM12]` V (**contrast case only**: no cross-subject
    validation)

## E. Labels: condition labels and self-report (Ch. 3 labels)

47. S. Gado, K. Lingelbach, M. Wirzberger, and M. Vukelić, "Decoding mental effort in a
    quasi-realistic scenario: A feasibility study on multimodal data fusion and
    classification," *Sensors*, vol. 23, no. 14, 6546, 2023, doi:10.3390/s23146546.
    `[LB1/LB4]` V (condition labels F1 0.69 vs self-report 0.35)
48. M. A. Hogervorst, A.-M. Brouwer, and J. B. F. van Erp, "Combining and comparing EEG,
    peripheral physiology and eye-related measures for the assessment of mental workload,"
    *Front. Neurosci.*, vol. 8, 322, 2014, doi:10.3389/fnins.2014.00322. `[EM9/LB5]` V (drop
    the middle level; 30 s segments)
49. I. Albuquerque *et al.*, "WAUC: A multi-modal database for mental workload assessment under
    physical activity," *Front. Neurosci.*, vol. 14, 549524, 2020,
    doi:10.3389/fnins.2020.549524. `[LB6]` V
50. M. P. Oppelt *et al.*, "ADABase: A multimodal dataset for cognitive load estimation,"
    *Sensors*, vol. 23, no. 1, 340, 2023, doi:10.3390/s23010340. `[LB7/X15]` V
51. P. Schmidt, A. Reiss, R. Dürichen, and K. Van Laerhoven, "Wearable-based affect
    recognition—A review," *Sensors*, vol. 19, no. 19, 4079, 2019,
    doi:10.3390/s19194079. `[CF6/LB8]` V
52. L. Fridman, B. Reimer, B. Mehler, and W. T. Freeman, "Cognitive load estimation in the
    wild," *Proc. CHI*, 2018, doi:10.1145/3173574.3174226. `[LB9]` A
53. A. Gorin *et al.*, *Sensors*, 2024, doi:10.3390/s24061759 (title to confirm). `[LB12]` V
54. H. P. Martínez, G. N. Yannakakis, and J. Hallam, "Don't classify ratings of affect; rank
    them!," *IEEE Trans. Affect. Comput.*, vol. 5, no. 3, pp. 314–326, 2014,
    doi:10.1109/TAFFC.2014.2352268. `[LB3]` V (self-report scale differences)

## F. Preprocessing raw eye-tracking data (Ch. 3 preprocessing)

55. S. Mathôt, J. Fabius, E. Van Heusden, and S. Van der Stigchel, "Safe and sensible
    preprocessing and baseline correction of pupil-size data," *Behav. Res. Methods*, 2018,
    doi:10.3758/s13428-017-1007-2. `[NM7]` A (subtractive baseline)
56. S. R. Steinhauer, M. M. Bradley, G. J. Siegle, K. A. Roecklein, and A. Dix, "Publication
    guidelines and recommendations for pupillary measurement in psychophysiological studies,"
    *Psychophysiology*, vol. 59, no. 4, e14035, 2022, doi:10.1111/psyp.14035. `[CF2]` V
57. D. C. Niehorster *et al.*, "The impact of slippage on the data quality of head-worn eye
    trackers," *Behav. Res. Methods*, vol. 52, pp. 1140–1160, 2020,
    doi:10.3758/s13428-019-01307-0. `[DS4]` A (data quality)

*Preprocessing IDs (PP1–PP19) are listed here and used in `methodology-colet.md` §3b.*

113. D. D. Salvucci and J. H. Goldberg, "Identifying fixations and saccades in eye-tracking
     protocols," in *Proc. ETRA*, 2000, pp. 71–78, doi:10.1145/355017.355028. `[PP1]` V
     (I-VT/I-DT; 100–200 ms minimum)
114. O. V. Komogortsev, D. V. Gobert, S. Jayarathna, D. H. Koh, and S. M. Gowda, "Standardization
     of automated analyses of oculomotor fixation and saccadic behaviors," *IEEE Trans. Biomed.
     Eng.*, vol. 57, no. 11, pp. 2635–2645, 2010, doi:10.1109/TBME.2010.2057429. `[PP2]` A
     (results depend on thresholds)
115. R. Andersson, L. Larsson, K. Holmqvist, M. Stridh, and M. Nyström, "One algorithm to rule
     them all? An evaluation and discussion of ten eye movement event-detection algorithms,"
     *Behav. Res. Methods*, vol. 49, no. 2, pp. 616–637, 2017, doi:10.3758/s13428-016-0738-9.
     `[PP3]` V
116. K. Holmqvist *et al.*, *Eye Tracking: A Comprehensive Guide to Methods and Measures*.
     Oxford, U.K.: Oxford Univ. Press, 2011. `[PP4]` M (quote only from your own copy)
117. M. Kassner, W. Patera, and A. Bulling, "Pupil: An open source platform for pervasive eye
     tracking and mobile gaze-based interaction," in *Proc. UbiComp Adjunct*, 2014,
     pp. 1151–1160, doi:10.1145/2638728.2641695. `[PP5]` V (author version; defines confidence)
118. B. V. Ehinger, K. Groß, I. Ibs, and P. König, "A new comprehensive eye-tracking test battery
     concurrently evaluating the Pupil Labs glasses and the EyeLink 1000," *PeerJ*, vol. 7,
     e7086, 2019, doi:10.7717/peerj.7086. `[PP6]` V (independent accuracy check; blink-detector
     caveat)
119. Y. Faraji *et al.*, "A toolkit for wide-screen dynamic area of interest measurements using the
     Pupil Labs Core Eye Tracker," *Behav. Res. Methods*, vol. 55, no. 7, pp. 3820–3830, 2023,
     doi:10.3758/s13428-022-01991-5. `[PP7]` V (confidence < 0.8 = gap; < 75 ms interpolation;
     fused 240 Hz stream)
120. P. Hausamann, C. Sinnott, and P. R. MacNeilage, "Positional head-eye tracking outside the
     lab: An open-source solution," in *Proc. ETRA*, 2020, doi:10.1145/3379156.3391365. `[PP8]`
     partial (confidence < 0.8 excluded; > 1000°/s rejected)
121. B. Petersch and K. Dierkes, "Gaze-angle dependency of pupil-size measurements in
     head-mounted eye tracking," *Behav. Res. Methods*, vol. 54, no. 2, pp. 763–779, 2022,
     doi:10.3758/s13428-021-01657-8. `[PP9]` V (use `diameter_3d`; both authors are from Pupil
     Labs)
122. R. Hershman, A. Henik, and N. Cohen, "A novel blink detection method based on pupillometry
     noise," *Behav. Res. Methods*, vol. 50, no. 1, pp. 107–114, 2018,
     doi:10.3758/s13428-017-1008-1. `[PP10]` V (merge blinks < 100 ms apart)
123. M. E. Kret and E. E. Sjak-Shie, "Preprocessing pupil size data: Guidelines and code," *Behav.
     Res. Methods*, vol. 51, no. 3, pp. 1336–1342, 2019, doi:10.3758/s13428-018-1075-y.
     `[PP11]` V (1.5–9 mm, MAD, 4 Hz low-pass)
124. B. Birawo and P. Kasprowski, "Review and evaluation of eye movement event detection
     algorithms," *Sensors*, vol. 22, no. 22, 8810, 2022, doi:10.3390/s22228810. `[PP12]` V
     (I-DT 0.5–1°, 100–200 ms)
125. J. Trabulsi *et al.*, "Optimizing fixation filters for eye-tracking on small screens,"
     *Front. Neurosci.*, vol. 15, 578439, 2021, doi:10.3389/fnins.2021.578439. `[PP13]` V
     (I-VT 30°/s; 60 ms)
126. B. Shiferaw, L. Downey, J. Westlake *et al.*, "Stationary gaze entropy predicts lane departure
     events in sleep-deprived drivers," *Sci. Rep.*, vol. 8, 2220, 2018,
     doi:10.1038/s41598-018-20588-7. `[PP14]` V (entropy grid and normalization)
127. K. Krejtz *et al.*, "Gaze transition entropy," *ACM Trans. Appl. Percept.*, vol. 13, no. 1,
     5, pp. 1–20, 2015, doi:10.1145/2834121. `[PP15]` A
128. K. Krejtz, T. Szmidt, A. T. Duchowski, and I. Krejtz, "Entropy-based statistical analysis of
     eye movement transitions," in *Proc. ETRA*, 2014, pp. 159–166,
     doi:10.1145/2578153.2578176. `[PP16]` M

**Not peer-reviewed; cite only as technical documentation:** Pupil Labs Core documentation
(confidence ≈ 0.6 rule of thumb; `diameter` in px vs `diameter_3d` in mm).

## G. Windows (Ch. 3 temporal segmentation)

58. H. Rahman, M. U. Ahmed, S. Barua, P. Funk, and S. Begum, "Vision-based driver's cognitive
    load classification considering eye movement using machine learning and deep learning,"
    *Sensors*, vol. 21, no. 23, 8019, 2021, doi:10.3390/s21238019. `[EM8]` V (30 s best of
    15/30/60)
59. T. Chihara and J. Sakamoto, "Effect of time length of eye movement data analysis on the
    accuracy of mental workload estimation during automobile driving," *Proc. IEA*, LNNS,
    pp. 593–599, 2021, doi:10.1007/978-3-030-74608-7_72. `[WN3]` A (counter-evidence)
60. J. Tervonen, K. Pettersson, and J. Mäntyjärvi, "Ultra-short window length and feature
    importance analysis for cognitive load detection from wearable sensors," *Electronics*,
    vol. 10, no. 5, 613, 2021, doi:10.3390/electronics10050613. `[WN4/X21]` V
61. A. R. Bentivoglio *et al.*, "Analysis of blink rate patterns in normal subjects," *Mov.
    Disord.*, vol. 12, no. 6, pp. 1028–1034, 1997, doi:10.1002/mds.870120629. `[ET5]` A

## H. Confounds: luminance and shortcut learning (Ch. 3 checks; Ch. 4 discussion)

62. M. Lohani, B. R. Payne, and D. L. Strayer, "A review of psychophysiological measures to assess
    cognitive states in real-world driving," *Front. Hum. Neurosci.*, vol. 13, 57, 2019,
    doi:10.3389/fnhum.2019.00057. `[CF1]` V
63. B. Pfleging, D. K. Fekety, A. Schmidt, and A. L. Kun, "A model relating pupil diameter to
    mental workload and lighting conditions," *Proc. CHI*, pp. 5776–5788, 2016,
    doi:10.1145/2858036.2858117. `[CF3]` A
64. R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nat. Mach. Intell.*,
    vol. 2, no. 11, pp. 665–673, 2020, doi:10.1038/s42256-020-00257-z. `[CF7]` V
65. J. R. Zech *et al.*, "Variable generalization performance of a deep learning model to detect
    pneumonia in chest radiographs: A cross-sectional study," *PLoS Med.*, vol. 15, no. 11,
    e1002683, 2018, doi:10.1371/journal.pmed.1002683. `[CF8]` V

## I. Per-person normalization (Ch. 3)

66. I. Albuquerque, J. Monteiro, O. Rosanne, and T. H. Falk, "Estimating distribution shifts for
    predicting cross-subject generalization in electroencephalography-based mental workload
    assessment," *Front. Artif. Intell.*, vol. 5, 2022, doi:10.3389/frai.2022.992732.
    `[NM1/NM4]` V
67. J. Fdez, N. Guttenberg, O. Witkowski, and A. Pasquali, "Cross-subject EEG-based emotion
    recognition through neural networks with stratified normalization," *Front. Neurosci.*,
    vol. 15, 626277, 2021, doi:10.3389/fnins.2021.626277. `[NM2]` V
68. Tognotti, Otesteanu, Anceschi, and Menon, *Front. Digit. Health*, vol. 8, 1827279, 2026,
    doi:10.3389/fdgth.2026.1827279 (title to confirm). `[NM5]` V (leak-free normalization)
69. G. Bargary *et al.*, "Individual differences in human eye movements: An oculomotor
    signature?" *Vision Res.*, vol. 141, pp. 157–169, 2017,
    doi:10.1016/j.visres.2017.03.001. `[ET2]` A
70. W. Poynter, M. Barber, J. Inman, and C. Wiggins, "Individuals exhibit idiosyncratic
    eye-movement behavior profiles across tasks," *Vision Res.*, vol. 89, pp. 32–38, 2013,
    doi:10.1016/j.visres.2013.07.002. `[ET3]` A

## J. Models: LR, XGBoost, and comparing them (Ch. 2–3)

71. C. M. Bishop, *Pattern Recognition and Machine Learning*. New York: Springer, 2006. `[M1]` M
72. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM
    SIGKDD*, 2016, pp. 785–794, doi:10.1145/2939672.2939785. `[M2]` V
73. H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *J. R.
    Stat. Soc. B*, vol. 67, no. 2, pp. 301–320, 2005,
    doi:10.1111/j.1467-9868.2005.00503.x. `[M8]` A
74. E. Christodoulou *et al.*, "A systematic review shows no performance benefit of machine
    learning over logistic regression for clinical prediction models," *J. Clin. Epidemiol.*,
    vol. 110, pp. 12–22, 2019, doi:10.1016/j.jclinepi.2019.02.004. `[M3]` A
75. S. Nusinovici *et al.*, "Logistic regression was as good as machine learning for predicting
    major chronic diseases," *J. Clin. Epidemiol.*, vol. 122, pp. 56–69, 2020,
    doi:10.1016/j.jclinepi.2020.03.002. `[M4]` A
76. R. Couronné, P. Probst, and A.-L. Boulesteix, "Random forest versus logistic regression: A
    large-scale benchmark experiment," *BMC Bioinformatics*, vol. 19, 270, 2018,
    doi:10.1186/s12859-018-2264-5. `[M5]` A
77. L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep
    learning on typical tabular data?," in *Proc. NeurIPS (Datasets & Benchmarks)*, 2022,
    pp. 507–520. `[M6]` A

## K. Validation, leakage, and tuning (Ch. 3)

78. S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation,
    detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, 15, 2012,
    doi:10.1145/2382577.2382579. `[EV1]` A
79. S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based
    science," *Patterns*, vol. 4, no. 9, 100804, 2023, doi:10.1016/j.patter.2023.100804.
    `[EV2]` V
80. S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, "The need to approximate the
    use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, 2017,
    doi:10.1093/gigascience/gix019. `[EV3]` V
81. M. A. Little, G. Varoquaux, S. Saeb *et al.*, "Using and understanding cross-validation
    strategies. Perspectives on Saeb et al.," *GigaScience*, vol. 6, no. 5, 2017,
    doi:10.1093/gigascience/gix020. `[EV4]` V
82. G. Brookshire *et al.*, "Data leakage in deep learning studies of translational EEG," *Front.
    Neurosci.*, vol. 18, 1373515, 2024, doi:10.3389/fnins.2024.1373515. `[EV5]` A
83. G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection
    bias in performance evaluation," *J. Mach. Learn. Res.*, vol. 11, pp. 2079–2107, 2010.
    `[EV7]` V
84. S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model
    selection," *BMC Bioinformatics*, vol. 7, 91, 2006, doi:10.1186/1471-2105-7-91. `[EV8]` V
85. G. Varoquaux *et al.*, "Assessing and tuning brain decoders: Cross-validation, caveats, and
    guidelines," *NeuroImage*, vol. 145, pp. 166–179, 2017,
    doi:10.1016/j.neuroimage.2016.10.038. `[EV9]` V
86. Demirezen, Taşkaya Temizel, and Brouwer, *Front. Neuroergon.*, vol. 5, 1346794, 2024,
    doi:10.3389/fnrgo.2024.1346794 (title to confirm). `[LB13]` V (reporting checklist)

## L. Missing data (Ch. 3)

87. A. Perez-Lebel, G. Varoquaux, M. Le Morvan, J. Josse, and J.-B. Poline, "Benchmarking
    missing-values approaches for predictive models on health databases," *GigaScience*,
    vol. 11, giac013, 2022, doi:10.1093/gigascience/giac013. `[MD1]` V
88. M. Van Ness, T. M. Bosschieter, R. Halpin-Gregorio, and M. Udell, "The missing indicator
    method: From low to high dimensions," in *Proc. ACM KDD*, 2023, pp. 5004–5015,
    doi:10.1145/3580305.3599911. `[MD2]` A

## M. Metrics and statistical comparison (Ch. 3)

89. T. Fawcett, "An introduction to ROC analysis," *Pattern Recognit. Lett.*, vol. 27, no. 8,
    pp. 861–874, 2006, doi:10.1016/j.patrec.2005.10.010. `[EV11]` M
90. G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars,"
    *NeuroImage*, vol. 180, pp. 68–77, 2018, doi:10.1016/j.neuroimage.2017.06.061. `[EV10]` A
91. G. Forman and M. Scholz, "Apples-to-apples in cross-validation studies: Pitfalls in
    classifier performance measurement," *ACM SIGKDD Explor.*, vol. 12, no. 1, pp. 49–57, 2010,
    doi:10.1145/1882471.1882479. `[EV13]` A
92. J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *J. Mach. Learn.
    Res.*, vol. 7, pp. 1–30, 2006. `[EV12]` M
93. T. G. Dietterich, "Approximate statistical tests for comparing supervised classification
    learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998,
    doi:10.1162/089976698300017197. `[M9]` A
94. C. Nadeau and Y. Bengio, "Inference for the generalization error," *Mach. Learn.*, vol. 52,
    no. 3, pp. 239–281, 2003, doi:10.1023/A:1024068626366. `[M10]` M
95. A. Benavoli, G. Corani, J. Demšar, and M. Zaffalon, "Time for a change: A tutorial for
    comparing multiple classifiers through Bayesian analysis," *J. Mach. Learn. Res.*, vol. 18,
    no. 77, pp. 1–36, 2017. `[M11]` A
96. A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning,"
    in *Proc. ICML*, 2005, pp. 625–632, doi:10.1145/1102351.1102430. `[M13]` V
97. B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg,
    "Calibration: The Achilles heel of predictive analytics," *BMC Med.*, vol. 17, 230, 2019,
    doi:10.1186/s12916-019-1466-7. `[M14]` A

## N. Feature importance analysis (Ch. 3; title component)

98. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in
    *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017. `[FI1]` A
99. S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI
    for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020,
    doi:10.1038/s42256-019-0138-9. `[FI3]` V
100. L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001,
     doi:10.1023/A:1010933404324. `[FI4]` A
101. A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a
     variable's importance by studying an entire class of prediction models simultaneously,"
     *J. Mach. Learn. Res.*, vol. 20, no. 177, pp. 1–81, 2019. `[FI5]` A
102. C. Strobl, A.-L. Boulesteix, A. Zeileis, and T. Hothorn, "Bias in random forest variable
     importance measures: Illustrations, sources and a solution," *BMC Bioinformatics*, vol. 8,
     25, 2007, doi:10.1186/1471-2105-8-25. `[FI6]` V
103. C. Strobl, A.-L. Boulesteix, T. Kneib, T. Augustin, and A. Zeileis, "Conditional variable
     importance for random forests," *BMC Bioinformatics*, vol. 9, 307, 2008,
     doi:10.1186/1471-2105-9-307. `[FI7]` V
104. G. Hooker, L. Mentch, and S. Zhou, "Unrestricted permutation forces extrapolation: Variable
     importance requires at least one more model, or there is no free variable importance,"
     *Stat. Comput.*, vol. 31, 82, 2021, doi:10.1007/s11222-021-10057-z. `[FI8]` V
105. C. Molnar *et al.*, "General pitfalls of model-agnostic interpretation methods for machine
     learning models," in *xxAI – Beyond Explainable AI*, LNAI 13200, Springer, 2022,
     pp. 39–68, doi:10.1007/978-3-031-04083-2_4. `[FI9]` V
106. K. Aas, M. Jullum, and A. Løland, "Explaining individual predictions when features are
     dependent: More accurate approximations to Shapley values," *Artif. Intell.*, vol. 298,
     103502, 2021, doi:10.1016/j.artint.2021.103502. `[FI10]` A
107. A. Altmann, L. Toloşi, O. Sander, and T. Lengauer, "Permutation importance: A corrected
     feature importance measure," *Bioinformatics*, vol. 26, no. 10, pp. 1340–1347, 2010,
     doi:10.1093/bioinformatics/btq134. `[FI11]` A
108. A. Gelman, "Scaling regression inputs by dividing by two standard deviations," *Stat. Med.*,
     vol. 27, no. 15, pp. 2865–2873, 2008, doi:10.1002/sim.3107. `[FI12]` A
109. C. F. Dormann *et al.*, "Collinearity: A review of methods to deal with it and a simulation
     study evaluating their performance," *Ecography*, vol. 36, no. 1, pp. 27–46, 2013,
     doi:10.1111/j.1600-0587.2012.07348.x. `[FI13]` A
110. S. Nogueira, K. Sechidis, and G. Brown, "On the stability of feature selection algorithms,"
     *J. Mach. Learn. Res.*, vol. 18, no. 174, pp. 1–54, 2018. `[FI14]` A
111. S. B. Shafiei, S. Shadpour, and J. L. Mohler, "An integrated EEG and eye-tracking analysis
     using XGBoost for mental workload evaluation in surgery," *Hum. Factors*, 2025,
     doi:10.1177/00187208241285513. `[EM15/X16]` V (XGB + SHAP; random split)
112. R. Xu *et al.*, "Consumer-grade wearable sensors for classifying pilot workload and stress
     during real flight training: A leave-one-subject-out validation study," *Sensors*, vol. 26,
     no. 12, 3627, 2026, doi:10.3390/s26123627. `[LB2/X22]` V (nested LOSO; SHAP caution)

## O. Added after the independent review (2026-10-01)

Metadata confirmed on Crossref. These fill sources the methodology and researcher notes cite
but that were missing from this list (review R3/R4/R6).

129. C. Wu, J. Cha, J. Sulek, T. Zhou, C. Sundaram, J. Wachs, and D. Yu, "Eye-tracking metrics
     predict perceived workload in robotic surgical skills training," *Hum. Factors*, vol. 62,
     no. 8, pp. 1365–1386, 2020 (online 2019), doi:10.1177/0018720819874544. `[EM14]` V (only
     extremes kept: top/bottom NASA-TLX quartiles)
130. M. A. Recarte and L. M. Nunes, "Effects of verbal and spatial-imagery tasks on eye
     fixations while driving," *J. Exp. Psychol. Appl.*, vol. 6, no. 1, pp. 31–43, 2000,
     doi:10.1037/1076-898X.6.1.31. `[ET16]` A (visual field narrows under load)
131. S. Scannella, V. Peysakhovich, F. Ehrig, E. Lepron, and F. Dehais, "Assessment of ocular
     and physiological metrics to discriminate flight phases in real light aircraft," *Hum.
     Factors*, vol. 60, no. 7, pp. 922–935, 2018, doi:10.1177/0018720818787135. `[ET20]` A
     (saccade rate discriminates load)
132. F. Nenna, V. Orso, D. Zanardi, and L. Gamberini, "The virtualization of human–robot
     interactions: A user-centric workload assessment," *Virtual Reality*, vol. 27, no. 2,
     pp. 553–571, 2023 (online 2022), doi:10.1007/s10055-022-00667-x. `[HR4]` V (trials with
     > 35% missing data dropped: the C3 threshold source)
133. P. Pluchino *et al.*, "Advanced workstations and collaborative robots: Exploiting
     eye-tracking and cardiac activity indices to unveil senior workers' mental workload in
     assembly tasks," *Front. Robot. AI*, vol. 10, 1275572, 2023,
     doi:10.3389/frobt.2023.1275572. `[HR2]` V (spoken serial-subtraction dual task; blink
     rate rose; talking not controlled: C2b precedent)
134. S. G. Hart, "NASA-Task Load Index (NASA-TLX); 20 years later," *Proc. Hum. Factors
     Ergon. Soc. Annu. Meet.*, vol. 50, no. 9, pp. 904–908, 2006,
     doi:10.1177/154193120605000909. `[CL8]` M (reviews TLX use, including the unweighted
     "raw TLX"; **read before citing specific claims**)
135. J. C. Byers, A. C. Bittner, and S. G. Hill, "Traditional and raw task load index (TLX)
     correlations: Are paired comparisons necessary?," in *Advances in Industrial Ergonomics
     and Safety I*. London: Taylor & Francis, 1989, pp. 481–485. `[CL9]` M (the RTLX source
     COLET cites, ref [41]; no DOI; check against a library copy)
136. M. Stolte, B. Gollan, and U. Ansorge, "Tracking visual search demands and memory load
     through pupil dilation," *J. Vis.*, vol. 20, no. 6, 21, 2020, doi:10.1167/jov.20.6.21.
     `[X9]` V (pupil-only LR, Pupil Labs; P2 precedent, V2-1; added for round-3 review V3-5)
137. I. T. C. Hooge, R. S. Hessels, and M. Nyström, "Do pupil-based binocular video eye trackers
     reliably measure vergence?," *Vision Res.*, vol. 156, pp. 1–9, 2019,
     doi:10.1016/j.visres.2019.01.004. `[PP17]` A (vergence depth unreliable at 77 cm; R4-1)
138. A. Velisar and N. M. Shanidze, "Noise estimation for head-mounted 3D binocular eye tracking
     using Pupil Core eye-tracking goggles," *Behav. Res. Methods*, vol. 56, no. 1, pp. 53–79,
     2024, doi:10.3758/s13428-023-02150-0. `[PP18]` A (Pupil Core gaze-depth errors; R4-1)
139. R. Kothari *et al.*, "Gaze-in-wild: A dataset for studying eye and head coordination in
     everyday activities," *Sci. Rep.*, vol. 10, 2539, 2020, doi:10.1038/s41598-020-59251-5.
     `[PP19]` V (velocity from angles between unit gaze vectors, Pupil Labs; R4-1)

**ID alias:** `X5` in the support matrix refers to the Kaczorowska studies, entries 36–38
(EM1/EM2/EM18).

---

## Not used in the COLET methodology

These remain in `rrl-master-list.md`.

- **GAZELOAD-specific sources:** DS0; HRC studies HR3, HR5, HR6; GAZELOAD feature notes. (HR2 and HR4 are now used: entries 132–133.)
- **Other wearable-device sources:** DS2, DS5.
- **Weak or contrast-only studies:** EM10, EM11 (Karunathilake; dropped as V2-1 support, V3-5), EM20–EM23, X8, X17, X20, X23, X24.
- **Never cite:** the "Do not cite" list in `rrl-master-list.md` §11 (including Parasuraman &
  Manzey 2010, the wrong Marinescu title, and Kaczorowska *Sensors* 2021 for aggregation).

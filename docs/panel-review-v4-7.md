# Panel review of draft v4-7 (Chapters 1-5)

Reviewer role: strict panelist, ML / eye-tracking. Read: v4-7.md, reference list in v4-6.md, all figures, `outputs/final/results`, `logs`, `features`, and `src/` (topn.py, run_model.py, evaluate.py, models.py, config.py). Extra numbers marked "reviewer check" were computed from the saved files only (features.csv, log_recordings.csv, predictions.csv); nothing was re-run.

## Verdict

The paper is honest, well controlled and much better than the proposal version. The pipeline has no leakage that I could find: ranking, scaling, imputation and tuning are all inside the folds, and the permutation test re-runs feature selection. Almost every number matches the result files. The weak points are not the ML but what the features measure. Blink rate, the top feature, follows speaking and also tracks data loss. The second feature (lowest vertical gaze angle) comes from a few extreme samples and from recording length. The paper should say this more plainly and fix one wrong recommendation. Recommendation: pass with minor revisions.

## Number check

I checked every number in Ch. 4-5 and the data numbers in Ch. 1-3 (Tables 2, 3, 7-13; text values for margins, N chosen, C, depth, trees, SHAP shares, selection counts, correlations, Friedman, blink rates by activity, durations, exclusions). **All match the files except the items below.**

| Where | Paper says | File says | Severity |
|---|---|---|---|
| Ch. 3 Feature Extraction | Blink-duration statistics "are blank whenever blinks are few, which happens almost only in low load" | `log_features.csv`: blank in 43 of 44 A1 **and 36 of 44 A4** recordings (79 of 88). Only 9 recordings have ≥ 20 blinks (1 A1, 8 A4). The reason in `log_decisions.md` ("13 A1 vs 0 A4") is from before the 20-value rule. | Minor (wrong reason; the drop decision itself is still correct) |
| Ch. 4 Number of Features | LR "gave identical results from N = 40 upward" | `results_by_n.csv`: AUC 0.8755 (N = 40) vs 0.8760 (N = 140); Brier 0.227 vs 0.226. Your own Table 12 shows the different Brier. | Minor |
| Ch. 4 Feature Importance | LR used only five features because "the penalty set all others to zero" | `chosen_params_logistic_regression.csv`: N = 5 in 42 folds (the selector removed the others), N = 140 in 2 folds (there the penalty zeroed them). `shap_logistic_regression.csv`: 5 non-zero features, correct. | Minor (wrong mechanism) |
| Ch. 4 Feature Importance | "the top five gave 73%" (XGBoost) | 72.5% | Rounding only, OK |
| Ch. 3 Design Choices | Blinks inside fixations "may raise the fixation rate" in A4 | `features.csv` (reviewer check): fixation rate is **lower** in A4 (median 3.39 → 3.14 per s; higher in only 10 of 44) | Minor (cut the clause) |
| Figure 8 | "Up/down gaze angle", "Saccade size", "Fixation length", "Pupil size" | The text uses vertical gaze angle, saccade amplitude, fixation duration, pupil diameter | Term mismatch |

Values confirmed (short list): Table 11 (all cells), comparison 0.000 (−0.054 to 0.048), Wilcoxon p = 5.5e-6, margins 0.20 vs 0.40 and 36 of 44, 73 vs 65 correct, Table 12 (all 72 cells), Table 13 (all cells), N chosen (LR 5 in 42; XGB 5 in 17, 80 in 16), C = 0.07, L1 ratio 0.82, depth 1 in 40 folds, 141 trees, SHAP 87% / 52%, 121 features, pupil range 2.4%, 5 features in 44 folds, 160 pairs with r > 0.9, median single-feature AUC 0.61, Table 10 (all), blink rate A2 0.0 / A3 11.7, Tables 7-9, Friedman χ² = 82.72, 43 of 43, median difference 31.7, Table 3 durations and means, exclusions (57.4 / 38.8 / 48.6%), demographics (26 F / 21 M, 32 ± 8). The `{guyon}` placeholder exists only in the .md file. The .docx resolves it, and all 75 references in the .docx are cited.

## Major issues

**M1. Speech confound (major; addressed well, with one error).** A4 = counting aloud + time pressure. Blink rate is 0.0/min in silent A2 and 11.7 in spoken A3, the same as A4 (11.7). The paper says this in the scope, the definitions, the discussion, the limitations and the conclusion. Good. Two things remain:
- Recommendation 2 is wrong. A1 vs A3 and A2 vs A4 do **not** "differ only in speaking". They also differ in the secondary counting task, which is the workload manipulation. In COLET, speaking and the secondary task always come together. The only contrast with no difference in speaking is **A1 vs A2** (time pressure only). A panelist will catch this.
- The title and general objective still say "cognitive load detection". This is acceptable because the Definitions section redefines the term, but answer this risk in the defense (Q1).

**M2. The gaze-angle features measure extremes and recording length, not "where people looked" (major; only partly addressed).** Three of the five features chosen in every fold are gaze-angle statistics: vertical minimum, horizontal range and horizontal kurtosis. Reviewer checks on `features.csv` / `log_recordings.csv`:
- The vertical minimum separates the classes (single-feature AUC 0.79), but the vertical 5th percentile barely does (0.61; medians −13.9° vs −15.6°). Horizontal range: 0.75, but IQR 0.60. So the signal comes from **a few extreme samples**, not from a shift in where people looked.
- Extremes grow with recording length, and A4 is longer (median 51 s vs 37 s). Spearman with duration: x-range 0.57, y-min −0.43 (−0.46 within A4 alone).
- The median vertical range in A4 is 24.3°. A 24-inch 16:9 screen at 80 cm spans only about 21° vertically, so some A4 samples are off the screen or are measurement error. In A1 the range is 11.3°.
- Your own decision log (D3) says "min, max and range vs recording length: kept; **reported as a limitation**", but the paper never mentions recording length.

Fix (no new analysis): in Ch. 3 replace the caveat with "Minimum, maximum and range also grow with recording length, and A4 recordings were longer (Table 3)". In Ch. 4 add the half-sentence "the 5th percentile differed by only about 2°". Update the limitation bullet the same way.

**M3. Blink rate may partly measure tracking loss (major; only partly addressed).** Blinks come from the tracker's confidence-based detector. Reviewer checks: the median recording has 1 blink in A1 and 0 in A2, and 30% of A1 and 55% of A2 recordings have **no blinks at all** in 30-40 s. That is far below normal blinking, so the detector probably misses many blinks. Within each activity, blink rate correlates with the invalid-sample fraction (Spearman 0.65-0.73). The authors correctly refused to use valid-sample % as a feature because "it would reveal the label", but blink rate is closely tied to it. The paper only says "blinks were taken from the tracker's own detector". Fix: one sentence in Limitations, e.g. "Blink counts were low (30% of A1 recordings had none), so the detector may miss small blinks, and blink rate rose with data loss; part of its effect may reflect tracking loss while talking."

**M4. The model comparison: say what was found, no more (moderate; mostly addressed).**
- The ROC-AUC tie is solid. The CI (−0.054 to 0.048) is informative: it makes a difference larger than about 0.05 unlikely. Say this, instead of only "no equivalence margin was set".
- "LR had higher accuracy" is threshold-dependent and is not tested in the paper. LR's strong penalty pulls its probabilities toward 0.5 (mean 0.37 / 0.61 vs XGBoost 0.27 / 0.71), and its recall is lopsided (0.93 low / 0.73 high). Keep the claim descriptive ("at the 0.5 threshold"), as Ch. 4 already does. The Conclusions sentence "reached the same ROC-AUC with five features and higher accuracy" should add "at the 0.5 threshold".
- The Wilcoxon test on margins: the paper itself says the margins depend on each model's probability scale, so the test does not answer the research question. Recommend **cutting it** (Ch. 3, Ch. 4, Figure 2). This shortens the paper and removes an easy attack.
- Equal budget: 30 candidates cover a 3-setting space for LR but an 8-setting space for XGBoost. This is a weak point but not fatal: XGBoost chose depth-1 stumps anyway, and the fixed-N results agree. Be ready to answer (Q6); no change needed.

## Minor issues

1. **Permutation test uses default settings (N = 20)**, not the tuned pipeline in Table 11. This is stated in Ch. 3. Add "(default settings)" to the Table 11 row so the table is not misread.
2. **Majority baseline vs AUC.** "ROC-AUC of 0.862, well above the majority-class baseline (balanced accuracy 0.500)" compares two different metrics. Compare accuracy with the baseline, or say "above chance (0.5)".
3. **Balanced accuracy equals accuracy by design** (44/44). Cut the row from Tables 11-13 and from Objective 3. One sentence in the definitions already explains it.
4. **No missing values** remain, so "median imputation" (LR) and "handles missing values natively {77}" (XGBoost) do nothing. Cut both, or add "(no values were missing)".
5. **Two NASA-RTLX tables with different n** (Table 3: 47 people; Table 7: 43). Cut the RTLX column from Table 3.
6. **Pooled AUC vs within-person ranking.** 42 of 44 people have A4 ranked above A1 (a within-person AUC of 0.95), but the pooled AUC is 0.86. The gap comes from differences between people, and a fixed threshold suffers from them. The paper reports the count but never interprets it. One clause in the Discussion would help.
7. **Deliverable "tuned cognitive load classification models" (Fig. 1).** LOPO produces 44 fold models, and no final model trained on all 44 people is saved (`run_model.py`, `demo.py` save nothing). Either rename the deliverable to "the tuned modeling pipeline (settings per fold, Tables 6 and 11)" or be ready to explain.
8. **Figure 2** shows "rank features" before "inner CV: tune N", but the ranking is re-done inside every inner fold (it is a pipeline step). Put the ranking box inside the tuning box, or label it "(repeated in each inner fold)". For LR the real order is impute → select → scale.
9. **Event sanity check:** with I-VT, saccades and fixations alternate, so a saccade/fixation ratio near 1 is almost automatic. Not worth adding text; just do not oversell it.
10. **Citations to verify** (I cannot confirm them from the files): Nerella et al. {12} is called "participant-independent" in the Discussion, but Ch. 2 does not say so. {38} (saccadic peak velocity) supports "gaze speed" in Table 5, and {37} (saccade extent) supports "saccade duration". {66} supports a 35% exclusion rule that Design Choices calls the authors' own choice. {76} (collinearity review) supports "range copies max" only generally.

## Previous panel requests

| Request | Done? | Convincing? |
|---|---|---|
| Flatten every signal into many statistics | Yes: 10 signals × 17 statistics + 3 rates = 173. 33 removed for clear, logged data reasons, leaving 140. | Yes. The removal rules are stated and logged. Fix the blink-duration reason (Number check). |
| Try different top-N to find a sweet spot | Yes: N tuned inside the folds from {5, 10, 20, 40, 80, 140}, plus a fixed-N table and Figure 7 | Yes, but **say the answer in plain words**: there is no sweet spot. The curve is flat (0.84-0.88), 5 features are enough, and blink rate alone (0.884) matches. N below 5 was not on the grid except blink-alone. |
| Report accuracy, precision, recall, F1 | Yes: per class and macro, plus Brier, for every model and every N | Yes |
| High score = leakage or one feature? | Leakage: in-fold ranking, scaling and tuning, nested CV, a permutation test that re-runs selection, and label-revealing features removed. One feature: blink rate alone gives 0.884. | Mostly. The evidence is all there, but the paper never links it back to the panel's warning. Add one sentence in the Discussion: "The high score is not from leakage (all steps inside the folds; permutation p = 0.005 with selection re-run). It is because one feature, blink rate, already gives most of the answer." Be ready for M3, where blink rate may itself be a data-loss proxy. |

## Writing fixes

| Quote | Fix |
|---|---|
| "Variances (9) are exact copies of standard deviations" | "Variances (9) are the squares of the standard deviations and carry the same information" |
| "Blink-duration statistics (17) are blank whenever blinks are few, which happens almost only in low load, so their blanks would reveal the label {15}." | "Blink-duration statistics (17) were blank in 79 of 88 recordings (fewer than 20 blinks); 8 of the 9 recordings with values were high load, so whether a value exists would reveal the label {15}." |
| "Logistic Regression gave identical results from N = 40 upward, because its penalty set the extra features to zero." | "Logistic Regression gave almost the same results from N = 40 upward (ROC-AUC 0.876), because its penalty set most extra features to zero." |
| "Logistic Regression used only these five features; the penalty set all others to zero." | "Logistic Regression used only these five features: it kept five in 42 folds, and in the other two its penalty set the rest to zero." |
| "people looked about 9° lower at some point in A4, which may come from the stimuli or the headset rather than load" | "the lowest vertical gaze angle was about 9° lower in A4, but the 5th percentile differed by only about 2°, so this reflects a few samples, which may come from the stimuli, the headset or the longer A4 recordings rather than load" |
| "Three of the five leading features describe gaze location and carry the caveat stated in Chapter 3." | "Three of the five leading features are gaze-angle statistics (minimum, range, kurtosis) and carry the caveat stated in Chapter 3." |
| "Both reached a pooled ROC-AUC of 0.862, well above the majority-class baseline (balanced accuracy 0.500)." | "Both reached a pooled ROC-AUC of 0.862; both accuracies (0.830 and 0.739) were well above the majority-class baseline (0.500)." |
| "Blink rate alone did as well as the full models (ROC-AUC 0.884; Table 13)." | "Blink rate alone matched the full models in ROC-AUC (0.884; Table 13)." |
| "No equivalence margin was set, so the results do not show that the two models are equal." | Add: "; the interval does, however, make a difference larger than about 0.05 unlikely." |
| "XGBoost did not outperform the simpler Logistic Regression, which reached the same ROC-AUC with five features and higher accuracy." | "... and higher accuracy at the 0.5 threshold." |
| "Saccade rate fell under high load, whereas the COLET authors reported an increase {18}; saccade amplitude has been reported to decrease under load {37}." | The second clause is about amplitude, not rate, so it does not explain the first. Cut it: "Saccade rate fell under high load, whereas the COLET authors reported an increase {18}; this may come from ..." |
| "...blinks were far more frequent in A4, which may raise the fixation rate there." | Cut this clause (fixation rate was lower in A4). |
| Rec. 2: "Classify COLET activities that differ only in speaking (A1 versus A3, A2 versus A4) to measure the speech effect directly." | "Classify A1 versus A2, the only COLET contrast without a difference in speaking, to see how much signal remains without speech." |
| "For practitioners, Logistic Regression on a few well-established features such as blink rate is a reasonable choice..." | Cut, or: "For practitioners, blink rate should not be used as a load index in tasks that involve speaking unless the speech effect is controlled." |
| "The recordings are cleaned following published guidelines for Pupil Core data" (Ch. 1) | "...following published guidelines where they exist; the remaining settings are listed as design choices." |
| Figure 8 labels | Use the text terms: vertical/horizontal gaze angle, saccade amplitude, fixation duration, pupil diameter. |
| Table 11 row "Permutation test p" | "Permutation test p (default settings, N = 20)" |

## Likely panel questions with answers

1. **"Did you detect cognitive load or just talking?"** We cannot separate them. A4 adds counting aloud, and the top feature, blink rate, is 0.0/min in silent A2 and 11.7 in spoken A3. So we claim to detect COLET's dual-task condition, which includes speech, not load separated from speech. Without blink rate the models still reached 0.73-0.79, but those features may also be affected by speaking. In COLET only A1 vs A2 has no difference in speaking.

2. **"0.86 is high. Leakage, or one feature?"** One feature, not leakage. Every learning step (ranking, scaling, tuning) is fitted on the 43 training people only. The permutation test re-runs feature selection and gives p = 0.005, the smallest possible. We removed features whose blanks or data loss reveal the label. Blink rate alone gives 0.884, so one feature already gives the answer.

3. **"What is the sweet spot for the number of features?"** There is none. ROC-AUC stays at 0.84-0.88 from 5 to 140 features (Table 12), and LR chose 5 in 42 of 44 folds. Many statistics of one signal are near-copies (160 pairs with r > 0.9), and a typical feature alone reaches only about 0.6, so more features add noise, not information.

4. **"Same AUC, so which model is better?"** They rank recordings equally well (difference 0.000, CI −0.054 to 0.048, so a gap larger than about 0.05 is unlikely). At the 0.5 threshold LR got 73 of 88 correct and XGBoost 65, but that depends on the threshold. XGBoost gave no benefit, so the simpler 5-feature LR is preferred. (If pressed: on the saved predictions, LR was right and XGBoost wrong in 9 recordings, the reverse in 1. An exact McNemar test gives p ≈ 0.02. This is a reviewer check, not in the paper.)

5. **"Why is the lowest vertical gaze angle so important? Is that load?"** We do not claim it is. It is an extreme value from a few samples (the 5th percentile barely differs), and extremes grow with recording length; A4 was longer. It may also reflect stimulus positions, the headset, or looking away from the screen while counting. It is reported with this caveat.

6. **"Was XGBoost given a fair chance?"** Same folds, same features, same 30 candidates and nested tuning. XGBoost's space is larger, but it chose single-split trees in 40 of 44 folds, and its fixed-N results (0.84-0.88) agree. With 86 training samples and one dominant feature there is little non-linearity to learn.

7. **"Your blink rates are very low. Is the blink detector valid?"** We used the tracker's blink events with a 50-500 ms filter. Small blinks may be missed in silent tasks, and blink rate rises with data loss. This is a stated limitation. Talking makes blinking and tracking loss rise together, which is part of the speech confound.

8. **"42 of 44 people have A4 above A1, yet accuracy is 0.83. Why?"** Within a person the model ranks the two activities correctly 95% of the time. Across people, baselines differ: one person's A4 can score below another person's A1. Pooled AUC and a fixed 0.5 threshold suffer from that.

9. **"Why test chance with default settings instead of the tuned model?"** Re-running nested tuning 200 times was too costly. The default pipeline still includes feature selection, so the test covers the selection step. Its AUCs (0.838 and 0.830) are only slightly below the tuned ones.

10. **"Is 44 participants enough?"** For detecting load vs no load, yes: the CI lower bounds are about 0.79. For telling the two models apart, only differences larger than about 0.05 in AUC could be detected. Results come from one lab task with a chin rest and should be tested on another dataset.

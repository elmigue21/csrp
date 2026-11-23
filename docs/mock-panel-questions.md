# Mock Panel: Shallow to Deep Questions

For v4-5. Each question has a short answer to give out loud. Levels go from basic understanding
(Level 1) to the hardest validity challenges (Level 5). Numbers are from the paper and the
official tables. Say the limits before the panel does.

---

## Level 1: Basics (anyone on the team must answer)

**1. In one sentence, what did you do?**
We compared Logistic Regression and XGBoost for telling COLET's low-load activity (A1) from its
high-load activity (A4) using ten eye-tracking features, tested on participants whose labels were
held out, and checked which features each model relied on.

**2. What is COLET?**
A public eye-tracking dataset (Ktistakis et al., 2022): 47 participants, Pupil Core at 240 Hz,
visual-search puzzles in four activities (2×2: counting aloud × time pressure), with NASA-RTLX
ratings per activity.

**3. Why only 44 participants and 88 samples?**
Three recordings had more than 35% invalid samples (P06 A3, P06 A4, P17 A4), so P06 and P17 lost
their A1–A4 pair. P31's recordings are byte-identical to P28's, so they were kept once. 44 people ×
A1 and A4 = 88.

**4. What are the ten features?**
Pupil mean and SD, blink rate, fixation rate and duration, saccade rate, amplitude and peak
velocity, and horizontal and vertical gaze spread.

**5. Why Logistic Regression and XGBoost?**
LR is the simple, interpretable baseline used in the COLET paper and clinical prediction work;
XGBoost is a strong model for tabular data. The question is whether the extra flexibility helps.

**6. What does ROC-AUC 0.99 mean?**
Given one low-load and one high-load recording, the model ranks the high-load one higher 99% of
the time. 0.5 is guessing.

---

## Level 2: Method choices

**7. Why A1 vs A4 and not all four activities?**
They are COLET's lowest and highest load conditions, every participant in the test rated A4
higher on NASA-RTLX (43 of 43), and it is the contrast the COLET authors classified. A2 and A3
are used for standardization and the manipulation check.

**8. Why leave-one-participant-out?**
So every test is on a person whose labels the model never saw. Random splits put the same person
in training and test and inflate results.

**9. What is the inner loop for? Isn't LOPO enough?**
LOPO measures performance; the inner loop (5-fold, grouped by participant, 30 random settings)
chooses settings using only the 43 training people. Tuning on the test person would be like
taking the exam 30 times and reporting the best score.

**10. Why per-person standardization?**
Differences between people (for example, natural pupil size) are larger than the load effect.
Comparing each person with their own average removes that. It follows the closest prior studies.

**11. Why whole activities instead of short windows?**
It follows the COLET paper, which said trials were too short to analyze separately. We checked
10 s windows post hoc and the pattern held (Appendix B).

**12. What is P2 and why?**
The same analysis with only the two pupil features. A4 involves counting aloud, which raises
blinking, so P2 checks that the result does not rest on blinks alone.

**13. What is the ceiling rule?**
Stated before the results: if both models exceed 0.95 in P1, the model comparison is read mainly
from P2, because near-perfect models tend to tie.

---

## Level 3: Results and interpretation

**14. Which model won?**
Neither clearly. P1: LR 0.992 vs XGBoost 0.980; P2: 0.831 vs 0.812. Both differences have
intervals that include zero. There is no evidence XGBoost is better.

**15. Then why does the Wilcoxon test give p = 0.007 for XGBoost in P2?**
It compares per-person probability margins, which measure how confident each model is, not how
well it separates. LR was heavily regularized in P2, which pulls its probabilities toward 0.5.
It is secondary to ROC-AUC.

**16. Which features mattered most?**
Blink rate, mean pupil diameter and fixation rate by SHAP for both models (Kendall's τ = 0.69).
Permutation agreed on blink rate; XGBoost relied much less on pupil size than LR. This is
descriptive: it shows what the models used, not what causes load.

**17. Did tuning help?**
No. Untuned models scored about the same (P1 0.991 / 0.977; P2 0.852 / 0.819).

**18. How did you check the labels mean low and high load?**
NASA-RTLX: Friedman χ²(3) = 82.72, p < 0.001, n = 43, and all 43 rated A4 above A1 (median +31.7
points).

**19. How does your result compare with COLET's 85%?**
Not directly comparable. COLET's split was not participant-independent and scaled and selected
features on all data. Ours is participant-independent but uses per-person standardization.
Without it, we get 0.85–0.88.

---

## Level 4: Validity challenges

**20. 0.99 is too good. Is there leakage?**
Labels never leak: LOPO and tuning only on training people. Shuffled labels drop to about 0.5.
The one known source of optimism is standardization, which uses the test person's own unlabeled
recordings. We disclose it and report results without it.

**21. Isn't 0.99 just your standardization?**
Partly. Using only a separate activity (A2) as the baseline still gives 0.90–0.92; without any
standardization, 0.85–0.88. So it helps a lot, but the signal is there without it.

**22. Who is the 0.99 for? Would it work on a new user?**
A calibrated user who has done all four activities. With one calibration activity, about 0.91.
For a completely new user, about 0.86. We report all three and do not claim the headline for
new users.

**23. Are you detecting cognitive load or just talking?**
Mostly the talking condition. A4 adds counting aloud, and blinks follow talking (A1 1.7, A2 0.0,
A3 11.7, A4 11.7 per minute). With talking held equal (A3 vs A4), detection drops to 0.60–0.62.
We describe the result as detecting COLET's task conditions, which include speech.

**24. But P2 has no blinks. Doesn't that prove it's load?**
It proves it is not only blinks (0.81–0.83). Speaking also enlarges the pupil, and P2 drops to
0.61–0.66 for A3 vs A4. So P2 rules out blinks as the only cue, not speech.

**25. Is there any speech-free signal?**
Yes, but weaker. Between the two silent activities (A1 vs A2), about 0.75 with standardization.

**26. Your title says "cognitive load detection". Isn't that overclaiming?**
We use cognitive load as COLET defines it, as task conditions, and say in the paper that our
models detect those conditions, which include speaking. We do not claim speech-free detection.

**27. Why not use A1 vs A2 as the main analysis, since it has no speech?**
Time pressure is weak: it adds only about 8 RTLX points (counting adds about 27), and about a
quarter of participants did not rate A2 higher. A2 is also shorter and mostly zero blinks.
Switching after seeing results would be choosing the analysis by outcome.

---

## Level 5: Deep and adversarial

**28. Your post-hoc analyses were chosen after seeing the results. Why trust them?**
We do not treat them as confirmatory. They are labeled post hoc and exploratory, were not
corrected for multiple comparisons, and do not change the main analysis. Their role is to show
the limits of the main result, and most of them weaken our claims rather than strengthen them.

**29. A3 vs A4 only varies time pressure. How can you say speech drives the result?**
We can't prove it with these comparisons; we say it "suggests". The direct test is A1 vs A3 and
A2 vs A4 (speech only), which is Recommendation 2. Time pressure itself is a weak manipulation.

**30. You say LR beat XGBoost in three settings. Isn't that cherry-picking?**
Those are three of about twenty uncorrected comparisons, and we report them as exploratory. A
few intervals excluding zero is roughly what chance alone gives. The firm claim is only that
there is no evidence XGBoost is better.

**31. Can you claim the two models are equivalent?**
No. No equivalence margin was set in advance. Post hoc, P1 is within ±0.05 AUC but not ±0.02,
and that is partly forced by the ceiling.

**32. Your confidence intervals treat 44 LOPO models as one. Is that valid?**
It is the standard pooled approach, with a paired participant bootstrap. It ignores that the
training sets overlap, so the intervals are likely too narrow. This is a known limitation of
LOPO.

**33. Permutation importance shuffles across people with paired data. Is that valid?**
It describes how much each model relies on a feature under that shuffle, not a feature's unique
effect. We report it as descriptive and note the limit.

**34. Your shuffled-label test used untuned models. What does it actually show?**
Only that the procedure does not find patterns in random labels. It cannot detect duplicated
recordings or the standardization issue. It is a limited negative control.

**35. Blink rate is 0.0 per minute in A2. Is your blink detector broken?**
Visual search suppresses blinking, so low rates are expected, but missed blinks by the tracker's
detector are possible. We did not validate the detector and list that as a limitation.

**36. Recordings differ in length (A4 longer than A1). Does that bias features?**
Rates are per unit time, but standard-deviation features can depend on length. The 10 s windows,
which equalize length, gave the same pattern.

**37. Could fatigue or practice explain A1 vs A4?**
Order varied: A4 came after A1 for only 18 of 44 (inferred from timestamps). Fixation rate
drifts slightly with position, but the predictions were unrelated to position.

**38. The data quality exclusion removed only counting recordings. Is that a selection effect?**
Yes, possibly. All three excluded recordings were from counting activities, so the retained
people may be easier to track while talking. It is stated in the limitations.

**39. What would you do differently with more time?**
Run the speech-only contrasts (A1 vs A3, A2 vs A4); evaluate with a neutral calibration
recording; use a dataset with a silent high-load task; handle pupil-model refits explicitly.

**40. What is the single most important limitation?**
That the high-load condition includes speaking, so we cannot separate mental load from speech in
this dataset. Everything else is secondary to that.

---

## Rules for answering

- Say the limit first, then the evidence.
- Use "suggests", not "proves", for anything post hoc.
- Never say XGBoost or LR is "better"; say "no evidence XGBoost is better".
- Keep the three calibration numbers ready: 0.99 (all four), about 0.91 (A2 only), about 0.86 (none).
- If you don't know, say so and offer what the paper does report.

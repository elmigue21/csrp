# Thesis Review Findings, Explained Like You Question Everything

**Project:** Comparative Analysis of Logistic Regression and XGBoost for Cognitive Load Detection Using Eye-Tracking Features With Feature Importance Analysis  
**Main manuscript reviewed:** `CSRP-Chapter-1-to-5-draft-COLET-v3.docx`  
**Review date:** October 2026

---

## 1. What did the reviewers decide overall?

The thesis is **defensible**, but several sentences are more certain than the evidence allows.

That means the main experiment does not need to be thrown away. The reviewers did **not** find proof that the reported scores are wrong, that the researchers cheated, or that the models secretly saw the answers. They found places where the thesis says something stronger than it can prove.

The safest description of the study is:

> The study compares Logistic Regression and XGBoost in identifying two different COLET task conditions, A1 and A4, using eye-tracking features. The models are tested on participants whose labeled data were not used for training, but the method uses each test participant's own unlabeled recordings for normalization.

The study should **not** be described as proving that it detects pure cognitive load without any influence from speaking. A4 includes both greater task demand and speaking aloud, so the model can learn signals caused by either or both.

The judge found five issues:

| Finding | Judge's decision | How serious? |
|---|---|---:|
| Blink gaps cannot be ruled out as a cause of the fixation-rate result | Sustained | Medium |
| The confidence intervals and Wilcoxon test need stronger qualifications | Partly sustained | Medium |
| The shuffled-label test does not prove that the whole pipeline has no leakage | Partly sustained | Medium |
| The manuscript says no non-normalized run happened, but adviser notes say one did | Sustained | Medium |
| The description of TreeSHAP is technically too broad | Sustained | Low |

“Sustained” means the prosecutor showed that the criticism is valid. “Partly sustained” means part of the criticism is valid, but the strongest possible accusation is not supported.

---

## 2. Tiny dictionary before we begin

### What is cognitive load?

Cognitive load is how much mental work a task asks from a person. Remembering numbers while searching for something is usually harder than searching alone.

### What are A1 and A4?

They are two activities in the COLET dataset:

- **A1:** visual search without time pressure and without the spoken counting task.
- **A4:** visual search with time pressure while counting aloud.

The thesis calls A1 “low load” and A4 “high load.” The important catch is that A4 changes more than one thing: difficulty, time pressure, multitasking, and speaking.

### What is a feature?

A feature is a number given to a model. Examples are average pupil size, blink rate, fixation rate, and gaze spread.

Think of the model as a detective. The features are clues. The model does not see the raw human experience; it sees the clue numbers.

### What are Logistic Regression and XGBoost?

- **Logistic Regression**, or LR, combines the feature values using a weighted formula. It is relatively simple and easier to inspect.
- **XGBoost** combines many small decision trees. It can learn more complicated patterns and interactions.

The thesis asks whether XGBoost's extra flexibility helps more than the simpler LR model.

### What is ROC-AUC?

ROC-AUC measures ranking ability. Imagine randomly selecting one A1 recording and one A4 recording. The AUC is the chance that the model gives the A4 recording a higher score.

- `0.50` is like random guessing.
- `1.00` is perfect ranking.
- `0.99` is nearly perfect ranking.

A high AUC does not automatically reveal **why** the model succeeded. It might recognize cognitive load, speaking, the experiment's condition structure, or a mixture of these.

### What is leave-one-participant-out validation?

Suppose there are 45 participants. The model trains on 44 people and predicts the remaining person. This repeats until every person has been the held-out person once.

This is called **LOPO**, or leave-one-participant-out validation. It prevents the held-out person's labeled examples from being used to train that fold's model.

### What is normalization?

People naturally have different pupil sizes, blink rates, and eye-movement habits. Normalization changes a person's values into “lower or higher than this person's usual value.”

Example:

- Ana's usual pupil size is 3 mm.
- Ben's usual pupil size is 5 mm.
- Both increase by 0.5 mm during A4.

Raw pupil sizes make Ana and Ben look very different. Per-person normalization highlights that both increased relative to their own usual levels.

The study calculates “usual” using the same participant's unlabeled recordings, including recordings belonging to the held-out participant. That is not labeled training leakage, but it means the system already knows the new person's baseline. It is closer to a calibrated system than a system meeting a stranger with no prior data.

### What is leakage?

Leakage happens when information that should be unavailable during prediction sneaks into model development or evaluation.

An obvious example is putting the correct answer in an input column. A less obvious example is choosing features after looking at which ones best separate the final test labels.

Leakage can make a weak model appear much better than it really is.

### What is a confidence interval?

A confidence interval is a range used to show uncertainty around an estimate. It is not a magical fence guaranteeing that the true answer is inside.

Its meaning depends on how the interval was calculated and what sources of uncertainty that calculation includes.

### What is a bootstrap?

A bootstrap repeatedly makes new pretend samples by drawing from the observed participants, with replacement. “With replacement” means a participant can appear more than once in one pretend sample while another participant is absent.

The spread of the pretend results is used to estimate uncertainty.

### What is a permutation or shuffled-label test?

The correct A1/A4 labels are mixed up, and the model is tested again. If performance falls close to `0.50`, the model is not succeeding with those shuffled answers.

This is a useful check. It is not an all-purpose certificate that every earlier research decision was safe.

### What is SHAP?

SHAP explains how features move a model's prediction away from a baseline prediction. It answers a question like, “For this model and this recording, how much did blink rate push the score?”

SHAP describes what a model relied on. It does not prove that a feature caused cognitive load.

---

## 3. Finding 1: Blink gaps can change fixation counts in either direction

### What did the thesis say?

The thesis explains that a blink can split one fixation into two pieces. It then says this splitting should increase the fixation count, so the lower fixation rate observed in A4 cannot be caused by blink splitting.

### What is a fixation?

A fixation is a period when the eyes remain fairly steady on one area.

The pipeline calls a slow-eye-movement segment a fixation only when it lasts at least **55 milliseconds**.

### Why does blinking matter?

The camera cannot track gaze normally while the eye is closed. That creates a hole in the gaze record.

Imagine one fixation like this:

```text
steady gaze for 150 ms
```

Without a blink, the detector may count one fixation.

Now put an 80 ms blink in the middle:

```text
35 ms visible | 80 ms blink | 35 ms visible
```

Each visible piece is shorter than 55 ms. The program throws both pieces away. The count changes from one fixation to zero.

In another case, both pieces could be longer than 55 ms. Then one fixation could become two. Therefore, a blink can increase, decrease, or leave the accepted count unchanged.

### Did the reviewers test this idea?

Yes. A small synthetic example was passed through the current event detector:

- An uninterrupted 250 ms steady gaze produced **one fixation**.
- Adding a long missing section that left two short pieces produced **zero fixations**.

This is a counterexample. A counterexample is one valid example that proves an “always” or “cannot” claim is false.

### Does this prove that the real A4 fixation result is wrong?

No.

It proves only that the thesis cannot rule out blink-related processing as part of the explanation. The review did not measure how large this effect was in the real recordings.

### Why is this a problem?

The thesis uses a logical argument to exclude an artifact. The actual program does not behave as that argument assumes.

An **artifact** is a pattern created or changed by measurement or processing rather than by the human behavior the researchers want to study.

### What is the solution?

Replace the categorical sentence with:

> Blink masking and tracking gaps can alter fixation counts and durations; their contribution to the observed condition difference was not quantified.

This fix is honest and does not require changing the models or rerunning the main analysis.

### What should we say if a panel asks?

> Yes, blinks may affect fixation counts. A blink splits the gaze record, and fragments shorter than 55 milliseconds are discarded. We report this as an unquantified limitation and do not claim that the fixation-rate difference is purely physiological.

---

## 4. Finding 2: The uncertainty ranges do not include every kind of uncertainty

### What did the thesis do?

The thesis saved one probability for every held-out recording. It then repeatedly resampled participants from those already-created predictions to calculate confidence intervals.

This is called a **participant bootstrap over fixed out-of-fold predictions**.

### What does “out-of-fold” mean?

Each prediction came from a model that did not train on that participant's labeled examples. The prediction is outside that participant's training fold.

### What does “fixed predictions” mean?

During the bootstrap, the program rearranges the saved predictions. It does not redo feature preparation, hyperparameter tuning, or model training for every bootstrap sample.

Think of baking one cake and repeatedly asking different groups of people to rate slices from that same cake. This measures how ratings vary across sampled people. It does not measure how much the cake would change if the baker repeated the entire recipe many times.

### So what uncertainty is included?

It includes variation from which observed participants are selected in a bootstrap sample and keeps each participant's A1/A4 pair together.

### What uncertainty is missing?

It does not fully include variation caused by:

- training a new model on a different dataset;
- selecting different hyperparameters;
- having a different small sample of participants;
- repeatedly executing the whole model-development procedure.

### What are overlapping training sets?

In LOPO, one fold trains on participants 2–45, another trains on participants 1 and 3–45, and so on. Those training sets share almost everybody.

The fold results are therefore connected. They are not 45 completely independent experiments.

### What is the Wilcoxon test?

The thesis compares the two models' participant-level probability margins using a Wilcoxon signed-rank test.

A **margin** here is:

```text
predicted probability for A4 − predicted probability for A1
```

Larger positive margins mean the model separated that participant's two recordings more strongly on its probability scale.

The Wilcoxon test normally assumes that the paired differences being tested are independent. Because the LOPO models share most of their training participants, that independence is not clearly established. The two models can also produce differently calibrated probability scales, making margin size a secondary comparison rather than the main performance comparison.

### Does this mean the reported numbers are calculated incorrectly?

No.

The reviewers recalculated the saved confidence intervals and Wilcoxon results from the saved predictions. They match the tables. The problem is how strongly the numbers may be interpreted, not an arithmetic mismatch.

### Does this prove that the intervals are too narrow?

No.

Research on cross-validation explains why overlapping training sets complicate uncertainty estimates and can cause undercoverage. It does not prove the exact amount of error in this thesis's intervals.

**Undercoverage** means an interval method contains the target value less often than its advertised percentage over many repeated studies.

### What is the solution?

Add this explanation:

> Bootstrap intervals resample participants from fixed out-of-fold predictions. They are approximate summaries and do not include the full variability of refitting and hyperparameter selection. The Wilcoxon comparison is exploratory because the LOPO models share training participants and probability margins depend on model calibration.

Call the Wilcoxon finding **exploratory** or **supporting**, not final proof that one model is better.

### What should we say if a panel asks?

> The intervals describe participant variation in our saved out-of-fold predictions. They do not reproduce the complete training and tuning process, so we treat them as approximate. Our main conclusion is modest: we observed no clear XGBoost advantage under this protocol.

---

## 5. Finding 3: Shuffled labels are a useful check, not a “no leakage anywhere” certificate

### What did the thesis do?

The researchers shuffled the A1/A4 labels and refitted untuned models. Their AUCs dropped near `0.50`.

That is reassuring. It means those downstream models could not keep producing near-perfect scores when the answers were randomized.

### Then why is there still a problem?

The features had already been prepared before the labels were shuffled in the diagnostic script.

Imagine a teacher creates a worksheet after seeing the real answer key. Later, the teacher swaps the answer labels and asks whether a student can solve the scrambled worksheet. Failure on the scrambled worksheet does not prove that the original worksheet was designed without seeing the original answers.

This analogy does **not** accuse the researchers of cheating. It shows the logical limit of the test.

### What kinds of problems can this test miss?

It cannot by itself prove that:

- every earlier feature decision was label-blind;
- no historical preprocessing choice used knowledge of the real conditions;
- test-participant normalization has no optimistic effect;
- speaking has no effect on the features;
- the complete workflow would work in a future deployment.

### Did the review find actual leakage?

No.

The review found no direct evidence that the pipeline leaked the labels into model fitting. The issue is that the phrase “there is no leak” claims more than this diagnostic can prove.

### Why does wording matter?

Scientific tests answer limited questions. A good thesis says exactly what a test checked.

“We did not find evidence of leakage in this diagnostic” is different from “we proved that no leakage exists anywhere.” The first is supported. The second is too broad.

### What is the solution?

Use this wording:

> The shuffled-label diagnostic returned approximately chance performance for the untuned downstream models. It does not independently audit all upstream processing or establish that the entire workflow is free of leakage.

Also change the researcher notes that say:

- “There is no leak.”
- The shuffled-label check “rules out leakage.”
- P2 proves a load signal “beyond talking.”

P2 removes blink, fixation, saccade, and gaze-spread features, but pupil measurements may still respond to speaking or tracking changes. The safer statement is:

> P2 shows that the two conditions can still be distinguished using pupil features alone; it does not prove that the pupil signal is independent of speaking.

### What should we say if a panel asks?

> The shuffle test is a negative control for the downstream modeling procedure. Its chance-level result is reassuring, but it cannot certify every upstream decision. We found no direct evidence of label leakage, and we state the test's limits.

---

## 6. Finding 4: The records disagree about whether a no-normalization run happened

### What does the manuscript say?

The manuscript says no analysis or run without per-person normalization was performed.

### What do the adviser notes say?

`docs/adviser-questions.md` says an internal, post-hoc comparison was performed using default, untuned models, both with and without normalization.

It records these raw-feature AUCs:

| Analysis without normalization | Logistic Regression | XGBoost |
|---|---:|---:|
| P1, all 10 features | 0.870 | 0.874 |
| P2, pupil only | 0.602 | 0.647 |

The notes say the scratch script was not preserved in the repository.

### What is “post-hoc”?

Post-hoc means the check was decided or performed after seeing the main results. That does not make it useless or dishonest. It means it should be labeled as an exploratory follow-up rather than a pre-planned confirmatory analysis.

### What is “untuned”?

Untuned means the models used default settings rather than searching for the best hyperparameters inside the training data.

### What are hyperparameters?

Hyperparameters are model settings chosen by the researchers or a search procedure, such as tree depth or penalty strength. They control how the model learns.

### Why can't we simply add those numbers to the thesis as final results?

The script that produced them is not preserved, the run was untuned, and it has no matching confidence intervals. The review could not independently reproduce its full execution history.

The numbers are documented internal evidence, but they are not verified to the same level as the saved main results.

### Why is this still important?

The recorded values suggest that normalization may contribute greatly to performance, especially in P2. Raw pupil-only performance was much closer to chance in that internal comparison.

This fits the correct interpretation: P2 detects condition differences **relative to each participant's own baseline**. It does not demonstrate strong pupil-only performance on an entirely uncalibrated stranger.

### Is using normalization automatically wrong?

No.

It is a defensible calibrated protocol when clearly described. The problem is inconsistency about whether a comparison was run and overgeneralizing the result beyond a calibrated setting.

### What is the solution?

Replace “no analysis without normalization was run” with:

> No tuned sensitivity analysis without per-person standardization is reported. An internal post-hoc comparison using default hyperparameters suggested substantial dependence on standardization, particularly for P2, but its execution script was not preserved in the repository.

The team may omit the unverified values from the main results, but it should not say that no such exploratory run happened.

### What should we say if a panel asks?

> Our reported tuned results use each participant's own activity baseline. An internal untuned comparison suggested that this step helps substantially, especially for pupil-only P2, but that scratch run does not have the same preserved provenance as the main results. We therefore describe the dependence honestly without treating the scratch values as headline results.

---

## 7. Finding 5: TreeSHAP does not simply “follow feature dependencies”

### What did the thesis say?

The thesis says tree-path-dependent SHAP “follows the dependencies between features in the training data.”

### What is a feature dependency?

Two features are dependent when knowing one tells you something about the other.

For example, saccade amplitude and saccade velocity may be related. If a method ignores this relationship, it can divide explanatory credit between them in a debatable way.

### What does TreeSHAP actually do here?

The code creates `TreeExplainer` without supplying a separate background dataset. In the installed SHAP behavior used by the project, this selects a tree-path-dependent approach.

That approach uses the number of training examples that traveled down each tree path to represent the background distribution.

It is too broad to say this generally understands or preserves all real relationships between features.

### Is the SHAP analysis invalid?

No.

The manuscript already admits that LR SHAP and XGBoost SHAP use different assumptions for correlated features. The correction is about explaining TreeSHAP precisely, not deleting the analysis.

### Are SHAP values causal effects?

No.

If blink rate receives a large SHAP value, the correct conclusion is:

> The model relied strongly on blink rate for these predictions.

The incorrect conclusion would be:

> Blink rate caused the person's cognitive load.

Prediction importance is not proof of cause.

### What is the solution?

Replace the broad explanation with:

> TreeSHAP is run without background data and uses the tree-path-dependent convention, which represents the background distribution through recorded training counts along tree paths.

Keep the existing warning that the LR and XGBoost explanations use different assumptions.

### What should we say if a panel asks?

> SHAP describes model reliance, not causation. Both models are explained in log-odds units, but their explainers handle feature dependence differently. We compare the rankings descriptively and disclose that limitation.

---

## 8. Important limitations that were already disclosed

These were not judged to be newly discovered fatal errors. They are boundaries on what the results mean.

### Speaking is mixed with cognitive load

A4 requires counting aloud. Speaking can change blinking and possibly other eye signals. Therefore, P1 detects the complete A4 condition, not pure mental effort separated from speech.

P2 uses only mean pupil size and pupil-size variability. It removes the most obvious speech-related blink and event features, but it cannot prove that speaking has no effect on the pupil measurements.

### Normalization uses the held-out person's recordings

The held-out person's labels are not used for training, but their unlabeled recordings help establish their personal mean and standard deviation.

This makes the study relevant to a system with a calibration session. It is not evidence for a completely uncalibrated system meeting a stranger for the first time.

### The event settings were not validated specifically for this tracker

The pipeline uses a 45°/s velocity threshold and a 55 ms minimum fixation duration. These choices have literature and COLET precedent, but they were not validated against manually labeled events for this exact 240 Hz Pupil Core setup.

The sanity check shows the resulting event summaries look plausible. Plausible is not the same as proven ground truth.

### Pupil size can respond to light

The puzzle images may differ in brightness. The original COLET study found a brightness-related pupil pattern in only a small number of participants, but this thesis did not perform its own luminance correction.

### The sample is small and specialized

The final main analysis has 45 participants and 90 A1/A4 recordings from a laboratory task with a chin rest. Results may not transfer directly to real classrooms, vehicles, hospitals, or uncontrolled webcams.

### Whole-activity features hide moment-to-moment changes

Each activity becomes one row of features. This reduces noise but cannot show when cognitive load changed inside an activity.

### Importance means reliance, not cause

LR coefficients, SHAP, and permutation importance show how the fitted models use features. None proves that changing a feature would cause cognitive load to change.

---

## 9. What the review verified

The saved results contain 45 participants and 90 main A1/A4 samples.

| Analysis | Model | Saved ROC-AUC |
|---|---|---:|
| P1, 10 features | Logistic Regression | 0.9901 |
| P1, 10 features | XGBoost | 0.9832 |
| P2, pupil only | Logistic Regression | 0.8509 |
| P2, pupil only | XGBoost | 0.8040 |

The review recalculated the following from the saved prediction files using the current evaluation code:

- all four AUCs;
- balanced accuracy;
- macro-F1;
- recall for both classes;
- Brier scores;
- participant-bootstrap AUC intervals;
- model-difference intervals;
- Wilcoxon statistics.

They matched the saved result tables.

### What is balanced accuracy?

It is the average recall across the two classes. It gives equal importance to correctly recognizing A1 and A4.

### What is macro-F1?

F1 balances precision and recall. Macro-F1 calculates it separately for each class and then averages the classes equally.

### What is a Brier score?

The Brier score measures the squared distance between predicted probabilities and actual answers. Smaller is better.

For example, confidently predicting `0.99` when the answer is `0` is punished much more than predicting `0.55`.

### What else was checked?

- Static code inspection found the intended participant-grouped outer and inner validation.
- SHAP explanations are computed on held-out rows using the corresponding fold models.
- The saved event sanity checks pass for all four activities.
- All Python source and test files passed syntax parsing.
- The blink-gap counterexample was executed using the current detector.

### What was not verified?

- The entire model-training pipeline was not rerun.
- The real-data feature pipeline was not rerun.
- The test suite was not run because `pytest` was unavailable in the checked environments.
- The saved metadata does not contain a Git revision, execution timestamp, or input-data hashes.
- The 40-repeat shuffled-label output is described in notes and has a script, but the individual saved repeat results are absent.
- The scratch no-normalization script is not preserved.

This means the saved numerical outputs are internally consistent with the current scoring code. It does not completely reconstruct the history of how every artifact was originally produced.

---

## 10. What the main results safely mean

### Can the models distinguish A1 from A4?

Yes, very well under the study's offline, participant-normalized protocol.

### Does that mean they detect pure cognitive load?

No. They distinguish task conditions that differ in cognitive demand, time pressure, multitasking, and speaking.

### Did XGBoost beat Logistic Regression?

No clear XGBoost advantage was found.

In both P1 and P2, XGBoost's observed AUC was lower than LR's, but the approximate confidence interval for the difference included zero. The study should say “no clear added benefit was observed,” not “the two models are proven equal” or “LR will always be better.”

### Does P2 solve the speaking problem?

No. P2 asks a narrower question: can pupil features alone still distinguish the conditions? The answer is yes under participant normalization. That is useful, but speaking-related pupil effects and tracker refits remain possible.

### Do both models use similar features?

Their reported importance rankings are similar, especially for the leading features. This is evidence that the fitted models relied on similar clues under their respective explanation methods. It is not proof that the features have identical causal importance.

---

## 11. A very simple “what do we change?” checklist

### Must correct before defense

- [ ] Remove the claim that blink splitting cannot explain a lower fixation rate.
- [ ] State that blink and gap effects on fixation counts were not quantified.
- [ ] Describe the bootstrap intervals as approximate fixed-prediction summaries.
- [ ] Label the Wilcoxon model comparison exploratory or supporting.
- [ ] Remove “there is no leak” and “rules out leakage.”
- [ ] Say exactly what the shuffled-label test checked.
- [ ] Stop saying that P2 proves a signal independent of talking.
- [ ] Reconcile the manuscript with the documented untuned no-normalization run.
- [ ] Do not present the scratch ablation values as verified headline results.
- [ ] Correct the explanation of tree-path-dependent SHAP.

### Good provenance improvements

- [ ] Add the Git commit hash to future run metadata.
- [ ] Add input file hashes or a dataset version identifier.
- [ ] Add the run date and time.
- [ ] Save the individual outputs from post-hoc shuffle checks.
- [ ] Preserve any future ablation scripts and their exact settings.

These provenance improvements make future results easier to audit. They are not proof that the current results are false.

---

## 12. Panel questions and child-clear answers

### “Is your model detecting cognitive load or detecting speech?”

> It detects the difference between COLET's A1 and A4 task conditions. A4 includes higher task demand and speaking, so speech contributes to the full-feature result. The pupil-only analysis removes blink and eye-event features, but it still cannot prove complete independence from speaking.

### “Did the held-out participant really stay unseen?”

> Their labels were not used for model training, but their own unlabeled recordings were used to calculate their personal normalization baseline. The result therefore represents a calibrated new participant, not an entirely unknown person with no baseline data.

### “Why is AUC almost perfect?”

> A1 and A4 are strongly different conditions. Blink rate alone separates them very well, partly because A4 includes speaking. Per-person normalization also highlights within-person changes. The high score should be interpreted as condition discrimination under this protocol.

### “Does your shuffled-label check prove there was no leakage?”

> No single check can prove that. It showed that the untuned downstream models fell to chance when labels were randomized. We found no direct evidence of label leakage, but the diagnostic does not audit every earlier decision.

### “Could blinking distort your fixation results?”

> Yes. Blinks create missing sections. Those sections split fixations, and pieces shorter than 55 milliseconds are discarded. We did not measure the size or direction of that bias in the real condition difference.

### “Are your confidence intervals exact?”

> They are approximate participant-bootstrap intervals over fixed held-out predictions. They show participant variation in those predictions but do not include all uncertainty from repeating training and tuning.

### “Why use normalization if it makes the task easier?”

> Eye measurements differ greatly between people. Normalization lets the model compare each person with their own baseline. That is appropriate for a calibrated system, but it limits claims about an uncalibrated stranger.

### “What happens without normalization?”

> An internal untuned check suggested lower performance, especially for pupil-only P2, but its scratch script was not preserved and it is not equivalent to the tuned main analysis. The final claim is therefore about participant-normalized detection.

### “Did XGBoost outperform Logistic Regression?”

> No clear advantage was observed. XGBoost's measured AUC was lower, but the approximate difference interval included zero. We do not claim that the models are universally equal or that LR always wins.

### “Do SHAP values show which eye behavior causes cognitive load?”

> No. SHAP shows which inputs the model relied on. It does not prove that those inputs caused cognitive load.

### “Are your fixation labels definitely correct?”

> They pass a plausibility check and use thresholds with published precedent, but they were not compared with human-labeled fixations for this exact tracker and sampling setup.

---

## 13. Final child-sized explanation

Imagine two detectives trying to tell whether a person did Task A1 or Task A4 by looking at eye clues.

Both detectives were very good at telling the tasks apart. The simple detective, Logistic Regression, did at least as well as the complicated detective, XGBoost, in this experiment.

But Task A4 was not just “more thinking.” It also included speaking and time pressure. The detectives may have noticed those differences too. They were also allowed to learn what was normal for each new person before judging that person.

The calculations in the saved tables match the saved predictions. The main problems are sentences that say, “This definitely cannot be an artifact,” “This proves there is no leak,” or “No other run happened.” Those sentences go farther than the evidence.

The thesis becomes easier to defend when it says precisely what was tested:

> Under an offline protocol that uses each participant's own recordings for normalization, eye-tracking features distinguished COLET A1 from A4 very well. XGBoost showed no clear advantage over Logistic Regression. Speaking, calibration, event-detection artifacts, and the small laboratory dataset limit how broadly the findings can be generalized.

That conclusion is narrower than “we built a universal cognitive-load detector,” but it is much stronger scientifically because it says exactly what the evidence supports.

---

## 14. Sources used to evaluate the disputed technical claims

- C. H. H. Hessels et al., “Fixation classification: how to merge and select fixation candidates,” *Behavior Research Methods*. [Publisher page](https://link.springer.com/article/10.3758/s13428-021-01723-1); [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9729319/).
- Y. Bengio and Y. Grandvalet, “No Unbiased Estimator of the Variance of K-Fold Cross-Validation,” *Journal of Machine Learning Research*. [Paper page](https://www.jmlr.org/papers/v5/grandvalet04a.html).
- S. Bates, T. Hastie, and R. Tibshirani, “Cross-validation: what does it estimate and how well does it do it?” [Preprint](https://arxiv.org/abs/2104.00673).
- SciPy, “Wilcoxon signed-rank test.” [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
- S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*. [DOI page](https://doi.org/10.1016/j.patter.2023.100804).
- SHAP, “TreeExplainer.” [Official documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).


# Questions for the Adviser

Open decisions that need the adviser's sign-off before the paper (Chapters 1–5) is finalized.
Each entry gives the question, the evidence, the options and the team's recommendation.

**Source:** mock-defense review (prosecutor vs defender) of
`CSRP-Chapter-1-to-5-draft-COLET-v2.docx`, 2026-10-02. Line numbers (L…) refer to the
text extracted from that draft.

Updated 2026-10-02.

---

## Q1. Should we report a "no per-person standardization" result? (review item B2)

**Status:** ⏳ Open

### Background

Before training, every feature is converted to a z-score within each participant, using the
mean and SD of that participant's own four activities (`src/run_colet.py:62`,
`src/normalize.py`). This includes the two recordings (A1, A4) that are being tested in that
fold. No labels are used, but the model effectively learns "higher than this person's usual"
rather than "high".

The paper discloses this (L44, L172, L300) and states that no run without standardization
was made (L298). It is also our own Recommendation 3 (L308).

### Evidence

A post-hoc re-run of the pipeline with default (untuned) hyperparameters, LOPO, with and
without the standardization step. The standardized numbers reproduce the paper's untuned
permutation-test values exactly (L245, L256). Untuned, no confidence intervals; not in the
paper.

| Pooled ROC-AUC (LR / XGB), untuned | Raw features | Standardized (as in paper) |
|---|---|---|
| A1 vs A4, P1 (10 features) | 0.870 / 0.874 | 0.991 / 0.976 |
| A1 vs A4, P2 (pupil only) | **0.602 / 0.647** | 0.854 / 0.819 |
| A1 vs A4, blink rate only | 0.886 / 0.832 | 0.964 / 0.922 |

**Update 2026-10-02:** the full tuned run (nested tuning, bootstrap CIs, permutation test), done
both ways, is in **Q2** below. The tuned numbers differ from these untuned ones (for example, tuned
raw P1 LR is 0.853, not 0.870), so cite only the tuned run.

### What this means

- **P1 holds up.** About 0.87 without standardization is still strong; the step adds about
  0.12.
- **P2 largely depends on it.** Raw pupil-only drops to about 0.60–0.65. The claim that "a
  load-related signal remains when the speech-sensitive features are removed" (L279, L290)
  holds only when each person is compared with their own baseline.

### Options

1. **Report it** as a post-hoc table (labelled as such, like the shuffled-label control), and
   reword the P2 interpretation: pupil carries a load signal *relative to each person's own
   baseline*; across people, raw pupil size barely separates the conditions.
2. **Do not report it;** keep the current disclosure (L298) and the limitation only.

### Team recommendation

**Option 1.** A panel is likely to ask "what happens without standardization?", since L298
already invites it. Disclosing the drop ourselves is safer than having it found. The reframed
P2 result is still a real finding and is consistent with the limitation that a deployed system
needs a calibration session (L44, L300).

### If approved, the follow-up work is

- Move the ablation script into the repo so the result is reproducible (it currently exists
  only as a scratch script), and decide whether to run it tuned (nested) with bootstrap CIs
  like the main results.
- Add the table to Chapter 4 (post hoc) and remove "no analysis without per-person
  standardization was run" from L298.
- Reword L279, L280, L290 and the P2 paragraph (L256) to match.

### Adviser decision

> _(to be filled in)_

---

## Q2. Which route should the study take: standardized or unstandardized features?

**Status:** ✅ Decided 2026-11-03: Route A (per-person standardized) approved by the adviser

### The two routes

| | **Route A: per-person standardized** (current plan) | **Route B: unstandardized** (raw features) |
|---|---|---|
| What the model sees | Each feature as "higher or lower than this person's own average" over their 4 activities | The raw values (scaled the same way for everyone) |
| Real-world meaning | A **calibrated** user: needs a few recordings of the new person first | A **total stranger**: no calibration |
| Literature | The closest studies (Božak 2026, Oppelt 2023, Gado 2023, Appel 2023) | COLET paper, Choi & Nam 2026, He 2025 (global scaling) |
| Known cost | Uses the test person's own unlabelled data, so the test is easier (disclosed) | Large differences between people swamp the load effect |

### Results (full pipeline, identical apart from the standardization step)

Same 44 participants (rerun 2026-10-15 after the P28/P31 duplicate fix; researcher-notes N31),
leave-one-participant-out, nested tuning with 30 candidates per model,
participant-bootstrap 95% CIs, permutation test (default hyperparameters). Route A is the
main run; Route B was run on 2026-10-02 (`docs/colet-eda/no_standardization.py`, outputs in
`outputs/colet/tables/optional/raw_*`).

| Pooled ROC-AUC (95% CI) | Route A: standardized | Route B: unstandardized |
|---|---|---|
| **P1, LR** (10 features) | **0.992** (0.979-1.000) | **0.853** (0.783-0.923) |
| **P1, XGBoost** | **0.980** (0.957-0.997) | **0.881** (0.822-0.941) |
| **P2, LR** (pupil only) | **0.831** (0.735-0.912) | **0.611** (0.573-0.657) |
| **P2, XGBoost** | **0.812** (0.697-0.912) | **0.649** (0.565-0.732) |

| Other results | Route A | Route B |
|---|---|---|
| Balanced accuracy, P1 (LR / XGB) | 0.966 / 0.920 | 0.739 / 0.818 |
| People whose A4 was ranked above their A1, P1 (LR / XGB) | 44/44 / 44/44 | 42/44 / 42/44 |
| Ceiling rule (both > 0.95 in P1) | **Triggered**: compare models in P2 | Not triggered: compare models in P1 |
| Model comparison (XGB - LR) | P2: -0.019 (-0.086 to 0.047); margins larger for XGB in 29/44 (p = 0.007) | P1: **+0.028** (-0.034 to 0.094); margins larger for XGB in 41/44 (p < 0.001) |
| Permutation p (P1 / P2) | 0.005 / 0.005 | 0.005 / 0.015 (LR), 0.04 (XGB) |
| Top features, P1 | Blink rate, pupil mean, fixation rate (both models; tau 0.69) | LR: fixation duration, blink rate, saccade rate; XGB: blink rate, fixation rate, saccade amplitude; tau 0.24 |

### What this means

1. **Within each person, the signal is strong either way.** Even without standardization, the
   model ranked a person's A4 above their A1 for 42 of 44 people. What drops is the
   *pooled* AUC, because people differ a lot from each other (for example in pupil size).
   Standardization removes those between-person differences.
2. **Route A** gives near-perfect results, but they assume calibration, and the model comparison
   has to be read from the weaker pupil-only test (ceiling rule).
3. **Route B** is the stricter "stranger" test. P1 is still good (0.84-0.88), and because it is
   not at the ceiling, it gives the **clearer model comparison**. XGBoost scores slightly higher
   there, but the difference is not significant (its CI includes zero). The pupil-only result
   (P2) becomes weak (0.61-0.66), so raw pupil size barely separates the conditions across people.
4. **The overall LR-vs-XGBoost answer is similar on both routes:** no significant difference in
   pooled AUC. The direction differs: LR slightly ahead with standardization, XGBoost slightly
   ahead without. The optional no-blink run (Route A) favored LR (researcher-notes N29).

### Options

1. **Keep Route A as the main analysis and report Route B in full** as a secondary analysis
   (it is the pre-listed optional S4). The thesis then reports both readings: with calibration,
   detection is near-perfect; without it, detection is good but lower, and the pupil-only signal
   depends on each person's own baseline.
2. **Switch the main analysis to Route B.** This gives a stricter test and a model comparison
   that is not at the ceiling, but it changes the pre-specified primary analysis after results
   are known, which a panel can call outcome-driven. It also departs from the closest studies.
3. **Report both routes as co-primary**, with the research question answered for each.

### Team recommendation

**Option 1** (or 3 if the adviser prefers). Route A was fixed before any results and follows the
closest studies, so keeping it as primary is the defensible choice. Reporting Route B in full
answers the panel's most likely question ("isn't 0.99 inflated by the normalization?") with
numbers, and shows the model comparison without the ceiling. Option 2 needs explicit adviser
approval, because it changes the frozen methodology.

### If approved, the follow-up work is

- Option 1 or 3: add a Route B table and paragraph to Chapter 4; reword the P2 interpretation
  ("relative to each person's own baseline") and the conclusions; replace "no analysis without
  per-person standardization was run" in the limitations.
- Option 2: also rewrite Chapter 3 (§7, analysis plan), the ceiling-rule section and Chapter 5,
  and record the change and its reason in the decision log.

### Adviser decision

> **2026-11-03: Route A (per-person standardized) approved** as the main analysis (reported by
> the team). Still to confirm with the adviser: whether Route B is also reported as a
> secondary analysis (Option 1) or not reported (this also answers Q1).


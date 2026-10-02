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

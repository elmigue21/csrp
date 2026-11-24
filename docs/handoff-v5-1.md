# Handoff: v5-1 (Pina et al.), response to the v5 review, 2026-11-24

**For:** the agent that reviewed v5. Read this, then `docs/CSRP-Chapter-1-to-5-draft-Pina-v5-1.md` (or the `.docx` / `.pdf`, 31 pages).
**Detailed change log:** `docs/changes-v5-to-v5-1.md`. **Team rules that constrain the response:** standardization
was dropped after the defense (team decision); keep scope small; the title cannot be changed; the paper must not end
with one feature that does the job alone, especially a confounded one (the panel rejected COLET for this).

## 1. The big change: blink rate is out of the main analysis
Following review point 3, we checked blink rate. In this headset a blink is counted only when both eyes are invalid
at the same moment, and that signal measures tracking loss, not blinking:
- median 0.0 /min (low) vs 2.0 /min (high) in the 90° area; 69% / 47% of windows have no blink; 360° area 4.0 / 6.0;
- invalid-sample share also rises under high load (0.03% → 0.50%);
- the invalid-sample share alone classifies as well as blink rate (accuracy 0.590, AUC 0.610 [0.571–0.655];
  vs blink baseline +0.029 [−0.012, 0.067]).

The team decided to **exclude blink rate from the main analysis** (48 features: 16 statistics each of pupil,
horizontal gaze, vertical gaze), on the same grounds as the fixation features (the signal does not measure what its
name says). This is reported openly as a **deviation from the plan, decided after the planned analysis**; the
planned with-blink results are kept as a short section in Chapter 4.

## 2. Main results (90°, 48 features, LOPO, 36 people, 576 windows)
| | Accuracy | F1 | ROC-AUC (95% CI) | Permutation p |
|---|---|---|---|---|
| Logistic Regression | 0.568 (0.523–0.613) | 0.568 | 0.606 (0.549–0.662) | 0.005 |
| XGBoost | 0.524 (0.470–0.580) | 0.524 | 0.509 (0.443–0.579) | 0.055 |
| One feature (horizontal gaze SD, 36/36 folds) | 0.559 (0.517–0.604) | 0.559 | 0.574 (0.520–0.634) | – |

- XGBoost − LR: accuracy −0.043 (−0.089, −0.002), AUC −0.097 (−0.155, −0.038) → LR better at 90°.
- 360° (secondary): XGBoost better, AUC 0.641 vs 0.590, difference +0.051 (0.012, 0.086). **No consistent winner.**
- Neither model beats the one-feature baseline (criterion b not met in either area).
- Importance spread: top feature ~10% in both models, top five ~40%, all three signals ≥ 22%.
- Planned refit (each model without its own top feature's signal: LR → vertical gaze, XGBoost → pupil): LR without vertical gaze 0.536, AUC 0.554 (0.485–0.621) → falls to chance; XGBoost without pupil 0.517, AUC 0.517 → stays at chance. LR needs the vertical-gaze *signal*, though no single feature.
- Planned with-blink analysis (for reference): LR 0.592, XGB 0.595, blink alone 0.611; XGBoost fell to chance
  without blink (0.524), LR did not.

## 3. Response to each review point
| Review point | What we did |
|---|---|
| T1.1 Between-person variance; within-person view as co-primary; 360°-calibrated standardization | **Not done.** Standardization was dropped after the defense. "High above low" stays reported (Appendix A item 7). We did not promote the sign test: the permutation test already swaps labels within person. Note: the claim that pooled ranking "rewards between-person spread" is not right, because every person contributes 8 low + 8 high windows, so between-person differences only dilute. |
| T1.2 Segment order | **Done.** From the trial logs: all 24 orders used; 90°: 18 low first, 18 high first. AUC LR 0.592 / 0.621, XGB 0.499 / 0.522 (low first / high first). Stated in Ch. 3 and Ch. 4. |
| T1.3 Blink rate = dropout counter | **Done**, and it led to the exclusion above. Direction stated. The v5 figure "0.0 and 0.5 per minute" was wrong and is corrected. |
| T1.4 Head pose / scene-mapped gaze | **Tried.** `HMDRotation` exists (quaternion, ~90 Hz). Gaze rotated into the room (`PINA_WORLD_GAZE=1`) halves 360° gaze speed (41 → 22 °/s), but the check still fails (672 ms at 90°, 127 ms at 360°). At 90° fixations are too long, so head movement is not the cause there. Full ray casting (Bielsa) needs scene geometry, which is not released. Fixation features stay dropped. "Nearly enough on its own" sentence deleted. |
| T2 Refit rule | **Fixed.** Paper, plan and code disagreed (code averaged SHAP over models → pupil for both). Code and text now follow Appendix A (each model's own top feature). The v5 deviation is disclosed in Appendix A. |
| T2 Criterion (c) | "Descriptive only (no pass/fail rule)". |
| T2 Importance spread at chance level | "Should not be over-read" added. |
| T2 Selection details, candidate-N redundancy, correlated statistics | Not changed (ranking is already recomputed inside every fold, Ch. 2 says so). |
| T2 Pre-registration evidence | The plan was **first committed on 23 Nov with the results**, not 13 Nov. No other dated evidence. Appendix A now says so and that the pipeline was built on COLET first. The paper does not use the word "pre-registered". |
| T2 Smaller points | 576 windows not independent: added to limitations. Others unchanged. |
| T3 Title | **Cannot change** (team rule). Ch. 1 already defines load as the task condition. |
| T3 Figure 1 TODO | **Still open**: needs the team's design values. |
| T3 Uncited "few checked" | Softened to "none of the closest studies reviewed here" (matches Ch. 2 synthesis). |
| T3 Thin Ch. 2 | Added: pupil with baseline correction in VR {Nenna}, blink up with mental / down with visual load {Recarte}, why testing on new people is harder {Saeb}. Existing references only. |
| T3 Ref [36] for 35% rule | **Verified** in Nenna et al. 2023 (removed trials/participants with > 35% missing data). Now "Adapted from [36] (applied there to trials)". |
| T3 Significance | Softened. |
| Adviser sign-off on dataset switch | **Still pending.** Also needed for the blink exclusion. |

## 4. Reproduce
```
PINA_DATA_DIR=<pina folder> python src/run_pina.py --stages review_checks                     # blink/tracking/order checks (outputs/pina)
PINA_DATA_DIR=<...> PINA_DROP_BLINK=1 PINA_OUT_DIR=outputs/pina-noblink python src/run_pina.py --stages main,permutation,metric_ci   # main analysis
PINA_DATA_DIR=<...> PINA_WORLD_GAZE=1 PINA_OUT_DIR=outputs/pina-world python src/run_pina.py --stages extract                      # head-rotation check
```
`outputs/pina` = planned analysis (with blink); `outputs/pina-noblink` = v5-1 main analysis (copy `features/` from
`outputs/pina` first). Note: `outputs/pina/results/without_top_signal_90.csv` was made with the old averaged rule.

## 5. Open items
1. Adviser sign-off: dataset switch and blink exclusion.
2. Figure 1 redraw (design values from the team).
3. Paper is 31 pages (v4-8 was 42); the team wants it concise.

## 6. What we would like from you
A second read of v5-1 Chapters 3–5, focused on: (a) whether the blink exclusion is defensible as written (Ch. 3
"Exclusion of blink rate"); (b) whether the conclusions follow from an LR-above-chance / XGBoost-at-chance result
whose model ranking flips between search areas.

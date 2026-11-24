# Changes: v5 → v5-1 (Pina et al.), 2026-11-24

**Paper:** `CSRP-Chapter-1-to-5-draft-Pina-v5-1.docx` (and `.pdf`, 31 pages), built by `build_v5_1_docx.py` from the `.md` of the same name.
v5 moved to `docs/old/`. **Why:** follow-up to an external review of v5. **Still needs adviser sign-off** on the
dataset change.

## Main decision: blink rate excluded from the main analysis (team decision, 2026-11-24)
Blink rate measures tracking loss in this headset (see checks below), and the panel rule is that no single,
confounded feature may carry the result. The main analysis now uses **48 features** (pupil, horizontal gaze,
vertical gaze). Reported as a **deviation from the plan, decided after the planned analysis**; the planned
with-blink results stay as a short Chapter 4 section. Run: `PINA_DROP_BLINK=1 PINA_OUT_DIR=outputs/pina-noblink
python src/run_pina.py --stages main,permutation,metric_ci` (copy `features/` from `outputs/pina` first).

| 90°, 48 features | Accuracy | ROC-AUC (95% CI) | Permutation p |
|---|---|---|---|
| Logistic Regression | 0.568 (0.523–0.613) | 0.606 (0.549–0.662) | 0.005 |
| XGBoost | 0.524 (0.470–0.580) | 0.509 (0.443–0.579) | 0.055 |
| One feature (horizontal gaze SD, 36/36) | 0.559 (0.517–0.604) | 0.574 (0.520–0.634) | – |

- XGBoost − LR at 90°: accuracy −0.043 (−0.089, −0.002), AUC −0.097 (−0.155, −0.038). At 360°: XGBoost better,
  AUC +0.051 (0.012, 0.086). No consistent winner.
- Criterion (a): LR yes, XGBoost no. Criterion (b): not met. Top feature ~10% in both models.
- Planned refit: LR without vertical gaze 0.536, AUC 0.554 (0.485–0.621) → chance; XGBoost without pupil 0.517.

## Earlier in this round: with-blink checks (all post-hoc, labeled as such in the paper)
Run: `PINA_DATA_DIR=<pina folder> python src/run_pina.py --stages review_checks` → `outputs/pina/results/review_checks_90.csv`.

| Check | Result | What it means |
|---|---|---|
| XGBoost without blink rate (the planned refit: blink rate is XGBoost's top feature) | Accuracy 0.524, AUC 0.509 (0.443–0.579) | XGBoost falls to chance: its result depended on blink rate |
| LR without blink rate (48 features) | Accuracy 0.568, AUC 0.606 (0.549–0.662) | LR stays above chance without blink |
| One feature: share of invalid samples (tracking quality) | Accuracy 0.590, AUC 0.610 (0.571–0.655); vs blink baseline +0.029 (−0.012 to 0.067) | Tracking loss classifies as well as blink rate → blink rate behaves like a tracking-loss measure |
| Blink rate by load, 90° | Median 0.0 /min low vs 2.0 high; 28 of 36 higher under high load; 360°: 4.0 vs 6.0 | Direction now stated. **v5's "0.0 and 0.5 per minute" was wrong**, corrected |
| Invalid-sample share by load, 90° | Median 0.03% low vs 0.50% high | Rises with the button-press task |
| Condition order (from the trial logs) | All 24 orders used; 90°: 18 low first, 18 high first | Counterbalanced. AUC 0.55–0.59 (low first) vs 0.64–0.65 (high first): small time effect, cancels out |
| Head rotation applied before event detection (`PINA_WORLD_GAZE=1`, `outputs/pina-world/`) | 360° gaze speed 41 → 22 °/s; fixation median 672 ms (90°), 127 ms (360°) | Event check still fails; head movement does not explain the 90° failure. Fixation features stay dropped |

## Plan deviation found
Appendix A says "refit without the **top feature's** signal"; v5 Chapter 3 said "the signal with the largest share";
the code averaged SHAP shares over both models (→ pupil for both). v5-1 follows Appendix A: LR without pupil, XGBoost
without blink. The XGBoost-without-pupil result is kept "for completeness" and called a deviation.

## Text changes
| Where | Change |
|---|---|
| Ch. 1 Context | "few checked…" (uncited) → "none of the closest studies reviewed here checked…" |
| Ch. 1 Significance | "could one day support adaptive training" softened |
| Ch. 2 | Pupil in VR with baseline correction {66}; blink rises with mental, falls with visual load {36}; why testing on new people is harder {14} |
| Ch. 3 Labels | Condition order sentence |
| Ch. 3 Table 4 | 35% rule: "Adapted from {66} (applied there to trials)". Checked in the paper: Nenna et al. removed trials/participants with > 35% missing data |
| Ch. 3 Importance | Refit rule matches Appendix A; post-hoc checks listed |
| Ch. 4 Table 8 | Criterion (c): "Yes (descriptive)" → "Descriptive only (no pass/fail rule)" |
| Ch. 4 Importance | Planned refits, the deviation, "should not be over-read" |
| Ch. 4 Secondary | Post-hoc check results paragraph |
| Ch. 4 Discussion | Blink = tracking loss; order; event-check failure not explained by head movement; deleted "nearly enough on its own, which is why they were left out" |
| Ch. 5 | Findings 4–5, conclusion, limitations (fixation, blink, windows not independent, post-hoc checks), Recommendation 1 (scene geometry) |
| Appendix A | Plan not timestamped; first committed 23 Nov with the results; pipeline built on COLET first |

## Code
- `src/run_pina.py`: stage `review_checks`, `condition_order()`.
- `src/pina.py`: `parse_quaternions`, `head_rotation_per_sample` (HMDRotation is logged at ~90 Hz, repeated for 2–3
  eye samples, so it is interpolated), `rotate`; used only when `PINA_WORLD_GAZE=1`.
- `src/features.py`: events and gaze speed use `world_normal*` columns when present; gaze angle features stay
  head-relative. Main results unchanged (flag off by default).
- `src/config.py` `PINA_DROP_BLINK`, `src/run_pina.py` `area_data` honours it; refit rule in `stage_main` now per model
  (Appendix A). `outputs/pina/results/without_top_signal_90.csv` was made with the old averaged rule.
- `tests/test_pina.py`: tests for the rotation helpers and the drop-blink switch. 116 tests pass.

## Still open
1. **Figure 1** still the COLET version (needs the team's design values).
2. **Adviser sign-off** on the dataset change.
3. Title unchanged (not allowed to change).

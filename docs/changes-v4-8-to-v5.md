# Changes: v4-8 (COLET) → v5 (Pina et al.), 2026-11-19

**Paper:** `CSRP-Chapter-1-to-5-draft-Pina-v5.docx`, built by `build_v5_docx.py` from the `.md` of the same name.
Old drafts are in `docs/old/`. **Needs adviser sign-off** (dataset change after the defense).
**Why:** on COLET, blink rate alone matched both models and the high-load task involved speaking. Pina et al.'s
high-load task has no speaking.

## What changed
| Where | v4-8 (COLET) | v5 (Pina) |
|---|---|---|
| Data | COLET, 44 people, 88 whole recordings | Pina et al., 36 people, 576 × 30 s windows (90° main, 360° separate) |
| Label | A1 vs A4 (A4 includes speaking) | Visual search vs search + auditory task (button press on odd numbers) |
| Ch. 1 | – | Says openly why the dataset changed; the plan was written before loading eye data |
| Objective 3 | vs majority-class baseline | vs **one-feature baseline chosen inside each fold** |
| Objective 4 (number of features) | Included | Cut (keep scope small) |
| Metrics | ROC-AUC "primary"; CI on AUC only | **Accuracy, precision, recall, F1 lead**; every metric has a participant-bootstrap CI; paired differences on accuracy, F1, AUC |
| Ch. 3 | – | **Table 3: data shape before/after processing** (13,185,280 × 23 raw → 1,152 × 140 → 576 × 49) |
| Features | 140 (event check passed) | 49: event check failed (fixation 665 ms), gaze speed / fixation / saccade dropped as planned |
| Results | AUC 0.862 both | Accuracy LR 0.592, XGB 0.595; blink rate alone 0.611; no difference anywhere |
| New sections | – | Pass-criteria table, paired-difference table, 360° check, sensitivity check, refit without pupil |
| Ch. 2 | COLET prior use | VR eye-tracking section (Kothari, Nenna, Nerella, Pina, Bielsa) |
| References | 74 | 54 (`rrl-review-pina-v5.md`); Faraji and Velisar (Pupil Core) dropped; Pina and Bielsa added |
| Figures | 8 | 1 (Figure 1, still the COLET version: **needs redrawing**) |
| Appendix A | – | The analysis plan |

## Proposed title (for the adviser)
"Logistic Regression and XGBoost Compared with a Single-Feature Baseline for Detecting a High-Load Task
Condition from Virtual Reality Eye-Tracking Features". The title page still has the old title.

## Still open
1. **Figure 1** must be redrawn for Pina (needs the team's design values). The docx still shows the COLET figure,
   with a TODO in its caption.
2. **Title page** keeps the old title until the adviser approves a new one.
3. **gaze_y direction:** assumed +y = up (the headset position data uses y as height). Not checked against Varjo
   documentation. Only the wording of "vertical gaze, minimum" depends on it, not any number.

## Numbers added in this round
- Demographics (36): 23 male, 13 female; age 31.9 ± 11.3 (22–62). `outputs/pina/results/demographics.json`.
- CIs for every metric and paired differences: `metric_ci_<area>.csv`, `metric_differences_<area>.csv`
  (stage `metric_ci` in `src/run_pina.py`).
- Blink rate median 0.0 (low) / 0.5 (high) per minute: the headset rarely marks both eyes invalid at once
  (one participant, one 4-min segment: 7 such periods, 3 of them 50–100 ms). Reported as a limitation; the blink
  rule was not changed after seeing this.

## Code and outputs
- `src/pina.py`, `src/run_pina.py` (stages: extract, manipulation, main, permutation, sensitivity, metric_ci),
  `tests/test_pina.py`; plan in `docs/pina-analysis-plan.md`; outputs in `outputs/pina/`.
- Run: `PINA_DATA_DIR=<folder with gaze/, trial/, tlx/, demo/> python src/run_pina.py`.
- The repo run reproduced the scratchpad numbers exactly, except the permutation test, which now uses the repo's
  `permutation_test` (XGBoost p 0.020 → 0.005).

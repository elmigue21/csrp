# Changes: v4-7 → v4-8 (wording fixes from the panel review, 2026-11-12)

**Paper:** `CSRP-Chapter-1-to-5-draft-COLET-v4-8.docx`. Source text: the `.md` of the same name.
**Source of the fixes:** `panel-review-v4-7.md` (number check, writing fixes, minor issues 1-5).
No result changed and nothing was re-run.

## Fixed (wrong or imprecise statements)
| Where | Was | Now |
|---|---|---|
| Ch. 3 Feature Extraction | Blink-duration statistics blank "almost only in low load" | Blank in 79 of 88 recordings; 8 of the 9 with values were high load, so whether a value exists reveals the label. The drop decision is unchanged |
| Ch. 3 Feature Extraction | Variances are "exact copies" of SDs | The squares of the SDs, same information |
| Ch. 3 Design Choices | Blinks "may raise the fixation rate" in A4 | Clause cut (fixation rate is lower in A4) |
| Ch. 4 Number of Features | LR "identical" from N = 40, "because its penalty..." | "almost the same (ROC-AUC 0.876)"; unverified reason cut |
| Ch. 4 Feature Importance | "the penalty set all others to zero" | Kept five in 42 folds; the penalty did it in the other two |
| Ch. 4 Feature Importance | "top five gave 73%" | 72.5% |
| Ch. 4 Classification Results | AUC compared with "balanced accuracy 0.500" | Accuracies compared with the baseline (0.500) |
| Ch. 4, Ch. 5 | "Gaze location", "looked lower at some point" | "Gaze-angle statistics", "minimum vertical gaze angle" |
| Ch. 4 Discussion | Saccade-amplitude clause after the saccade-rate result | Cut (it is about amplitude, not rate) |
| Rec. 2 | A1 vs A3 and A2 vs A4 "differ only in speaking" (wrong: they also differ in the counting task) | Classify A1 vs A2, the only contrast without speaking |
| Practitioners | LR on blink rate "is a reasonable choice" | Blink rate should not be used alone: speaking raises it, visual tasks can lower it |
| Ch. 1 Purpose | Cleaning follows "published guidelines for Pupil Core data" | "where they exist", the rest listed as design choices |
| Model Comparison, Conclusions, Summary | LR "higher accuracy" | "at the 0.5 threshold"; the CI makes a gap above about 0.05 unlikely |
| Figure 8 | "Up/down gaze angle", "Saccade size" ... | The paper's terms: vertical gaze angle, saccade amplitude, fixation duration, pupil diameter |

## Cut
- Balanced accuracy (it equals accuracy here): Objective 3, Ch. 3 metrics, Table 11 row, definition.
- Imputation mentions (no values were missing). The XGBoost missing-values citation went with
  it, so the reference list is 74 (was 75) and renumbered.
- NASA-RTLX column of Table 3 (Table 7 has the analyzed participants' ratings).
- Table 11: the permutation row now says "(default settings, N = 20)".

## Same fix outside the paper
The blink-duration reason was also wrong in `src/flatten.py` (`_D1`), `src/processing_log.py`
(D1 rule), `post-defense-data-processing.md` (both "How to explain it" lines) and
`post-defense-modeling-plan.md`. `outputs/final` was rebuilt, so `logs/log_features.csv`,
`logs/log_decisions.md` and `previews/features_preview.xlsx` carry the new text. The features
are unchanged (same `features_sha256`).

## Not done (not wording; still open from the review)
- Limitation sentences: gaze extremes grow with recording length (D3 promised it); blink rate
  may partly track tracking loss.
- Cutting the Wilcoxon margin test; the Discussion sentence linking back to the panel's
  leakage question; the within-person ranking clause.
- Figure 1 deliverable wording ("tuned models"; no final model is saved) and Figure 2
  (ranking is repeated in every inner fold).
- Citations to verify: Nerella {12} "participant-independent", {38}, {37}, {66}, {76}; Guyon;
  Fawcett for precision/recall/F1.

## Rebuild
`python src/paper_figures.py`, then `python docs/build_v4_8_docx.py`.

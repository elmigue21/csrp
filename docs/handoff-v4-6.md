# Handoff: Latest Paper (v4-6)

**Date:** 2026-11-08
**For:** whoever picks up the thesis paper next (teammate or agent).
**Paper:** `docs/CSRP-Chapter-1-to-5-draft-COLET-v4-6.docx` (Chapters 1–5, references, Appendices A–B).
**Status:** text is final for the 7 Nov documentation deadline; presentation is **Sun 8 Nov, 8:00 AM**.

> `docs/HANDOFF.md` (2026-10-01) is out of date: it says the pipeline has not been run. Use
> this file for the paper and `HANDOFF.md` §6 only for how to work with the user.

---

## 1. The paper in one paragraph

- **Title:** Comparative Analysis of Logistic Regression and XGBoost for Cognitive Load Detection
  Using Eye Tracking Features with Feature Importance Analysis.
- **Team:** Caacbay, Magkasi, Uaje, Ventura (BS Computer Science).
- **Method:**
  - COLET data, A1 (low load) vs A4 (high load);
  - 44 participants, 88 whole-activity samples, 10 eye-tracking features;
  - per-person z-score over all four activities (Route A, approved by the adviser on 2026-11-03);
  - elastic-net LR vs XGBoost, nested LOPO (44 folds, 5-fold grouped inner, 30 candidates);
  - pooled ROC-AUC with participant-bootstrap CI;
  - SHAP, permutation importance and LR coefficients.
- **Main message:**
  - Both models separate A1 from A4 almost perfectly, and XGBoost is not better than LR.
  - The separation is mostly speaking (blink rate) plus per-person calibration, so the paper
    claims to detect "COLET's task condition", not speech-free cognitive load.

## 2. Version lineage

| Version | What changed | Change notes |
|---|---|---|
| v4 / v4-1 | Draft checked by the defense-panel review | `panel-review-handoff.md` |
| v4-2, v4-2-1 | Panel-review text and figure fixes; A3 vs A4 and A1 vs A2 results added | `changes-v4-2-1-to-v4-4.md` |
| v4-3 | Post-hoc analyses added (Tables 15, 16, Appendix B) | `changes-v4-2-1-to-v4-4.md`, `extra-analyses-notes.md` |
| v4-4 | Second review's fixes | `changes-v4-2-1-to-v4-4.md` |
| v4-5 | Text-only fixes (see §3) | this file |
| **v4-6** | v4-5 plus the **new Figure 1** (vertical flowchart, `docs/figure1-framework.png`) | this file |

**Checks on v4-6:**
- **Text:** the same as v4-5, paragraph for paragraph (checked from `document.xml`).
- **Images:** the only change is that `image1.png` (the old IPO-style Figure 1) was replaced by
  `figure1-framework.png`.
- **Markdown:** `CSRP-…-v4-6.md` is byte-identical to the v4-5 markdown. It was copied, not
  regenerated, so its embedded Figure 1 is still the old one. Use the docx for figures.

## 3. Changes from v4-4 to v4-5 (text)

| Where | Change |
|---|---|
| Ch 1 Significance | "fully specified evaluation" → "documented evaluation" |
| Ch 3 Per-person standardization | New sentence: the main evaluation corresponds to a calibrated user who has completed all four activities, not a new user |
| Ch 3 Metrics | Brier score explained (0.25 for a constant 0.5 prediction; not a full calibration assessment) |
| Ch 3 Model comparison | Bootstrap CI is now described as **paired** (same participants drawn for both models), and used for every model difference including post hoc |
| Ch 4 Manipulation check | Explains n = 43 (P28 dropped because the owner of the P28/P31 recordings is unknown; P31 already removed) and adds "all 47 also rated A4 > A1" in the full released data |
| Ch 4 Model comparison | "practically identical" → "no clear difference was detected" |
| Ch 5 Summary item 2 | Brier context ("against 0.25 for a constant prediction"); "majority-class baseline **balanced accuracy** 0.500" |

## 4. Paper structure

| Chapter | Contents | Tables / figures |
|---|---|---|
| 1 | Context, 5 specific objectives, significance (SDG 3, 4, 9), scope and limits, conceptual framework | Figure 1 |
| 2 | RRL: cognitive load, eye-tracking indicators, COLET, LR vs XGB, LOPO and leakage, feature importance, synthesis | Table 1 |
| 3 | Design, dataset, labels, preprocessing, event detection, features, standardization, models, validation, metrics, importance, analysis plan, implementation, ethics, design choices | Tables 2–7, Figures 2–4 |
| 4 | Manipulation check, data quality, sanity check, features by condition, P1, P2, model comparison, feature importance, shuffled-label check, **post-hoc robustness**, discussion | Tables 8–16, Figures 5–8 |
| 5 | Summary (6 items), conclusions, limitations (15 bullets), recommendations (5) | Table 17 |
| App. A / B | XGB gain importance; post-hoc 10 s windows | Tables A1, B1 |

## 5. Numbers to know

| Analysis | LR ROC-AUC (95% CI) | XGB ROC-AUC (95% CI) | Source |
|---|---|---|---|
| P1, 10 features (headline) | 0.992 (0.979–1.000) | 0.980 (0.957–0.997) | Table 12 |
| P2, pupil only | 0.831 (0.735–0.912) | 0.812 (0.697–0.912) | Table 12 |
| P1 / P2 not standardized | 0.853 / 0.611 | 0.881 / 0.649 | Table 12 |
| P1 without blink rate | 0.929 | 0.886 | Ch 4 text |
| A2-baseline centering, P1 (post hoc) | 0.903 | 0.921 | Table 15 |
| Blink rate alone, standardized / raw | 0.963 / 0.883 | not run | Table 15 |
| Default hyperparameters, P1 | 0.991 | 0.977 | Table 15 |
| A3 vs A4, P1 (both speaking) | 0.624 | 0.602 | Table 16 |
| A1 vs A2, P1 (both silent) | 0.752 | 0.765 | Table 16 |
| 10 s windows, P1 (App. B) | 0.928 | 0.901 | Table B1 |

**Other results:**
- **Balanced accuracy:** P1 0.966 / 0.920; P2 0.807 / 0.773.
- **Brier score:** P1 0.032 / 0.060.
- **A4 > A1:** P1 44/44 for both models.
- **XGB − LR AUC difference:**

  | Analysis | Difference (95% CI) | Wilcoxon on margins |
  |---|---|---|
  | P1 | −0.012 (−0.030 to 0.002) | p = 0.112 |
  | P2 | −0.019 (−0.086 to 0.047) | p = 0.007, but this only reflects XGB's more extreme probabilities |
  | No blink | −0.043 (−0.084 to −0.007) | the only pre-specified interval that excludes 0 |

- **Importance:**
  - SHAP top three for both models: blink rate, pupil mean, fixation rate.
  - Kendall τ = 0.69 (0.60–0.73).
  - Permutation drop for blink rate: 0.30 (LR), 0.27 (XGB).
  - LR blink coefficient: +5.34.
- **Manipulation check:** Friedman χ²(3) = 82.72, p < 0.001, n = 43. A4 > A1 for 43/43; median
  difference 31.7.
- **Shuffled labels:** mean AUC 0.46–0.53 over 40 runs.

All figures come from `outputs/colet/tables/` (main: `05_results.csv`, `06_comparison.csv`,
`07_importance_*`) and `outputs/colet/tables/extra/` (post hoc).

## 6. Panel-review items: where they stand in v4-6

| Item from `panel-review-handoff.md` | Status in v4-6 |
|---|---|
| F1 blink-AUC swap | ✅ Fixed ("0.89 on raw values and 0.97 after per-person standardization") |
| F3 stale Figures 2 and 4 | ✅ Replaced (`figure2-fixed.png`, `figure4-fixed.png`) |
| D1 headline framing | ✅ (a): Scope, Ch 3 and Discussion state the raw drop and the "calibrated user" reading |
| D2 speech confound | ✅ Ran A3 vs A4 and A1 vs A2 post hoc (Table 16). The claim is now "task condition", not load |
| D3 model comparison | ✅ "No detectable difference at this sample size". The no-equivalence-margin limitation is stated. Default-hyperparameter result reported |
| D4 practitioner recommendation | ✅ Limited to speaking tasks; silent tasks flagged (A2 blink median 0.0) |
| D5 Wilcoxon p = 0.007 | ✅ Reframed as confidence/probability scale, secondary to AUC |
| D6 importance wording | ✅ "Agreement clearest for blink rate"; XGB pupil drop and gain ranking reported |
| D7 COLET-detector speculation | ✅ Removed |
| D8 run-once wording | ✅ "Run once … repeated once, with no change to any setting, after two participant IDs were found to hold the same recordings" |
| Section C wording | ✅ None of the flagged phrases remain ("transportations", "work load", "calibration metrics", "verified ranking", "marginally", etc.). Ref [55] is now cited |
| Equivalence margin | Spoken answer only. Post hoc (`extra/6_equivalence.csv`): P1 is equivalent within ±0.05; P2 is inconclusive. Not in the paper, by decision |
| A2+A3-only standardization | Notes only; not in the paper |

## 7. Known issues and unverified points (not fixed)

These are small and can wait until after the presentation unless the user says otherwise.

1. **Ch 4 manipulation-check paragraph repeats itself.**
   - "In the full released data, all 47 participants also rated A4 higher than A1 (Chapter 3)"
     is followed by "All 43 participants rated A4 higher than A1…".
   - Suggest merging the two sentences.
   - The 43/43 and 31.7 figures match `03_manipulation_check.json`. The 47/47 claim was not
     re-checked.
2. **Ch 5 limitation: "A4 recordings were usually longer than A1 (median 51 vs 37 s)".**
   - Not verified. The local Python has no pandas, so it could not be checked here.
   - Table 9 and the earlier check give **valid time** medians of 34.8 s (A1) and 48.4 s (A4).
   - If 51/37 is total recording length, say so. If it was meant as valid time, change it to
     48/35.
3. **Ch 5 summary item 3** says both "Neither model proved superior consistently" and "there
   was no evidence that XGBoost performed better". This is redundant but not wrong.
4. **Table 7 "Features" cell** reads "1, 2, 10 or 2 features" for the baselines row, which is
   confusing.
5. **The markdown export has `undefined` rows under every table header.** This is a conversion
   artefact and not in the docx. Don't copy those rows into the paper.
6. **Still open from the panel review:**
   - Section D framing points: no hypotheses; Objective 1 is a task.
   - No search protocol for the novelty claim.
   - Mixed citation style; 2026 references not verified.

   All are deliberate scope choices or untouched; prepare spoken answers.
7. **Word lock files.** `~$RP-…-v4-6.docx` exists, so v4-6 is probably open in Word. Close it
   before editing or regenerating.

## 8. Presentation prep (today)

| File | Use |
|---|---|
| `docs/mock-panel-questions.md` | Questions from easy (Level 1) to hard (Level 5), each with a short spoken answer, plus answering rules. Written for v4-5; still valid for v4-6 |
| `docs/panel-questions.md` | The longer panel Q&A bank |
| `docs/CSRP2-Initial-Presentation.pptx` | Deck (commit `c9fa61a`) |
| `src/demo.py` | Live demo script |
| `docs/extra-analyses-notes.md` | Post-hoc analyses 1–7: method, results, caveats, how to reproduce |

**Points the panel is most likely to press** (say the limits first):

1. **0.99 is a within-person ranking with calibration.**
   - Raw: 0.85–0.88.
   - A2 as the only reference: 0.90–0.92.
2. **It is mostly speaking.**
   - Blink rate alone: 0.963.
   - With speaking held constant (A3 vs A4): 0.60–0.66.
3. **"No difference" is not "equal".**
   - No margin was set in advance.
   - P1's upper bound is 0.002; P2's CI is wide.
4. **Tuning added nothing at n = 88.** Default hyperparameters gave 0.991 / 0.977.

## 9. Files

| File | What it is |
|---|---|
| `docs/CSRP-Chapter-1-to-5-draft-COLET-v4-6.docx` | **The paper** |
| `docs/figure1-framework.png`, `figure2-fixed.png`, `figure4-fixed.png` | Current Figure 1, 2 and 4 sources |
| `docs/changes-v4-2-1-to-v4-4.md` | Change notes for v4-2-1 to v4-4 |
| `docs/panel-review-handoff.md` | 7 Nov panel-review check and decisions D1–D11 |
| `docs/adviser-questions.md` | Q1 open (report the unstandardized result: now done in the paper); Q2 decided (Route A) |
| `docs/methodology-colet.md`, `rrl-decision-log.md`, `researcher-notes.md` (N0–N31; next is **N32**), `leakage-audit.md` | Method record. Keep it consistent with the paper |
| `src/` (`run_colet.py`, `config.py`, …) | Main pipeline |
| `docs/colet-eda/extra_*.py` | Post-hoc analysis scripts (outputs in `outputs/colet/tables/extra/`) |

## 10. Working rules (unchanged)

- **Never commit.** Give `git add` commands and a suggested commit message.
- **RRL:**
  - Every method claim needs a verified peer-reviewed source.
  - Label the team's own choices as such (Ch 3 "Design choices").
- **Labelling and scope:**
  - Post-hoc analyses stay labelled as post hoc and never replace P1/P2.
  - Ask before new pipeline runs, package installs or long jobs.
- **Version names:** follow the existing pattern (`…-COLET-v4-7.docx`).
- **Record changes:**
  - Add a change note per version.
  - Update `rrl-decision-log.md` and `researcher-notes.md` for any method decision.

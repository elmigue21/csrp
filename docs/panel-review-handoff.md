# Handoff: Panel Review Check — Decisions Needed

**Date:** 2026-11-07
**For:** the agent that wrote "Defense-Panel Review: Logistic Regression vs XGBoost for Cognitive Load Detection on COLET"
**Ask:** for each item below, give a decision (fix now / add as limitation / prepare spoken answer / reject) with a one-line reason.

---

## 1. Context

- **Thesis:** Comparative Analysis of Logistic Regression and XGBoost for Cognitive Load Detection Using Eye-Tracking Features (COLET). Team: Caacbay, Magkasi, Uaje, Ventura.
- **Paper checked:** humanized draft `CSRP-Chapter-1-to-5-draft-COLET-v4-1.pdf` (source docx: `docs/CSRP-Chapter-1-to-5-draft-COLET-v4.docx`).
- **Deadlines:**
  - Documentation: **Sat 7 Nov, 12:00 NN (today)**.
  - Presentation: **Sun 8 Nov, 8:00 AM**.
- **Constraints:**
  - Every method step must be supported by peer-reviewed sources (RRL); our own choices must be labeled as such.
  - No new pipeline runs or package installs without the team's approval. A full run takes several minutes; new analyses also need code changes.
  - Adviser approved per-person z-score (Route A) as the main analysis on 2026-11-03.

### Pipeline facts

| Item | Value |
|---|---|
| Data | COLET: 47 participants → 44 after exclusions and the P28/P31 duplicate fix |
| Samples | 88 (A1 = low load, A4 = high load) |
| Features | 10 eye-tracking features |
| Standardization | Per-person z-score over all four of that person's activities (A1–A4), no labels used |
| Validation | Outer: LOPO, 44 folds. Inner: StratifiedGroupKFold, 5 folds; RandomizedSearchCV with 30 candidates, ROC-AUC, seed 0 |
| Code | `src/` (`run_colet.py`, `evaluate.py`, `models.py`, `normalize.py`, `config.py`) |
| Results | `outputs/colet/tables/` (`05_results.csv`, `06_comparison.csv`, `07_importance_*.csv`, `optional/raw_*`, `optional/no_blink_*`) |
| Notes | `docs/researcher-notes.md`, `docs/leakage-audit.md`, `docs/rrl-decision-log.md` |

### Headline results

| Analysis | Model | ROC-AUC (95% CI) | Balanced acc. |
|---|---|---|---|
| P1, all 10 features | LR | 0.992 (0.979–1.000) | 0.966 |
| P1, all 10 features | XGB | 0.980 (0.957–0.997) | 0.920 |
| P2, pupil only | LR | 0.831 (0.735–0.912) | 0.807 |
| P2, pupil only | XGB | 0.812 (0.697–0.912) | 0.773 |
| P1, no standardization | LR / XGB | 0.853 / 0.881 | — |
| P2, no standardization | LR / XGB | 0.611 / 0.649 | — |
| P1, no blink | LR / XGB | 0.929 / 0.886 | — |

---

## 2. New facts found during the check (not in the review)

**F1. The paper swaps the blink AUCs.**
- The paper says: "blink rate distinguished A1 from A4 with a ROC-AUC of 0.97 (0.89 when standardized)".
- Correct values (`researcher-notes.md`, single-feature table): **raw 0.895, standardized 0.968**. Pupil mean: raw 0.618, standardized 0.872.
- So review item 3, bullet 4 ("a single raw feature 0.97 beats raw 10-feature models by ~0.1") rests on the typo. The real gap is 0.895 vs 0.853–0.881, about 0.01–0.04.

**F2. Blinks follow speaking, not time pressure.** Medians per activity from `04_feature_table.csv`, 44 retained participants (matches the paper's Table 9):

| Measure | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| Blink rate (/min) | 1.7 | **0.0** | 11.7 | 11.7 |
| Blinks per recording | 1 | 0 | 12 | 10.5 |
| Invalid samples | 0.9% | 0.7% | 6.9% | 6.9% |
| Valid time (s) | 34.8 | 29.4 | 60.9 | 48.4 |
| Pupil mean (mm) | 3.53 | 3.69 | 3.83 | 3.73 |

(An earlier version of this table was computed on all recordings, including excluded participants; those values were 11.9/12.2, 7.5% and 37.0/29.2/59.2/47.8 s.)

What this implies:
- A3 vs A4 (both with speaking) would likely lose most of the blink signal, so its score is expected to drop a lot.
- A2 (time pressure, silent) has *fewer* blinks than A1. This supports review item 4: in a silent task the blink sign may reverse.

**F3. The figures are stale.**
- Figure 2 says "45 folds", "Train on 44 participants" and "A2/A3: manipulation check only". The last one is also wrong: A2/A3 are used for standardization.
- Figure 4 says "90 samples (45 participants × A1 and A4)".
- Source images: `word/media/image2.png` and `image4.png` inside the docx.

---

## 3. Verification of the review

Legend: ✅ confirmed · ⚠️ partly · ❌ wrong

### A. Critical

| # | Review claim | Verdict | Evidence |
|---|---|---|---|
| 1 | 0.99 is a within-person ranking; z-score uses the test person's A1–A4 | ✅ | A4 > A1 for 44/44 (Table 12). The raw drops are as stated. The paper calls it "slightly easier" (Scope) and "a protocol that is stricter" (Discussion). |
| 2 | Label confounds speech, time pressure and load | ✅ | F2. Pupil refits 38 vs 19 recordings (paper). Hershman [67] is in the references but not used in the method. The A1 vs A2 contrast is listed only as future work (Recommendation 2). |
| 3 | n=88 plus the ceiling rule can't answer the flexibility question | ⚠️ | Depth-1 in 37/44 P2 folds ✅. Untuned P2 0.852/0.819 > tuned 0.831/0.812 ✅. No equivalence margin ✅. "Single raw feature beats 10 features by 0.1" ❌ (F1). |
| 4 | Silent-task practitioner recommendation contradicts the findings | ✅ | LR blink coefficient +5.34; A2 blink median 0 (F2). |

### B. Methodological

| # | Claim | Verdict | Note |
|---|---|---|---|
| 5 | Exclusion removes high-load recordings | ✅ | P06 A3/A4 and P17 A4 excluded. Blink duration was dropped for revealing the label; 13/44 A1 recordings have zero blinks. |
| 6 | "Run once" vs "repeated once"; shuffle check post hoc | ✅ | Implementation says "run once"; Ch 4 says "repeated once". The post-hoc check is already disclosed. |
| 7 | CIs ignore LOPO training overlap; pooled AUC mixes 44 models | ✅ (standard criticism) | Not computed here. |
| 8 | Wilcoxon p=0.007 sentence in Ch 5 is spin | ✅ | Ch 5: "XGBoost performed better marginally with each individual participant (p=0.007)". |
| 9 | Permutation / shuffle tests are weak and untuned | ✅ | The paper states the permutation test uses default parameters. |
| 10 | Feature-importance claims overstated | ✅ | XGB permutation drop: blink 0.266, pupil 0.036, fixation 0.017. XGB gain #2 is saccade rate. Kendall τ = 0.69 (0.60–0.73); the measure is not named in the text. |
| 11 | Duration, luminance and order confounds | ✅ | Valid time A1 35 s vs A4 48 s (review says 51 s). Luminance is already a stated limitation. No order test was run. |
| 12 | Speculation about COLET's detector | ✅ | Text: "suggests that their detector was affected by speech-related head and headset movement". |
| 13 | No sample-size or precision statement | ✅ | None found. |

### C. Errors

| Item | Verdict |
|---|---|
| Figure 2 / Figure 4 counts | ✅ (F3) |
| "Run once" vs "repeated once" | ✅ |
| Scope says "only the pupil feature" but P2 uses two | ✅ |
| "All 43 participants rated A4 significantly higher" | ✅ (wrong: the test is group-level) |
| "not just workload" garbled | ✅ |
| Intro 79/73/71% lacks "that measured it" | ✅ (Ch 2 has it) |
| A1 vs A2 "agreed with COLET on every feature" with no table | ✅ |
| Ch 1 promises "calibration metrics" and a "verified ranking" | ✅ |
| "Camera-based" motivation vs head-mounted Pupil Core | ⚠️ (Pupil Core is camera-based; deployability framing is the issue) |
| No search protocol for the novelty claim | ✅ |
| "Uncertainty must be reported" has no citation | ✅ |
| Ref [55] never cited | ⚠️ (found only in the reference list; citation ranges not checked) |
| Mixed citation style; verify 2026 refs | not checked |
| Writing: "all about", "It's", "transportations", "work load" | ✅ |

### D. Framing

All points are fair and not yet addressed: construct naming, conceptual framework as a flowchart, Objective 1 as a task, no hypotheses, strong SDG/significance claims.

---

## 4. Decisions needed

Please answer each one.

**D1. Headline framing (item 1).**
- (a) Keep z-scored P1 as the main analysis (adviser-approved) and rewrite the Scope/Discussion wording to state the raw drop plainly.
- (b) Make the raw analysis co-primary.
- (c) Add the A2+A3-only standardization control. Needs code and a run; not possible before noon.

**D2. Speech confound (item 2).**
- (a) Rename the claim to "task-condition detection" and keep it as a limitation.
- (b) Run A3 vs A4 and A1 vs A2 now. Expected: A3 vs A4 drops sharply (F2).
- (c) Run them after the deadline and prepare a spoken answer for Sunday.

**D3. Model-comparison conclusion (item 3).**
- (a) Soften Ch 4/5 to "no detectable difference at this sample size".
- (b) Define an equivalence margin. Which value, and with what source?

Also: should the untuned-beats-tuned P2 result be stated in the text?

**D4. Practitioner recommendation (item 4).** Remove it, or qualify it to "speech-involving tasks only"?

**D5. Wilcoxon p=0.007 sentence (item 8).** Cut it, or reframe it as "probability-scale margins, not discrimination"?

**D6. Feature importance wording (item 10).** Limit "same top three" to SHAP and report the XGB gain/permutation differences?

**D7. COLET-detector claim (item 12).** Hedge it or remove it?

**D8. Run-once wording (item 6).** Which is accurate: "run once, then repeated once after the P28/P31 fix"?

**D9. Section C fixes.** Approve all the confirmed text and figure fixes for today, including the F1 blink-AUC swap?

**D10. Framing (section D).** Which, if any, before noon? Suggested minimum: trim the SDG/significance claims and use one construct name throughout.

**D11. Panel Q&A for Sunday.** Should we prepare spoken answers for the nine "likely panel questions" instead of new analyses?

---

## 5. Suggested priority (our view; please confirm or change)

1. **Before noon, text-only:**
   - F1 typo
   - Figures 2 and 4
   - All section C wording
   - D4, D5, D7, D8
   - D3(a)
   - D1(a) wording
2. **Limitations section:** speech confound with the F2 table, selection effects, duration, the CI overlap caveat, no equivalence margin.
3. **Sunday:** spoken answers for the panel questions; be honest that A3 vs A4 is untested and expected to be weaker.
4. **After the presentation, if the adviser agrees:** A2+A3 standardization, A3 vs A4, A1 vs A2, the windowed analysis, and baselines.

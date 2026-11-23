# Thesis changes needed after the defense

**Status (2026-11-11, later):** done in v4-7; see `changes-v4-6-to-v4-7.md`. Kept as the record
of what had to change.

**Original status:** checklist, 2026-11-11. Nothing in the thesis has been edited yet. This lists every
place in the latest draft (`CSRP-Chapter-1-to-5-draft-COLET-v4-6.md`, line numbers from that
file) that no longer matches the post-defense method.

**Source of truth:** `post-defense-data-processing.md` (data, final) and
`post-defense-modeling-plan.md` (modeling, decided; results recorded 2026-11-11 in
`outputs/final/results/`, so Ch. 4 can now be rewritten).

**Wording rule:** record the removal of per-person standardization neutrally ("removed after
the proposal defense"). Never say it inflated results.

## What changed, in one table

| | Defended draft | Now |
|---|---|---|
| Participants / rows | 44 / 88 | Same (the draft is already correct) |
| Features | 10 fixed features | 10 signals × 17 statistics + 3 rates = 173 extracted, **140** after D1–D10 |
| Per-person standardization | Every feature, over A1–A4 | **Removed**; LR scaling fitted on training people inside each fold |
| A2 / A3 | Manipulation check + standardization | Manipulation check only |
| Feature set | Fixed in advance | Top-N chosen inside each fold (modeling plan, MD1–MD2) |
| Saccade mean velocity, saccade duration | Dropped (duplicates) | Kept |
| Blink duration | Dropped | Still dropped (D1: blank in 43/44 A1 vs 36/44 A4 recordings under the 20-value rule) |
| Gaze position | Excluded | Kept, 22 features tagged `position` |
| Cleaning C1–C10 | — | **Unchanged** |
| Analyses | P1 (all features) + P2 (pupil only) | **P1 only** (MD3): remove P2 everywhere |
| All model results | From the 10-feature, standardized run | **Must be re-run** |

## Priority order
1. **Ch. 3: Feature Extraction, Table 5, Per-Person Standardization (L239–254, L265).** These
   describe the whole old method.
2. **Ch. 4, Ch. 5, Appendices A/B:** every model number is from the old run. **Re-run, don't
   reword.** Wait for the modeling stage.
3. **Figures 1, 2, 4 and Tables 3, 5, 7:** redraw or rebuild.
4. **Ch. 1 (L33–72) and the definitions of terms.**

## P2 (pupil-only analysis) is removed everywhere (MD3, 2026-11-11)

Remove P2 from: Ch. 1 L60 (analysis list) and any objective that names it; Ch. 3 L275 and
Table 7 (L281–288); Ch. 3 Figure 2 (P2 box); Ch. 4 L380 (P2 results) and P2 rows in Tables 12–13
and Figures 6–7; Ch. 5 findings and conclusions that cite P2. The pupil's role is reported
through SHAP and selection count instead.

## Permutation importance is removed (MD4, 2026-11-11, scope kept small)

Feature importance is now **SHAP + selection count** only. Remove permutation importance and
the SHAP-vs-permutation ranking agreement (Kendall's τ): Ch. 1 L52 (Objective 5), Ch. 3 L273,
Ch. 4 L394–411 (Figure 8, Table 14), Ch. 5 findings that cite τ. Add selection count where
importance is reported.

## Chapter 1

| Line | Now says | Change to |
|---|---|---|
| 33 | "ten features are computed for each recording" | Ten eye signals are each summarized into 17 statistics, plus 3 rates (173 features; 140 kept after data-quality checks). |
| 33 | "Each person's features are standardized using … all four activities" | Delete. Any scaling a model needs is fitted on the training participants only, inside each fold. |
| 44 | "(pupil, blink, fixation, saccade and gaze-dispersion measures)" | Summary statistics of pupil, gaze, fixation, saccade and blink signals, plus event rates. |
| 52 | Objective 5 (importance) | Optionally add: report how often each feature is selected across folds. |
| 56 | "the per-person standardization … is stated as a limitation" | All scaling and feature selection are fitted on training participants only. |
| 60 | "extraction of ten eye-tracking features, participant-wise normalization" | Flattening of ten signals into 140 features, with top-N features chosen inside each fold. |
| 60 | P1 "all ten features" / P2 "the two pupil features" | P1: all 140 features (top-N in fold). P2: pupil only (16 features). P3 if adopted (MD3). |
| 60 | "the second without participant-wise normalization"; post-hoc sentence | Remove the no-normalization analysis; keep post-hoc only if re-run. |
| 64 | Standardization discussion and AUCs (0.98–0.99, 0.85–0.88, 0.90–0.92, 0.60–0.66, 0.68–0.77) | Delete the standardization part. **Re-run** or remove the other numbers. |
| 70 | "The 10-dimensional Eye Tracking Feature set … normalized for each person" | Each signal is flattened into summary statistics (140 features); no per-person normalization. |
| 72 | "trained and tested with … LOPO" | Add: with the top-N features selected inside each training fold. |

### Definition of terms

| Line | Term | Change to |
|---|---|---|
| 79 | Blink rate: "one of the ten input features" | One of the three rate features. |
| 93 | Fixation: "fixation rate and fixation duration are input features" | Fixation rate and 15 statistics of fixation duration. |
| 97 | Gaze Dispersion | Redefine as "Gaze angle (horizontal/vertical)", summarized by 17 statistics; location statistics are tagged `position`. |
| 107 | Per-Person Standardization | Delete. Add "Summary statistics (flattening)" and "Top-N feature selection". |
| 111 | Pupil: "mean and standard deviation … are input features" | 16 statistics of pupil diameter. |
| 115 | Saccade: "rate, amplitude and peak velocity" | Saccade rate plus statistics of amplitude, peak velocity, mean velocity and duration. |

## Chapter 2

| Line | Now says | Change to |
|---|---|---|
| 148 | "many cross-participant workload studies standardize features within each participant … should be acknowledged" | Keep the citations as background; change the last sentence to explain that this study fits all scaling on training participants only. Low priority. |

## Chapter 3

| Line | Now says | Change to |
|---|---|---|
| 175 | **Figure 2** boxes: A2/A3 for standardization; P1 all 10 features; P2 pupil mean and SD; per-person standardization | **Redraw:** A2/A3 for the manipulation check only; P1 = 140 features; no standardization box; "top-N selection on training participants" inside the LOPO box. |
| 179 | "P1 with all ten features"; "All thresholds … fixed before any model was trained" | P1 with 140 features (top-N in fold). Add: data-quality decisions D1–D10 were made before any model was trained on the new table. |
| 193 | "one row of ten features for each of the 88 … recordings" | One row of 173 extracted features (140 used) for each of the 88 recordings. |
| 210, 211 | Table 3, A2/A3: "Manipulation check; included in within-person standardization" | "Manipulation check only". |
| 214 | "…and computing features" | "…and flattening each signal into summary statistics". |
| 216 | **Figure 4** boxes: "Compute 10 features"; "Compute 5 features" | **Redraw:** "Flatten 10 signals × 17 statistics + 3 rates (173 → 140 after D1–D10)"; fallback = drop all fixation/saccade features. |
| 239 | Fallback "five features … (pupil mean, pupil SD, blink rate and gaze dispersion" | If the check fails, all fixation, saccade and eye-speed features are removed for everyone. |
| 241 | "Ten features are computed …"; "The feature set is fixed in advance and is the same in every fold" | Ten signals × 17 statistics + 3 rates; top-N chosen inside each training fold, with selection frequency reported. |
| 243–251 | **Table 5** (10 features) | **Rebuild:** the 10 signals with unit and type, the 17 statistics, the 3 rates (`post-defense-data-processing.md` §4.2–4.4). |
| 252 | Blink duration reason "(13 of 44 A1 … none of the A4"; saccade duplicates excluded; "higher-order statistics … omitted" | Use D1 (43/44 A1 vs 36/44 A4 blank under the 20-value rule; blink rate kept). Saccade mean velocity and duration are now kept. Replace the "omitted" sentence with the drop rules D6a, D8, D9. Keep the luminance sentence. |
| 253–254 | **Section "Per-Person Standardization"** | Replace with a short "No per-person standardization" note: features keep their units; A2/A3 are not in the feature table; LR scaling is fitted in-fold. |
| 256 | "there were no missing values in any of the features" | Re-check on the 140-feature table (currently 0 blanks), then reword. |
| 265 | "Per-person standardization is the one exception …" | Delete. Add: top-N selection is also fitted on training participants only. |
| 273 | Kendall's τ and weight stability | Add selection frequency and per-signal grouping of importance (modeling plan MD4). |
| 275 | P1 "all ten features"; P2 "two pupil features" | Per MD3. |
| 281–288 | **Table 7** (analysis plan) | **Rebuild:** new P1/P2 (and P3); drop the "without per-person standardization" row; mark old post-hoc rows re-run or remove. |
| 291 | Run history | Add the post-defense extraction (`src/run_extract.py`, `outputs/final/`) and the modeling re-run. |
| 295 | Design-choices list | Add the 20-value minimum (D2) and drop rules D6a, D8, D9. |

## Chapter 4: re-run everything that comes from a model

| Line | Item | Action |
|---|---|---|
| 300 | Run history | Describe the post-defense run. |
| 302–312 | NASA-RTLX check, Figure 5 | **Keep** (unchanged). |
| 316–325 | Data quality, Table 9 | **Keep** (cleaning unchanged). |
| 326 | "None of the ten features had missing values" | "None of the 140 model features had missing values". |
| 328 | "P1 utilized all ten features … instead of the five-feature backup" | "so all fixation and saccade features were kept". Table 10 numbers still hold (212–235 ms, 0.93–0.99). |
| 339–354 | Table 11, median feature values "before standardization" | **Re-run** from the new table; drop "before standardization". |
| 356–378 | P1 results, Table 12, Figures 6–7, tuning summary | **Re-run and redraw.** |
| 380 | P2 results | **Re-run.** |
| 382–392 | Table 13; standardization comparison (0.853 / 0.881); blink-rate removal | **Re-run.** Remove the standardization paragraph. |
| 394–411 | Figure 8, Table 14 (importance) | **Re-run and redraw**; add selection frequency. |
| 413 | Shuffled-label check | **Re-run** if kept; drop the standardization comparison. |
| 415–446 | Tables 15–16: A2-baseline centering, A3 vs A4, A1 vs A2 | **Re-run or remove.** A2-baseline centering has no counterpart now. |
| 449–455 | Discussion of standardization | Rewrite after the re-run, neutral wording. |
| 457 | "mean speed of saccades and saccade duration, which were disregarded" | They are now included; overlap handled by grouping importance. |

## Chapter 5

| Line | Item | Action |
|---|---|---|
| 463 | "Ten whole-activity features" | Ten signals flattened into 173 features (140 kept) for 44 participants. |
| 465–477 | Findings 2–6, conclusions | **Re-run** and rewrite. |
| 479, 502 | Limitation: normalization uses all four activities | Delete. Optionally: no test-participant data enter scaling or feature selection. |
| 481 | Speaking limitation numbers | Keep the point; re-run or remove the numbers. |
| 498 | "Without per-person standardization, ROC-AUC fell to …" | Delete that sentence. |
| 504 | Ten-feature vs five-feature fallback | "Whether to keep fixation/saccade features". |
| 510 | Recording length affects SD features | Extend to min, max and range (D3); p05/p95 are the length-robust versions. |
| 522, 528, 530, 536 | Post-hoc analyses and recommendations built on standardization | Rewrite after the re-run; recommendation 3 becomes future work on calibration from a neutral baseline recording. |
| New | Limitations not yet in the draft | Add: (a) pupil quality: 7 A4 recordings keep under 80% of pupil data; P44 A4's right eye reads too small (data doc §3). (b) Gaze-position features may reflect headset or screen position; there is no per-person correction (D10, gaze cv). (c) 140 features on 88 rows: handled by regularization and in-fold selection. |

## Appendices

| Line | Item | Action |
|---|---|---|
| 711–727 | Table A1, XGBoost gain over 10 features | **Re-run.** |
| 729–741 | Table B1, standardization over four activities | **Re-run without standardization, or remove.** |

## Figures and tables

| Item | Status |
|---|---|
| Figure 1 (conceptual framework, only in the .docx) | **Redraw:** 10 signals × 17 statistics + 3 rates (173 → 140); no per-person normalization; top-N inside LOPO |
| Figures 2, 4 | **Redraw** (Ch. 3 rows above) |
| Figures 6, 7, 8 | **Redraw after the re-run** |
| Figures 3, 5 | Still correct |
| Tables 3, 5, 7 | **Rebuild** |
| Tables 11–16, A1, B1 | **Re-run** |
| Tables 1, 2, 4, 8, 9, 10, 17 | Still correct |

## Order of work
1. Decide MD1–MD4 (`post-defense-modeling-plan.md`).
2. Edit Ch. 1, 3 and the definitions now; they only depend on the data decisions, which are final.
3. Run the models, then rewrite Ch. 4, 5 and the appendices from the new results.
4. Redraw Figures 1, 2, 4, then 6–8 after the run.

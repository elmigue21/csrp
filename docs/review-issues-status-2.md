# Independent Review: Round 2 Issues

**Reviewed:** `methodology-colet.md`, `researcher-notes.md`, `review-issues-status.md`,
`colet-eda/` and `rrl-colet-list.md`, as of 2026-10-01 (after the fix pass).

**Round 1:** "COLET Methodology Review — Issues" doc and `review-issues-status.md`.

**Reviewer stance:**
- independent; no repo files were edited, and this file is new;
- **label-blind:** no feature was tested for how well it separates A1 from A4, to keep the
  team's feature selection unbiased.

Updated 2026-10-01.

**Status key** (same as round 1):
- ✅ **Resolved** (no team decision needed).
- 🟡 **Resolved in part;** the rest needs a decision.
- ⏳ **Needs a team decision.**
- ⏸ **On hold / pinned** by the team.

**Severity:** Critical / Major / Moderate / Housekeeping, as defined in round 1.

---

## 1. Verdict

**Conditional approval, with fewer conditions than in round 1.** The design and validation
are strong, and the data checks are real. The main remaining threat is **construct validity:
the spoken secondary task now runs through most of the feature set.**

| Aspect | Round 1 | Round 2 |
|---|---|---|
| Design and validation | Strong | Strong |
| RRL authenticity | Clean (116/116) | Clean (122/122) |
| Data grounding | Untested | Good: real checks done |
| Construct validity (speech) | Major | **Main threat**: up to 6 of 10 features affected (V2-1) |
| Feature validity (event detection) | — | Unresolved: no pass/fail rule (V2-2) |
| Scope vs contribution | Too wide | Too thin unless S3 is required (V2-5) |
| Defensibility | Not yet | Close: fix V2-1, V2-2, V2-3 and V2-5 |

**Conditions for approval:**
1. One talking-robustness analysis is **required** (V2-1).
2. Pre-stated pass/fail criteria and a fallback for the gaze-event features (V2-2).
3. A rule for 3D eye-model refits (V2-3).
4. S3 required, or removed from the stated contributions (V2-5).

## 2. Checks run in round 2

| Check | Result |
|---|---|
| Crossref, all DOIs in `rrl-colet-list.md` | **122/122 resolve;** author, year and volume match. **No fabricated references.** |
| New entries 129–135 (incl. Hart 2006, CL8) | Verified |
| Non-DOI entries (Kahneman, Bishop, Holmqvist, Byers 1989, …) | Real, well-known works |
| Year notes | Oppelt (LB7) and Faraji (PP7): online 2022, issue 2023. Cite one year consistently (R7, pinned). |
| Ethics number 110/12-02-2021 | Matches the COLET PDF |
| `norm_pos` coordinate frame (M1) | Conclusion holds: world-camera relative |
| Gaze columns available | `gaze_point_3d_*`, plus **`gaze_normal0/1_*`** and `eye_center0/1_3d_*` |
| 3D eye-model refits (`model_id`) | **63 of 188 recordings** have more than one model for an eye (max 10) |
| Blink events | Pupil Labs confidence-based detector (`filter_response` column) |

Crossref confirms that a paper exists and that its metadata is right. It does not confirm
that a claim matches the paper; the abstract-only (A/M) rows in Appendix A still carry
that risk.

## 3. Round-1 issues: reviewer verdict

| # | Issue | Verdict |
|---|---|---|
| M1 | Coordinate frame | ✅ Checked in the data. D-M1 decided; see V2-2 for the gaze-normal option. |
| M2 | Speech confound | 🟡 Data-loss difference measured (0.9% vs 7.3%, 41/45). Escalated as V2-1. |
| M3 | Missingness leaks the label | ✅ Blink duration dropped (D-M3). The right call. |
| M4 | Normalization exploits the balanced design | 🟡 Choice (b) with disclosure is defensible. Residual risk: V2-8. |
| M5 | P1 ≈ S1 | ✅ Justified in N0b. S1's role is unclear: V2-H6. |
| M6 | Windows and durations | ✅ Whole-activity unit; durations measured |
| M7 | S3 replication + ladder | 🟡 Well designed (§16), but optional: V2-5 |
| M8 | Novelty claims | 🟡 Rephrased; search protocol added. Two claims at risk: V2-6. |
| M9 / C5 | Adviser sign-off | ⏸ Still pending; all decisions were made ahead of it |
| R1 | S2 bins | 🟡 Bins fixed; comparability still overstated: V2-7 |
| R2–R6 | RRL fixes | ✅ |
| R7, R8 / C7 | Years, author lists, full-text reads | ⏸ Pinned |
| N1 | Pruning varies per fold | ✅ Fixed feature set |
| N2 | Saccade peak velocity | 🟡 Kept as Core (D-N2). See V2-M5. |
| N3, N4, N8, N9 | Entropy, permutation, AUC, time-on-task | ✅ |
| N5 | Scope | 🟡 Trimmed, perhaps too far: V2-5, V2-8 |
| N6 | LR penalty | ✅ Elastic net (D-N6) |
| N7 | Framing | ✅ "LR ≈ XGB is legitimate" |
| N10 | Ethics and licence | ✅ Verified against the PDF |
| H1–H10 | Contradictions | ✅ In the methodology. New ones in the other files: §6. |
| C8 | Stimulus luminance | ⏸ Pinned |

## 4. New issues: Major

### V2-1. The speech confound runs through most of the features ⏳ (Critical)

**What:**
- COLET's blink events come from Pupil Labs' **confidence-based** blink detector
  (`filter_response`).
- Invalid samples rise from 0.9% (A1) to 7.3% (A4) under talking (N15). Talking-related
  tracking dropouts can therefore be logged as **blinks**.
- So P11's rule that data quality is never a feature is partly undone: data quality comes
  back in through blink rate.
- On gaze, jaw movement and headset slippage (DS4, Niehorster 2020) add jitter, which looks
  like extra, faster saccades, and offset drift, which inflates dispersion.

**Affected Core features (up to 6 of 10):** blink rate, saccade rate, saccade amplitude,
saccade peak velocity, gaze dispersion x and y.

**Why it matters:** the panel question "is the model detecting load or talking?" becomes
hard to answer, and feature importance will likely rank these features at the top.

**Recommended:**
- Make **one talking-robustness model required** (not optional), for example:
  - pupil-only (pupil mean, SD); or
  - no blink and no saccade features.
- **Label-free check:** within each condition, correlate blink rate with invalid-sample %
  per recording. A strong correlation means blink rate is partly data loss.
- Name the construct in Chapters 1 and 3 as dual-task load with a spoken secondary task
  (already in N0b; carry it into the title framing and the scope).

### V2-2. Gaze-event features have no pass/fail rule ⏳ (Major)

**What:**
- §6 marks the fixation and saccade features "Core, after the event detection is
  validated", but no criteria or fallback are stated.
- The first pass gave **saccade rate 4.0/s > fixation rate 3.4/s** (EDA README). These
  should be roughly equal, so this suggests noise saccades.
- **The main-sequence check is weak on its own:** noise "saccades" also show peak velocity
  rising with amplitude, because both come from the same velocity trace.
- `gaze_point_3d` depends on the binocular vergence depth estimate, which is noisy.
  **`gaze_normal0/1`** (eye-centred gaze directions) are in the data and measure eye
  rotation more directly.

**Recommended (pre-state before running):**
1. **Pass criteria**, for example:
   - saccade-to-fixation count ratio about 1;
   - median fixation duration 150–400 ms;
   - main-sequence slope within the published range;
   - the two conversion methods (D-M1) agree within a stated tolerance.
2. **Fallback:** if the criteria fail, P1 runs on the 5 non-event features (pupil mean, pupil
   SD, blink rate, dispersion x and y).
3. Use `gaze_normal0/1` as the primary or as a third check, and justify the choice.

### V2-3. 3D eye-model refits can step the pupil size ⏳ (Major)

**What:**
- 63 of 188 recordings contain more than one 3D eye model (`model_id` changes; up to 10 in
  one recording).
- `diameter_3d` can jump when the model refits. P7 does not handle this.

**Recommended:**
- Check whether refits differ by condition. This is a data-quality check, like invalid %.
- Add a rule, for example:
  - compute pupil statistics within model segments; or
  - drop samples near a refit; or
  - cross-check with 2D `diameter` (px).
- Report refit counts in the retention table (P14).

### V2-4. P1 may hit a ceiling ⏳ (Major)

**What:**
- COLET's published means show blink rate going from about 3 to 14 per minute (Table 4).
- With per-person z-scoring and one A1/A4 pair per person, P1 reduces to "which of my two
  recordings blinks more".
- Both models may reach AUC ≥ 0.95. Then the LR-vs-XGBoost comparison cannot differ, and
  importance will say "blinks", which is partly V2-1.
- The reviewer did not compute this (label-blind). It follows from the published values.

**Recommended:**
- Pre-state how a ceiling result is interpreted.
- The V2-1 talking-robust model is where the models can actually differ, so it doubles as
  the informative comparison.

### V2-5. Contribution vs scope contradiction ⏳ (Major)

**What:**
- §1 lists "quantify how much COLET's published accuracies depend on its random split" as
  the secondary contribution.
- D-N5 and §16 make S3 **optional**.
- Without S3, the novelty is "LR vs XGBoost + SHAP on COLET with LOPO", which is thin for a
  panel.

**Recommended:** make S3 (at least Run 1 vs Run 2) **required**. It costs two runs on the
same 90 samples. Otherwise remove it from §1.

### V2-6. Two gap claims rest on abstract-only reading ⏳ (Major)

| Claim (§1) | Risk | Action |
|---|---|---|
| "No peer-reviewed COLET study used participant-held-out validation" | Fenoglio 2023 ×2 (federated learning; abstract only) probably evaluates on held-out clients or users | Read both full texts; reword or keep |
| "None combined LR-vs-XGBoost with LOPO and an importance comparison between the two models" | X1 Rolon-Merette 2026 is listed as "LR vs XGB, LOPO" | Confirm it does not report importance for both models |

One counterexample found at the defense would sink the claim.

### V2-7. S2 comparability still overstated ⏳ (Major)

**What:**
- COLET's 0.88 (GNB, C1 vs C3) used the random-split, all-data-selection protocol.
- S2 under LOPO is **not comparable** to it, only under the S3 (COLET) protocol.
- RTLX bins are not balanced per person, so some LOPO test folds hold one class only, and
  per-person metrics are undefined.

**Recommended:**
- State "comparable only under COLET's protocol"; or
- run S2 under both protocols;
- report pooled metrics only.

## 5. New issues: Moderate

| # | Issue | Recommended |
|---|---|---|
| V2-8 | The no-normalization run (S4) is optional, but N7 cites 3–13 points of inflation (NM5). A panel will ask how large the inflation is here. | Make S4 required; it is one run |
| V2-9 | XGBoost search space (depth 2–6, 100–500 trees) is large for 88 training samples; the ranges come from larger datasets (X2, X14) | Consider depth 1–3 and fewer trees; keep the budget equal |
| V2-10 | Kendall τ over 10 features has a very wide confidence interval | Report a CI or a permutation p-value with τ |
| V2-11 | The 35% exclusion cut was chosen after inspecting the invalid-sample distribution (label-blind, which is fine). §4 says "fixed before any model is run". | Say "chosen after label-blind quality inspection" in Chapter 3 |
| V2-12 | Saccade peak velocity as Core (D-N2): COLET's direction is opposite to the literature (N17), and talking jitter is a likely cause (V2-1) | Team's call; prepare the answer, or move it to exploratory |
| V2-13 | Correlations used to drop features (ρ = 0.95, 0.83) come from the discarded first-pass detector (N21 note) | Recheck with the final pipeline before the feature list is final |

## 6. Housekeeping: contradictions between files

| # | Where | Problem |
|---|---|---|
| V2-H1 | `review-issues-status.md` | Lists M3, M4, N2, N5, N6 and C4 as "need decision"; `methodology-colet.md` says all are decided. M7 points to "§9 / S3"; it is now §16. |
| V2-H2 | researcher-notes N13 | Says D-M1 is "pending"; N18 and methodology §5 say it is decided |
| V2-H3 | researcher-notes N1; `colet-eda/README.md` | Point to "methodology-colet.md §3c", which no longer exists (now §5) |
| V2-H4 | N1 and the EDA README vs N18 and §5 | N1: the detector "must reproduce COLET's Table 4". EDA README: "14° is the target". N18 and §5: Table 4 is compared descriptively only. Keep N18. |
| V2-H5 | `colet-eda/README.md` "Suggested decisions" | C4 set still includes blink duration; mark it superseded by D-M3 |
| V2-H6 | methodology §11 | S1 is "Secondary (D-M5: kept alongside P1)", but D-N5 says everything except P1 is optional |
| V2-H7 | `review-issues-status.md` M2 vs researcher-notes N3 | M2 says "revisit C2b?"; N3 says blinks stay (C2b). Unresolved. |

## 7. Priority checklist

- [ ] V2-1: make one talking-robustness model required; run the label-free blink vs invalid-% check
- [ ] V2-2: write the event-detection pass criteria and the 5-feature fallback; decide on `gaze_normal0/1`
- [ ] V2-3: check refits by condition; add a pupil refit rule to P7
- [ ] V2-5: make S3 Run 1 vs Run 2 required, or remove it from §1
- [ ] V2-6: read Fenoglio ×2 and Rolon-Merette in full; adjust the gap claims
- [ ] V2-7: qualify S2 comparability
- [ ] V2-4: pre-state the ceiling interpretation
- [ ] V2-8: make S4 (no normalization) required
- [ ] V2-9 to V2-13: moderate items
- [ ] V2-H1 to V2-H7: align the files
- [ ] C5: adviser sign-off

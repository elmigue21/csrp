# Independent Review: Round 4 Issues

**Reviewed:** `methodology-colet.md`, `researcher-notes.md` (N24), `rrl-decision-log.md`
(V3-x rows), `review-issues-status.md`, `HANDOFF.md` and `colet-eda/README.md`, as of
2026-10-01, plus label-blind checks on `data/colet/parquet/`.

**Earlier rounds:** round 1 (review doc), `review-issues-status-2.md`,
`review-issues-status-3.md`.

**Reviewer stance:**
- independent; no repo files were edited, and this file is new;
- label-blind: no feature was tested for how well it separates A1 from A4;
- **Critical and Major only**, and each fix is the **smallest one that keeps P1/P2 valid**
  (the team's scope rule, N23/N24).

**Code:** not reviewed. `src/` still holds the GAZELOAD code, which matches `HANDOFF.md`
("pipeline not yet written").

Updated 2026-10-01.

**Status key** (same as before): ✅ Resolved · 🟡 Resolved in part · ⏳ Needs a team
decision · ⏸ On hold / pinned.

---

## 1. Verdict

**The round-3 fixes (V3-2 to V3-6) are applied correctly,** and N24's scope reasoning is sound.

**One Critical issue was found in the gaze input itself (R4-1).** It was found by testing the
V3-4 fix against the data.

**All three remaining fixes are minimal:**
- R4-1 is one input column changed and one step **removed**, so the scope shrinks.
- R4-2 is one phrase.
- V3-1 is one sentence.

After these three, the reviewer considers the methodology **frozen.** Nothing further is
added unless the data shows another Critical problem when the pipeline runs.

| | Round 3 | Round 4 |
|---|---|---|
| Critical | 0 | 1 (R4-1) |
| Major | 2 | 1 new (R4-2, wording) + V3-1 still open |
| New analyses requested | 0 | **0** |

## 2. Round-3 issues: reviewer verdict

| # | Team outcome | Verdict |
|---|---|---|
| V3-1 | Open (team chose not to add the refit-free P2 run) | ⏳ Minimum fix still needed: §4, item 3 |
| V3-2 | Importance still reported from P1 under the ceiling rule (§11) | ✅ |
| V3-3 | Ratio 0.8–1.25; conversions within ±15%; any failure → fallback | 🟡 The ±15% rule is superseded by R4-1 (the two conversions are not independent) |
| V3-4 | Gaze spread computed after the > 1000°/s rejection | 🟡 **Does not remove the artifact** (R4-1) |
| V3-5 | Karunathilake dropped; Stolte added; Hefron marked EEG | ✅ |
| V3-6 | Status and wording fixes | ✅ |

## 3. Issues

### R4-1. `gaze_point_3d` is unreliable; the V3-4 fix does not remove the artifact ⏳ (Critical)

**Label-blind data checks:**

| Check | Result |
|---|---|
| The "102° artifact" | Not one recording: **6 recordings** have gaze spread over 30° (P37 A1–A4, P9 A3, P47 A3). The median across recordings is 4.6°. |
| Does the > 1000°/s rejection (V3-4) fix it? | **No.** P37 A4: 102.1° before, 102.4° after. P37 A1: 90.3° before and after. |
| Cause | Gaze points placed **behind the camera** (`gaze_point_3d_z` ≤ 0), which flips the direction by about 180°. 44–80% of P37's valid samples; present in **39 of 188** recordings. |
| Estimated gaze depth | Median **111 mm**, against a true screen distance of **800 mm**. Per-recording medians range from −1064 to +484 mm. |

**Why it matters:**
- `gaze_point_3d` comes from the binocular vergence-depth estimate, which is failing here.
- With the depth wrong, eye–camera parallax skews the gaze direction, and depth noise becomes
  direction noise. That probably explains the first-pass excess of saccades (saccade rate
  4.0/s > fixation rate 3.4/s).
- This affects **7 of the 10 Core features:** fixation rate, fixation duration, saccade rate,
  saccade amplitude, saccade peak velocity, and gaze spread x and y.
- **The D-M1 cross-check is not independent.** `norm_pos` (option (a)) is the projection of the
  same 3D point into the camera image, so it inherits the same depth error. The ±15% agreement
  test (V3-3) could pass while both inputs are wrong.

**Smallest fix (scope shrinks):**
1. **P8 input:** compute gaze angles from **`gaze_normal0/1`**, each eye's own gaze direction,
   which does not depend on depth. Average the two eyes when both are valid; otherwise use the
   valid one.
2. **Delete** cross-check option (a) and the ±15% agreement criterion (§5 step 7, §13). They no
   longer test anything independent.
3. Keep everything else as it is: the I-VT recipe, the > 1000°/s rejection, the sanity check
   (fixation 150–400 ms, ratio 0.8–1.25) and the 5-feature fallback.

**Files to update:** `methodology-colet.md` §4 P8, §5 (D-M1 paragraph and step 7), §13,
Appendix A row 8a; researcher-notes N13 and N18; decision-log D-M1 and V3-3; `HANDOFF.md` §3.

**Alternatives considered (worse for scope or validity):**

| Option | Why not preferred |
|---|---|
| Keep `gaze_point_3d`, treat z ≤ 0 as invalid | Also one line, and it brings max spread to 11.9° (median 4.5°). But the depth bias (111 vs 800 mm) remains, so the event features may still fail the sanity check, which lands on the fallback anyway. |
| Declare the 5-feature model now | Smaller still, but it drops RRL-supported features without testing them, and gaze spread still needs a fixed input |

**Note on scope history:** in round 2, `gaze_normal0/1` was suggested as an *additional*
check, and the team rightly scoped it out. It is now needed because the primary input fails,
not as an extra.

**Supporting statement for Chapter 3 (suggested):** *"Gaze angles were computed from the
tracker's per-eye gaze-direction vectors (`gaze_normal`), because the binocular 3D gaze point
depends on a vergence-depth estimate that was implausible at the 80 cm viewing distance
(median 111 mm) and placed some samples behind the camera."*

### R4-2. The sanity check is pooled, but the artifact depends on condition ⏳ (Major; wording only)

**What:**
- §5 step 7 applies the sanity check "over all retained recordings".
- Talking-related jitter and data loss are concentrated in A3/A4 (N22). A pooled ratio can pass
  while A4 alone fails, and the saccade features would then carry the artifact into P1.

**Smallest fix:** in §5 step 7, change "computed over all retained recordings" to **"computed
for each activity (A1–A4) separately; a failure in any activity triggers the 5-feature
fallback."**
- No new analysis.
- It is label-blind QC, the same kind as invalid % by condition (P11).

### 3. V3-1 (carried over). Refits vs P2 ⏳ (Major; one sentence)

**Smallest fix:** add to the limitations (and researcher-notes N23, V2-3):
> *"The pupil-only model (P2) is not fully independent of the spoken secondary task: Pupil
> Core's 3D eye-model refits, which can shift pupil-size estimates, were more frequent in the
> multitask activities (researcher-notes N22)."*

No new analysis. The refit-free P2 run stays optional or future work.

## 4. Still open (unchanged)

| # | Item | Status |
|---|---|---|
| V2-6 | Full-text reads of Fenoglio 2023 ×2 and Rolon-Merette 2026 (gap claims) | ⏳ |
| V2-13 | Recheck ρ = 0.95 / 0.83 after the pipeline runs | ⏳ |
| C5 / M9 | Adviser sign-off | ⏸ |
| C7, C8 | Reference checks; stimulus luminance | ⏸ |

## 5. Checklist to freeze the methodology

- [ ] R4-1: P8 input changed to `gaze_normal0/1`; cross-check (a) and the ±15% rule deleted
- [ ] R4-2: sanity check applied per activity
- [ ] V3-1: limitation sentence added
- [ ] Then: freeze the methodology; next review is of the pipeline and results, not the plan

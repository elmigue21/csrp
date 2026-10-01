# Independent Review: Round 3 Issues

**Reviewed:** `methodology-colet.md` (incl. §0 scope rule and §17 Out of scope),
`researcher-notes.md` (N22, N23), `rrl-decision-log.md` (SCOPE, V2-x rows),
`review-issues-status.md` and `colet-eda/README.md`, as of 2026-10-01.

**Earlier rounds:**
- Round 1: "COLET Methodology Review — Issues" doc and `review-issues-status.md`.
- Round 2: `review-issues-status-2.md`.

**Reviewer stance:**
- independent; no repo files were edited, and this file is new;
- label-blind: no feature was tested for how well it separates A1 from A4.

Updated 2026-10-01.

**Status key** (same as before):
- ✅ **Resolved** (no team decision needed).
- 🟡 **Resolved in part;** the rest needs a decision.
- ⏳ **Needs a team decision.**
- ⏸ **On hold / pinned** by the team.

**Severity:** Critical / Major / Moderate / Housekeeping, as in rounds 1–2.

---

## 1. Verdict

**Conditional approval, on one real fix (V3-1); the rest is wording.**

The scope rule (§0) and the Out of scope table (§17) are legitimate, defensible practice. "The
RRL tells us what is acceptable, not what is required", plus a written list of exclusions with
reasons, means a panel cannot call those items omissions. Most exclusions are accepted as made
(§3).

One scope call weakens another: the lean refit handling (V2-3) undermines the only required
talking check (P2).

| Aspect | Round 2 | Round 3 |
|---|---|---|
| Scope definition | Unclear | **Clear and defensible** |
| Contribution vs scope | Contradiction | Consistent |
| Talking confound | Main threat | **Partly handled:** P2 is required, but its pupil features carry a refit artifact (V3-1) |
| Event-feature validity | No rule | Rule exists; needs numeric tolerances (V3-3) |
| RRL authenticity | Clean (122/122) | Clean; new sources real; one WEAK source used as support (V3-5) |
| Defensibility | Close | **Close:** fix V3-1; the rest is wording |

**Conditions for approval:**
1. V3-1: add the refit-free P2 run, **or** state plainly that P2 is not fully talking-free.
2. V3-3: give the sanity check numeric tolerances.

## 2. Checks run in round 3

| Check | Result |
|---|---|
| Crossref, sources newly cited in decisions: Karunathilake 2025, Stolte 2020, Appel 2018, Laut 2026, Aygun 2022, Hefron 2018, Rizzo 2022, Diaz-Piedra 2019 | **All 8 DOIs resolve;** titles and authors match. No fabricated references. |
| Are those sources in `rrl-colet-list.md`? | Karunathilake (EM11) and Stolte (X9) are **not** (V3-5) |
| Evidence grade of Karunathilake (EM11) in `rrl-master-list.md` | **WEAK**, abstract only, K-means labels |
| Hefron 2018 modality | **EEG** (cited in N7 alongside eye-tracking studies) |
| Stale references across files | One left: `colet-eda/README.md:101` (V3-6) |

## 3. Scope exclusions: reviewer verdict

| Out-of-scope item (§17) | Verdict | Residual risk |
|---|---|---|
| Cross-dataset; GAZELOAD; ADABase; other models; non-eye signals; real-time or calibration | ✅ Accepted; the title justifies them | — |
| Windows; trial-level; self-report as main label; A2/A3 in the main analysis | ✅ Accepted; justified by data and RRL | — |
| COLET replication (S3) optional and removed from contributions | ✅ Accepted; V2-5 closed | Panel: "why is yours lower than 0.85?" is answered by argument (N6), not shown |
| No-normalization run (S4) optional | ✅ Accepted with the N7 disclosure | Panel: "how big is the inflation here?" is answered by citation (Tognotti 2026), not a number |
| Third gaze-conversion method; formal main-sequence test | ✅ Accepted, given the sanity check and the 5-feature fallback | See V3-3 |
| Deep XGBoost settings | ✅ Resolved (depth 1–3, 50–300 trees) | — |
| Blink duration, duplicates, higher-order statistics; entropy in the main model; luminance correction | ✅ Accepted | — |
| Statistical control for talking beyond P2 | ✅ Accepted as scope | Only if P2 is valid: see V3-1 |
| **Per-segment pupil handling for refits** | **🟡 Not accepted as is** | Conflicts with P2: see V3-1 |

## 4. Round-2 issues: reviewer verdict

| # | Issue | Team outcome | Reviewer verdict |
|---|---|---|---|
| V2-1 | Speech confound across features | Required P2 pupil-only (N22 confirms the talking effect) | 🟡 Right response, but P2 is weakened by V3-1 |
| V2-2 | Event-detection pass/fail | Lean sanity check + 5-feature fallback | 🟡 Accepted; needs numbers (V3-3) |
| V2-3 | 3D eye-model refits | Lean: speed filter + refit count + limitation | 🟡 Not enough for P2 (V3-1) |
| V2-4 | Ceiling | Rule pre-stated (§11) | 🟡 Fallback is a 2-feature comparison (V3-2) |
| V2-5 | Contribution vs scope | Replication removed from contributions | ✅ |
| V2-6 | Gap claims, abstract-only | Pending full-text reads; claims softened | ⏳ Still open |
| V2-7 | S2 comparability | Wording fixed | ✅ |
| V2-8 | No-normalization optional | Kept optional (scope) | ✅ Accepted with residual risk (§3) |
| V2-9 | XGBoost search space | Depth 1–3, 50–300 trees | ✅ |
| V2-10 | Kendall τ CI | Added | ✅ |
| V2-11 | Exclusion wording | "Label-blind inspection" | ✅ |
| V2-12 | Peak velocity as Core | Kept (D-N2) | ✅ Team's call; answer prepared in N17 |
| V2-13 | Recheck correlations | Pending final pipeline | ⏳ Still open |
| V2-H1–H7 | File contradictions | Fixed | ✅ Except V3-6 |

## 5. New issues

### V3-1. The lean refit handling undermines P2 ⏳ (Major)

**What:**
- **P2 (pupil-only) is the one required talking check** (V2-1).
- N22 shows refits **differ by condition:** 19–21 of 45 multitask recordings vs 8–11
  single-task; A4 > A1 in 17 participants vs 3; Wilcoxon p = 0.001. The likely cause is
  talking moving the headset.
- V2-3 keeps only the speed filter. It removes the **moment** of a jump, not the **level
  shift** after it.
- So pupil mean, and **especially pupil SD**, can carry a talking-linked artifact. Those are
  exactly P2's two features.

**Why it matters:** P2 is not the talking-free check it is presented as. "No RRL study did
per-segment handling" is not a reason against it here; no RRL study had to deal with this
artifact on this dataset.

**Recommended:** pick one (both fit the scope rule, because P2 cannot be valid without it):
1. **Preferred, cheapest:** also run P2 on participants with **no refit in A1 or A4**, and
   report both. No new method is needed.
2. **Minimum:** the limitation sentence states that P2 is *not fully* talking-free, because
   refits were more frequent in multitask activities and can shift pupil-size estimates.

### V3-2. The ceiling rule falls back to a 2-feature comparison ⏳ (Major)

**What:**
- §11: if both models exceed 0.95 AUC in P1, the comparison is read mainly from P2.
- P2 has 2 features. LR vs XGBoost on 2 features is a weak test, because XGBoost has almost no
  interactions to exploit.
- Feature importance on 2 features is nearly trivial, and importance is in the title.

**Recommended:** add one sentence to §11: *"In that case, the feature-importance analysis is
still reported from P1, noting that top-ranked blink and saccade features may partly reflect
talking (researcher-notes N17, N22)."* This keeps the title component alive.

### V3-3. The sanity check is not numeric yet ⏳ (Moderate)

**What:**
- "Saccade:fixation count ratio **about 1**" (§5 step 7; N23) has no tolerance, so it is not
  yet a pass/fail rule.
- The first pass would have **passed** the fixation check (median 196 ms, inside 150–400 ms)
  while its saccades were likely noise. The ratio check does the real work, so it needs a
  number.
- §5 D-M1: option (a) "must agree" with option (b), with no tolerance and no action if they
  disagree.

**Recommended:**
- State a tolerance, for example **ratio 0.8–1.25**.
- State the agreement tolerance for (a) vs (b), for example median fixation duration within
  ±15%. Treat disagreement as a fail, leading to the 5-feature fallback.

### V3-4. Gaze-dispersion cleaning rule unspecified ⏳ (Moderate)

**What:** §6 lists gaze dispersion x/y as "Core, after cleaning (one 102° artifact)", but the
cleaning rule is not defined.

**Recommended:** name the rule, for example drop samples outside a plausible screen angle, or
apply the > 1000°/s velocity rejection before computing the SD. Fix it before running.

### V3-5. V2-1 support cites a WEAK source; two IDs missing from the COLET list ⏳ (Moderate)

**What:**
- `rrl-decision-log.md` V2-1 cites Karunathilake 2025 and Stolte 2020 as support for
  pupil-only models.
  - **Karunathilake (EM11)** is graded **WEAK** in `rrl-master-list.md` (abstract only;
    K-means labels).
  - **Neither EM11 nor X9 (Stolte) is in `rrl-colet-list.md`** (the R3 pattern again).
- researcher-notes N7 lists **Hefron 2018** among per-person standardization precedents. It
  is an **EEG** study. That is fine as a method precedent, but it should not be grouped with
  the eye-tracking studies.

**Recommended:**
- Rest V2-1 on Appel 2018 (EM6), Rolon-Merette 2026 (X1) and Tao 2019 (CL7). Drop or
  downgrade Karunathilake.
- Add Stolte (X9) to `rrl-colet-list.md` if kept.
- Mark Hefron as EEG in N7.

All DOIs involved are real (Crossref-checked, §2).

### V3-6. Status and wording accuracy (Housekeeping)

| # | Where | Problem |
|---|---|---|
| V3-6a | `review-issues-status.md` round-2 table | V2-1 and V2-3 are marked ✅. Given V3-1, they should be 🟡 until V3-1 is decided. |
| V3-6b | `colet-eda/README.md:101` | Suggested C4 set still includes blink duration. Mark it superseded by D-M3. |
| V3-6c | `methodology-colet.md` Appendix A row 15 | "Per-person normalization (+ no-normalization)" graded STRONG, but the no-normalization run is now optional. Reword it. |

## 6. Still open from earlier rounds

| # | Item | Status |
|---|---|---|
| V2-6 | Full-text reads of Fenoglio 2023 ×2 and Rolon-Merette 2026; the gap claims depend on them | ⏳ |
| V2-13 | Recheck ρ = 0.95 / 0.83 once the final pipeline runs | ⏳ |
| C5 / M9 | Adviser sign-off | ⏸ |
| C7 | Year consistency, author lists, full-text reads of A/M rows | ⏸ |
| C8 | Stimulus-luminance check | ⏸ |

## 7. Priority checklist

- [ ] V3-1: refit-free P2 run, or a limitation stating that P2 is not fully talking-free
- [ ] V3-3: numeric tolerances (ratio, method agreement) and the fail action
- [ ] V3-2: one sentence keeping importance reported from P1 under the ceiling rule
- [ ] V3-4: define the gaze-dispersion cleaning rule
- [ ] V3-5: rebase V2-1 support; add the missing IDs; mark Hefron as EEG
- [ ] V3-6: status and wording fixes
- [ ] V2-6: full-text reads before the defense
- [ ] V2-13: recheck correlations after the pipeline runs
- [ ] C5: adviser sign-off

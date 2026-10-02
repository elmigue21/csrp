# Status of the "COLET Methodology Review — Issues" doc

**Review doc:** https://claude.ai/artifact/AXyhbekkL2Fr6oVtVd83Kb (independent review,
2026-10-01).

**This file:** tracks each issue against the repo, the data and the decisions.

Updated 2026-10-01, after the fix pass.

**Status key:**
- ✅ **Resolved** (no team decision needed).
- 🟡 **Resolved in part;** the rest needs a decision.
- ⏳ **Needs a team decision.**
- ⏸ **On hold / pinned** by the team.

## Resolved in the fix pass (no decision needed)

| # | Issue | What was done |
|---|---|---|
| R1 | S2 comparability claim | S2 now uses **COLET's own bins** (low 0–29 vs high 50–100), comparable with their C1/C3 0.88. Their ≥50/<50 result (0.74) is noted. |
| R2 | Haag preprint cited | Removed from `methodology-colet.md` and researcher-notes N6 |
| R3 | IDs missing from `rrl-colet-list.md` | After the rewrite, all cited IDs are listed. Added EM14, ET16, ET20, HR2, HR4 (entries 129–133) and the X5 alias. |
| R4 | HR2–HR4 marked "not used" | Reconciled: HR2 (Pluchino) and HR4 (Nenna) are now used and listed |
| R5 | Over-stated STRONG grades | Matrix rebuilt as `methodology-colet.md` Appendix A. Every row shows the evidence level (V/A/M); rows 12, 24, 27 lowered to ADEQUATE. |
| R6 | RTLX not cited | Added Hart 2006 (CL8, Crossref-verified) and Byers et al. 1989 (CL9, COLET's own RTLX source). Both metadata-only; read before citing specific claims. |
| M6 | Durations / windows / contradictions | Durations measured; whole-activity unit decided (C1); text cleaned |
| N1 | In-fold pruning varies features per fold | Feature set fixed in advance (§6) |
| N3 | Sparse transition entropy | Moot: whole activity, and entropy is exploratory |
| N4 | Per-fold permutation importance | Now on pooled out-of-fold predictions (§10) |
| N8 | Coarse per-person AUC | Pooled AUC + participant cluster bootstrap is now the primary comparison (§10) |
| N9 | Time-on-task across windows | Moot (no windows) |
| N10 | Ethics/licence | Added to §2 and researcher-notes N16, with a ready Chapter 3 sentence |
| H1–H10 | Internal contradictions | `methodology-colet.md` rewritten cleanly. The old layered version is at `archive/methodology-colet-v1-layered.md`. |
| D1 | Stray P18 pupil samples | Preprocessing step P1 (trim to the gaze time range); researcher-notes N14 |
| D2 | Long gaps in P16/P18 | Preprocessing step P4 (rates over valid time); researcher-notes N14 |

**Also fixed:** three wrong author initials found via Crossref.
- Wu (C., not Y.)
- Nenna (F., not E.)
- Pluchino (P., not L.)

## Round-1 items now decided (previously "needs decision")

| # | Decision |
|---|---|
| M1 | D-M1: per-eye gaze directions `gaze_normal0/1` (revised in round 4, R4-1; N18, N25) |
| M2 | D-M2: follow the RRL; round 2 added the required pupil-only model P2 (V2-1) |
| M3 | D-M3: blink duration dropped |
| M4 | D-M4: per-person standardization, bias disclosed (N7) |
| M5 | D-M5: keep A1 vs A4 (P1); S1 optional |
| M7 | D-M7: COLET replication optional, now `methodology-colet.md` §16; not a stated contribution |
| M8 | Gap claims softened; D-M8: SHAP for both models; full-text checks pending (V2-6) |
| N2 | D-N2: peak velocity kept as core |
| N5 | D-N5 + scope rule: only P1 and P2 required |
| N6 | D-N6: elastic net |
| N7 | Framing: LR ≈ XGB is a legitimate result |
| C4 | 10 features (N21) |

## Round 2 (`review-issues-status-2.md`): outcomes

| # | Outcome |
|---|---|
| V2-1 | 🟡 Required **P2, pupil-only model** (data check N22 confirmed the talking effect). Round 3: P2 weakened by refits until V3-1 is decided |
| V2-2 | ✅ Lean sanity check + 5-feature fallback (N23); numeric tolerances added (V3-3) |
| V2-3 | 🟡 Lean: speed filter + refit count + limitation (N22, N23). Round 3: not enough for P2 until V3-1 is decided |
| V2-4 | ✅ Ceiling rule pre-stated (§11) |
| V2-5 | ✅ Replication optional; removed from stated contributions |
| V2-6 | ✅ Done 2026-10-01: Fenoglio 2023a (V) uses unseen test users, so the third gap claim was reworded; Rolon-Merette (V) has no importance, so claims 1–2 hold; Fenoglio 2023b abstract only (no full text) |
| V2-7 | ✅ S2 wording: comparable only under COLET's protocol |
| V2-8 | ✅ No-standardization run optional (scope rule); answer via Tognotti 2026 / N7 |
| V2-9 | ✅ XGBoost depth 1–3, 50–300 trees |
| V2-10 | ✅ Kendall τ reported with a CI or permutation p-value |
| V2-11 | ✅ "Chosen after a label-blind inspection" wording |
| V2-12 | ✅ Peak velocity kept (D-N2); panel answer in N17 |
| V2-13 | ⏳ Recheck the correlations once the final pipeline runs (N21) |
| V2-H1–H7 | ✅ File contradictions fixed |

## Round 3 (`review-issues-status-3.md`): outcomes

Team rule for round 3: approve only items that add no new analysis (researcher-notes N24).

| # | Outcome |
|---|---|
| V3-1 | ✅ Limitation sentence only (decided in round 4; §11, N23). Refit-free P2 run = future work |
| V3-2 | ✅ Under the ceiling rule, importance still reported from P1 (`methodology-colet.md` §11) |
| V3-3 | ✅ Ratio 0.8–1.25; any failure → 5-feature fallback (§5 step 7; our own choice, §13). The ±15% part was removed by R4-1 |
| V3-4 | ⤴ Superseded by R4-1 (the velocity rejection did not remove the artifact) |
| V3-5 | ✅ Karunathilake dropped from V2-1 support; Stolte (X9) added as entry 136; Hefron marked EEG in N7 |
| V3-6 | ✅ (a) V2-1/V2-3 set to 🟡 above; (b) `colet-eda/README.md` blink duration marked dropped; (c) Appendix A row 15 reworded |

## Round 4 (`review-issues-status-4.md`): outcomes

| # | Outcome |
|---|---|
| R4-1 | ✅ Gaze input changed to `gaze_normal0/1` (two eyes averaged); `norm_pos` cross-check and ±15% rule removed. Reviewer's numbers reproduced (N25). ✅ One-eye samples as missing (decided 2026-10-02). RRL: PP17–PP19 added (vergence depth unreliable; vector-angle velocity); the field choice itself is a design choice (§13) |
| R4-2 | ✅ Sanity check per activity (§5 step 7) |
| V3-1 | ✅ Limitation sentence (§11) |

With V3-1 decided, V2-1 and V2-3 above are effectively closed by the limitation sentence.
**Freeze:** the reviewer proposes freezing the plan after round 4; the team's call.

**All deliberate exclusions** are listed in `methodology-colet.md` §17 (Out of scope).

## On hold / pinned

| # | Item |
|---|---|
| M9 / C5 | Adviser sign-off (on hold) |
| R7, R8 / C7 | Year consistency, full author lists, abstract-only full-text checks (pinned) |
| C8 | Stimulus-luminance check: optional (decided 2026-10-02) |

# Independent Review: Round 5 Issues

**Reviewed:** the v3 draft (`CSRP-Chapter-1-to-5-draft-COLET-v3.docx`), `leakage-audit.md`,
`researcher-notes.md` (to N30), `adviser-questions.md` (Q1, Q2), saved outputs in
`outputs/colet/tables/`, and the COLET paper (Ktistakis *et al.* 2022, full PDF in the repo
root). This round combines four external critiques (a panel verdict, a methods critique of
Chapters 1–5, a v3 draft review and a COLET-labels review) with checks run on the repo.

**Earlier rounds:** round 1 (review doc), `review-issues-status-2.md` to `-4.md`.

Updated 2026-10-15 (statuses updated after the rerun and v4 draft).

**Status key** (same as before): ✅ Resolved · 🟡 Resolved in part · ⏳ Needs a team
decision · ⏸ On hold / pinned.

**"Needs a fix?" key:**
- **Required:** the defense is at risk without it.
- **Recommended:** a cheap, high-value check.
- **Optional:** only if time allows.
- **Wording:** a text fix, no new analysis.
- **No:** considered and not adopted.

---

## 1. Verdict

**The methods are more careful than some of the claims.** The evaluation design (nested,
participant-grouped tuning; in-fold scaling; a pre-stated ceiling rule) holds up. The problems
are (a) one data error, (b) claims that go beyond the prediction setup, and (c) a model
comparison that has little power at the ceiling.

**Two checks were run this round (§2):**
- **R5-1, P28/P31: confirmed as a dataset error.** Every raw gaze, pupil and blink file is
  byte-identical, but demographics and NASA-RTLX differ, so they are two people sharing one
  person's eye data.
- **R5-25, feature directions vs COLET: largely resolved.** Our pipeline agrees with COLET on
  every time-pressure effect. It disagrees only on multitask (speech) effects, where COLET's
  own numbers are internally inconsistent.

**A correction to earlier review notes (R5-19):** the P2 Wilcoxon (p = 0.006) favors
**XGBoost**, not LR. P2's pooled AUC favors LR while its within-person margins favor XGBoost.

## 2. Checks run this round

### 2.1 P28/P31 (R5-1)

| Check | Result |
|---|---|
| MD5 of `p28_t{1..4}_{gaze,pupil,blinks}.parquet` vs P31 | **All 12 files identical** |
| `subject_info.csv` | P28: F, 23 y, acuity −0.01 · P31: F, 26 y, acuity −0.05 |
| `annotation.csv` (RTLX mean A1–A4) | P28: 30.8, 39.2, 44.2, 37.5 · P31: 23.3, 42.5, 40.0, 62.5 |
| `data/colet/convert_colet.py` | Indexes participants by position; reads eye data and annotation from the same entry |

**Conclusion:** the duplication is in COLET v3's data, as decoded from `data_v3.mat`, not in
our conversion. Two real people share one eye recording. Which of the two it belongs to cannot
be known.

**Caveat:** a decoding fault in `mat-io` cannot be fully ruled out without reading
`data_v3.mat` with a second reader (e.g. `h5py`) and comparing entries 28 and 31.

### 2.2 Feature directions vs COLET (R5-25)

Script: `colet-eda/trace_check.py`. Outputs: `colet-eda/trace_check.png` (I-VT detections on
5 s of P01, P10 and P20, A1 and A4) and `colet-eda/trace_features.csv`.

| Feature | COLET A1 / A2 / A3 / A4 | Ours A1 / A2 / A3 / A4 | Time pressure (A1→A2) | Multitask (A1→A3) |
|---|---|---|---|---|
| Pupil (mm) | 3.49 / 3.66 / 3.80 / 3.82 | 3.50 / 3.67 / 3.76 / 3.83 | ✅ agree | ✅ agree |
| Blinks (/min) | 3.0 / 1.8 / 14.4 / 14.4 | 3.0 / 1.9 / 13.7 / 13.8 | ✅ agree | ✅ agree |
| Fixations (/s) | 2.50 / 2.80 / 2.24 / 2.35 | 3.31 / 3.56 / 2.98 / 3.08 | ✅ agree | ✅ agree |
| Fixation duration, median (ms) | 273 / 244 / 269 / 254 | 221 / 213 / 239 / 230 | ✅ agree | ❌ COLET flat, ours longer |
| Saccades (/s) | 1.71 / 1.85 / 3.40 / 3.35 | 3.25 / 3.53 / 2.77 / 2.90 | ✅ agree | ❌ COLET doubles, ours lower |
| Saccade amplitude, median (°) | 14.1 / 14.0 / 14.2 / 14.1 | 2.5 / 2.7 / 2.7 / 2.8 | — | — |
| Saccade velocity, mean (°/s) | 146 / 133 / 261 / 268 | 102 / 106 / 104 / 108 | — | ❌ COLET +80%, ours +2% |

**Why our numbers are the more plausible ones:**
1. **Fixations and saccades alternate,** so their rates should be close. Ours are (3.25 vs
   3.31 in A1). COLET's are not (1.71 vs 2.50 in A1), and COLET's saccade rate overtakes its
   fixation rate in A3 (3.40 vs 2.24).
2. **COLET's median saccade amplitude (14°) exceeds the whole gaze range** seen in the traces
   (about ±6° around the screen centre).
3. **The traces are physiologically clean:** step-like saccades with peaks of 100–250 °/s,
   flat fixations, and a 45 °/s threshold well clear of fixation noise.
4. **COLET's disagreements are confined to the speaking activities.** That fits jaw and
   headset movement being picked up as saccades by their detector.
5. Our longer fixations under multitasking match the RRL expectation (Tao 2019), and N17
   marked COLET's direction as "opposite".

**Still open:**
- Small speed bumps just above 45 °/s occasionally split one fixation in two (P01 A1 ~12.7 s;
  P20 A4 ~20.5 s).
- A4 has more tracking gaps, which cut fixations. With the 55 ms minimum, fragments can be
  discarded (see R5-38).
- Neither effect has been quantified.

**Wording for Chapter 4:**
> Our time-pressure effects matched those reported by the dataset authors on every feature.
> Under multitasking, our saccade rate and velocity changed little, whereas COLET reported a
> doubling. COLET's saccade and fixation rates are inconsistent with alternating events and
> its median saccade amplitude exceeds the observed gaze range, which suggests its detector
> was affected by speech-related head and headset movement.

## 3. Issues

### 3.1 Data and evaluation

| # | Issue | Needs a fix? | Suggested fix | Status |
|---|---|---|---|---|
| R5-1 | P28 and P31 share one eye recording (§2.1) | **Required** | For the models: keep the recording **once**. Labels come from the activity, so they stay valid. For RTLX analyses: **exclude both** (the owner is unknown). Rerun everything. Optionally confirm with `h5py` and report the error to the COLET authors / Zenodo. | ✅ Done (N31) |
| R5-2 | Per-person standardization uses the test person's own A1/A4 | **Required** (reporting) | Report the standardized and unstandardized runs side by side, after R5-1. Name the main protocol "participant-disjoint training with within-person standardization using all four activities". | ✅ Reported in v4 |
| R5-3 | Standardization uses each sample's paired counterpart | Recommended | Variant standardized on **A2+A3 only**, the activities not evaluated. COLET has no rest baseline, so this is the closest available. | ⏳ |
| R5-4 | Feature set chosen using all participants (leakage audit §3) | Wording | Disclose as a limitation. | ✅ Applied in v4 (tracked) |
| R5-5 | All excluded recordings are counting activities | Optional | Sensitivity run with a looser exclusion threshold, so P06/P17 come back in. | ⏳ |
| R5-6 | Many threshold choices of our own | Optional | Sensitivity table over confidence 0.8, 35% exclusion, 45 °/s and 55 ms. | ⏳ |
| R5-7 | Run metadata lacks git commit and input hashes | Recommended | Record both in the rerun metadata. | ✅ Done |
| R5-8 | Feature cache not keyed on the data source | Optional | Add the data-root identity to the cache key. | ⏳ |

### 3.2 Speech and labels

| # | Issue | Needs a fix? | Suggested fix | Status |
|---|---|---|---|---|
| R5-9 | A1 vs A4 bundles speech, counting, time pressure, duration and tracking quality | **Required** (framing) | Reframe the target as "workload condition" classification; add the speech limitation paragraph. | ⏳ |
| R5-10 | No contrast without a speech difference | Recommended | A1 vs A2 and A3 vs A4 under the same LOPO pipeline. Call A1 vs A2 a **time-pressure sensitivity analysis**, not a definitive speech-free test: COLET's time-pressure manipulation was weak by the authors' own account. | ⏳ |
| R5-11 | The model may detect tracking quality | Recommended | Quality-only baseline: % invalid samples, refit count, duration. | ⏳ |
| R5-12 | Unknown whether predictions follow workload or speech | Recommended | Dose-response: apply the A1/A4 model to A2 and A3; compare the ordering with RTLX. | ⏳ |
| R5-13 | P2 does not rule out speech | Wording | "Not solely dependent on blink-rate input; speech-related effects cannot be ruled out." Note that the signal depends on calibration (0.85 → 0.61–0.66 raw). | ✅ Applied in v4 (tracked) |
| R5-14 | A4 recordings are longer than A1 | Optional | Fixed-window features (first ~11 s). See R5-41 for the wording. | ⏳ |
| R5-15 | Switch to NASA-RTLX labels as the main target | **No** | Keep condition labels (decision C2). RTLX bins as optional S2 only; cite COLET as the precedent, since the bins' own source (Nur 2020) is weak. | ✅ |
| R5-39 | The draft says RTLX shows "a clear difference in experienced workload" | Wording | "NASA-RTLX ratings confirmed that participants reported greater workload in A4 than A1; the labels therefore represent a validated difference in subjective workload between these task conditions." | ✅ Applied in v4 (tracked) |

### 3.3 Model comparison

| # | Issue | Needs a fix? | Suggested fix | Status |
|---|---|---|---|---|
| R5-16 | P1 is at the ceiling and cannot separate the models | **Required** (reporting) | Put the comparisons with headroom in the main results: P2, no-blink, unstandardized, A1 vs A2. | ✅ Reported in v4 |
| R5-17 | No blink-only baseline | Recommended | Single-feature blink-rate model, to show what the ML models add. | ⏳ |
| R5-18 | Pooled AUC and within-person metrics disagree | **Required** (reporting) | Report both and state that they disagree. Do not switch the primary metric after seeing results. | ✅ Reported in v4 |
| R5-19 | P2 Wilcoxon direction misread in earlier notes | Wording | P2 margins favor XGBoost: 31/45, medians 0.497 vs 0.341, p = 0.006. Pooled AUC favors LR. | ✅ Applied in v4 (tracked) |
| R5-20 | The model ranking flips with standardization | **Required** (reporting) | "No clear AUC advantage for either model under either protocol; the direction of the point estimate changes with the protocol." Report AUC differences with CIs for both, and keep the tuned S4 run (N30) separate from the untuned Q1 check. | ✅ Reported in v4 |
| R5-21 | The tuning budget favors LR | Wording + Recommended | Call it "an equal number of candidate settings", not an equal budget. Sensitivity run: ~100 XGBoost draws plus `min_child_weight`. Conclude "LR at least matches a lightly tuned XGBoost". | 🟡 Wording in v4; run optional |
| R5-22 | XGBoost picked depth-1 stumps in P2 | Wording | State that P2 cannot test XGBoost's interaction advantage. | ✅ Applied in v4 (tracked) |
| R5-23 | No equivalence margin; CIs understated; many tests | Wording | "No evidence of a difference", never "equivalent". Bootstrap CIs ignore refit variance. | ✅ Applied in v4 (tracked) |
| R5-24 | The "Control for Leakage" section overstates the shuffle test | **Required** (wording) | Rename it "Shuffled-label check". It tests label leakage for the untuned procedure only, not the full workflow (R5-1, R5-2). | ✅ Applied in v4 (tracked) |
| R5-40 | The draft says the Brier score measures calibration | Wording | Brier mixes discrimination, calibration and uncertainty; a lower Brier alone does not show better calibration. | ✅ Applied in v4 (tracked) |

### 3.4 Features and importance

| # | Issue | Needs a fix? | Suggested fix | Status |
|---|---|---|---|---|
| R5-25 | Feature directions disagree with COLET's | **Required** (check) | Checked (§2.2). Add the Chapter 4 wording. Optionally quantify fixation splitting and gap fragmentation. | ✅ Checked; wording in v4 |
| R5-38 | Chapter 4 says blink splitting can only raise the fixation count | **Required** (wording) | Fragments under 55 ms are discarded, so splitting can lower the count too: "Blinks and tracking gaps can split fixations; because short segments are discarded, this may raise or lower the detected fixation count. The contribution of these effects to the A1–A4 difference was not quantified." | ✅ Applied in v4 (tracked) |
| R5-26 | Importance may reflect the speech confound | Wording | "What the model relies on", not "indicators of cognitive load". | ✅ Applied in v4 (tracked) |
| R5-27 | Permutation importance shuffles across participants | Wording | Present it as model reliance under that shuffle; it breaks the participant pairing and feature correlations, so it is not a feature's unique or causal contribution. Optionally permute within the held-out fold. | ✅ Applied in v4 (tracked) |
| R5-28 | Kendall τ is driven by the top features | Recommended | Add top-3 overlap next to τ. | ✅ In v4 (3/3) |
| R5-29 | SHAP explainers differ between LR and XGBoost | Wording | Rankings are descriptive; keep the caveat. | ✅ Applied in v4 (tracked) |

### 3.5 Writing and documents

| # | Issue | Needs a fix? | Suggested fix | Status |
|---|---|---|---|---|
| R5-30 | Title and Chapter 1 overclaim "cognitive load detection" | **Required** | Retitle around comparing LR and XGBoost on workload conditions. | ⏳ Adviser |
| R5-31 | "Unseen participants", "data not used in training", "stricter protocol" | **Required** | Labels were held out, but the person's unlabeled data were used for standardization. "Stricter" applies only to participant overlap. | ✅ Applied in v4 (tracked) |
| R5-32 | The draft says "no run without standardization was made" | **Required** | Replace with the N30 results (rerun after R5-1). | ✅ Replaced in v4 |
| R5-33 | Terms used interchangeably | Wording | Define one construct ("workload condition") once and use it throughout. | 🟡 Defined once in v4; not renamed throughout |
| R5-34 | Defensive Discussion tone; decorative SDG paragraphs | Wording | Call design flaws flaws; trim the SDG section. | ✅ Applied in v4 (tracked) |
| R5-35 | COLET citation details | Wording | RTLX = overall score, not Mental Demand. A1/A2: 80% (NB). A1/A4: 86% (k-NN; LR 85%). Both under a random 80/20 split with 5-fold CV. Time pressure was weak by the authors' account. | ✅ Applied in v4 (tracked) |
| R5-36 | Tao *et al.* figures and 2026 references unverified | Recommended | Check against the originals. | ⏸ Team to verify |
| R5-37 | Docs out of sync | **Required** | Commit N29/N30; update `thesis-review-eli5.md` and `leakage-audit.md` (§2.1 now answers its open question); set EM7 to VERIFIED in `rrl-master-list.md`. | 🟡 Docs updated; commit pending |
| R5-41 | The limitations say SD features "partly grow with recording length" | Wording | Not automatic: "recording length may affect how stable or representative the estimates are." | ✅ Applied in v4 (tracked) |
| R5-42 | Table 3 labels A2/A3 "standardization baseline" | Wording | They are task conditions, not neutral baselines: "included in within-person standardization". | ✅ Applied in v4 (tracked) |
| R5-43 | No statement of the study's limits near the conclusion | **Required** | Add a "What this study can and cannot claim" paragraph (§4). | ✅ Applied in v4 (tracked) |

## 4. Suggested "What this study can and cannot claim" paragraph

> This study evaluates whether eye-tracking features distinguish COLET's A1 and A4 task
> conditions under participant-disjoint training. The within-person protocol uses each
> participant's recordings from all four activities for standardization. Results do not
> establish real-time performance, cold-start performance under that protocol, or
> speech-independent detection of cognitive load. Model rankings differed in direction across
> the standardized and raw-feature analyses, and neither protocol established a clear AUC
> advantage for either model.

## 5. Order of work

1. R5-1: fix the duplicate, then rerun everything (main, no-blink, unstandardized).
2. R5-37: bring the documents in line.
3. Reporting fixes once the new numbers are in: R5-2, R5-16, R5-18, R5-20.
4. Text fixes: R5-24, R5-30 to R5-32, R5-38, R5-43, then the remaining Wording items.
5. Recommended analyses in one script: R5-10, R5-11, R5-12, R5-17 (then R5-3, R5-21).
6. Optional items if time allows.

## 6. Scope check against the title, objectives and RRL (2026-10-15)

**Wording approval:** the team approved all text-only fixes on 2026-10-15 (marked
"✅ Approved (wording)" in §3). R5-30 (title) still needs the adviser; R5-32 needs the rerun
numbers first.

**Test used:** the scope rule (N23/N24). The research question is the **LR vs XGBoost
comparison** (title; general objective). An item is in scope if it is needed for a valid P1/P2
comparison or for the feature-importance objective. Anything else becomes optional, a
limitation, or future work.

| # | Item | In scope? | Why | Treatment |
|---|---|---|---|---|
| R5-1 | Duplicate fix and rerun | **In** | Without it P1/P2 aren't valid | ✅ Done (N31) |
| R5-2, R5-20 | Report standardized vs unstandardized | **In** | The S4 run already exists and changes the comparison's direction | Report (no new analysis) |
| R5-16, R5-18 | Comparisons with headroom; metrics that disagree | **In** | Reporting of existing results | Report |
| R5-7 | Commit and input hashes in metadata | **In** | Reproducibility of the comparison | ✅ Done |
| R5-28 | Top-3 overlap next to Kendall τ | **In** | Objective 5 (ranking agreement), from existing outputs | ✅ In v4 (3/3) |
| R5-36 | Verify Tao *et al.* and 2026 references | **In** | RRL quality bar | ⏸ Team to verify |
| R5-21 | Larger XGBoost search | Borderline | About comparison fairness, but a new run | 🟡 Wording in v4; run optional |
| R5-17 | Blink-only baseline model | Out | New model; N27 already reports blink rate's single-feature AUC (0.97) | Cite N27 descriptively |
| R5-10 | A1 vs A2, A3 vs A4 | Out | Outside the A1/A4 scope (Ch. 1); a construct question, not a model comparison | Future work or optional |
| R5-11 | Quality-only baseline | Out | Construct validity | Limitation (shortcut learning, Geirhos 2020) |
| R5-12 | Dose-response on A2/A3 | Out | Construct validity | Future work |
| R5-3 | A2+A3-only standardization | Out | New protocol; no RRL precedent | Future work |
| R5-5, R5-6, R5-14 | Exclusion, threshold, fixed-window sensitivity | Out | Robustness of features, not the comparison | Limitations |
| R5-8 | Cache key | Out | Code hygiene, no thesis impact | Optional |
| R5-25 | Feature directions vs COLET | **In** (done) | Needed to trust the features both models use | ✅ Checked; wording in v4 |

**RRL fit (`rrl-colet-list.md`, 140 entries):**
- **Core question (§J, §D, §M):** supported. Direct LR–XGBoost comparisons (Rolon-Merette,
  Nerella, Chakraborty, Shao), "ML no better than LR" (Christodoulou, Nusinovici), trees on
  tabular data (Grinsztajn, Couronné), and classifier-comparison statistics (Demšar,
  Dietterich, Nadeau & Bengio, Benavoli).
- **Feature importance (§N):** supported, and covers the R5-27/R5-29 caveats (Hooker, Molnar,
  Strobl, Aas).
- **Background beyond the question:**
  - §A cognitive-load theory and §B eye indicators: keep, but trim to what interprets the
    features.
  - §G windows (whole activities are used): keep only to justify not windowing.
  - Tangential entries, candidates for removal: Fenoglio 2024 federated personalisation (25),
    FAIR principles (30), Foulsham gaze allocation (28), Valtakari human interaction (29).
- **Ch. 1 Significance (SDGs, education, healthcare):** not supported by the comparison's
  evidence. Trim (R5-34).
- **Research gap (Ch. 2 Synthesis):** currently a combination-novelty claim ("none combined
  LR, XGBoost, LOPO and SHAP on COLET"). Lead with the question the RRL actually raises:
  **does XGBoost add value over a regularized LR on small, tabular eye-tracking data under
  participant-independent evaluation?** The cited studies disagree (Rolon-Merette vs Shao;
  Christodoulou). Keep the COLET combination as the setting, not the contribution.

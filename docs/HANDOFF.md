# Handoff: Thesis Project (read this first)

**Thesis title:** COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION AND XGBOOST FOR COGNITIVE LOAD
DETECTION USING EYE TRACKING FEATURES WITH FEATURE IMPORTANCE ANALYSIS

**Team:** Caacbay, Magkasi, Uaje, Ventura (BS Computer Science). Handoff written 2026-10-01.

---

## 1. Where things stand (one paragraph)

- **Dataset:** **COLET** (Ktistakis *et al.*, 2022). It replaced GAZELOAD; the earlier
  COLET + GAZELOAD cross-dataset plan was rejected by the panel.
- **Methodology:** **all decisions are made** and recorded.
- **Reviews:** four independent reviews were done, all resolved or deliberately scoped out.
  Round 4 changed the gaze input (R4-1). The reviewer proposes freezing the plan now (team's
  call).
- **Data:** downloaded, decoded and converted. Exploration done (first pass only).
- **Not yet started:**
  - **pipeline code has not been written or run** (the user said: do not run code yet);
  - **adviser sign-off on COLET is pending** (on hold, team's call).

## 2. Files, in reading order

| File | What it is |
|---|---|
| `docs/HANDOFF.md` | This file |
| `docs/methodology-colet.md` | **The plan.** §0 scope rule; §§2–11 method; §16 optional replication; §17 **out of scope**; Appendix A RRL support matrix |
| `docs/methodology-flowcharts.md` | Mermaid flowcharts: simple and detailed preprocessing, full methodology; RRL source per preprocessing step. **Update it when the method changes.** |
| `docs/rrl-decision-log.md` | **Every decision**, with options, choice, sources and date. The decision register near the top; action log at the bottom. |
| `docs/researcher-notes.md` | **Plain-language notes N0–N26** the team must understand and defend: dataset and tasks, label justification, talking confound, normalization bias, event detection, feature choices, scope |
| `docs/rrl-colet-list.md` | Full IEEE citations for the 139 sources used by the COLET method, with IDs (e.g. EM7, PP1) |
| `docs/rrl-master-list.md` | Every source found (superset), the "do not cite" list (§11), and dataset surveys (§9b) |
| `docs/review-issues-status.md` | Status of review rounds 1–4 |
| `docs/review-issues-status-2.md`, `-3.md`, `-4.md` | Round-2 to round-4 independent reviews (reviewer's files; don't edit) |
| `docs/colet-eda/` | `README.md` (exploration findings), `eda_colet.py` (first-pass, **hand-rolled; its fixation/saccade numbers must not be used**), `check_r4_1.py` + `.csv` (gaze-input check, N25), `dataset-overview.html`, figures |
| `src/` | **The COLET pipeline** (written, **not run on real data**): `python src/run_colet.py`; `run_colet.py` is the entry point, `config.py` holds the settings. Tables written to `outputs/colet/tables/`: `01_retention`, `01b_activities_per_participant`, `02_sanity_check`, `03_manipulation_check`, `04_feature_table`, `05_results`, `06_comparison`, `07_importance_*`, `08_redundancy_v2_13`, `09_chosen_params_*` (per-fold hyperparameters), `10_missingness` |
| `notebooks/colet_pipeline.ipynb` | Colab notebook: clones the repo from GitHub, installs pinned packages, runs Stages 1–5 on the parquet data in Drive |
| `requirements-colab.txt` | Pinned package versions for Colab (the notebook checks them) |
| `archive/` | GAZELOAD-era docs, outputs, old drafts, and the earlier methodology version (see `archive/README.md`) |
| `1-s2.0-S0169260722003716-main.pdf` (repo root) | COLET paper full text |

**External links:**
- Round-1 review doc: https://claude.ai/artifact/AXyhbekkL2Fr6oVtVd83Kb
- Dataset overview page: https://claude.ai/artifact/AER29vvMmtxAufBHiHjq5N
- ADABase email draft: https://claude.ai/artifact/AN3Ju9LPK7wscrHZcTtF6D

## 3. The study, as decided

- **Labels:**
  - **A1** (single task, no time pressure) = **low load**;
  - **A4** (counting aloud + time pressure) = **high load**;
  - A2/A3 are left out of the main analysis (C2; justification N0b).
- **Unit:** one sample per **whole activity** (C1).
- **Exclusion:** recordings with > 35% gaze samples below 0.8 confidence (C3). P06 and P17 are
  dropped, leaving **45 participants, 90 samples**.
- **Preprocessing:** steps P0–P14 in `methodology-colet.md` §4. Includes:
  - trimming the stray P18 pupil samples;
  - computing rates over valid time;
  - the Kret & Sjak-Shie pupil filter;
  - counting 3D-model refits.
- **Gaze to angles:** each eye's **gaze direction** (`gaze_normal0/1`), averaged over the two
  eyes (D-M1 revised, R4-1). **Not** `gaze_point_3d` (depth estimate broken: median 113 mm vs
  800 mm, points behind the camera) and **not** `norm_pos` (world-camera coordinates, same
  faulty point). One-eye samples treated as missing (decided 2026-10-02; each eye alone is about
  10° off; N25).
- **Event detection:** I-VT 45°/s, minimum fixation 55 ms (COLET's settings). **Sanity check,
  per activity:** median fixation 150–400 ms and saccade:fixation ratio 0.8–1.25. If either
  fails in any activity, use the 5-feature fallback (V2-2, V3-3, R4-2).
- **10 features (C4; N21):**
  1. pupil mean;
  2. pupil SD;
  3. blink rate;
  4. fixation rate;
  5. fixation duration;
  6. saccade rate;
  7. saccade amplitude;
  8. saccade peak velocity;
  9. gaze spread x;
  10. gaze spread y.

  Dropped: blink duration, mean saccade velocity, saccade duration, higher-order stats.
  Entropy exploratory.
- **Normalization:** per-person z-score over each person's 4 activities. Label-free; the bias
  is disclosed (D-M4; N7).
- **Models:**
  - **elastic-net LR** (D-N6);
  - **XGBoost**, depth 1–3, 50–300 trees (V2-9);
  - equal tuning budget.
- **Validation:** leave-one-participant-out, with participant-grouped inner tuning.
  Preprocessing fitted in-fold.
- **Metrics:**
  - pooled AUC with participant cluster bootstrap CI;
  - balanced accuracy, F1, majority baseline, Brier score.
- **Importance:**
  - LR weights;
  - **SHAP for both models** (D-M8);
  - permutation importance on pooled out-of-fold predictions;
  - Kendall τ with a CI.
- **Required analyses:**
  - **P1** (10 features);
  - **P2** (pupil-only: the talking-robustness check, V2-1).
- **Ceiling rule:** if both models exceed 0.95 AUC in P1, interpret the comparison from P2;
  importance is still reported from P1 (V3-2).
- **P2 limitation:** P2 is not fully talking-free (refits more frequent in multitask; V3-1).
- **Everything else is optional** (N19) or **out of scope** (§17).

## 4. Data locations (not in git; `data/` is git-ignored)

| Path | Contents |
|---|---|
| `data/colet/COLET_v3.zip` | Zenodo record 7766785 (MD5 `5a9e73bf4ba109f99c30f7c1e24e4617`) |
| `data/colet/data_v3.mat` | 3.8 GB, MATLAB v7.3 with `table` objects; needs the `mat-io` package |
| `data/colet/colet_loaded.pkl` | Decoded snapshot (2.35 GB). **Use this;** decoding the .mat takes about 50 minutes. |
| `data/colet/parquet/` | 564 files `pXX_tY_{gaze,pupil,blinks}.parquet` + `annotation.csv` (NASA-RTLX) + `subject_info.csv` |
| `data/colet/inventory.csv` | Per-recording durations, rates, confidence, blinks |
| `data/colet/convert_colet.py` | pkl → parquet + inventory. **Not version-controlled** (inside `data/`); planned move to `src/`. |
| `data/colet/images/` | The 21 puzzle images |

**Key data facts:**
- 47 participants × 4 activities; gaze stream ≈ 242 Hz (binocular, measured); pupil ≈ 121 Hz per
  eye (each sample appears twice, as 2d and 3d rows; N26).
- Activities last 11–141 s; no rest baseline; no fixation/saccade events in the release.
- Pupil rows are duplicated as `2d c++` and `3d c++`; use the 3d rows for `diameter_3d`.

## 5. Open items

**For the next agent (no user decision needed):**
1. ~~V2-6~~ done (gap claims updated in `methodology-colet.md` §1).
2. ~~Write the COLET pipeline code~~ **done: code written, not run on real data** (`src/`;
   see §2). **Do not run it until the user says so.**
3. Move `data/colet/convert_colet.py` into `src/` (ask the user first).
4. ~~Colab notebook with pinned versions (N11)~~ **done: code written, not run on real data**
   (`notebooks/colet_pipeline.ipynb`, `requirements-colab.txt`).
5. **V2-13:** recheck the correlations behind the dropped features, after the pipeline runs.

**For the user (decision needed):**
- ~~How Colab gets the code~~ **decided 2026-10-02: from GitHub** (`elmigue21/csrp`; token if the
  repo is private). Parquet data on Google Drive.
- **Decided 2026-10-02:** methodology **frozen**; a **Colab notebook is required** (the team runs
  the pipeline on Colab); one-eye samples → missing; C8 optional.

**For the user (on hold or pinned):**
- C5 adviser sign-off;
- C6 ADABase email;
- C7 reference checks (years, author lists, abstract-only full texts);
- ~~C8 stimulus-luminance check~~ (optional, 2026-10-02).

**Later:** rewrite thesis Chapters 1–3 for COLET; Chapters 4–5 from the results.

## 6. How to work with this user

- **Never commit.** Give `git add` commands and suggest a commit message (user rule). Match
  existing code conventions.
- **Every methodology step must be RRL-supported:**
  - peer-reviewed sources only, existence confirmed on a publisher or DOI page,
    claim-matched;
  - if support is weak, propose changing the method, not padding citations;
  - say clearly when something is our own choice rather than from the RRL.
- **Scope rule:** keep the scope small. Only what P1/P2 need; everything else is optional,
  a limitation, or future work. Don't add analyses just because a study or reviewer
  suggested them. Test for each reviewer item: "is it needed for a valid P1/P2 result?"
  (N24).
- **Explain simply,** with examples and tables. Expect follow-ups like "what did the RRL do?"
  and "explain simpler". Answer each with the RRL evidence.
- **Record every decision** in `rrl-decision-log.md`, and **record awareness items** in
  `researcher-notes.md` as new numbered notes (next is **N29**). Update
  `methodology-colet.md` so the files never contradict each other; reviewers check.
- **Ask before** downloading, installing packages, or running long jobs.
- Don't choose features by how well they separate A1 from A4. Stay label-blind.

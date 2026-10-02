# Methodology Flowcharts — COLET

Flowcharts of the plan in `methodology-colet.md` (as of 2026-10-01, after the round-4 review).
They render in GitHub, VS Code (Mermaid extension), Obsidian, or https://mermaid.live.

- **Figure 1** is the simple version for the thesis (Chapter 3).
- **Figure 2** is the detailed version, a reference for writing the pipeline code.
- **Figure 3** is the full methodology.
- The table after Figure 2 lists the RRL source for each preprocessing step.

Not shown: optional analyses (S1–S5) and the pinned luminance check (C8), which are outside the
required scope (`methodology-colet.md` §0, §17).

---

## Figure 1. Preprocessing (simple, for the thesis)

```mermaid
flowchart TD
    RAW["COLET raw data<br/>gaze ≈ 242 Hz · pupil ≈ 121 Hz per eye · blinks"]
    RAW --> S1["1 · Remove bad data<br/>low-confidence samples;<br/>recordings over 35% bad"]
    S1 --> S2["2 · Clean each signal<br/>blinks · pupil · gaze direction"]
    S2 --> S3["3 · Detect fixations and saccades<br/>I-VT 45 °/s + sanity check"]
    S3 --> CHK{"Sanity check<br/>passed?"}
    CHK -- "Yes" --> F10["4 · Compute 10 features<br/>per activity"]
    CHK -- "No" --> F5["4 · Compute 5 features<br/>(no fixation/saccade features)"]
    F10 --> OUT["90 samples<br/>45 participants × A1 and A4"]
    F5 --> OUT
```

| Stage | Plain words | Steps (`methodology-colet.md` §4) |
|---|---|---|
| 1. Remove bad data | Drop samples the tracker wasn't sure about, and recordings that are mostly bad | P1, P2, P3 |
| 2. Clean each signal | Blinks: keep real ones. Pupil: standard cleaning. Gaze: average the two eyes, fill tiny gaps | P5, P6, P7, P8 |
| 3. Find fixations and saccades | When the eye holds still vs jumps; check the result looks normal | P9 (§5) |
| 4. Compute features | One number per feature per activity | P4, P10 |

P0 (inventory), P11 (quality report) and P14 (retention table, incl. refit counts) only report
numbers; they do not change the data.

## Figure 2. Preprocessing (detailed, for the code)

```mermaid
flowchart TD
    RAW["COLET raw data<br/>47 participants × 4 activities<br/>gaze ≈ 242 Hz · pupil ≈ 121 Hz per eye · blinks"]

    RAW --> P0["P0 · Inventory<br/>duration, sampling rate, gaps"]
    P0 --> P1["P1 · Time alignment<br/>trim pupil and blinks to the gaze time range"]
    P1 --> P2["P2 · Sample validity<br/>confidence below 0.8 = invalid"]
    P2 --> P3{"P3 · More than 35% of<br/>samples invalid?"}
    P3 -- "Yes" --> EXC["Exclude recording<br/>P06 A3, P06 A4, P17 A4<br/>→ P06 and P17 dropped"]
    P3 -- "No" --> KEEP["45 participants kept"]

    KEEP --> BL
    KEEP --> GZ
    KEEP --> PU

    subgraph BLINKS["Blinks"]
        BL["P5 · COLET blink events<br/>keep 50–500 ms<br/>merge blinks less than 100 ms apart"]
    end

    subgraph GAZE["Gaze"]
        GZ["Blink periods → missing"]
        GZ --> G1["One-eye samples → missing"]
        G1 --> G2["P8 · Average both eyes' directions<br/>gaze_normal0/1"]
        G2 --> G3["P6 · Fill gaps under 75 ms<br/>longer gaps stay missing"]
        G3 --> G4["Angle between consecutive<br/>directions, in degrees"]
    end

    subgraph EVENTS["P9 · Event detection"]
        E1["Velocity, 5-tap FIR filter"]
        E1 --> E2["Reject over 1000 °/s"]
        E2 --> E3["I-VT at 45 °/s"]
        E3 --> E4["Keep fixations of 55 ms or more"]
        E4 --> SC{"Sanity check, per activity<br/>fixation median 150–400 ms<br/>saccade:fixation 0.8–1.25"}
    end

    subgraph PUPIL["Pupil"]
        PU["Blink periods → missing"]
        PU --> U1["P7 · diameter_3d from 3d rows<br/>keep 1.5–9 mm"]
        U1 --> U2["MAD speed-outlier filter"]
        U2 --> U3["Average both eyes<br/>fill gaps of 250 ms or less"]
        U3 --> U4["4 Hz low-pass"]
        U4 --> U5["Count 3D-model refits"]
    end

    G4 --> E1
    BL --> VT
    G4 --> VT
    U5 --> VT

    VT["P4 · Valid time<br/>exclude gaps over 1 s"]

    SC -- "Pass in all activities" --> F10["P10 · 10 features<br/>per whole activity, as rates"]
    SC -- "Fail in any activity" --> F5["P10 · 5-feature fallback<br/>pupil mean, pupil SD, blink rate,<br/>gaze spread x and y"]
    VT --> F10
    VT --> F5

    F10 --> OUT["90 samples<br/>45 participants × A1 and A4"]
    F5 --> OUT

    OUT -.-> QR["P11 · Quality report by condition<br/>(not a feature)"]
    OUT -.-> RT["P14 · Retention table<br/>incl. refit counts"]
```

### Where each preprocessing step comes from

- ✅ **Done by a study:** an eye-tracking study did this step.
- 📘 **Guideline:** a methods paper recommends it.
- ⚙️ **Our own:** no source; justified by our data check (design choice, `methodology-colet.md` §13).

No single RRL study did the whole pipeline; each step has its own source. Full citations:
`rrl-colet-list.md`.

| Step | Type | Source |
|---|---|---|
| Confidence < 0.8 = bad sample | ✅ | Faraji 2023 (Pupil Core) |
| Drop recordings > 35% bad | ✅ + ⚙️ | Nenna 2023 (35% per trial); applying it per recording is our choice |
| Trim stray pupil samples (P18) | ⚙️ | Data check (researcher-notes N14) |
| Rates over valid time (gaps > 1 s out) | ⚙️ | Data check (N14); the 1 s cutoff is our choice |
| Blinks 50–500 ms; merge < 100 ms apart | 📘 | Hershman 2018 (merge); range derived from Steinhauer 2022 and Hershman 2018 |
| Fill gaze gaps < 75 ms | ✅ | Faraji 2023 (Pupil Core) |
| Pupil cleaning (1.5–9 mm, MAD, 4 Hz) | 📘 | Kret & Sjak-Shie 2019; Mathôt 2018 |
| Average both eyes' directions (`gaze_normal`) | ✅ + ⚙️ | Kothari 2020 (velocity from direction vectors, Pupil Labs); Hooge 2019 and Velisar & Shanidze 2024 (depth guess unreliable); the exact column is our choice (N25) |
| One-eye samples → missing | ⚙️ | Data check (N25); follows Faraji 2023 (unreliable samples = gaps); decided 2026-10-02 |
| 5-tap filter, I-VT 45°/s, fixations ≥ 55 ms | ✅ | COLET (Ktistakis 2022), following Duchowski 2017, Salvucci & Goldberg 2000, Andersson 2017, Trabulsi 2021 |
| Reject > 1000°/s | ✅ | Hausamann 2020 |
| Sanity check (150–400 ms; ratio 0.8–1.25) | 📘 + ⚙️ | Komogortsev 2010 (detection must be checked); typical values from COLET and Salvucci & Goldberg 2000; the rule and tolerance are ours |
| 5-feature fallback | ✅ | Božak 2026 (feature-subset model) |
| One value per whole activity | ✅ | COLET (Ktistakis 2022); Kaczorowska 2021; Wu 2020 |
| Count pupil-model refits | ⚙️ | No RRL study discusses refits (N23) |

Sources read at abstract level only so far (to be read in full before the defense, C7): Hooge
2019, Velisar & Shanidze 2024, Di Stasi 2011, among others (`methodology-colet.md` Appendix A).

## Figure 3. Methodology

```mermaid
flowchart TD
    D["COLET dataset<br/>47 participants, 4 activities"]

    D --> LAB{"Activity"}
    LAB -- "A1: single task,<br/>no time pressure" --> LOW["Low load"]
    LAB -- "A4: counting aloud<br/>+ time pressure" --> HIGH["High load"]
    LAB -- "A2, A3" --> MC["Manipulation check only<br/>NASA-RTLX, Friedman test"]

    LOW --> PRE["Preprocessing<br/>(Figures 1–2)"]
    HIGH --> PRE
    PRE --> DATA["90 samples · 10 features<br/>balanced 1:1"]

    DATA --> P1["P1 · All 10 features"]
    DATA --> P2["P2 · Pupil mean and SD only<br/>talking-robustness check"]

    P1 --> LOPO
    P2 --> LOPO

    subgraph LOPO["Leave-one-participant-out · 45 folds"]
        direction TB
        Z["Per-person z-score<br/>over each person's own activities, label-free"]
        Z --> SPLIT["Train: 44 participants<br/>Test: 1 participant"]
        SPLIT --> FIT["Fit scaling and imputation<br/>on training fold only"]
        FIT --> TUNE["Inner participant-grouped CV<br/>equal tuning budget"]
        TUNE --> LR["Logistic regression<br/>elastic net"]
        TUNE --> XGB["XGBoost<br/>depth 1–3, 50–300 trees"]
        LR --> PRED["Predict held-out participant"]
        XGB --> PRED
    end

    PRED --> POOL["Pooled out-of-fold predictions"]

    POOL --> MET["Metrics<br/>ROC-AUC + participant cluster bootstrap CI<br/>balanced accuracy, macro-F1, Brier,<br/>majority baseline"]
    POOL --> CMP["Model comparison<br/>AUC difference + CI<br/>paired Wilcoxon<br/>permutation test vs chance"]
    POOL --> IMP["Feature importance<br/>LR weights · SHAP for both models<br/>permutation importance<br/>Kendall τ with CI"]

    MET --> CEIL{"Ceiling rule:<br/>both models above<br/>0.95 AUC in P1?"}
    CEIL -- "No" --> R1["Compare models from P1"]
    CEIL -- "Yes" --> R2["Compare models from P2<br/>importance still from P1"]

    R1 --> REP["Report and discuss<br/>limitations: talking, refits in P2,<br/>per-person normalization bias"]
    R2 --> REP
    CMP --> REP
    IMP --> REP
```

**Reading notes:**
- The per-person z-score sits inside the loop because it is applied to every person, the test
  person included, using only that person's own unlabelled recordings. This is the disclosed
  bias (researcher-notes N7).
- RRL for each methodology step: `methodology-colet.md` §§2–11 and Appendix A.

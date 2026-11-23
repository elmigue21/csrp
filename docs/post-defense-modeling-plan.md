# Post-defense modeling plan

**Status:** all four decisions made by the team on 2026-11-11 (MD1–MD4, scope kept small).
Nothing has been run yet; the code changes in §4 come next. Decide everything here **before** the
first model runs, and do not change a decision because of results (same rule as the data
decisions, `post-defense-data-processing.md` §6).

**Input:** `outputs/final/features/features.csv`: 88 rows (44 participants × A1 low / A4 high),
140 features, no blanks, no per-person standardization.

**Related files:**
- `post-defense-data-processing.md`: how the feature table was made (all decisions final).
- `methodology-colet.md`: the defended modeling setup. Most of it carries over (M1–M4 below).
- Code that exists today: `src/models.py`, `src/evaluate.py`, `src/run_colet.py` (built for the
  old 10-feature, z-scored table; it needs adapting, see §4).

## 1. The idea in one paragraph

We train a model on 43 people and test it on the 1 person it has never seen, then repeat
until every person has been the test person once (LOPO, 44 rounds). In each round, the model
may only learn from the 43 training people: which features to use, how many, how to scale
them, and the model settings. The test person is touched only once, at the very end of the
round, to get a prediction. All 44 predictions together give the final score.

- **How to explain it:** "Every person is tested by a model that never saw them, and every
  choice the model made (features, settings) was made without them."

## 2. What carries over from the defended version (no decision needed)

| # | Item | Setting | Where |
|---|---|---|---|
| M1 | Models | Logistic regression with elastic-net penalty vs XGBoost (shallow trees) | `src/models.py` |
| M2 | Evaluation | LOPO (44 outer folds); 5 participant-grouped inner folds for tuning | `src/evaluate.py` |
| M3 | Tuning | Random search, 30 candidates per model (equal budget), scored by ROC-AUC | `config.N_ITER` |
| M4 | Reporting | Pooled ROC-AUC with a participant-bootstrap 95% CI (2,000 resamples), permutation test against chance (200), majority-class baseline, LR vs XGBoost difference | `src/evaluate.py` |
| M5 | Scaling for LR | Each feature rescaled to mean 0, SD 1 using the **training people only**, inside the pipeline. This is not the dropped per-person z-score: it never sees the test person | `make_logistic_regression()` |

- **How to explain M5:** "Logistic regression needs features on one scale, so we rescaled
  them, learning the scale from the training people only."

## 3. Decisions to make

### MD1. How features are ranked inside each fold — **Decided 2026-11-11 (team): (a) simple filter**
The top-N needs a ranking. The ranking must be computed on the 43 training people only.

| Option | How it works | For | Against |
|---|---|---|---|
| **(a) Simple filter** (recommended) | Score each feature alone: how well does it separate A1 from A4 in the training people (univariate ROC-AUC, distance from 0.5). Keep the top N. | Same ranking for both models, so the comparison stays fair. Easy to explain. Fast | Ignores that two features may carry the same information |
| (b) Model-based | Fit the model, rank by its own importance (LR coefficients, XGBoost gain), keep the top N, refit | Uses the model's view | Each model picks different features; slower; harder to explain |
| (c) No selection | Give all 140 to the model and let regularization shrink the useless ones | Simplest; no extra step | Doesn't answer "which top 20?" |

- **Decision:** (a). "A feature is ranked by how well it alone separates low from high load
  in the training people."
- **Why not SHAP for picking:** SHAP explains a model that is already trained, so it is used
  *after* training (MD4). Using it to pick features would mean training twice per round, and LR
  and XGBoost would each pick different features, which makes the comparison less fair.
- **How to explain it:** "In each round, we ranked the features using only the training
  people, kept the best N, and then trained the model on those."

### MD2. How many features (N) — **Decided 2026-11-11 (team): tuned inside each fold**
You want to try different N (10, 20, 30, 40, …). Picking the N with the best test score is
the same mistake as ranking on all the data: the test people would have chosen N.

| Option | How it works |
|---|---|
| **(a) N chosen inside the fold, plus a curve** (recommended) | N is one more setting tuned in the inner folds (like the model settings), from {5, 10, 20, 40, 80, 140}. That gives the main result. Separately, run the whole LOPO once per fixed N and draw "AUC vs N" as a descriptive curve |
| (b) Fixed N decided now | Pick one N today (for example 20) before any run |
| (c) Try N values, report the best | **Not allowed:** the reported number would be chosen by the test results |

- **Decision:** N is found by the random search, together with the model settings, using only
  the 43 training people in each round ("try randomly until we find the sweet spot", done by the
  computer inside the training data). Candidate values: {5, 10, 20, 40, 80, 140}.
- **Not allowed:** looking at test results and keeping the N that scored best.
- The descriptive AUC-vs-N curve is not planned; add it only if the team asks.
- **How to explain it:** "The number of features was tuned like any other setting, using only
  the training people. We also show how performance changes with N, reporting every value
  we tried."

### MD3. Which analyses to run — **Decided 2026-11-11 (team): P1 only**

| Name | Features | Question it answers |
|---|---|---|
| **P1** | All 140 (top-N inside each fold) | Main question: can eye features tell low from high load? |
| **P2** | Pupil only (16 features) | Does pupil size alone do it? (kept from the defended version) |
| **P3** (new, recommended) | 140 minus the 22 `position`-tagged features = 118 | Does the result survive without features that could reflect headset or screen position (D10, gaze cv)? |

- **Decision:** run **P1 only** (all 140 features, top-N inside each fold). P2 (pupil only) and
  P3 (without `position` features) are not run.
- **Consequence:** the defended thesis has P2 in its objectives, method and results; those
  parts are removed (`thesis-changes-post-defense.md`). If a `position` feature ranks highly, it
  is reported with the headset/screen caveat (D10) instead of a separate P3 run.

### MD4. Feature importance — **Decided 2026-11-11 (team): SHAP + selection count only**
- **What stays:** SHAP and permutation importance on the outer folds (as defended).
- **New, recommended:**
  1. **Selection frequency:** how often each feature made the top N across the 44 folds
     (for example, "pupil SD was picked in 44 of 44 folds"). This is the most direct answer to
     "which features matter?".
  2. **Grouped by signal:** sum the importance of all statistics of one signal (pupil, blink
     rate, saccades, …). Many statistics are near-copies (mean and median), and copies split the
     importance between them, so a signal can look weaker than it is when read one feature at
     a time.
  3. **Caveats when reporting:** a `position`-tagged feature that ranks highly is reported with
     the headset/screen caveat (D10); `min`/`max`/`range` are checked against `p05`/`p95`
     because A4 recordings are longer (D3).
- **Decided:** keep **SHAP** (it delivers the title's "feature importance analysis" and is the
  only tool that shows what *each* model relied on, so LR and XGBoost can be compared) and add
  **selection count** (how often each feature made the top N across the 44 rounds).
- **SHAP with top-N:** each round may pick different features. A feature not picked in a round
  counts as 0 importance for that round; importance is then averaged over all 44 rounds.
- **Dropped to keep the scope small:** permutation importance (so Objective 5's "ranking
  agreement" between SHAP and permutation importance is removed from the thesis) and
  per-signal totals.
- **How to explain it:** "SHAP shows which features each model relied on and in which
  direction. We also counted how often each feature was chosen across the 44 rounds."

### MD5. Thesis numbers to replace
Every result in the defended draft (AUCs, CIs, SHAP plots, importance tables) comes from the
old 10-feature, z-scored table and must be re-run. `docs/thesis-changes-post-defense.md` lists
where they appear.

## 4. Code (built 2026-11-11)

| File | What it does |
|---|---|
| `src/topn.py` | `TopNByAUC`: scores each feature alone on the rows it is fitted on (MD1) and keeps the best N. The LR pipeline is impute → top N → scale → elastic-net LR; the XGBoost pipeline is top N → XGBoost. N is in both search spaces, from {5, 10, 20, 40, 80, 140} (MD2) |
| `src/run_model.py` | Reads `outputs/final/features/features.csv`, runs LOPO for both models with the existing `evaluate.py` (inner tuning, bootstrap CI, comparison, chance test, baseline), then SHAP and selection count (MD4). No per-person z-score, no P2 |
| `outputs/final/results/` | Everything it writes (see its README) |

- **Run it:** `python src/run_model.py` (after `python src/run_extract.py`). It runs three
  stages: `main` (`results.csv` etc.), `by_n` (`results_by_n.csv`) and `checks` (`checks.csv`);
  `--stages by_n,checks` runs only some. The by-N and check rows are saved one at a time, so
  a crash loses at most one run, and rerunning skips rows already saved for the same features
  file. About 1.5 hours in total with 6 workers (`COLET_JOBS`, default 6; 20 ran out of memory).
- **Safety:** re-running the extraction keeps `results/`. If the features changed since the
  models ran, `results/STALE.md` appears, and the models must be re-run.
- **Chance test:** run with default settings (N = 20), as in the defended version, because
  tuned settings were chosen with the true labels.
- **SHAP:** each fold's model explains only the participant it never saw; a feature that fold
  did not pick counts as 0.
- `src/run_colet.py` and `src/models.py` stay as the defended record; `run_model.py` reuses
  their model settings.

## 5. Decision summary

| # | Decision | Recommendation | Status |
|---|---|---|---|
| MD1 | How to rank features in each fold | Univariate ROC-AUC on training people | **Decided** |
| MD2 | How many features | N found by random search inside the training people, from {5, 10, 20, 40, 80, 140} | **Decided** |
| MD3 | Analyses | P1 only (all 140 features) | **Decided** |
| MD4 | Importance | SHAP + selection count only (no permutation importance, no per-signal totals) | **Decided** |

## 6. Results (recorded run 2026-11-11, `outputs/final/results/`)

**These are the recorded numbers; nothing needs to be re-run to quote them.** All three stages
ran on the same features file (`features_sha256` 7249a59b…, in every results row and in
`run_info.json`): main 18:26, by-N 19:21, checks 19:41. Settings: 44 people, 88 rows, 140
features, 30 search candidates, 2,000 bootstrap resamples, 200 chance-test shuffles. A rerun
gives the same numbers (fixed random seed; the main stage matched the 16:29 run exactly).

| File | What is in it |
|---|---|
| `results.csv` | Main result: both models, N tuned inside each fold (§6 table) |
| `results_by_n.csv` | Same run with N fixed at 5, 10, 20, 40, 80, 140 (§6.0) |
| `checks.csv` | Blink rate alone, and both models without blink rate (§6.1) |
| `comparison.csv` | LR vs XGBoost |

**Main result, every metric** (threshold 0.5; "low" = A1, "high" = A4):

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC (95% CI) | 0.862 (0.794–0.933) | 0.862 (0.795–0.919) |
| Accuracy | 0.830 | 0.739 |
| Balanced accuracy | 0.830 | 0.739 |
| Precision, macro | 0.844 | 0.740 |
| Precision, low / high | 0.774 / 0.914 | 0.723 / 0.756 |
| Recall, low / high | 0.932 / 0.727 | 0.773 / 0.705 |
| F1, macro | 0.828 | 0.738 |
| F1, low / high | 0.845 / 0.810 | 0.747 / 0.729 |
| Brier score (lower = better) | 0.172 | 0.167 |
| Participants with A4 ranked above A1 | 42 of 44 | 40 of 44 |
| Chance test (200 shuffles) | p = 0.005 (the smallest possible) | p = 0.005 |
| Number of features chosen (median) | 5 (5 in 42 of 44 folds) | 10 (5 to 140) |
| Majority-class baseline (balanced accuracy) | 0.500 | 0.500 |

- Accuracy equals balanced accuracy because every person has one A1 and one A4 (44 of each).
- **How to explain it:** "Accuracy is how many recordings it got right. Precision is: when it
  says 'high load', how often that is true. Recall is: of the real high-load recordings, how
  many it found. F1 combines precision and recall into one number."

- **LR vs XGBoost:** same pooled AUC (difference 0.000, 95% CI −0.054 to 0.048). The paired
  Wilcoxon on per-participant margins is significant (p < 0.001): XGBoost separates each
  person's two recordings by a wider margin (median 0.40 vs 0.20; wider for 36 of 44 people),
  but neither model ranks the recordings better overall. Margins depend on each model's
  probability scale (LR is strongly regularized, median C = 0.07), so this is secondary.
  (Corrected 2026-11-11: an earlier version of this line said LR had the wider margin.)
- **Picked in all 44 folds by both models:** `blink_rate`, `saccade_rate`, `gaze_y__min`,
  `gaze_x__range`, `gaze_x__kurt`. No pupil feature was picked by LR in more than 2 folds.
- **SHAP, top 2 for both models:** `blink_rate`, then `gaze_y__min`.
- **Directions (medians, A1 → A4):** blink rate 1.7 → 11.7 per min (higher in A4 for 42 of 44);
  saccade rate 3.4 → 2.9 per s (lower for 38 of 44); lowest up/down gaze −14.5° → −23.2° (lower
  for 34 of 44); left/right gaze range 14.9° → 21.8°.
- **Caveat that applies now (D10):** `gaze_y__min` is tagged `position`. People looked about 9°
  lower at some point in A4. That may reflect where something appeared on the screen in A4, or
  how the headset sat, rather than mental load. It must be reported with that caveat.
- **Ceiling rule** (AUC > 0.95 for both models): not triggered.

### 6.0 Results by number of features (the "top 5 / top 20 / top 140" check)

Same LOPO, but N is fixed instead of tuned; the other settings are still tuned inside each fold.
File: `results_by_n.csv`. Every metric:

| Model | N | ROC-AUC (95% CI) | Accuracy | Precision macro (low / high) | Recall low / high | F1 macro (low / high) | Brier | A4 > A1 |
|---|---|---|---|---|---|---|---|---|
| LR | 5 | 0.855 (0.786–0.927) | 0.807 | 0.827 (0.745 / 0.909) | 0.932 / 0.682 | 0.804 (0.828 / 0.779) | 0.195 | 42/44 |
| LR | 10 | 0.855 (0.784–0.928) | 0.795 | 0.812 (0.741 / 0.882) | 0.909 / 0.682 | 0.793 (0.816 / 0.769) | 0.219 | 42/44 |
| LR | 20 | 0.868 (0.799–0.939) | 0.807 | 0.836 (0.737 / 0.935) | 0.955 / 0.659 | 0.803 (0.832 / 0.773) | 0.218 | 42/44 |
| LR | 40 | 0.876 (0.810–0.943) | 0.807 | 0.836 (0.737 / 0.935) | 0.955 / 0.659 | 0.803 (0.832 / 0.773) | 0.227 | 42/44 |
| LR | 80 | 0.876 (0.810–0.943) | 0.807 | 0.836 (0.737 / 0.935) | 0.955 / 0.659 | 0.803 (0.832 / 0.773) | 0.226 | 42/44 |
| LR | 140 | 0.876 (0.810–0.943) | 0.807 | 0.836 (0.737 / 0.935) | 0.955 / 0.659 | 0.803 (0.832 / 0.773) | 0.226 | 42/44 |
| XGBoost | 5 | 0.884 (0.821–0.940) | 0.818 | 0.819 (0.833 / 0.804) | 0.795 / 0.841 | 0.818 (0.814 / 0.822) | 0.137 | 42/44 |
| XGBoost | 10 | 0.861 (0.798–0.921) | 0.761 | 0.761 (0.767 / 0.756) | 0.750 / 0.773 | 0.761 (0.759 / 0.764) | 0.152 | 42/44 |
| XGBoost | 20 | 0.849 (0.785–0.916) | 0.773 | 0.773 (0.761 / 0.786) | 0.795 / 0.750 | 0.773 (0.778 / 0.767) | 0.157 | 41/44 |
| XGBoost | 40 | 0.864 (0.796–0.928) | 0.750 | 0.752 (0.729 / 0.775) | 0.795 / 0.705 | 0.749 (0.761 / 0.738) | 0.152 | 41/44 |
| XGBoost | 80 | 0.853 (0.773–0.928) | 0.807 | 0.808 (0.787 / 0.829) | 0.841 / 0.773 | 0.807 (0.813 / 0.800) | 0.151 | 42/44 |
| XGBoost | 140 | 0.838 (0.759–0.917) | 0.773 | 0.773 (0.761 / 0.786) | 0.795 / 0.750 | 0.773 (0.778 / 0.767) | 0.160 | 41/44 |

- **The sweet spot is flat.** Every N from 5 to 140 lands between AUC 0.84 and 0.88, and all the
  confidence intervals overlap heavily. No N is clearly better than another.
- **LR is identical from N = 40 up** because elastic-net LR switches off the extra features
  itself, so adding more makes no difference.
- **XGBoost is best at N = 5** (AUC 0.884, accuracy 0.818), but its CI (0.821–0.940) overlaps
  every other N.
- **Do not report N = 5 as the result.** This table was made by looking at the test people's
  scores, so picking the best row from it would be choosing on the test set. The honest result
  is the main table above, where N was picked inside each fold without seeing the test person.
  This table only shows that the result does not depend much on N.
- **How to explain it:** "We tried keeping the top 5, 10, 20, 40, 80 and all 140 features. The
  score barely changes, so a handful of features already carry almost all the information."

**Why it barely moves (2026-11-11).**
- One feature does most of the work. On its own, blink rate has AUC 0.89. Only 1 of 140 features
  reaches 0.80 alone, and 9 reach 0.70. Blink rate is ranked first in every fold, so it is in the
  model at every N. (Single-feature AUCs on all 88 rows; a description only, not used by the models.)
- Most features repeat each other. The 17 statistics of one signal (mean, median, p75 …) move
  together: 160 pairs of features have correlation above 0.9.
- The models ignore extras themselves. Elastic-net LR sets them to zero (identical rows from N = 40);
  XGBoost uses depth-1 trees that keep reusing the strongest features.
- The differences between rows are noise: with 88 recordings, one recording is 0.011 of accuracy.

**How much each feature contributed (SHAP, main run; share of the total mean |SHAP|).**
Files: `shap_logistic_regression.csv`, `shap_xgboost.csv`, `selection_count.csv`.

| Feature | LR share | XGBoost share |
|---|---|---|
| blink_rate | 55.9% | 32.6% |
| gaze_y__min (position) | 31.4% | 19.6% |
| saccade_rate | 4.9% | 9.4% |
| gaze_x__kurt | 2.6% | 5.5% |
| gaze_x__range | 5.2% | 5.4% |
| saccade_amplitude__median | 0 | 4.3% |
| fixation_duration__iqr | 0 | 3.5% |
| gaze_x__cv (position) | 0 | 2.7% |
| gaze_x__min (position) | 0 | 2.4% |
| pupil__range | 0 | 2.4% |
| gaze_y__range (position) | 0 | 1.8% |
| gaze_y__iqr (position) | 0 | 1.6% |
| All other features | 0 (the other 128) | 8.8% (the other 128 together; 109 of them above 0) |

- **LR used exactly 5 features**; the other 135 got a weight of 0 and contributed nothing.
- **XGBoost spread the credit wider**: the top 5 give 73%, the top 10 give 88%, and 121 features
  got some credit, each tiny.
- **Blink rate plus lowest up/down gaze = 87% (LR) and 52% (XGBoost)** of the prediction.
- **How to explain it:** "SHAP splits each prediction into how much each feature pushed it. For
  LR, blink rate and where people looked vertically did almost all the pushing; the other
  features added little or nothing."

**What happened to the other statistics (mean, median, SD …).** None were removed by the model
step. All 140 were offered in every fold; most simply lost.
- **33 were dropped before modeling** (173 → 140, `logs/log_features.csv`): 17 blink-duration
  stats (D1, blank in 79 of 88 recordings; 8 of the 9 with values are high load), 9 variances (D8, exact copies of
  SD), 6 ranges (D9, copies of max), 1 saccade-duration minimum (D6a, stuck at one sample).
- **The rest were tested and were weaker.** Per signal (AUC alone = that statistic by itself;
  description only):

| Signal | Stats in model | LR share | XGBoost share | Best stat alone (AUC) | Typical stat alone (median AUC) |
|---|---|---|---|---|---|
| blink_rate | 1 | 55.9% | 32.6% | 0.89 | 0.89 |
| gaze_y | 16 | 31.4% | 23.4% | 0.79 (min) | 0.58 |
| gaze_x | 16 | 7.8% | 16.4% | 0.78 (kurt) | 0.59 |
| saccade_rate | 1 | 4.9% | 9.4% | 0.76 | 0.76 |
| fixation_duration | 15 | 0 | 6.1% | 0.72 (iqr) | 0.63 |
| saccade_amplitude | 15 | 0 | 5.8% | 0.68 | 0.61 |
| pupil | 16 | 0 | 2.9% | 0.71 (range) | 0.62 |
| gaze_speed | 15 | 0 | 1.2% | 0.70 | 0.59 |
| saccade_peak_velocity | 15 | 0 | 0.8% | 0.65 | 0.62 |
| saccade_mean_velocity | 15 | 0 | 0.7% | 0.68 | 0.63 |
| fixation_rate | 1 | 0 | 0.4% | 0.69 | 0.69 |
| saccade_duration | 14 | 0 | 0.3% | 0.67 | 0.59 |

- **Example, pupil:** mean 0.62, median 0.61, p25 to p95 0.62–0.63, SD 0.65, range 0.71. They all
  say the same thing ("pupil a bit bigger in A4"), so XGBoost took pupil range (2.4%) and the other
  15 pupil stats got almost nothing. Same pattern for every signal: one statistic carries the
  signal and its siblings are copies.
- **How to explain it:** "We gave the model every statistic. For each signal, the statistics
  mostly say the same thing, so the model kept the best one and ignored its copies. The typical
  statistic alone is only a little better than guessing (AUC about 0.6)."

### 6.1 Panel check: leakage, or "one feature is enough"? (2026-11-11)

The panel warned that a high score means either leakage or a problem one feature can solve.

**Leakage: none found.**
- Every step that learns from data (picking the top N, scaling, filling blanks, tuning) is
  fitted on the 43 training people only. There is no per-person standardization any more.
- Each recording's features are computed from that recording alone.
- The chance test reruns the whole pipeline, picking included, with shuffled labels. None of
  200 shuffled runs reached the real AUC (p = 0.005). A leak in the pipeline would let shuffled
  runs score high too.

**One feature is enough: yes, this concern holds.**

| Model | LOPO ROC-AUC | Balanced accuracy |
|---|---|---|
| Blink rate alone (one-feature LR, same LOPO) | **0.884** | 0.807 |
| Lowest up/down gaze alone | 0.783 | 0.670 |
| Full LR (top N of 140, tuned) | 0.862 | 0.830 |
| Full XGBoost | 0.862 | 0.739 |

- Blink rate alone does as well as the full models; it is the only feature that reaches 0.80
  alone. Blink rate is higher in A4 for 42 of 44 people.
- **Why that matters:** A4 includes counting backwards **aloud**, and A1 does not. People blink
  more while speaking (`researcher-notes.md` N3, Bentivoglio 1997: 17 → 26 per minute). So the
  strongest signal may partly be talking, not mental load.
- **Note:** the pupil-only analysis (P2) dropped in MD3 was the defended thesis's required
  check for talking (V2-1). Without it, nothing in the current run separates load from talking.
- Recorded in `checks.csv` (row "blink_rate alone"). The "lowest up/down gaze alone" row
  above came from a one-off scratch check and is not recorded.

**Same run without blink rate (139 features), recorded in `checks.csv`. All metrics:**

| Metric | Full LR | Full XGBoost | Blink rate alone | LR without blink rate | XGBoost without blink rate |
|---|---|---|---|---|---|
| ROC-AUC | 0.862 | 0.862 | 0.884 | 0.727 | 0.792 |
| 95% CI | 0.794–0.933 | 0.795–0.919 | 0.822–0.942 | 0.629–0.825 | 0.714–0.865 |
| Accuracy | 0.830 | 0.739 | 0.807 | 0.716 | 0.727 |
| Balanced accuracy | 0.830 | 0.739 | 0.807 | 0.716 | 0.727 |
| Precision, macro | 0.844 | 0.740 | 0.815 | 0.717 | 0.728 |
| Precision, low / high | 0.774 / 0.914 | 0.723 / 0.756 | 0.765 / 0.865 | 0.702 / 0.732 | 0.717 / 0.738 |
| Recall, low / high | 0.932 / 0.727 | 0.773 / 0.705 | 0.886 / 0.727 | 0.750 / 0.682 | 0.750 / 0.705 |
| F1, macro | 0.828 | 0.738 | 0.806 | 0.716 | 0.727 |
| F1, low / high | 0.845 / 0.810 | 0.747 / 0.729 | 0.821 / 0.790 | 0.725 / 0.706 | 0.733 / 0.721 |
| Brier (lower = better) | 0.172 | 0.167 | 0.147 | 0.227 | 0.197 |
| A4 ranked above A1 | 42/44 | 40/44 | 42/44 | 35/44 | 39/44 |
| Chance test p (200 shuffles) | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| Features used (median N) | 5 | 10 | 1 | 60 | 80 |

- The recorded run reproduced the earlier scratch numbers exactly, and now has chance tests:
  all three checks beat every one of the 200 shuffled runs.

- Without blink rate both models drop but stay well above chance (CIs above 0.5): the other
  eye features carry some signal on their own. Most picked without blink rate: gaze_x range and
  kurtosis, gaze_y min, saccade rate (all 44 folds), fixation-duration IQR, pupil range (XGBoost).

**What the 95% CI means (plain language).** With only 44 people, the score would come out a bit
different with another group. The CI is the range the true score probably falls in: we resample
the 44 people with replacement 2,000 times, recompute the score each time, and keep the middle
95%. Narrow = stable; wide = shaky; entirely above 0.5 = better than guessing; two CIs that
overlap heavily (LR 0.79–0.93 vs XGBoost 0.80–0.92) mean neither model is shown to be better.
- **How to explain it:** "The 95% confidence interval is the range the true score likely falls
  in if we had tested a different group of people."

### 6.1a Answering "the model is not needed, one feature gives it away" (2026-11-11)

**Do not claim the model beats blink rate.** It does not: blink rate alone has AUC 0.884,
accuracy 0.807 and Brier 0.147, against the full LR's 0.862, 0.830 and 0.172 (`checks.csv`). The
panel can check this, so concede it and reframe:

1. **Finding blink rate was itself the model's job.** Nobody chose blink rate in advance. The
   pipeline ranked all 140 features using only the training people, and blink rate came first
   in all 44 folds. "One feature is enough" is a result of the study, not something known
   beforehand.
2. **"Blink rate alone" is still a model.** The cutoff between low and high load has to be
   learned from other people. Our blink-only check is a one-feature logistic regression tested
   the same way (LOPO). The real question is how complex the model must be, and the answer is
   "very simple", which matches our finding that LR does as well as XGBoost (AUC difference
   0.000).
3. **The other features are not empty.** Without blink rate, both models still beat chance
   (AUC 0.727 and 0.792, chance test p = 0.005, 35 and 39 of 44 people ranked correctly).
   Combining features gives a higher balanced accuracy (XGBoost 0.727) than the best remaining
   single feature, lowest up/down gaze (0.670), though the AUCs are close (0.792 vs 0.783).
4. **Why we still report it as a limitation.** A4 includes counting aloud, and talking raises
   blink rate. So blink rate may be partly measuring talking, and it may not carry over to
   tasks without speech. A model that also uses gaze and saccade features is less tied to that
   one cue.

- **How to explain it:** "Yes, blink rate alone does as well as the full model, and we report
  that openly. But we only know that because the model tested all 140 features on people it
  had not seen and picked blink rate every time. Even one feature needs a learned cutoff. And
  without blink rate, the other eye features still predict above chance. Since blink rate may
  partly reflect talking during the hard task, we list it as a limitation."

### 6.1b What the flattening added, and are 5 features enough? (2026-11-11)

**The new statistics did matter.** The defended version used 10 features (pupil mean and SD,
blink rate, fixation rate, mean fixation duration, saccade rate, mean saccade amplitude, mean peak
velocity, gaze SD x and y). Every one of them is still in the 140. Share of SHAP credit:

| | LR | XGBoost |
|---|---|---|
| The old 10 | 60.8% (blink rate 55.9 + saccade rate 4.9; the other 8 got 0) | 43.2% (blink rate 32.6 + saccade rate 9.4; the other 8 about 1% together) |
| The 130 new statistics | 39.2% | 56.8% |

- 3 of the 5 features picked in all 44 folds are new: `gaze_y__min`, `gaze_x__range`,
  `gaze_x__kurt`. The old means and SDs were beaten by other statistics of the same signals
  (for example, gaze SD lost to gaze min and range).
- All three new winners are gaze position features (D10 caveat, §6).
- **How to explain it:** "Flattening found statistics the old 10 features did not have: the
  lowest and the spread of where people looked. These carry about 40–57% of the prediction."

**Are 5 features enough for a thesis? Yes, if presented as a result.**
- We did not start with 5. We studied 140, and the procedure chose 5 inside each fold, using
  only the training people. Finding that a few features are enough is the result.
- §6.0 shows this is not luck: every N from 5 to 140 gives about the same score.
- With 88 recordings (44 of each class), fewer features means less risk of overfitting. A
  common rule of thumb is about 10 cases of the smaller class per feature, which is about
  4 features here. Verify the source (Peduzzi et al. 1996) before citing it.
- LR used 5 features; XGBoost used a median of 10.
- **Caveats to state:** 3 of the 5 are gaze position features, and blink rate may partly reflect
  talking (§6.1, §6.1a).
- **How to explain it:** "We tested 140 features and the model only needed about 5. A simpler
  model that does as well is better, especially with 44 people."

### 6.2 Compared with other studies (verified sources, `rrl-master-list.md`)

Only studies that test on people the model never saw are a fair comparison. Metrics differ
between studies (AUC, accuracy, F1), so compare like with like.

| Study | People / task | Tested on unseen people? | Eye-only result |
|---|---|---|---|
| **Ours (COLET, A1 vs A4)** | 44, visual search | **Yes, LOPO** | **AUC 0.862 (LR and XGB); accuracy 0.83 / 0.74; precision 0.84 / 0.74; F1 0.83 / 0.74** |
| He 2025 (EM16) | 30, construction, heat | Yes, LOSO | AUC 0.844 |
| Božak 2026 (EM5) | 54 | Yes, LOSO | AUC 0.848 (LightGBM); majority baseline 76% |
| Choi & Nam 2026 (EM17) | 26, VR | Yes, LOSO | Accuracy 78.9% (XGB) |
| Rolon-Merette 2026 (EM4) | 89, n-back | Yes, LOPO | XGB > 75%; LR 52% (LR given raw time series) |
| Nerella 2026 (X2) | 31, AR search | Yes, participant hold-out | AUC LR 0.76 vs XGB 0.85 |
| Chen 2025 (X13) | 78, collaborative learning | Yes, participant-wise | F1 LR 0.71 vs XGB 0.84 |
| Shao 2026 (X12) | 26, driving | No, random split | AUC LR 0.866 vs XGB 0.86, "no significant differences" |
| COLET authors (EM7) | 47, same data | No, random split, selection on all data | LR accuracy 0.85 |

- **Where we sit:** in the same range as the other unseen-person studies (AUC about 0.84–0.85,
  accuracy about 75–79%). The result is no longer unusually high.
- **LR vs XGBoost:** most studies found XGBoost ahead; we found a tie in AUC, like Shao 2026.
  Our LR has the higher balanced accuracy.
- **Without blink rate** (§6.1) we are below that range (AUC 0.73–0.79).
- **Caution:** different tasks, sensors and metrics. A1 vs A4 compares the two extreme
  activities, which is an easier contrast than graded load levels.
- **How to explain it:** "Tested on people the model never saw, our models reach about the same
  level as comparable studies (AUC around 0.85), so the result is in the expected range, not
  suspiciously high."

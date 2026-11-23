# How to Explain the Code

For presenting the code to the panel. Each step: **what it does**, **what to say**, and likely
**questions**. Updated 2026-11-11 for the post-defense method (140 features, no per-person z-score).

Show `src/demo.py`, and keep `outputs/final/code_preview.html` open for the code behind each step.

```bash
python src/demo.py
```

The full run takes about 8 minutes and prints exactly the paper's numbers: start it early in the
presentation. `--fast` (5 tuning tries instead of 30) takes about 2.5 minutes, but its numbers
differ (e.g. LR accuracy 0.852, not 0.830), so say "quick version" if you use it.

## 0. The big picture (say this first)

> "We want to know if a computer can tell low from high mental load just from the eyes. We compare
> Logistic Regression and XGBoost, and test them only on people they have never seen."

```
1. Load features → 2. Train + tune + test (LOPO) → 3. Every metric → 4. SHAP → results
```

| Word | Plain meaning |
|---|---|
| **Feature** | One number about the eyes in one recording, e.g. blink rate or the mean pupil size. We have 140. |
| **Label** | The answer: low load (A1) or high load (A4). |
| **Training / testing** | Learning from examples with answers / guessing on new ones. |
| **Tuning** | Trying different settings and keeping the best. |
| **Leakage** | The test person's information sneaking into training. |

## 1. Load features (demo.py:47)

**What it does:** Reads `outputs/final/features/features.csv`: 88 rows (44 people × A1 and A4),
140 features. The cleaning and flattening ran earlier in `src/run_extract.py`: every cleaned eye
signal is summarized with 17 statistics (min, max, mean, median, SD, percentiles…).

**Say:** "Each recording is cleaned and summarized into 140 numbers. We did not pick statistics by
hand; every signal gets all 17."

**Questions:**
- *Why 44 of 47?* Three recordings had over 35% bad data, and P31 was an exact copy of P28.
- *Why 140, not 173?* 33 were copies or unusable (for example, variance is just SD squared).

## 2. Train, tune and test (demo.py:27-42)

**Outer loop, LOPO (line 31):** 44 rounds. Hide one person, train on the other 43, test on the
hidden person.

**Inner loop, tuning (lines 34-37):** on the 43 training people only, try 30 random settings
(including **N, how many features to keep**) and keep the best by ROC-AUC over 5 inner folds.

**Inside the model:** the features are ranked by how well each one alone separates low from high
load, and only the top N are kept. This ranking is redone in every round, on training people only
(see `TopNByAUC` in `src/topn.py`).

**Say:** "In each round the hidden person is never used for picking features, scaling or tuning.
So every score is for a brand-new person."

**Questions:**
- *Why two loops?* LOPO measures how good the model is; the inner loop picks its settings. Using
  LOPO for both would be like taking the exam 30 times and reporting the best score.
- *Fair to both models?* Same features, same rounds, same 30 tries.

## 3. Every metric (demo.py:57)

| Metric | Plain meaning |
|---|---|
| **ROC-AUC** | How often the model rates the high-load recording above the low-load one. 0.5 = guessing. |
| **Accuracy** | Share of recordings guessed right. |
| **Precision** | When it says "high load", how often that is true. |
| **Recall** | Of the real low (or high) recordings, how many it found. |
| **F1** | Precision and recall combined into one number. |
| **Brier** | How close the probabilities are to the truth; lower is better. |

## 4. SHAP (demo.py:65)

**What it does:** Each round's model explains the person it never saw. We average and rank.

**Say:** "Both models relied most on blink rate and on the lowest up/down gaze angle."

**Question:** *Does blinking cause load?* No: the model used it. A4 includes counting aloud, and
talking raises blinking; we state this as a limitation.

## Results it prints

| | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.862 | 0.862 |
| Accuracy | 0.830 | 0.739 |
| Precision | 0.844 | 0.740 |
| Recall low / high | 0.932 / 0.727 | 0.773 / 0.705 |
| F1 | 0.828 | 0.738 |
| Brier | 0.172 | 0.167 |

**Say:** "Both models rank the recordings equally well. The simpler model is just as good, and
classifies more recordings correctly."

## If they ask "is it leakage, or is one feature enough?"

1. **No leakage:** every step that learns (feature picking, scaling, tuning) uses training people
   only. The chance test reruns everything with shuffled labels: none of 200 runs reached the real
   score (p = 0.005).
2. **One feature is nearly enough, and we report it:** blink rate alone gives 0.884. We only know
   that because the model tested all 140 features and picked blink rate in every round.
3. **The other features still work:** without blink rate, 0.73 (LR) and 0.79 (XGBoost), above chance.

## Other files (if asked)

| File | Job |
|---|---|
| `run_extract.py` | Cleans the raw data and builds the 140 features (`outputs/final/`) |
| `run_model.py` | The full study run: results, every N, checks, SHAP (`outputs/final/results/`) |
| `topn.py` | Picks the top N features inside each round, and builds the two models |
| `evaluate.py` | LOPO, metrics, confidence intervals, chance test |
| `config.py` | All fixed numbers in one place (thresholds, 30 tries, 5 folds, seed 0) |
| `paper_figures.py` | Draws the paper's charts from the saved results |

**Same numbers every time?** Yes: the random seed is fixed (`RANDOM_STATE = 0`).

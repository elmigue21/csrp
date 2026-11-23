"""Extra analysis 7: activity order (position 1-4) and fatigue/practice.

annotation.csv has no order column. Order is INFERRED from the first gaze timestamp of each
recording: Pupil Capture timestamps run on one clock per session, so within a participant the
earlier start was recorded first (our own inference; checked: no ties, gaps of minutes).
Tests (our own choice):
  (i)  feature drift: per-person z-scored features (main pipeline) minus the activity mean
       across people (removes the condition effect); per participant, Spearman rho between
       position and this residual over the 4 activities; Wilcoxon signed-rank of the 44 rhos vs 0.
  (ii) model output: Spearman between position and P1 out-of-fold probability within A1 and within A4
       (official predictions_P1.csv, read only), and Mann-Whitney of the per-person margin
       p(A4) - p(A1) between people who did A4 after A1 and people who did A4 before A1.
Output: outputs/colet/tables/extra/7_*.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from extra_common import EXTRA_DIR

from config import CACHE_DIR, DATA_DIR, TABLE_DIR
from normalize import per_participant_zscore
from run_colet import prepare

prep = prepare(DATA_DIR, CACHE_DIR)
rows = prep["rows"].copy()
P1 = prep["features_used"]

start = {(p, a): pd.read_parquet(DATA_DIR / f"p{p:02d}_t{a}_gaze.parquet", columns=["gaze_timestamp"])
         ["gaze_timestamp"].min() for p, a in rows[["participant", "activity"]].itertuples(index=False)}
rows["start"] = [start[(p, a)] for p, a in rows[["participant", "activity"]].itertuples(index=False)]
rows["position"] = rows.groupby("participant")["start"].rank(method="first").astype(int)
gaps = rows.sort_values(["participant", "start"]).groupby("participant")["start"].diff().dropna()
print(f"gap between consecutive starts (s): min {gaps.min():.0f}, median {gaps.median():.0f}, max {gaps.max():.0f}")

order = rows.pivot(index="participant", columns="activity", values="position")
order.to_csv(EXTRA_DIR / "7_inferred_order.csv")
pos_table = pd.crosstab(rows["activity"], rows["position"])
print("activity x position counts:\n", pos_table.to_string())
pos_table.to_csv(EXTRA_DIR / "7_activity_by_position.csv")
a4_after_a1 = order[4] > order[1]
print(f"A4 after A1: {int(a4_after_a1.sum())}/{len(order)}")

# (i) feature drift with position, condition effect removed
z = per_participant_zscore(rows[P1], rows["participant"])
resid = z - z.groupby(rows["activity"]).transform("mean")
out = []
for f in P1:
    rhos = []
    for p, idx in rows.groupby("participant").groups.items():
        r = spearmanr(rows.loc[idx, "position"], resid.loc[idx, f]).statistic
        if np.isfinite(r):
            rhos.append(r)
    rhos = np.array(rhos)
    out.append({"feature": f, "n": len(rhos), "median_rho": float(np.median(rhos)),
                "mean_rho": float(rhos.mean()), "wilcoxon_p": float(wilcoxon(rhos).pvalue)})
drift = pd.DataFrame(out)
print(drift.round(3).to_string(index=False))
drift.to_csv(EXTRA_DIR / "7_feature_drift_by_position.csv", index=False)

# (ii) model output vs position
pred = pd.read_csv(TABLE_DIR / "predictions_P1.csv")
pred["position"] = [order.loc[p, a] for p, a in pred[["participant", "activity"]].itertuples(index=False)]
res = []
for model in ("p_lr", "p_xgb"):
    for act in (1, 4):
        d = pred[pred["activity"] == act]
        s = spearmanr(d["position"], d[model])
        res.append({"test": f"spearman position vs {model} within A{act}", "n": len(d),
                    "statistic": float(s.statistic), "p": float(s.pvalue)})
    m = pred.pivot(index="participant", columns="activity", values=model)
    margin = m[4] - m[1]
    u = mannwhitneyu(margin[a4_after_a1.loc[margin.index]], margin[~a4_after_a1.loc[margin.index]])
    res.append({"test": f"mann-whitney margin {model}: A4 after A1 vs before", "n": len(margin),
                "statistic": float(u.statistic), "p": float(u.pvalue),
                "median_after": float(margin[a4_after_a1.loc[margin.index]].median()),
                "median_before": float(margin[~a4_after_a1.loc[margin.index]].median())})
res = pd.DataFrame(res)
print(res.round(4).to_string(index=False))
res.to_csv(EXTRA_DIR / "7_order_vs_predictions.csv", index=False)

"""Optional robustness analysis (pre-listed in researcher-notes N19): P1 without blink rate."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config
import evaluate

evaluate.SEARCH_JOBS = 12
evaluate.PERM_JOBS = 12

from config import CACHE_DIR, DATA_DIR, FEATURES, OUT_DIR
from run_colet import prepare, run_analysis

t0 = time.time()
prep = prepare(DATA_DIR, CACHE_DIR)
feats = [f for f in FEATURES if f != "blink_rate"]
out = run_analysis("P1_no_blink", feats, prep["rows"])
d = OUT_DIR / "tables" / "optional"
d.mkdir(parents=True, exist_ok=True)
out["metrics"].to_csv(d / "no_blink_results.csv", index=False)
(d / "no_blink_comparison.json").write_text(json.dumps(out["comparison"], indent=2, default=float))
for kind, df in out["importance"].items():
    df.to_csv(d / f"no_blink_importance_{kind}.csv", index=False)
out["predictions"].to_csv(d / "no_blink_predictions.csv", index=False)
cols = ["model", "roc_auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy", "brier",
        "share_a4_above_a1", "perm_observed_auc", "perm_p_value"]
print(out["metrics"][cols].round(3).to_string(index=False))
print(json.dumps({k: out["comparison"][k] for k in ("auc_difference", "auc_difference_ci_low",
      "auc_difference_ci_high", "wilcoxon_p", "shap_rank_kendall_tau")}, indent=1, default=float))
print("SHAP LR:", out["importance"]["shap_lr"][["feature", "mean_abs_shap"]].round(3).values.tolist()[:5])
print("SHAP XGB:", out["importance"]["shap_xgb"][["feature", "mean_abs_shap"]].round(3).values.tolist()[:5])
print(f"runtime {time.time() - t0:.0f} s")

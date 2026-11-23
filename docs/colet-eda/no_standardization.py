"""Optional analysis S4 (pre-listed): P1 and P2 WITHOUT per-person standardization.

Identical to the main run (nested tuning, bootstrap CIs, permutation test, importance) except that
the per-person z-score is replaced by the identity, so the models see raw feature values. Scaling
inside the LR pipeline still happens in-fold, as in the main run.
Outputs: outputs/colet/tables/optional/raw_*  (raw = unstandardized).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import evaluate
import run_colet

evaluate.SEARCH_JOBS = 12
evaluate.PERM_JOBS = 12
run_colet.per_participant_zscore = lambda X, participants: X.copy()   # switch the step off

from config import CACHE_DIR, DATA_DIR, OUT_DIR, PUPIL_FEATURES

t0 = time.time()
prep = run_colet.prepare(DATA_DIR, CACHE_DIR)
d = OUT_DIR / "tables" / "optional"
d.mkdir(parents=True, exist_ok=True)
cols = ["analysis", "model", "roc_auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy", "brier",
        "share_a4_above_a1", "perm_observed_auc", "perm_p_value"]
for name, feats in (("P1_raw", prep["features_used"]), ("P2_raw", PUPIL_FEATURES)):
    out = run_colet.run_analysis(name, feats, prep["rows"])
    out["metrics"].to_csv(d / f"raw_{name}_results.csv", index=False)
    (d / f"raw_{name}_comparison.json").write_text(json.dumps(out["comparison"], indent=2, default=float))
    for kind, df in out["importance"].items():
        df.to_csv(d / f"raw_{name}_importance_{kind}.csv", index=False)
    out["predictions"].to_csv(d / f"raw_{name}_predictions.csv", index=False)
    print(out["metrics"][cols].round(3).to_string(index=False))
    c = out["comparison"]
    print(f"  XGB-LR {c['auc_difference']:+.3f} [{c['auc_difference_ci_low']:+.3f}, {c['auc_difference_ci_high']:+.3f}]"
          f"  Wilcoxon p={c['wilcoxon_p']:.3f}  tau={c['shap_rank_kendall_tau']:.2f}")
    print("  SHAP LR:", out["importance"]["shap_lr"]["feature"].tolist()[:4])
    print("  SHAP XGB:", out["importance"]["shap_xgb"]["feature"].tolist()[:4])
print(f"runtime {time.time() - t0:.0f} s")

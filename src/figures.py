"""Chapter 4 figures, drawn only from the tables run_colet.py wrote (outputs/colet/tables).

Every figure has a table twin, so no value is reachable only from a picture.
Usage: python src/figures.py   (writes PNGs to outputs/colet/figures/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, OUT_DIR
from data import load_annotation
from viz_style import INK_MUTED, INK_SECONDARY, SERIES, SERIES_ORDER, apply_style, strip_spines

TABLES = OUT_DIR / "tables"
FIGS = OUT_DIR / "figures"
PROBA = {"Logistic Regression": "p_lr", "XGBoost": "p_xgb"}
LABELS = {
    "pupil_mean": "Pupil diameter (mean)", "pupil_sd": "Pupil diameter (SD)",
    "blink_rate": "Blink rate", "fixation_rate": "Fixation rate",
    "fixation_duration": "Fixation duration", "saccade_rate": "Saccade rate",
    "saccade_amplitude": "Saccade amplitude", "saccade_peak_velocity": "Saccade peak velocity",
    "gaze_sd_x": "Gaze spread (horizontal)", "gaze_sd_y": "Gaze spread (vertical)",
}


def roc_figure() -> Path:
    """ROC curves of the pooled out-of-fold predictions, P1 and P2 side by side."""
    results = pd.read_csv(TABLES / "05_results.csv").set_index(["analysis", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, analysis in zip(axes, ("P1", "P2")):
        pred = pd.read_csv(TABLES / f"predictions_{analysis}.csv")
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=":", label="Chance")
        for model in SERIES_ORDER:
            fpr, tpr, _ = roc_curve(pred["y"], pred[PROBA[model]])
            auc = results.loc[(analysis, model), "roc_auc"]
            ax.plot(fpr, tpr, color=SERIES[model], label=f"{model} (AUC {auc:.3f})")
        ax.set_title("P1: all ten features" if analysis == "P1" else "P2: pupil features only")
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7.5)
        strip_spines(ax)
    axes[0].set_ylabel("True positive rate")
    return _save(fig, "fig_roc_p1_p2.png")


def auc_figure() -> Path:
    """Pooled ROC-AUC with the participant-bootstrap 95% CI for each analysis and model."""
    res = pd.read_csv(TABLES / "05_results.csv")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ys = {("P1", m): i for i, m in enumerate(SERIES_ORDER)}
    ys.update({("P2", m): i + 3 for i, m in enumerate(SERIES_ORDER)})
    for _, r in res.iterrows():
        y = ys[(r["analysis"], r["model"])]
        ax.errorbar(r["roc_auc"], y, xerr=[[r["roc_auc"] - r["auc_ci_low"]], [r["auc_ci_high"] - r["roc_auc"]]],
                    fmt="o", color=SERIES[r["model"]], capsize=3, lw=1.5)
        ax.text(r["auc_ci_high"] + 0.01, y, f"{r['roc_auc']:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax.set_yticks([0, 1, 3, 4])
    ax.set_yticklabels([f"P1 {m}" for m in SERIES_ORDER] + [f"P2 {m}" for m in SERIES_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.06)
    ax.set_xlabel("Pooled ROC-AUC (95% participant-bootstrap CI); dotted line = chance")
    strip_spines(ax)
    return _save(fig, "fig_auc_ci.png")


def shap_figure() -> Path:
    """Mean |SHAP| per feature for both models in P1 (log-odds scale)."""
    lr = pd.read_csv(TABLES / "07_importance_P1_shap_lr.csv").set_index("feature")["mean_abs_shap"]
    xgb = pd.read_csv(TABLES / "07_importance_P1_shap_xgb.csv").set_index("feature")["mean_abs_shap"]
    order = lr.add(xgb, fill_value=0).sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(y + 0.2, lr[order], height=0.38, color=SERIES["Logistic Regression"], label="Logistic Regression")
    ax.barh(y - 0.2, xgb[order], height=0.38, color=SERIES["XGBoost"], label="XGBoost")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[f] for f in order])
    ax.set_xlabel("Mean |SHAP value| (log-odds), held-out participants")
    ax.legend(loc="lower right")
    strip_spines(ax)
    return _save(fig, "fig_shap_p1.png")


def rtlx_figure() -> Path:
    """NASA-RTLX per activity for the analysed participants (manipulation check)."""
    keep = pd.read_csv(TABLES / "01b_activities_per_participant.csv")["participant"]
    ann = load_annotation(DATA_DIR)
    ann = ann[ann["participant"].isin(keep)]
    data = [ann.loc[ann["activity"] == a, "mean"].to_numpy() for a in (1, 2, 3, 4)]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.boxplot(data, widths=0.5, medianprops={"color": SERIES["Logistic Regression"], "lw": 2},
               flierprops={"markersize": 3})
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["A1\nsingle task", "A2\ntime pressure", "A3\ncounting aloud", "A4\ncounting + time"])
    ax.set_ylabel("NASA-RTLX (0-100)")
    strip_spines(ax)
    return _save(fig, "fig_rtlx.png")


def _save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> list[Path]:
    apply_style()
    return [roc_figure(), auc_figure(), shap_figure(), rtlx_figure()]


if __name__ == "__main__":
    for p in main():
        print(p)

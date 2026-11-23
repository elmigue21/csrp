"""Thesis figures for the post-defense paper (v4-8), drawn only from outputs/final.

Charts read the recorded results (outputs/final/results); the two flowcharts describe the
pipeline in src/run_extract.py and src/run_model.py. Every chart has a table twin in the
paper or in outputs/final/results, so no value is reachable only from a picture.
Usage: python src/paper_figures.py   (writes PNGs to outputs/final/paper_figures/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FINAL_DIR
from viz_style import INK_MUTED, INK_SECONDARY, SERIES, SERIES_ORDER, apply_style, strip_spines

RESULTS = FINAL_DIR / "results"
FIGS = FINAL_DIR / "paper_figures"
PROBA = {"Logistic Regression": "p_lr", "XGBoost": "p_xgb"}
RATE_LABELS = {"blink_rate": "Blink rate", "fixation_rate": "Fixation rate", "saccade_rate": "Saccade rate"}
# The paper's terms (Table 5), not the plain-language labels of the Excel previews
SIGNAL_LABELS = {
    "pupil": "pupil diameter", "gaze_x": "horizontal gaze angle", "gaze_y": "vertical gaze angle",
    "gaze_speed": "gaze speed", "fixation_duration": "fixation duration",
    "saccade_amplitude": "saccade amplitude", "saccade_peak_velocity": "saccade peak velocity",
    "saccade_mean_velocity": "saccade mean velocity", "saccade_duration": "saccade duration",
    "blink_duration": "blink duration",
}
STAT_LABELS = {"min": "minimum", "max": "maximum", "sd": "SD", "cv": "CV", "iqr": "IQR",
               "kurt": "kurtosis", "skew": "skewness"}


def feature_label(name: str) -> str:
    """'gaze_y__min' -> 'Vertical gaze angle, minimum'."""
    if name in RATE_LABELS:
        return RATE_LABELS[name]
    signal, stat = name.split("__")
    text = f"{SIGNAL_LABELS[signal]}, {STAT_LABELS.get(stat, stat)}"
    return text[0].upper() + text[1:]


# ------------------------------------------------------------------ charts
def roc_figure() -> Path:
    """ROC curves of the pooled out-of-fold predictions of both models (main result)."""
    res = pd.read_csv(RESULTS / "results.csv").set_index("model")
    pred = pd.read_csv(RESULTS / "predictions.csv")
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=":", label="Chance")
    for model in SERIES_ORDER:
        fpr, tpr, _ = roc_curve(pred["y"], pred[PROBA[model]])
        ax.plot(fpr, tpr, color=SERIES[model], label=f"{model} (AUC {res.loc[model, 'roc_auc']:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7.5)
    strip_spines(ax)
    return _save(fig, "fig6_roc.png")


def by_n_figure() -> Path:
    """ROC-AUC (95% CI) with the number of features fixed at each candidate N."""
    by_n = pd.read_csv(RESULTS / "results_by_n.csv")
    checks = pd.read_csv(RESULTS / "checks.csv")
    blink = checks.loc[checks["check"] == "blink_rate alone", "roc_auc"].iloc[0]
    ns = sorted(by_n["n_features"].unique())
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, model in enumerate(SERIES_ORDER):
        d = by_n[by_n["model"] == model].set_index("n_features").loc[ns]
        xs = x + (i - 0.5) * 0.18
        ax.errorbar(xs, d["roc_auc"], yerr=[d["roc_auc"] - d["auc_ci_low"], d["auc_ci_high"] - d["roc_auc"]],
                    fmt="o", color=SERIES[model], capsize=3, lw=1.5, label=model)
    ax.axhline(blink, color=INK_SECONDARY, lw=1, ls="--", label=f"Blink rate alone ({blink:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of features kept (N)")
    ax.set_ylabel("Pooled ROC-AUC (95% CI)")
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc="lower left", fontsize=7.5, ncol=3)
    strip_spines(ax)
    return _save(fig, "fig7_by_n.png")


def shap_figure(top: int = 10) -> Path:
    """Share of the total mean |SHAP| per feature, both models; top features by combined share."""
    share = {}
    for model, file in (("Logistic Regression", "logistic_regression"), ("XGBoost", "xgboost")):
        s = pd.read_csv(RESULTS / f"shap_{file}.csv").set_index("feature")["mean_abs_shap"]
        share[model] = 100 * s / s.sum()
    df = pd.DataFrame(share).fillna(0)
    order = df.sum(axis=1).sort_values(ascending=False).index[:top][::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(y + 0.2, df.loc[order, "Logistic Regression"], height=0.38,
            color=SERIES["Logistic Regression"], label="Logistic Regression")
    ax.barh(y - 0.2, df.loc[order, "XGBoost"], height=0.38, color=SERIES["XGBoost"], label="XGBoost")
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in order])
    ax.set_xlabel("Share of total mean |SHAP value| (%), held-out participants")
    ax.legend(loc="lower right")
    strip_spines(ax)
    return _save(fig, "fig8_shap.png")


# ------------------------------------------------------------------ flowcharts
def _box(ax, x, y, w, h, text, size=11, dashed=False, fill="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec="black", lw=1.6, ls="--" if dashed else "-"))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, family="Times New Roman")


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.6, "mutation_scale": 14})


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def method_figure() -> Path:
    """Figure 2: overview of the methodology."""
    fig, ax = _canvas(7.0, 9.0)
    c = 3.5
    _box(ax, c, 8.5, 4.2, 0.62, "COLET: A1 (low load) and A4 (high load)\nA2 and A3: manipulation check only")
    _arrow(ax, c, 8.19, c, 7.85)
    _box(ax, c, 7.55, 4.2, 0.55, "Preprocessing and feature extraction (Figure 4)\n140 features per recording")
    _arrow(ax, c, 7.27, c, 6.92)
    _box(ax, c, 4.95, 6.3, 3.9, "", dashed=True, fill="#f2f2f2")
    ax.text(c, 6.68, "Leave-one-participant-out (44 folds)", ha="center", va="center", fontsize=11,
            family="Times New Roman", weight="bold")
    _box(ax, c, 6.2, 5.5, 0.5, "Train on 43 participants; imputation and scaling fitted in-fold")
    _arrow(ax, c, 5.95, c, 5.72)
    _box(ax, c, 5.45, 5.5, 0.5, "Rank features by single-feature ROC-AUC; keep the top N")
    _arrow(ax, c, 5.2, c, 4.97)
    _box(ax, c, 4.7, 5.5, 0.5, "Inner participant-grouped CV: tune N and model settings")
    _arrow(ax, 2.6, 4.45, 2.0, 4.12)
    _arrow(ax, 4.4, 4.45, 5.0, 4.12)
    _box(ax, 2.0, 3.6, 2.6, 0.95, "Logistic Regression\n(elastic net)")
    _box(ax, 5.0, 3.6, 2.6, 0.95, "XGBoost\n(depth 1–3, 50–300 trees)")
    _arrow(ax, 2.0, 3.12, 2.9, 2.62)
    _arrow(ax, 5.0, 3.12, 4.1, 2.62)
    _box(ax, c, 2.35, 4.2, 0.5, "Pooled predictions for held-out participants")
    _arrow(ax, c, 2.1, 1.15, 1.62)
    _arrow(ax, c, 2.1, c, 1.62)
    _arrow(ax, c, 2.1, 5.85, 1.62)
    _box(ax, 1.15, 1.05, 2.2, 1.1, "Metrics\nROC-AUC + CI, accuracy,\nprecision, recall, F1,\nBrier", size=10)
    _box(ax, c, 1.05, 2.2, 1.1, "Model comparison\nAUC difference + CI,\nWilcoxon", size=10)
    _box(ax, 5.85, 1.05, 2.2, 1.1, "Feature importance\nSHAP,\nselection count", size=10)
    _arrow(ax, c, 0.5, c, 0.3)
    _box(ax, c, 0.0, 5.4, 0.5, "Checks: results for each N; blink rate alone; without blink rate", size=10)
    ax.set_ylim(-0.3, 9.0)
    return _save(fig, "fig2_method.png", tight=True)


def preprocessing_figure() -> Path:
    """Figure 4: preprocessing and feature extraction."""
    fig, ax = _canvas(7.0, 9.6)
    c = 3.5
    _box(ax, c, 9.15, 4.6, 0.6, "COLET raw data: gaze (≈ 242 Hz), pupil and blinks")
    _arrow(ax, c, 8.85, c, 8.6)
    _box(ax, c, 8.2, 4.6, 0.75, "1. Remove bad data\nsamples with confidence < 0.8;\nrecordings with > 35% invalid samples")
    _arrow(ax, c, 7.82, c, 7.57)
    _box(ax, c, 7.22, 4.6, 0.65, "2. Clean each signal\nblinks · pupil diameter · gaze direction")
    _arrow(ax, c, 6.89, c, 6.64)
    _box(ax, c, 6.29, 4.6, 0.65, "3. Detect fixations and saccades\nI-VT at 45 °/s; fixations ≥ 55 ms")
    _arrow(ax, c, 5.96, c, 5.6)
    ax.add_patch(Polygon([(c, 5.6), (c + 1.3, 5.1), (c, 4.6), (c - 1.3, 5.1)], closed=True,
                         fc="white", ec="black", lw=1.6))
    ax.text(c, 5.1, "Sanity check passed\nin every activity?", ha="center", va="center", fontsize=11,
            family="Times New Roman")
    _arrow(ax, c - 0.95, 4.82, 1.9, 4.2)
    _arrow(ax, c + 0.95, 4.82, 5.1, 4.2)
    ax.text(1.95, 4.72, "Yes", fontsize=10, family="Times New Roman")
    ax.text(4.85, 4.72, "No", fontsize=10, family="Times New Roman")
    _box(ax, 1.9, 3.7, 3.0, 0.95, "4. Summarize every signal\n17 statistics each + 3 rates\n(173 features)")
    _box(ax, 5.1, 3.7, 3.0, 0.95, "4. Same, without fixation\nand saccade signals")
    _arrow(ax, 1.9, 3.22, 3.0, 2.72)
    _arrow(ax, 5.1, 3.22, 4.0, 2.72)
    _box(ax, c, 2.35, 4.6, 0.72, "5. Remove copies and unusable statistics\n(33 features; Table 5)")
    _arrow(ax, c, 1.99, c, 1.74)
    _box(ax, c, 1.4, 4.6, 0.65, "88 samples (44 participants × A1 and A4)\n× 140 features")
    ax.set_ylim(0.9, 9.6)
    return _save(fig, "fig4_preprocessing.png", tight=True)


def _save(fig, name: str, tight: bool = True) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path


def main() -> list[Path]:
    apply_style()
    charts = [roc_figure(), by_n_figure(), shap_figure()]
    plt.rcdefaults()
    return charts + [method_figure(), preprocessing_figure()]


if __name__ == "__main__":
    for p in main():
        print(p)

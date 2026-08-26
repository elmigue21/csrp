"""Render every figure Chapter 4 needs, from the tables the experiment wrote.

Each figure has a table twin under outputs/tables holding the same numbers, so no
value in this study is reachable only from a picture.

Bar marks are drawn as butt-capped lines with a round marker at the data end, which
gives a rounded data-end anchored to a square baseline -- matplotlib has no native
rounded bar, and a FancyBboxPatch in data coordinates renders an ellipse rather than
a circle once the axes aspect is non-square.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIG_DIR, OUT_DIR, TABLE_DIR
from viz_style import (
    AXIS,
    BLUE_CMAP,
    DIVERGING_NEG,
    DIVERGING_POS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SERIES_ORDER,
    SURFACE,
    apply_style,
    caption,
    strip_spines,
)

PRED_DIR = OUT_DIR / "predictions"
BAR_LW = 9.0          # bar thickness in points
PAIR_LW = 1.0


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_per_").lower()


def rounded_hbar(ax, y, value, color, lw=BAR_LW, zorder=3):
    """Horizontal bar with a square baseline end and a rounded data end."""
    ax.plot([0, value], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([value], [y], marker="o", ms=lw, color=color, markeredgewidth=0, zorder=zorder)


def _pretty(name: str) -> str:
    """Make feature names readable in a figure without losing what they mean."""
    return (
        name.replace("_mean", " (mean)")
        .replace("_std", " (SD)")
        .replace("saccade_velocity_degree/s", "saccade velocity")
        .replace("saccade_amplitude_degree", "saccade amplitude")
        .replace("_", " ")
        .replace("gte", "GTE")
        .replace("fdi", "FDI")
        .replace("lux interpolated", "illuminance")
        .replace("EyeGaze", "gaze")
    )


def _load_oof(cfg: str) -> dict[str, pd.DataFrame]:
    out = {}
    for model in SERIES_ORDER:
        path = PRED_DIR / f"oof_{cfg}_{_safe(model)}.parquet"
        if path.exists():
            out[model] = pd.read_parquet(path)
    return out


# ---------------------------------------------------------------- 1. ROC curves
def fig_roc(cfg: str, subtitle: str) -> None:
    preds = _load_oof(cfg)
    if not preds:
        return
    fig, ax = plt.subplots(figsize=(4.7, 4.5))

    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=1)
    ax.annotate("chance", xy=(0.62, 0.62), xytext=(0.66, 0.55),
                color=INK_MUTED, fontsize=8, rotation=0)

    for model in SERIES_ORDER:
        if model not in preds:
            continue
        d = preds[model]
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_proba"])
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=SERIES[model], lw=2.0, zorder=3,
                label=f"{model}  (AUC {auc:.3f})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC, held-out participants — {subtitle}", loc="left", pad=12)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", handlelength=1.6)
    strip_spines(ax)
    caption(fig, "Pooled predictions from all 26 leave-one-participant-out folds. "
                 "Hyperparameters tuned inside each fold only.")
    fig.savefig(FIG_DIR / f"fig1_roc_{cfg}.png")
    plt.close(fig)


# -------------------------------------------------- 2. configuration comparison
def fig_config_comparison(main: pd.DataFrame) -> None:
    label_map = {
        "A": "A  as-is label\n     raw features",
        "B": "B  as-is label\n     no illuminance",
        "C": "C  as-is label\n     per-person features",
        "D": "D  per-person label\n     raw features",
        "E": "E  per-person label\n     per-person features",
        "F": "F  per-person label\n     no illuminance",
    }
    configs = [c for c in ["A", "B", "C", "D", "E", "F"] if c in set(main["config"])]
    fig, ax = plt.subplots(figsize=(7.4, 0.92 * len(configs) + 1.5))

    offset = 0.20
    for i, cfg in enumerate(configs):
        base_y = len(configs) - 1 - i
        for j, model in enumerate(SERIES_ORDER):
            row = main[(main["config"] == cfg) & (main["model"] == model)]
            if row.empty:
                continue
            v = float(row["pooled_roc_auc"].iloc[0])
            y = base_y + (offset if j == 0 else -offset)
            rounded_hbar(ax, y, v, SERIES[model])
            ax.text(v + 0.014, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=8.5, color=INK_SECONDARY)

    ax.axvline(0.5, color=AXIS, lw=0.9, zorder=2)
    ax.text(0.5, len(configs) - 0.35, " chance", color=INK_MUTED, fontsize=8, va="bottom")

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([label_map[c] for c in reversed(configs)], fontsize=8.5)
    ax.set_xlim(0, max(0.78, main["pooled_roc_auc"].max() + 0.06))
    ax.set_ylim(-0.65, len(configs) - 0.2)
    ax.set_xlabel("ROC-AUC, pooled over held-out participants")
    ax.set_title("Where the boundary is drawn matters more than which model is used",
                 loc="left", pad=14)
    ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], color=SERIES[m], lw=3.2, solid_capstyle="round",
                          label=m) for m in SERIES_ORDER]
    ax.legend(handles=handles, loc="lower right", handlelength=1.4, ncols=2)
    strip_spines(ax)
    caption(fig, "Every bar uses the same epochs, the same split and the same tuning "
                 "procedure; only the labelling and feature scheme change.")
    fig.savefig(FIG_DIR / "fig2_configuration_comparison.png")
    plt.close(fig)


# ------------------------------------------------------- 3. XGBoost SHAP / gain
def fig_shap(cfg: str, subtitle: str, top_n: int = 15) -> None:
    path = TABLE_DIR / f"06_importance_xgb_shap_{cfg}.csv"
    if not path.exists():
        return
    d = pd.read_csv(path).head(top_n).iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(d) + 1.6))
    gap = d["mean_abs_shap"].max() * 0.018
    for i, row in d.iterrows():
        rounded_hbar(ax, i, row["mean_abs_shap"], SERIES["XGBoost"], lw=7.5)
        ax.text(row["mean_abs_shap"] + gap, i, f"{row['mean_abs_shap']:.3f}",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([_pretty(f) for f in d["feature"]], fontsize=8.5)
    ax.set_xlim(0, d["mean_abs_shap"].max() * 1.16)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Mean |SHAP| on held-out participants")
    ax.set_title(f"XGBoost feature importance — {subtitle}", loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    strip_spines(ax)
    caption(fig, "SHAP values computed out-of-fold: each fold's model explains only "
                 "the participant it never trained on. Top "
                 f"{len(d)} of {len(pd.read_csv(path))} features.")
    fig.savefig(FIG_DIR / f"fig3_shap_importance_{cfg}.png")
    plt.close(fig)


# ------------------------------------------------ 4. Logistic Regression weights
def fig_lr_coefficients(cfg: str, subtitle: str, top_n: int = 15) -> None:
    path = TABLE_DIR / f"06_importance_lr_coefficients_{cfg}.csv"
    if not path.exists():
        return
    d = (pd.read_csv(path)
         .sort_values("abs_coefficient_mean", ascending=False)
         .head(top_n).iloc[::-1].reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(d) + 1.7))
    gap = d["coefficient_mean"].abs().max() * 0.035
    for i, row in d.iterrows():
        v = row["coefficient_mean"]
        color = DIVERGING_POS if v > 0 else DIVERGING_NEG
        ax.plot([0, v], [i, i], color=color, lw=7.5, solid_capstyle="butt", zorder=3)
        ax.plot([v], [i], marker="o", ms=7.5, color=color, markeredgewidth=0, zorder=3)
        ha = "left" if v > 0 else "right"
        ax.text(v + (gap if v > 0 else -gap), i, f"{v:+.2f}", va="center", ha=ha,
                fontsize=8, color=INK_SECONDARY)

    ax.axvline(0, color=AXIS, lw=0.9, zorder=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([_pretty(f) for f in d["feature"]], fontsize=8.5)
    lim = d["coefficient_mean"].abs().max() * 1.34
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Mean standardised coefficient across folds")
    ax.set_title(f"Logistic Regression coefficients — {subtitle}", loc="left", pad=12)
    ax.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], color=DIVERGING_POS, lw=3.2, solid_capstyle="round",
                   label="pushes towards HIGH load"),
        plt.Line2D([], [], color=DIVERGING_NEG, lw=3.2, solid_capstyle="round",
                   label="pushes towards LOW load"),
    ]
    ax.legend(handles=handles, loc="lower right", handlelength=1.4, fontsize=8.5)
    strip_spines(ax)
    caption(fig, "Features are standardised inside the model, so coefficients are "
                 "comparable to one another. Sign consistency across the 26 folds is "
                 "reported in the paired CSV.")
    fig.savefig(FIG_DIR / f"fig4_lr_coefficients_{cfg}.png")
    plt.close(fig)


# ------------------------------------------------ 5. per-participant variability
def fig_per_participant(cfg: str, subtitle: str) -> None:
    frames = {}
    for model in SERIES_ORDER:
        p = TABLE_DIR / f"05_per_participant_{cfg}_{_safe(model)}.csv"
        if p.exists():
            frames[model] = pd.read_csv(p)
    if len(frames) < 2:
        return

    a, b = frames[SERIES_ORDER[0]], frames[SERIES_ORDER[1]]
    m = a[["participant", "roc_auc"]].merge(
        b[["participant", "roc_auc"]], on="participant", suffixes=("_lr", "_xgb"))
    m = m.dropna().sort_values("roc_auc_xgb").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 0.30 * len(m) + 1.8))
    for i, row in m.iterrows():
        ax.plot([row["roc_auc_lr"], row["roc_auc_xgb"]], [i, i],
                color=GRID, lw=PAIR_LW, zorder=2)
        ax.plot(row["roc_auc_lr"], i, marker="o", ms=7,
                color=SERIES["Logistic Regression"], markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=4)
        ax.plot(row["roc_auc_xgb"], i, marker="o", ms=7,
                color=SERIES["XGBoost"], markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=4)

    ax.axvline(0.5, color=AXIS, lw=0.9, zorder=1)
    ax.text(0.5, len(m) - 0.2, " chance", color=INK_MUTED, fontsize=8, va="bottom")

    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f"P{int(p)}" for p in m["participant"]], fontsize=8)
    ax.set_ylim(-0.8, len(m) - 0.1)
    ax.set_xlabel("ROC-AUC on that participant's own held-out epochs")
    ax.set_title(f"Performance varies widely between people — {subtitle}",
                 loc="left", pad=12)
    ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], color=SERIES[mm], marker="o", ms=7, lw=0,
                          label=f"{mm}  (mean {m['roc_auc_lr' if mm == SERIES_ORDER[0] else 'roc_auc_xgb'].mean():.3f})")
               for mm in SERIES_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    strip_spines(ax)
    caption(fig, f"Sorted by XGBoost score. {len(m)} of 26 participants shown; the "
                 "remainder have only one class in their held-out data, so ROC-AUC "
                 "is undefined for them.")
    fig.savefig(FIG_DIR / f"fig5_per_participant_{cfg}.png")
    plt.close(fig)


# --------------------------------------------------------- 6. confusion matrices
def fig_confusion(cfg: str, subtitle: str) -> None:
    preds = _load_oof(cfg)
    if len(preds) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))

    for ax, model in zip(axes, SERIES_ORDER):
        d = preds[model]
        cm = confusion_matrix(d["y_true"], d["y_pred"])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=BLUE_CMAP, vmin=0, vmax=1, aspect="equal")

        for i in range(2):
            for j in range(2):
                dark = share[i, j] > 0.55
                ax.text(j, i - 0.10, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, color="#ffffff" if dark else INK)
                ax.text(j, i + 0.20, f"{100 * share[i, j]:.1f}%", ha="center",
                        va="center", fontsize=9,
                        color="#e8eef7" if dark else INK_SECONDARY)

        ax.set_xticks([0, 1], ["predicted\nLOW", "predicted\nHIGH"], fontsize=8.5)
        ax.set_yticks([0, 1], ["actual LOW", "actual HIGH"], fontsize=8.5)
        ax.set_title(model, loc="left", fontsize=10, pad=8)
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(f"Confusion matrices, held-out participants — {subtitle}",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    caption(fig, "Cell shading encodes the row percentage, i.e. recall per true class. "
                 "Counts are pooled across all 26 folds.")
    fig.savefig(FIG_DIR / f"fig6_confusion_{cfg}.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_table = pd.read_csv(TABLE_DIR / "04_main_results.csv")

    subtitles = {"A": "as-is label, raw features",
                 "E": "per-person label and features"}

    fig_config_comparison(main_table)
    for cfg, sub in subtitles.items():
        fig_roc(cfg, sub)
        fig_shap(cfg, sub)
        fig_lr_coefficients(cfg, sub)
        fig_per_participant(cfg, sub)
        fig_confusion(cfg, sub)

    written = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f"{len(written)} figures written to {FIG_DIR}")
    for name in written:
        print("  ", name)


if __name__ == "__main__":
    main()

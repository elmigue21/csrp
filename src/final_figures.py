"""Figures for the team to understand the data (post-defense-data-processing.md §7.1).

These are working figures, not paper figures.

Every figure is drawn from what run_extract.py already computed (the logs, the feature table,
the dictionaries) or from one raw recording, so a value in a picture is always also in a CSV.
Styling comes from viz_style (the validated palette the Chapter 4 figures use).

  fig_01_raw_tables        size of each original COLET table
  fig_02_raw_columns       every original column, used or not (the raw data dictionary)
  fig_03_data_flow         how the data shrinks from raw files to the final feature table
  fig_04_feature_table     the final table: features kept and dropped per signal
  fig_05_recording_signals what one recording looks like after cleaning
  fig_06_cleaning_effect   two cleaning steps on real recordings (C1 and C2/C5/C7)
  fig_07_a1_vs_a4          six core measures, low vs high load (descriptive only)
  fig_08_data_quality      invalid gaze per recording and the C3 cut-off
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from config import (
    CONFIDENCE_MIN, DUPLICATE_RECORDING_IDS, GAZE_GRID_HZ, HIGH_ACTIVITY, IVT_THRESHOLD_DEG_S,
    LOW_ACTIVITY, MAX_INVALID_FRACTION, PUPIL_MAX_MM, PUPIL_METHOD_TAG, PUPIL_MIN_MM,
)
from data import load_recording
from events import angular_speed_deg_s
from flatten import SIGNALS
from preprocess import clean_blinks, clean_pupil, gaze_directions, trim_to_gaze_range
from viz_style import (
    AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, NEUTRAL, SURFACE, apply_style, caption, strip_spines,
)

LOW_COLOR = "#2a78d6"     # A1 / kept / used: the validated blue of the two-series pair
HIGH_COLOR = "#eb6834"    # A4 / removed / excluded: the validated orange of the pair
ACTIVITY_NAMES = {1: "A1\nlow load", 2: "A2", 3: "A3", 4: "A4\nhigh load"}
CORE_FEATURES = {          # the defended thesis's core measures, fixed before any plot was drawn
    "pupil__mean": "Pupil size, mean (mm)",
    "pupil__sd": "Pupil size, SD (mm)",
    "blink_rate": "Blink rate (per min)",
    "fixation_rate": "Fixation rate (per s)",
    "fixation_duration__mean": "Fixation length, mean (ms)",
    "saccade_amplitude__mean": "Saccade size, mean (deg)",
}
SIGNAL_NAMES = {
    "pupil": "Pupil size", "gaze_x": "Gaze left/right", "gaze_y": "Gaze up/down",
    "gaze_speed": "Eye speed", "fixation_duration": "Fixation length",
    "saccade_amplitude": "Saccade size", "saccade_peak_velocity": "Saccade peak speed",
    "saccade_mean_velocity": "Saccade average speed", "saccade_duration": "Saccade length",
    "blink_duration": "Blink length", "rates": "Rates (blink, fixation, saccade)",
}


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _fmt(n: float) -> str:
    return f"{int(round(n)):,}"


# ------------------------------------------------------------------ shape of the data
def raw_tables_figure(shapes: pd.DataFrame):
    """The five original COLET tables as one readable table."""
    header = ["Table", "Files", "Columns", "Rows in total", "Rows per file (median)", "One row is"]
    body = [[r.table, _fmt(r.files), _fmt(r.columns), _fmt(r.rows_total), _fmt(r.rows_median),
             r.one_row_is] for r in shapes.itertuples()]
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * (len(body) + 1)))
    ax.axis("off")
    tab = ax.table(cellText=body, colLabels=header, cellLoc="left", colLoc="left",
                   colWidths=[0.11, 0.06, 0.08, 0.12, 0.19, 0.44], bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    for (row, _), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        cell.set_facecolor(NEUTRAL if row == 0 else SURFACE)
        cell.get_text().set_color(INK if row == 0 else INK_SECONDARY)
        if row == 0:
            cell.get_text().set_fontweight("semibold")
    ax.set_title("The original COLET data: five tables", loc="left")
    caption(fig, "Gaze, pupil and blinks come as one file per participant and activity "
                 "(47 participants x 4 activities). Source: raw/raw_table_shapes.csv.")
    return fig


def raw_columns_figure(dictionary: pd.DataFrame):
    """Every column of the original tables, marked by whether the pipeline uses it."""
    kinds = {"yes": ("Used", LOW_COLOR), "qc only": ("Quality check only", HIGH_COLOR),
             "check only": ("Quality check only", HIGH_COLOR), "no": ("Not used", INK_MUTED)}
    groups = [["gaze"], ["pupil"], ["blinks", "annotation", "subject_info"]]
    tallest = max(sum(len(dictionary[dictionary.table == t]) + 2 for t in g) for g in groups)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.17 * tallest + 1.0))
    for ax, tables in zip(axes, groups):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(tallest, -1)
        y = 0
        for t in tables:
            cols = dictionary[dictionary.table == t]
            ax.text(0, y, f"{t}  ({len(cols)} columns)", fontsize=9, fontweight="semibold",
                    color=INK, va="center")
            y += 1
            for c in cols.itertuples():
                label, color = kinds.get(str(c.used), ("Not used", INK_MUTED))
                filled = label != "Not used"
                ax.scatter([0.02], [y], s=22, marker="s", color=color if filled else SURFACE,
                           edgecolors=color, linewidths=1.0, zorder=3)
                ax.text(0.07, y, c.column, fontsize=7.6, va="center",
                        color=INK if filled else INK_MUTED)
                y += 1
            y += 1
    handles = [Patch(facecolor=LOW_COLOR, label="Used"),
               Patch(facecolor=HIGH_COLOR, label="Quality check only"),
               Patch(facecolor=SURFACE, edgecolor=INK_MUTED, label="Not used")]
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98)
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("Every column of the original tables", x=0.0, ha="left", y=1.02,
                 fontsize=11, fontweight="semibold", color=INK)
    caption(fig, "Meaning, unit and the reason each column is or is not used: "
                 "raw/raw_data_dictionary.csv.")
    return fig


def _box(ax, x, y, w, h, title, lines, color=AXIS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=SURFACE, edgecolor=color, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=9,
            fontweight="semibold", color=INK)
    ax.text(x + w / 2, y + h - 0.40, "\n".join(lines), ha="center", va="top", fontsize=7.6,
            color=INK_SECONDARY, linespacing=1.4)


def data_flow_figure(log: pd.DataFrame, n_people: int, n_rows: int, n_extracted: int,
                     n_kept: int):
    """From the raw files to the final feature table, with the size at every stage."""
    n_dup = int(log.participant.isin(DUPLICATE_RECORDING_IDS).sum())
    n_excl = int(log.excluded.astype(str).eq("True").sum())
    lost = log[log.analysis_reason.eq("participant lost A1 or A4 (C3)")].participant.nunique()
    stages = [
        ("Raw files", [f"{log.participant.nunique()} participants",
                       f"x 4 activities = {len(log)}",
                       f"{_fmt(log.gaze_rows.sum())} gaze rows",
                       f"{_fmt(log.pupil_rows_raw.sum())} pupil rows",
                       f"{_fmt(log.blinks_raw.sum())} blinks"]),
        ("Cleaning C1-C10", ["trim, drop unsure samples,",
                             "clean blinks and pupil",
                             f"{n_excl} recordings dropped (C3)",
                             f"{n_dup} duplicates left out (P31)",
                             f"{lost} people lose A1 or A4"]),
        ("10 signals", ["per recording, e.g.",
                        "pupil: ~6,000 values",
                        "fixations: ~130 values",
                        "saccades: ~120 values"]),
        ("Flatten", ["17 statistics per signal", "+ 3 rates",
                     f"= {n_extracted} features", "1 row per recording"]),
        ("A1 and A4 only", [f"{n_people} participants x 2", f"= {n_rows} rows",
                            "(A2, A3: no label)"]),
        ("Final table", [f"{n_rows} rows", f"x {n_kept} features",
                         "+ participant, activity,", "label (low/high)",
                         f"{n_extracted - n_kept} dropped (D1-D10)"]),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 2.3))
    ax.axis("off")
    w, h, gap = 1.75, 1.5, 0.3
    ax.set_xlim(-0.05, len(stages) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.1, h + 0.1)
    for i, (title, lines) in enumerate(stages):
        x = i * (w + gap)
        _box(ax, x, 0, w, h, title, lines, color=LOW_COLOR if i == len(stages) - 1 else AXIS)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.04, h / 2), (x - 0.04, h / 2),
                                         arrowstyle="-|>", mutation_scale=10, color=INK_MUTED,
                                         linewidth=1.2))
    ax.set_title("From raw files to the final feature table", loc="left")
    caption(fig, "Every count comes from logs/log_recordings.csv and run_info.json.")
    return fig


def feature_table_figure(feature_dict: pd.DataFrame, n_rows: int):
    """Features kept and dropped for each signal: the columns of the final table."""
    d = feature_dict.copy()
    d["group"] = np.where(d.statistic.eq("rate"), "rates", d.signal)
    d["kept"] = d.kept.astype(str).eq("True")
    order = [*SIGNALS, "rates"]
    kept = d[d.kept].groupby("group").size().reindex(order, fill_value=0)
    dropped = d[~d.kept].groupby("group").size().reindex(order, fill_value=0)
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(y, kept, height=0.62, color=LOW_COLOR, edgecolor=SURFACE, linewidth=1.5, label="Kept")
    ax.barh(y, dropped, left=kept, height=0.62, color=NEUTRAL, edgecolor=INK_MUTED,
            linewidth=0.8, label="Dropped (D1-D10)")
    for yi, k, dr in zip(y, kept, dropped):
        ax.text(k + dr + 0.3, yi, f"{k} kept" + (f", {dr} dropped" if dr else ""),
                va="center", fontsize=7.8, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels([SIGNAL_NAMES[g] for g in order])
    ax.set_xlim(0, 24)
    ax.set_xlabel("Number of features")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    strip_spines(ax)
    ax.set_title(f"The final feature table: {n_rows} rows x {int(kept.sum())} features", loc="left")
    caption(fig, f"Rows: each participant's A1 (low load) and A4 (high load) recording. "
                 f"{int(kept.sum() + dropped.sum())} features extracted; the reason for every drop "
                 "is in features/feature_dictionary.csv.")
    return fig


# ------------------------------------------------------------------ what the data looks like
def _example_recording(log: pd.DataFrame) -> tuple[int, int]:
    """A typical high-load recording: the analysed A4 recording with median invalid gaze."""
    a4 = log[log.in_analysis.astype(str).eq("True") & log.activity.eq(HIGH_ACTIVITY)]
    inv = a4.invalid_fraction.astype(float)
    row = a4.loc[(inv - inv.median()).abs().idxmin()]
    return int(row.participant), int(row.activity)


def recording_signals_figure(data_dir, log: pd.DataFrame, window_s: float = 20.0):
    """Pupil size, gaze angles and eye speed over time for one recording, blinks shaded."""
    pid, act = _example_recording(log)
    rec = trim_to_gaze_range(load_recording(pid, act, data_dir))
    t = rec["gaze"]["gaze_timestamp"].to_numpy(float)
    t0 = t.min()
    blinks = clean_blinks(rec["blinks"])
    grid, xyz = gaze_directions(rec["gaze"], blinks)
    pgrid, pupil = clean_pupil(rec["pupil"], blinks, t.min(), t.max())
    az = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 2]))
    el = np.degrees(np.arctan2(xyz[:, 1], np.hypot(xyz[:, 0], xyz[:, 2])))
    speed = angular_speed_deg_s(xyz, GAZE_GRID_HZ)
    gs, ps = grid - t0, pgrid - t0
    g_in, p_in = gs <= window_s, ps <= window_s

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 5.6), sharex=True)
    axes[0].plot(ps[p_in], pupil[p_in], color=LOW_COLOR, lw=1.6)
    axes[0].set_ylabel("Pupil size (mm)")
    axes[1].plot(gs[g_in], az[g_in], color=LOW_COLOR, lw=1.2, label="Left/right")
    axes[1].plot(gs[g_in], el[g_in], color=HIGH_COLOR, lw=1.2, label="Up/down")
    axes[1].set_ylabel("Gaze angle (deg)")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    sp = np.where(np.isfinite(speed), speed, np.nan)[: len(gs)]
    axes[2].plot(gs[g_in], sp[g_in], color=INK_SECONDARY, lw=0.8)
    axes[2].axhline(IVT_THRESHOLD_DEG_S, color=HIGH_COLOR, lw=1.2, ls="--")
    axes[2].text(1.0, 1.02, f"Dashed line: {IVT_THRESHOLD_DEG_S:.0f} deg/s. Above = saccade (jump), "
                 "below = fixation (look)", transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=7.6, color=INK_SECONDARY)
    axes[2].set_yscale("log")
    axes[2].set_ylim(1, 1000)
    axes[2].set_ylabel("Eye speed (deg/s)")
    axes[2].set_xlabel("Time from the start of the activity (s)")
    for ax in axes:
        for b in blinks.itertuples():
            if b.start - t0 <= window_s:
                ax.axvspan(b.start - t0, b.end - t0, color=NEUTRAL, zorder=0)
        ax.set_xlim(0, window_s)
        strip_spines(ax)
    axes[0].set_title(f"One recording after cleaning (P{pid:02d}, A{act}, first {window_s:.0f} s)",
                      loc="left")
    caption(fig, "Shaded bands: blinks (gaze is left blank there; pupil gaps up to 250 ms are "
                 "filled). The example is the analysed A4 recording with the median share of "
                 "invalid gaze.")
    return fig


def cleaning_effect_figure(data_dir, log: pd.DataFrame, window_s: float = 10.0):
    """(a) C1 on the recording that lost the most pupil rows; (b) C2/C5/C7 on the example."""
    inside = log[log.in_analysis.astype(str).eq("True")]
    cut = (inside.pupil_rows_raw - inside.pupil_rows_after_trim).astype(float)
    c1 = inside.loc[cut.idxmax()]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 5.4))

    rec = load_recording(int(c1.participant), int(c1.activity), data_dir)
    g = rec["gaze"]["gaze_timestamp"].to_numpy(float)
    p = rec["pupil"]["pupil_timestamp"].to_numpy(float)
    g0, g1 = g.min(), g.max()
    keep = (p >= g0) & (p <= g1)
    a.axvspan(0, (g1 - g0) / 60, color=NEUTRAL, zorder=0)
    a.scatter((p[keep] - g0) / 60, np.zeros(keep.sum()), s=3, color=LOW_COLOR, label="Kept: during the activity")
    a.scatter((p[~keep] - g0) / 60, np.zeros((~keep).sum()), s=3, color=HIGH_COLOR,
              label=f"Removed by C1: after the activity ({_fmt((~keep).sum())} rows)")
    a.set_yticks([])
    a.grid(axis="y", visible=False)
    a.set_xlabel("Minutes from the start of the activity")
    a.legend(loc="upper right", markerscale=4)
    a.set_title(f"(a) C1: stray pupil samples (P{int(c1.participant):02d}, A{int(c1.activity)}; "
                f"activity lasted {g1 - g0:.0f} s)", loc="left")
    strip_spines(a, keep=("bottom",))

    pid, act = _example_recording(log)
    rec = trim_to_gaze_range(load_recording(pid, act, data_dir))
    t = rec["gaze"]["gaze_timestamp"].to_numpy(float)
    t0 = t.min()
    blinks = clean_blinks(rec["blinks"])
    pgrid, pupil = clean_pupil(rec["pupil"], blinks, t.min(), t.max())
    raw = rec["pupil"]
    raw = raw[raw["method"].astype(str).str.contains(PUPIL_METHOD_TAG) & raw.eye_id.eq(0.0)]
    rt = raw.pupil_timestamp.to_numpy(float) - t0
    rd = raw.diameter_3d.to_numpy(float)
    bad = ((raw.confidence.to_numpy() < CONFIDENCE_MIN) | (rd < PUPIL_MIN_MM) | (rd > PUPIL_MAX_MM))
    win = rt <= window_s
    b.scatter(rt[win & ~bad], rd[win & ~bad], s=3, color=INK_MUTED, label="Raw samples (one eye)")
    b.scatter(rt[win & bad], np.clip(rd[win & bad], 0, 10), s=8, marker="x", color=HIGH_COLOR,
              linewidths=0.8, label="Removed: low confidence or impossible size")
    pw = (pgrid - t0) <= window_s
    b.plot(pgrid[pw] - t0, pupil[pw], color=LOW_COLOR, lw=2, label="Cleaned (both eyes, smoothed)")
    for bl in blinks.itertuples():
        if bl.start - t0 <= window_s:
            b.axvspan(bl.start - t0, bl.end - t0, color=NEUTRAL, zorder=0)
    b.set_xlim(0, window_s)
    b.set_ylim(0, 10)
    b.set_xlabel("Time from the start of the activity (s)")
    b.set_ylabel("Pupil size (mm)")
    b.legend(loc="upper right", fontsize=7.6, markerscale=2)
    b.set_title(f"(b) C2, C5, C7: pupil before and after cleaning (P{pid:02d}, A{act}, first "
                f"{window_s:.0f} s)", loc="left")
    strip_spines(b)
    fig.tight_layout(h_pad=1.6)
    caption(fig, "Shaded band in (b): a blink. Removed points are drawn at their raw value, "
                 "clipped to 0-10 mm.")
    return fig


def a1_vs_a4_figure(table: pd.DataFrame):
    """Six core measures per participant, low vs high load, one line per person."""
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6))
    wide = table.pivot(index="participant", columns="activity")
    shown = {f: t for f, t in CORE_FEATURES.items() if f in table}   # C9 failure drops event ones
    for ax in axes.flat[len(shown):]:
        ax.axis("off")
    for ax, (feat, title) in zip(axes.flat, shown.items()):
        lo, hi = wide[(feat, LOW_ACTIVITY)], wide[(feat, HIGH_ACTIVITY)]
        for l, h in zip(lo, hi):
            ax.plot([0, 1], [l, h], color=GRID, lw=0.8, zorder=1)
        ax.scatter(np.zeros(len(lo)), lo, s=14, color=LOW_COLOR, zorder=3)
        ax.scatter(np.ones(len(hi)), hi, s=14, color=HIGH_COLOR, zorder=3)
        for x, v in ((0, lo), (1, hi)):
            ax.plot([x - 0.18, x + 0.18], [v.median()] * 2, color=INK, lw=2, zorder=4)
        up = int((hi > lo).sum())
        ax.set_title(title, fontsize=9.5, loc="left", pad=15)
        ax.text(0.0, 1.02, f"higher in A4 for {up} of {len(lo)} participants",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6, color=INK_SECONDARY)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A1 low", "A4 high"])
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="x", visible=False)
        strip_spines(ax)
    fig.suptitle("Low vs high load for six core measures (one line = one participant)",
                 x=0.0, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    caption(fig, "Black bar: median. Descriptive only: these six are the defended thesis's core "
                 "measures, fixed in advance, not picked for how well they separate A1 from A4.")
    return fig


def data_quality_figure(log: pd.DataFrame):
    """Invalid gaze per recording and activity, with the C3 cut-off and dropped recordings."""
    d = log[~log.participant.isin(DUPLICATE_RECORDING_IDS)].copy()
    d["pct"] = d.invalid_fraction.astype(float) * 100
    d["out"] = d.excluded.astype(str).eq("True")
    rng = np.random.default_rng(0)
    d["x"] = d.activity + rng.uniform(-0.18, 0.18, len(d))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axhline(MAX_INVALID_FRACTION * 100, color=HIGH_COLOR, lw=1.2, ls="--")
    ax.text(0.55, MAX_INVALID_FRACTION * 100 + 1, f"C3 cut-off: {MAX_INVALID_FRACTION:.0%}",
            ha="left", va="bottom", fontsize=7.8, color=INK_SECONDARY)
    kept = d[~d.out]
    ax.scatter(kept.x, kept.pct, s=16, color=LOW_COLOR, label=f"Kept ({len(kept)})", zorder=3)
    out = d[d.out]
    ax.scatter(out.x, out.pct, s=30, color=HIGH_COLOR, edgecolors=SURFACE, linewidths=1.5,
               label=f"Dropped ({len(out)})", zorder=4)
    for r in out.itertuples():
        ax.text(r.x + 0.08, r.pct, f"P{int(r.participant):02d} A{int(r.activity)}", va="center",
                fontsize=7.8, color=INK_SECONDARY)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([ACTIVITY_NAMES[a] for a in (1, 2, 3, 4)])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel("Gaze samples below confidence 0.8 (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    strip_spines(ax)
    ax.set_title("Data quality: unreadable gaze per recording", loc="left")
    caption(fig, f"{len(d)} recordings (P31's duplicate files left out). Never used as a feature: "
                 "high-load recordings lose more data, so it would give away the label (C10).")
    return fig


def all_figures(data_dir, log: pd.DataFrame, table: pd.DataFrame, feature_dict: pd.DataFrame,
                raw_dict: pd.DataFrame, shapes: pd.DataFrame, n_extracted: int) -> dict[str, bytes]:
    """Every figure as PNG bytes, keyed by file name."""
    apply_style()
    n_people = table.participant.nunique()
    n_kept = table.shape[1] - 3
    return {
        "fig_01_raw_tables.png": _png(raw_tables_figure(shapes)),
        "fig_02_raw_columns.png": _png(raw_columns_figure(raw_dict)),
        "fig_03_data_flow.png": _png(data_flow_figure(log, n_people, len(table), n_extracted, n_kept)),
        "fig_04_feature_table.png": _png(feature_table_figure(feature_dict, len(table))),
        "fig_05_recording_signals.png": _png(recording_signals_figure(data_dir, log)),
        "fig_06_cleaning_effect.png": _png(cleaning_effect_figure(data_dir, log)),
        "fig_07_a1_vs_a4.png": _png(a1_vs_a4_figure(table)),
        "fig_08_data_quality.png": _png(data_quality_figure(log)),
    }

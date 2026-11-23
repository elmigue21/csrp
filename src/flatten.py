"""Flatten each cleaned signal into summary statistics (post-defense-data-processing.md §4).

Every signal from features.recording_signals becomes 17 statistics named
`<signal>__<statistic>`. The three rates are added as they are: 10 x 17 + 3 = 173 features.
`cv` of gaze_x/gaze_y is kept but tagged `position` with the other gaze-location statistics:
gaze angles have no true zero, so it depends on how the headset sat (§4.3, §4.6).

A statistic is NaN when the data is too small for its math: empty input -> all NaN;
sd/var/cv/iqr need n >= 2; skew n >= 3; kurt n >= 4; skew/kurt also need values that
vary (values equal up to float rounding count as constant);
cv needs a non-zero mean. On top of that, flat_features leaves every statistic of a signal
blank when the recording has fewer than MIN_VALUES values for it (decision D2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

STATISTICS = ("min", "max", "range", "mean", "median", "sd", "var", "cv", "iqr",
              "p05", "p10", "p25", "p75", "p90", "p95", "skew", "kurt")
_PERCENTILES = {"p05": 5, "p10": 10, "p25": 25, "p75": 75, "p90": 90, "p95": 95}

SIGNALS = {                       # name -> unit, in output order
    "pupil": "mm",
    "gaze_x": "deg",
    "gaze_y": "deg",
    "gaze_speed": "deg/s",
    "fixation_duration": "ms",
    "saccade_amplitude": "deg",
    "saccade_peak_velocity": "deg/s",
    "saccade_mean_velocity": "deg/s",
    "saccade_duration": "ms",
    "blink_duration": "ms",
}
RATES = {"blink_rate": "per min", "fixation_rate": "per s", "saccade_rate": "per s"}
POSITION_SIGNALS = ("gaze_x", "gaze_y")
MIN_VALUES = 20                   # D2: fewer values than this -> the signal's statistics are blank
POSITION_STATS = ("mean", "median", "min", "max", "p05", "p10", "p25", "p75", "p90", "p95", "cv")

# Feature decisions made by the team on data-quality grounds before any model was run
# (post-defense-data-processing.md §6, D1-D10, 2026-11-09). Extracted but not model inputs.
_D1 = ("D1: blank in 79 of 88 recordings (fewer than 20 blinks); 8 of the 9 with values are "
       "high load, so whether a value exists reveals the label")
_D6A = "D6a: stuck at one sample (4.17 ms) in 83 of 88 recordings; measures the tracker"
_D8 = "D8: variance is SD squared, an exact copy of the SD feature"
_D9 = "D9: minimum is fixed near zero or a detection threshold, so range copies max"
_RANGE_COPIES = ("gaze_speed", "fixation_duration", "saccade_amplitude", "saccade_peak_velocity",
                 "saccade_mean_velocity", "saccade_duration")
DROPPED: dict[str, str] = {
    **{f"blink_duration__{s}": _D1 for s in STATISTICS},
    "saccade_duration__min": _D6A,
    **{f"{sig}__var": _D8 for sig in SIGNALS if sig != "blink_duration"},
    **{f"{sig}__range": _D9 for sig in _RANGE_COPIES},
}

_LABELS = {
    "pupil": "pupil size",
    "gaze_x": "left/right gaze angle",
    "gaze_y": "up/down gaze angle",
    "gaze_speed": "eye movement speed",
    "fixation_duration": "fixation length",
    "saccade_amplitude": "saccade size",
    "saccade_peak_velocity": "saccade peak speed",
    "saccade_mean_velocity": "saccade average speed",
    "saccade_duration": "saccade length",
    "blink_duration": "blink length",
}
_RATE_MEANINGS = {
    "blink_rate": "Blinks per minute of usable recording time",
    "fixation_rate": "Fixations per second of time with a valid gaze direction",
    "saccade_rate": "Saccades per second of time with a valid gaze direction",
}


def summarize(values: np.ndarray, skip: tuple[str, ...] = ()) -> dict[str, float]:
    """The 17 statistics of one signal (minus `skip`), NaN where n is too small (module doc)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    nan = float("nan")
    out = dict.fromkeys(STATISTICS, nan)
    if n:
        mean = float(np.mean(x))
        pct = np.percentile(x, [25, 75, *_PERCENTILES.values()])
        out.update(min=float(x.min()), max=float(x.max()), range=float(x.max() - x.min()),
                   mean=mean, median=float(np.median(x)))
        out.update({k: float(v) for k, v in zip(_PERCENTILES, pct[2:])})
        if n >= 2:
            # Values equal up to float rounding (e.g. 12.5 vs 12.500000000000004) count as
            # constant: SD 0 and no skew/kurt, instead of a 1e-17 SD and a scipy warning.
            varies = not np.isclose(x.max(), x.min(), rtol=1e-12, atol=0.0)
            sd = float(np.std(x, ddof=1)) if varies else 0.0
            out.update(sd=sd, var=sd ** 2, iqr=float(pct[1] - pct[0]) if varies else 0.0,
                       cv=sd / mean if mean != 0 else nan)
            if varies and n >= 3:
                out["skew"] = float(skew(x, bias=False))
            if varies and n >= 4:
                out["kurt"] = float(kurtosis(x, fisher=True, bias=False))
    return {k: v for k, v in out.items() if k not in skip}


def flat_features(sig: dict) -> dict[str, float]:
    """The 173 features of one recording from recording_signals output, in feature_names order."""
    out = {}
    for signal in SIGNALS:
        values = np.asarray(sig["signals"][signal], dtype=float)
        if len(values) < MIN_VALUES:                       # D2: too few values to describe
            values = values[:0]
        stats = summarize(values)
        out.update({f"{signal}__{k}": v for k, v in stats.items()})
    out.update({rate: float(sig["rates"][rate]) for rate in RATES})
    return out


def feature_names() -> list[str]:
    names = [f"{signal}__{s}" for signal in SIGNALS for s in STATISTICS]
    return names + list(RATES)


def kept_feature_names() -> list[str]:
    """The model features: feature_names() minus the team's quality drops (DROPPED), in order."""
    return [f for f in feature_names() if f not in DROPPED]


def _meaning(signal: str, stat: str, unit: str) -> str:
    label = _LABELS[signal]
    if stat == "cv" and signal in POSITION_SIGNALS:
        return (f"SD of {label} divided by its average; gaze angles have no true zero, so this "
                "depends on how the headset sat (tagged position)")
    if stat in _PERCENTILES:
        return f"{label.capitalize()} ({unit}) that {_PERCENTILES[stat]}% of values stay below"
    return {
        "min": f"Smallest {label} ({unit})",
        "max": f"Largest {label} ({unit})",
        "range": f"Largest minus smallest {label} ({unit})",
        "mean": f"Average {label} ({unit})",
        "median": f"Middle value of {label} ({unit})",
        "sd": f"How much {label} varies around its average ({unit})",
        "var": f"SD of {label}, squared ({unit}²)",
        "cv": f"SD of {label} divided by its average: variation relative to size",
        "iqr": f"Spread of the middle half of {label} values ({unit})",
        "skew": f"Lopsidedness of {label}: positive = a long tail of high values",
        "kurt": f"Peakedness of {label}: higher = more extreme values",
    }[stat]


def feature_dictionary() -> pd.DataFrame:
    """One row per extracted feature: name, signal, statistic, unit, tag ('position' or ''),
    plain meaning, and whether the team kept it (with the drop reason if not)."""
    rows = []
    for signal, unit in SIGNALS.items():
        for stat in STATISTICS:
            stat_unit = {"var": f"{unit}²", "cv": "ratio", "skew": "", "kurt": ""}.get(stat, unit)
            tag = "position" if signal in POSITION_SIGNALS and stat in POSITION_STATS else ""
            name = f"{signal}__{stat}"
            rows.append({"feature": name, "signal": signal, "statistic": stat,
                         "unit": stat_unit, "tag": tag, "meaning": _meaning(signal, stat, unit),
                         "kept": name not in DROPPED, "drop_reason": DROPPED.get(name, "")})
    for rate, unit in RATES.items():
        rows.append({"feature": rate, "signal": rate, "statistic": "rate", "unit": unit,
                     "tag": "", "meaning": _RATE_MEANINGS[rate], "kept": True, "drop_reason": ""})
    return pd.DataFrame(rows)

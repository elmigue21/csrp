"""Turn the 1-10 self-report into a binary high/low target.

Both labellings use only `selfreport_mental_load(1-10)`. They differ solely in
where the boundary between low and high sits:

  absolute      -- one fixed cut for everyone (rating >= ABSOLUTE_THRESHOLD)
  within_person -- each participant is cut at their own median rating

The second exists because participants do not share a scale. In this dataset P12
never rated above 3 while P15 never rated below 6, so a fixed cut makes every P12
recording low and every P15 recording high, and a classifier can score well by
recognising the person rather than the workload.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import ABSOLUTE_THRESHOLD

DROP = -1  # marker for rows this labelling cannot assign


def absolute_label(ratings: pd.Series, threshold: float = ABSOLUTE_THRESHOLD) -> pd.Series:
    """High when the raw rating reaches `threshold`. Never drops a row."""
    return (ratings >= threshold).astype(int)


def within_person_label(ratings: pd.Series, participants: pd.Series) -> pd.Series:
    """High when the rating is above that participant's own median rating.

    Ratings sitting exactly at a participant's median are genuinely ambiguous --
    neither more nor less demanding than their typical task -- and are marked DROP
    rather than forced into a class.
    """
    # Median over recordings, not over epochs, so that a long recording does not
    # drag the participant's centre towards its own rating.
    per_recording = (
        pd.DataFrame({"participant": participants, "rating": ratings})
        .groupby("participant")["rating"]
        .median()
    )
    median = participants.map(per_recording)
    return pd.Series(
        np.where(ratings > median, 1, np.where(ratings < median, 0, DROP)),
        index=ratings.index,
        dtype=int,
    )


def build_labels(epochs: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "absolute":
        return absolute_label(epochs["rating"])
    if scheme == "within_person":
        return within_person_label(epochs["rating"], epochs["participant"])
    raise ValueError(f"unknown labelling scheme: {scheme!r}")


def rating_spread(epochs: pd.DataFrame) -> pd.DataFrame:
    """Per-participant rating range, to expose participants who barely varied.

    A participant whose five tasks all felt equally easy has no meaningful
    within-person contrast, and the within_person labelling would be inventing one.
    """
    per_rec = epochs.groupby(["participant", "task"])["rating"].first().reset_index()
    out = per_rec.groupby("participant")["rating"].agg(
        min_rating="min", max_rating="max", median_rating="median",
        rating_range=lambda s: s.max() - s.min(), n_tasks="size",
    )
    return out.reset_index()


def difficulty_agreement(epochs: pd.DataFrame, y: pd.Series) -> dict:
    """Do the high labels land on the objectively harder tasks?

    Task difficulty is externally fixed by the experimental protocol, so if a
    labelling is capturing real workload rather than relabelling noise inside each
    participant, its high labels should concentrate on the later tasks. Reported as
    a per-task breakdown plus a rank correlation.
    """
    keep = y != DROP
    per_rec = (
        pd.DataFrame({"participant": epochs["participant"], "task": epochs["task"],
                      "y": y, "rating": epochs["rating"]})[keep]
        .groupby(["participant", "task"])
        .agg(y=("y", "first"), rating=("rating", "first"))
        .reset_index()
    )
    by_task = per_rec.groupby("task").agg(
        n_recordings=("y", "size"), n_high=("y", "sum"), mean_rating=("rating", "mean"),
    )
    by_task["pct_high"] = (100 * by_task["n_high"] / by_task["n_recordings"]).round(1)
    rho, p = spearmanr(per_rec["task"], per_rec["y"])
    return {"by_task": by_task.reset_index(), "spearman_rho": float(rho), "p_value": float(p)}

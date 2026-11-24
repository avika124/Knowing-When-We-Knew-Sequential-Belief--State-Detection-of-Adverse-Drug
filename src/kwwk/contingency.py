"""Quarterly 2x2 contingency tables from daily openFDA report counts.

For a (drug, event) pair and a quarter:

    a = reports mentioning both the drug and the event
    b = reports with the drug but not the event
    c = reports with the event but not the drug
    d = reports with neither

The API gives three marginal series plus the joint one, so the table is stored
as (a, n_drug, n_event, n_total) and b, c, d are derived.
"""

from __future__ import annotations

import pandas as pd


def quarter_of(yyyymmdd):
    year, month = int(yyyymmdd[:4]), int(yyyymmdd[4:6])
    return f"{year}Q{(month - 1) // 3 + 1}"


def quarter_range(first, last):
    return [str(p) for p in pd.period_range(first, last, freq="Q")]


def quarter_end(quarter):
    """Last calendar day of a quarter; data for the quarter is complete after it."""
    return pd.Period(quarter, freq="Q").end_time.normalize()


def to_quarterly(daily_counts, quarters):
    """Aggregate ``{YYYYMMDD: n}`` into a list aligned with ``quarters``."""
    totals = dict.fromkeys(quarters, 0)
    for day, n in daily_counts.items():
        q = quarter_of(day)
        if q in totals:
            totals[q] += n
    return [totals[q] for q in quarters]


def add_cells(df):
    """Add b, c, d columns to a frame holding a, n_drug, n_event, n_total."""
    out = df.copy()
    out["b"] = out["n_drug"] - out["a"]
    out["c"] = out["n_event"] - out["a"]
    out["d"] = out["n_total"] - out["n_drug"] - out["n_event"] + out["a"]
    return out


def cumulative(df):
    """Cumulative counts per pair: the database as it stood at each quarter end."""
    cols = ["a", "n_drug", "n_event", "n_total"]
    out = df.sort_values(["pair_id", "quarter"]).copy()
    out[cols] = out.groupby("pair_id")[cols].cumsum()
    return out


def load_contingency(path, first=None, last=None):
    # Quarter labels are compared as strings here and throughout the replay.
    # The "YYYYQn" layout has a fixed-width year first and n in 1-4, so
    # lexicographic order is the same as chronological order; this is the
    # cut-off that enforces the no-look-ahead window, so the ordering matters.
    df = pd.read_csv(path)
    if first:
        df = df[df["quarter"] >= first]
    if last:
        df = df[df["quarter"] <= last]
    return df.sort_values(["pair_id", "quarter"]).reset_index(drop=True)

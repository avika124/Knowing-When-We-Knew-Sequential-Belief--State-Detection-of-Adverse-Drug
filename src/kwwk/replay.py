"""Retrospective replay: every method lives through history one quarter at a time.

At quarter t each method may use only reports received up to the end of t.

* PRR, ROR, BCPNN and GPS are static scores, so they are recomputed each
  quarter on the cumulative table (the database as it stood then).
* The HMM consumes the per-quarter counts as a sequence and carries its
  belief state forward.

The result is one row per (pair, quarter) holding every score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import gps as gps_mod
from . import hmm as hmm_mod
from .bcpnn import information_component
from .contingency import add_cells, cumulative
from .disproportionality import evans_signal, prr, ror


def static_scores(counts):
    """PRR, ROR and IC on cumulative counts for every (pair, quarter)."""
    cum = add_cells(cumulative(counts))
    out = cum[["pair_id", "quarter"]].copy()
    out["cum_a"] = cum["a"].to_numpy()
    out["cum_n_drug"] = cum["n_drug"].to_numpy()
    out["cum_expected"] = (cum["n_drug"] * cum["n_event"] / cum["n_total"].clip(lower=1)).to_numpy()
    for name, values in prr(cum["a"], cum["b"], cum["c"], cum["d"]).items():
        out[name] = values
    for name, values in ror(cum["a"], cum["b"], cum["c"], cum["d"]).items():
        out[name] = values
    for name, values in information_component(
        cum["a"], cum["n_drug"], cum["n_event"], cum["n_total"]
    ).items():
        out[name] = values
    out["evans"] = evans_signal(out["cum_a"], out["prr"], out["chi2"])
    return out


def gps_by_quarter(static, pool_ids):
    """EBGM / EB05 per quarter, with the prior refitted each quarter.

    The prior is fitted only on ``pool_ids`` (the tuning pairs) and only on
    data available at that quarter, then applied to every pair.
    """
    pieces = []
    prior = gps_mod.DEFAULT_PRIOR
    priors = {}
    for quarter, block in static.groupby("quarter", sort=True):
        pool = block[block["pair_id"].isin(pool_ids)]
        prior = gps_mod.fit_prior(pool["cum_a"], pool["cum_expected"], start=prior)
        priors[quarter] = prior
        scores = gps_mod.gps_scores(block["cum_a"], block["cum_expected"], prior)
        piece = block[["pair_id", "quarter"]].copy()
        for name, values in scores.items():
            piece[name] = values
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True), priors


def sequences(counts):
    """Per-pair observation sequences ``{pair_id: (quarters, a, expected)}``."""
    out = {}
    for pid, block in counts.sort_values("quarter").groupby("pair_id", sort=False):
        expected = hmm_mod.expected_count(
            block["a"], block["n_drug"], block["n_event"], block["n_total"]
        )
        out[pid] = (block["quarter"].to_numpy(), block["a"].to_numpy(dtype=float), expected)
    return out


def hmm_scores(counts, models):
    """Filtered belief state for every pair under each named parameter set."""
    rows = []
    for pid, (quarters, a, expected) in sequences(counts).items():
        piece = pd.DataFrame({"pair_id": pid, "quarter": quarters, "expected": expected})
        for name, params in models.items():
            piece[f"{name}_belief"] = hmm_mod.filtered_posterior(a, expected, params)
            piece[f"{name}_logodds"] = hmm_mod.filtered_log_odds(a, expected, params)
            piece[f"{name}_smoothed"] = hmm_mod.smoothed_posterior(a, expected, params)
            piece[f"{name}_level"] = hmm_mod.detection_level(a, expected, params)
        rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def run_replay(counts, models, gps_pool_ids):
    """Score every pair in every quarter with every method."""
    counts = counts.sort_values(["pair_id", "quarter"]).reset_index(drop=True)
    static = static_scores(counts)
    gps_scores, gps_priors = gps_by_quarter(static, set(gps_pool_ids))
    hmm = hmm_scores(counts, models)
    meta = counts[["pair_id", "drug", "event", "group", "label", "split", "quarter",
                   "a", "n_drug", "n_event", "n_total"]]
    out = (
        meta.merge(static, on=["pair_id", "quarter"])
        .merge(gps_scores, on=["pair_id", "quarter"])
        .merge(hmm, on=["pair_id", "quarter"])
    )
    return out, gps_priors

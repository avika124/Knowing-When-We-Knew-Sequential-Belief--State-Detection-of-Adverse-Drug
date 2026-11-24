"""Shared steps of the replay experiment: pair selection, splits, HMM fitting."""

from __future__ import annotations

from . import hmm
from .replay import sequences


def select_pairs(history, first, last, min_drug_reports):
    """Counts inside the replay window, dropping low-exposure control pairs.

    Historical cases are always kept.  Returns the counts and the number of
    control pairs that were dropped.
    """
    counts = history[(history["quarter"] >= first) & (history["quarter"] <= last)]
    exposure = counts.groupby("pair_id")["n_drug"].sum()
    enough = counts["pair_id"].map(exposure >= min_drug_reports)
    n_dropped = counts.loc[~enough & (counts["group"] == "omop"), "pair_id"].nunique()
    return counts[(counts["group"] == "case") | enough], int(n_dropped)


def split_ids(counts, group="omop"):
    """Pair ids of the reference set by split and label: tune_neg, tune_pos, test_neg, test_pos."""
    meta = counts.drop_duplicates("pair_id").set_index("pair_id")
    ref = meta[meta["group"] == group]
    return {
        f"{split}_{'pos' if label else 'neg'}":
            list(ref.index[(ref["split"] == split) & (ref["label"] == label)])
        for split in ("tune", "test") for label in (0, 1)
    }


def fit_hmms(counts, tune_pool, start=None):
    """The two HMM parameter sets: priors, and Baum-Welch on the unlabelled tuning pool.

    Returns ``(models, fit, train)`` where ``train`` is the list of sequences
    Baum-Welch saw.
    """
    seqs = sequences(counts)
    prior = start or hmm.default_params()
    train = [(seqs[p][1], seqs[p][2]) for p in tune_pool]
    fit = hmm.baum_welch(train, prior)
    return {"hmm_prior": prior, "hmm_bw": fit.params}, fit, train

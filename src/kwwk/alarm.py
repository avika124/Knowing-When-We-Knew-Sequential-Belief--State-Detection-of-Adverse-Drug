"""Alarm rules, lead time, and threshold tuning on negative controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contingency import quarter_end

MIN_CASES = 3  # no method may alarm on fewer than three joint reports


@dataclass(frozen=True)
class Method:
    key: str            # short identifier used in tables
    label: str          # display name
    score: str          # continuous alarm score (higher = more suspicious), used for tuning
    conv_score: str     # column the conventional rule is applied to
    conv_threshold: float
    conv_strict: bool   # True: score > threshold, False: score >= threshold
    rule: str           # the conventional rule in words
    family: str         # methods in one family share a colour in the figures
    has_conventional: bool = True


HMM_LABELS = {"hmm_prior": "HMM (prior)", "hmm_bw": "HMM (Baum-Welch)"}


def methods(hmm_names=("hmm_prior", "hmm_bw")):
    """The detectors being compared, with their conventional signalling rules.

    Each HMM appears twice.  The plain entry is tuned through its belief
    threshold, as proposed.  The "level" entry keeps the belief threshold at
    one half and is tuned through the baseline level instead (see
    ``hmm.detection_level``).  At their conventional setting the two are the
    same rule, so only the plain entry reports it.
    """
    out = [
        Method("prr", "PRR", "prr_lower", "evans", 0.5, False,
               "PRR >= 2, chi-square >= 4, cases >= 3", "prr"),
        Method("ror", "ROR", "ror_lower", "ror_lower", 1.0, True, "lower 95% bound > 1", "ror"),
        Method("bcpnn", "BCPNN", "ic025", "ic025", 0.0, True, "IC025 > 0", "bcpnn"),
        Method("gps", "GPS", "eb05", "eb05", 2.0, False, "EB05 >= 2", "gps"),
    ]
    for name in hmm_names:
        label = HMM_LABELS.get(name, name)
        out.append(Method(name, label, f"{name}_logodds", f"{name}_logodds", 0.0, False,
                          "belief >= 0.5", name))
    for name in hmm_names:
        label = HMM_LABELS.get(name, name).replace(")", ", level)")
        out.append(Method(f"{name}_level", label, f"{name}_level", f"{name}_level", 0.0, False,
                          "belief >= 0.5", name, has_conventional=False))
    return out


def eligible_scores(replay, score):
    """Score matrix (pairs x quarters) with ineligible quarters set to -inf."""
    wide = replay.pivot(index="pair_id", columns="quarter", values=score).astype(float)
    cases = replay.pivot(index="pair_id", columns="quarter", values="cum_a")
    values = wide.to_numpy(dtype=float)
    values = np.where(np.isnan(values) | (cases.to_numpy() < MIN_CASES), -np.inf, values)
    return pd.DataFrame(values, index=wide.index, columns=wide.columns)


def first_alarm(scores, threshold, strict=False):
    """Index of the first quarter whose score crosses the threshold, or -1."""
    hit = scores > threshold if strict else scores >= threshold
    idx = np.argmax(hit, axis=1)
    return np.where(hit.any(axis=1), idx, -1)


def alarm_date(quarter):
    """A quarter's data is complete the day after it ends; that is when an alarm can fire."""
    return quarter_end(quarter) + pd.Timedelta(days=1)


def lead_time_months(quarter, action_date):
    """Months by which an alarm at the end of ``quarter`` precedes the action date."""
    delta = pd.Timestamp(action_date) - alarm_date(quarter)
    return delta.days / 30.4375


def false_alarm_rate(scores, threshold, strict=False):
    """Share of pairs that alarm at any point in the replay window."""
    if len(scores) == 0:
        return float("nan")
    return float((first_alarm(scores.to_numpy(), threshold, strict) >= 0).mean())


# Sensitivity is the same computation -- the share of pairs that ever alarm --
# applied to known positives rather than negative controls.  The alias keeps
# the intent readable at the call site.
detection_rate = false_alarm_rate


def tune_threshold(negative_scores, target_far):
    """Smallest threshold whose false-alarm rate on the negatives is <= target.

    ``negative_scores`` is the eligible score matrix of the negative controls.
    A pair alarms when its running maximum reaches the threshold, so the
    threshold is placed just above the (m+1)-th largest per-pair maximum,
    where m is the number of false alarms the target allows.
    """
    peak = np.sort(negative_scores.to_numpy().max(axis=1))[::-1]
    allowed = int(np.floor(target_far * len(peak)))
    if allowed >= len(peak):
        return -np.inf
    boundary = peak[allowed]
    if not np.isfinite(boundary):
        # fewer pairs ever become eligible than the target allows
        finite = peak[np.isfinite(peak)]
        return float(finite.min()) if len(finite) else np.inf
    return float(np.nextafter(boundary, np.inf))

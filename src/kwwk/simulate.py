"""Simulated report streams with a known onset, for testing detectors.

Real data cannot say how fast a method detects a signal, because the true
onset is unknown.  Here the exposure is real and only the joint count is
simulated: each synthetic pair borrows the quarterly report volumes of a real
control pair, and its drug-event count is drawn around a relative reporting
rate that we choose.

    a_t ~ NegBin(mean = r_t * E_t, shape = k)

    r_t = baseline                 before the onset quarter
    r_t = baseline * effect        from the onset quarter on

``baseline`` is 1 for a clean pair.  Giving every pair its own baseline
(log-normal spread) mimics confounding by indication, where a drug's
reporting ratio sits stably away from 1 for reasons unrelated to harm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import expected_count


def exposure_profiles(counts):
    """Per-pair quarterly volumes from real data: (quarters, n_drug, c, n_total, E)."""
    profiles = []
    for _, block in counts.sort_values("quarter").groupby("pair_id", sort=True):
        n_drug = block["n_drug"].to_numpy(dtype=float)
        c = (block["n_event"] - block["a"]).to_numpy(dtype=float)  # event reports without the drug
        n_total = block["n_total"].to_numpy(dtype=float)
        expected = expected_count(0.0, n_drug, c, n_total)
        profiles.append((block["quarter"].to_numpy(), n_drug, c, n_total, expected))
    return profiles


def simulate(profiles, n_pairs, rng, effect=1.0, onset=None, baseline_sd=0.0, k=4.0,
             prefix="sim", label=0, split="test"):
    """Draw ``n_pairs`` synthetic pairs.

    ``onset`` is None for no change, an int for a fixed onset quarter index,
    or a ``(low, high)`` tuple for an onset drawn uniformly from that range.
    ``effect`` multiplies the reporting rate from the onset on; with
    ``onset=0`` the pair is elevated from the first quarter.

    Returns the counts in the same layout as the real contingency table, and
    a truth table with each pair's onset index and baseline.
    """
    frames, truth = [], []
    for i in range(n_pairs):
        quarters, n_drug, c, n_total, expected = profiles[rng.integers(len(profiles))]
        T = len(quarters)
        baseline = float(np.exp(rng.normal(0.0, baseline_sd))) if baseline_sd > 0 else 1.0
        if onset is None:
            start = T  # never
        elif isinstance(onset, tuple):
            start = int(rng.integers(onset[0], onset[1] + 1))
        else:
            start = int(onset)
        rate = np.where(np.arange(T) >= start, baseline * effect, baseline)
        mean = rate * expected
        # negative binomial as a gamma-mixed Poisson
        lam = rng.gamma(shape=k, scale=np.maximum(mean, 1e-12) / k)
        a = np.minimum(rng.poisson(lam), n_drug)
        a = np.where(expected > 0, a, 0.0)
        pid = f"{prefix}{i:05d}"
        frames.append(pd.DataFrame({
            "pair_id": pid, "drug": pid, "event": "simulated", "group": "sim",
            "label": label, "split": split, "quarter": quarters,
            "a": a.astype(int), "n_drug": n_drug.astype(int),
            "n_event": (c + a).astype(int), "n_total": n_total.astype(int),
        }))
        truth.append({"pair_id": pid, "onset": start if start < T else -1, "baseline": baseline})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth).set_index("pair_id")


def detection_delay(alarm_index, onset, horizon):
    """Quarters from onset to alarm for each pair, with three outcomes.

    Returns ``(delay, premature, missed)``.  ``delay`` is alarm minus onset,
    with a pair that never alarms counted as detected at the end of the
    window (``horizon - onset``).  ``premature`` marks alarms raised before
    the onset, which are false alarms and are excluded from the delay.
    """
    alarm_index = np.asarray(alarm_index)
    onset = np.asarray(onset)
    missed = alarm_index < 0
    premature = ~missed & (alarm_index < onset)
    delay = np.where(missed, horizon - onset, alarm_index - onset).astype(float)
    delay[premature] = np.nan
    return delay, premature, missed

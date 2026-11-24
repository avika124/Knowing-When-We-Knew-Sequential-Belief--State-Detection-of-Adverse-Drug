"""BCPNN Information Component (Bate et al., 1998).

    IC = log2 [ P(drug, event) / ( P(drug) * P(event) ) ]

The three probabilities get Beta priors chosen so that the prior expectation
of IC is zero, i.e. drug and event are assumed independent until the data say
otherwise.  That shrinks the estimate toward zero when counts are small, which
is what keeps sparse early quarters from raising false alarms.  A signal is
declared when the lower 95% credible bound (IC025) is above zero.
"""

from __future__ import annotations

import numpy as np

LN2 = np.log(2.0)


def information_component(a, n_drug, n_event, n_total, z=1.959963984540054):
    """Posterior mean, standard deviation and 95% bounds of the IC.

    Parameters are the joint count and the three margins of the 2x2 table.
    """
    a, n_drug, n_event, n = (np.asarray(x, dtype=float) for x in (a, n_drug, n_event, n_total))

    # Bate et al. (1998) hyperparameters: uniform Beta(1, 1) priors on the two
    # marginals, and a joint prior matched to independence.
    g11, a1, b1, alpha, beta = 1.0, 1.0, 1.0, 2.0, 2.0
    gamma = g11 * (n + alpha) * (n + beta) / ((n_drug + a1) * (n_event + b1))

    mean = np.log2(
        (a + g11) * (n + alpha) * (n + beta) / ((n + gamma) * (n_drug + a1) * (n_event + b1))
    )
    var = (
        (n - a + gamma - g11) / ((a + g11) * (1 + n + gamma))
        + (n - n_drug + alpha - a1) / ((n_drug + a1) * (1 + n + alpha))
        + (n - n_event + beta - b1) / ((n_event + b1) * (1 + n + beta))
    ) / LN2**2
    sd = np.sqrt(var)
    return {"ic": mean, "ic_sd": sd, "ic025": mean - z * sd, "ic975": mean + z * sd}

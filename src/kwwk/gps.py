"""Gamma-Poisson Shrinker (DuMouchel, 1999): EBGM and EB05.

Each pair's observed count N is Poisson with mean lambda * E, where E is the
count expected under independence and lambda is the true relative reporting
rate.  The lambdas across pairs share a prior that is a mixture of two gamma
distributions; its five hyperparameters are fitted by maximum marginal
likelihood (empirical Bayes).  The posterior for one pair is again a mixture
of two gammas, summarised by

    EBGM = exp(E[log lambda | N])     the shrunk geometric-mean estimate
    EB05 = 5th percentile of lambda   the lower bound used for signalling
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammainc, gammaln, logit

# Starting values recommended by DuMouchel (1999): (alpha1, beta1, alpha2, beta2, P)
DEFAULT_PRIOR = (0.2, 0.1, 2.0, 4.0, 1.0 / 3.0)


def _nb_logpmf(n, expected, alpha, beta):
    """Marginal log P(N = n) for lambda ~ Gamma(alpha, rate beta)."""
    return (
        gammaln(alpha + n) - gammaln(alpha) - gammaln(n + 1.0)
        + alpha * (np.log(beta) - np.log(beta + expected))
        + n * (np.log(expected) - np.log(beta + expected))
    )


def _unpack(theta):
    a1, b1, a2, b2 = np.exp(theta[:4])
    return a1, b1, a2, b2, expit(theta[4])


def _pack(prior):
    a1, b1, a2, b2, p = prior
    return np.array([np.log(a1), np.log(b1), np.log(a2), np.log(b2), logit(p)])


def neg_log_marginal(theta, n, expected):
    a1, b1, a2, b2, p = _unpack(theta)
    l1 = np.log(p) + _nb_logpmf(n, expected, a1, b1)
    l2 = np.log1p(-p) + _nb_logpmf(n, expected, a2, b2)
    return -np.sum(np.logaddexp(l1, l2))


def fit_prior(n, expected, start=DEFAULT_PRIOR):
    """Fit the two-gamma mixture prior to a pool of (N, E) pairs.

    Falls back to ``start`` if the optimiser fails, which can happen when the
    pool is tiny or almost every count is zero.
    """
    n = np.asarray(n, dtype=float)
    expected = np.asarray(expected, dtype=float)
    keep = expected > 0
    n, expected = n[keep], expected[keep]
    if keep.sum() < 20 or (n > 0).sum() < 10:
        return tuple(start)
    theta0 = _pack(start)
    bounds = [(-7, 7)] * 4 + [(-7, 7)]
    res = minimize(neg_log_marginal, theta0, args=(n, expected), method="L-BFGS-B", bounds=bounds)
    if not res.success and not np.isfinite(res.fun):
        return tuple(start)
    if res.fun > neg_log_marginal(theta0, n, expected):
        return tuple(start)
    return tuple(float(x) for x in _unpack(res.x))


def _posterior(n, expected, prior):
    a1, b1, a2, b2, p = prior
    l1 = np.log(p) + _nb_logpmf(n, expected, a1, b1)
    l2 = np.log1p(-p) + _nb_logpmf(n, expected, a2, b2)
    q = np.exp(l1 - np.logaddexp(l1, l2))  # posterior weight of component 1
    return q, (a1 + n, b1 + expected), (a2 + n, b2 + expected)


def _mixture_quantile(prob, q, comp1, comp2, iters=80):
    """Quantile of a two-gamma mixture by bisection on log(lambda)."""
    lo = np.full_like(q, -30.0)
    hi = np.full_like(q, 30.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x = np.exp(mid)
        cdf = q * gammainc(comp1[0], comp1[1] * x) + (1 - q) * gammainc(comp2[0], comp2[1] * x)
        below = cdf < prob
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.exp(0.5 * (lo + hi))


def gps_scores(n, expected, prior):
    """EBGM, EB05 and EB95 for each pair.  Pairs with E = 0 return NaN."""
    n = np.asarray(n, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ok = expected > 0
    e_safe = np.where(ok, expected, 1.0)
    q, c1, c2 = _posterior(n, e_safe, prior)
    log_mean = q * (digamma(c1[0]) - np.log(c1[1])) + (1 - q) * (digamma(c2[0]) - np.log(c2[1]))
    out = {
        "ebgm": np.exp(log_mean),
        "eb05": _mixture_quantile(0.05, q, c1, c2),
        "eb95": _mixture_quantile(0.95, q, c1, c2),
    }
    return {name: np.where(ok, values, np.nan) for name, values in out.items()}

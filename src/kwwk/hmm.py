"""Two-state hidden Markov model for temporal belief-state signal detection.

Hidden state per quarter:  z_t in {0: safe, 1: risky}.

Observation per quarter:   the number of reports a_t that mention both the
drug and the event, given the count E_t that would be expected if the drug
reported the event at the same rate as every other drug in the database.

Emission model:            a_t | z_t = j  ~  NegBin(mean = rho_j * E_t, shape = k_j)

    This is a Poisson count whose rate is Gamma-distributed around rho_j, so
    k_j controls how much quarter-to-quarter overdispersion state j tolerates
    (k -> infinity recovers a plain Poisson).  rho is the relative reporting
    rate: about 1 when the pair is unremarkable, above 1 when it is elevated.

Transition model:          safe -> risky with probability p per quarter, and
                           risky is absorbing (a one-way shift).

Inference runs entirely in log space.  The forward recursion multiplies one
likelihood per quarter, which underflows quickly in probability space, so the
sums over the previous state are done with the log-sum-exp trick.

The filtered posterior P(z_t = risky | o_1..t) is the belief state: what could
be known at quarter t using only the data that had arrived by then.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

SAFE, RISKY = 0, 1
NEG_INF = -np.inf


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass
class HMMParams:
    pi: np.ndarray   # initial state distribution, shape (2,)
    A: np.ndarray    # transition matrix, rows sum to one, shape (2, 2)
    rho: np.ndarray  # relative reporting rate per state, shape (2,)
    k: np.ndarray    # negative-binomial shape per state, shape (2,)

    def __post_init__(self):
        self.pi = np.asarray(self.pi, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        self.k = np.asarray(self.k, dtype=float)

    def copy(self):
        return HMMParams(self.pi.copy(), self.A.copy(), self.rho.copy(), self.k.copy())

    def as_dict(self):
        return {
            "pi_risky": float(self.pi[RISKY]),
            "p_safe_to_risky": float(self.A[SAFE, RISKY]),
            "p_risky_to_safe": float(self.A[RISKY, SAFE]),
            "rho_safe": float(self.rho[SAFE]),
            "rho_risky": float(self.rho[RISKY]),
            "k_safe": float(self.k[SAFE]),
            "k_risky": float(self.k[RISKY]),
        }


def default_params(pi_risky=0.05, p_shift=0.02, rho_safe=1.0, rho_risky=2.0, k=4.0):
    """Parameters set from priors rather than learned.

    "Risky" is anchored at a doubling of the reporting rate, the same effect
    size Evans' PRR rule screens for.  A pair starts out risky with small
    probability and otherwise shifts with a small per-quarter hazard.
    """
    return HMMParams(
        pi=[1 - pi_risky, pi_risky],
        A=[[1 - p_shift, p_shift], [0.0, 1.0]],
        rho=[rho_safe, rho_risky],
        k=[k, k],
    )


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #

def expected_count(a, n_drug, n_event, n_total):
    """Expected joint count E_t if the drug shared the background event rate.

    The background rate is the event's share of reports that do *not* mention
    the drug, c / (c + d), with a half-count of smoothing so that it is never
    exactly zero.  E_t = (reports with the drug) * (background rate).
    """
    a, n_drug, n_event, n_total = (
        np.asarray(x, dtype=float) for x in (a, n_drug, n_event, n_total)
    )
    c = n_event - a
    others = n_total - n_drug
    return n_drug * (c + 0.5) / (others + 1.0)


def nb_logpmf(a, mean, k):
    """log NegBin(a; mean, shape k), elementwise.  mean == 0 forces a == 0."""
    a = np.asarray(a, dtype=float)
    mean = np.asarray(mean, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (
            gammaln(a + k) - gammaln(k) - gammaln(a + 1.0)
            + k * (np.log(k) - np.log(k + mean))
            + a * (np.log(mean) - np.log(k + mean))
        )
    out = np.where((mean == 0) & (a == 0), 0.0, out)
    out = np.where((mean == 0) & (a > 0), NEG_INF, out)
    return out


def emission_loglik(a, expected, params):
    """log P(a_t | z_t = j) for every quarter and state, shape (T, 2).

    Quarters with no exposure (E_t = 0) carry no information and get a log
    likelihood of zero in both states.
    """
    a = np.asarray(a, dtype=float)
    expected = np.asarray(expected, dtype=float)
    out = np.zeros((len(a), 2))
    for j in (SAFE, RISKY):
        out[:, j] = nb_logpmf(a, params.rho[j] * expected, params.k[j])
    out[expected <= 0] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Log-space inference
# --------------------------------------------------------------------------- #

def logsumexp(x, axis=None):
    """log(sum(exp(x))) computed stably by factoring out the maximum.

    exp() of a large negative number underflows to zero, so the largest term
    is pulled out first:  log sum_i e^{x_i} = m + log sum_i e^{x_i - m}.
    """
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return out if axis is None and out.ndim == 0 else np.squeeze(out, axis=axis)


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, dtype=float))


def forward(loglik, pi, A):
    """Forward algorithm.

    Returns ``log_alpha`` with log_alpha[t, j] = log P(o_1..t, z_t = j) and the
    total log-likelihood log P(o_1..T).

        alpha_1(j) = pi_j * b_j(o_1)
        alpha_t(j) = [ sum_i alpha_{t-1}(i) * A_ij ] * b_j(o_t)
    """
    T = loglik.shape[0]
    log_pi, log_A = _log(pi), _log(A)
    log_alpha = np.empty((T, 2))
    log_alpha[0] = log_pi + loglik[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + loglik[t]
    return log_alpha, float(logsumexp(log_alpha[-1]))


def backward(loglik, A):
    """Backward algorithm: log_beta[t, i] = log P(o_{t+1}..T | z_t = i)."""
    T = loglik.shape[0]
    log_A = _log(A)
    log_beta = np.zeros((T, 2))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + (loglik[t + 1] + log_beta[t + 1])[None, :], axis=1)
    return log_beta


def filtered_posterior(a, expected, params):
    """Belief state P(z_t = risky | o_1..t) for every quarter t."""
    loglik = emission_loglik(a, expected, params)
    log_alpha, _ = forward(loglik, params.pi, params.A)
    log_norm = logsumexp(log_alpha, axis=1)
    return np.exp(log_alpha[:, RISKY] - log_norm)


def filtered_log_odds(a, expected, params):
    """log [ P(risky | o_1..t) / P(safe | o_1..t) ]; does not saturate like the probability."""
    loglik = emission_loglik(a, expected, params)
    log_alpha, _ = forward(loglik, params.pi, params.A)
    return log_alpha[:, RISKY] - log_alpha[:, SAFE]


def detection_level(a, expected, params, levels=None):
    """Largest baseline multiplier at which the pair would still be flagged.

    The belief threshold answers "how sure are we?".  This score answers a
    different question: "elevated relative to what?".  Both emission rates are
    multiplied by a level c, which moves the model's idea of normal reporting
    from rho_safe to c * rho_safe, and the score at quarter t is the largest c
    for which the belief is still at least one half.

    A level of 1 is the uncalibrated model, so ``level >= 1`` is exactly the
    rule ``belief >= 0.5``.  Requiring a higher level demands a larger effect
    rather than more evidence, which is what a confounded but stable reporting
    ratio calls for.  Returned on a log scale (log c); -inf if the pair is not
    flagged even at the lowest level on the grid.
    """
    if levels is None:
        levels = np.exp(np.linspace(np.log(1 / 16), np.log(64), 193))
    a = np.asarray(a, dtype=float)
    expected = np.asarray(expected, dtype=float)
    T, B = len(a), len(levels)

    # loglik[t, b, j]: quarter t, level b, state j
    loglik = np.zeros((T, B, 2))
    for j in (SAFE, RISKY):
        mean = params.rho[j] * levels[None, :] * expected[:, None]
        loglik[:, :, j] = nb_logpmf(a[:, None], mean, params.k[j])
    loglik[expected <= 0] = 0.0

    log_A = _log(params.A)
    log_alpha = _log(params.pi)[None, :] + loglik[0]
    out = np.full(T, NEG_INF)
    for t in range(T):
        if t > 0:
            log_alpha = logsumexp(log_alpha[:, :, None] + log_A[None, :, :], axis=1) + loglik[t]
        # Raising the level scales both emission rates up together, so it only
        # ever makes a quarter look more normal: ``flagged`` is True for an
        # unbroken run of the lowest levels and False above it.  The score is
        # the top of that run, so the largest flagged index is its boundary.
        flagged = log_alpha[:, RISKY] >= log_alpha[:, SAFE]
        if flagged.any():
            out[t] = np.log(levels[np.nonzero(flagged)[0].max()])
    return out


def smoothed_posterior(a, expected, params):
    """Hindsight estimate P(z_t = risky | o_1..T) using the whole sequence."""
    loglik = emission_loglik(a, expected, params)
    log_alpha, total = forward(loglik, params.pi, params.A)
    log_beta = backward(loglik, params.A)
    return np.exp(log_alpha[:, RISKY] + log_beta[:, RISKY] - total)


def viterbi(a, expected, params):
    """Most likely hidden state path (max-product in log space)."""
    loglik = emission_loglik(a, expected, params)
    T = loglik.shape[0]
    log_pi, log_A = _log(params.pi), _log(params.A)
    delta = np.empty((T, 2))
    back = np.zeros((T, 2), dtype=int)
    delta[0] = log_pi + loglik[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        back[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + loglik[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# --------------------------------------------------------------------------- #
# Baum-Welch (EM)
# --------------------------------------------------------------------------- #

@dataclass
class FitResult:
    params: HMMParams
    log_likelihood: list = field(default_factory=list)  # objective after each iteration
    converged: bool = False


def _e_step(a, expected, params):
    """Posterior state and transition responsibilities for one sequence."""
    loglik = emission_loglik(a, expected, params)
    log_alpha, total = forward(loglik, params.pi, params.A)
    log_beta = backward(loglik, params.A)
    gamma = np.exp(log_alpha + log_beta - total)
    log_A = _log(params.A)
    # xi[t, i, j] = P(z_t = i, z_{t+1} = j | o_1..T)
    log_xi = (
        log_alpha[:-1, :, None] + log_A[None, :, :]
        + (loglik[1:] + log_beta[1:])[:, None, :] - total
    )
    return gamma, np.exp(log_xi).sum(axis=0), total


def _fit_emission(a, expected, weights, rho0, k0, rho_bounds, k_bounds):
    """Weighted maximum likelihood for one state's (rho, k).

    The negative binomial has no closed-form M-step, so the expected complete
    log-likelihood is maximised numerically over (log rho, log k).
    """
    keep = (expected > 0) & (weights > 1e-12)
    a, expected, weights = a[keep], expected[keep], weights[keep]
    if weights.sum() < 1e-8:
        return rho0, k0

    def objective(theta):
        rho, k = np.exp(theta)
        return -np.sum(weights * nb_logpmf(a, rho * expected, k))

    start = np.log([rho0, k0])
    bounds = [tuple(np.log(rho_bounds)), tuple(np.log(k_bounds))]
    best = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(best.fun) or best.fun > objective(start):
        return rho0, k0
    rho, k = np.exp(best.x)
    return float(rho), float(k)


def baum_welch(
    sequences,
    init,
    max_iter=500,
    tol=1e-7,
    pseudo_count=1.0,
    learn_emissions=True,
    rho_bounds=(0.02, 50.0),
    k_bounds=(0.05, 500.0),
):
    """Learn HMM parameters from unlabelled sequences by EM.

    ``sequences`` is a list of ``(a, expected)`` array pairs, one per
    (drug, event) pair.  Each iteration runs forward-backward on every
    sequence (E-step) and then re-estimates the initial distribution, the
    transition matrix and the emission parameters from the expected counts
    (M-step).

    Transitions that start at zero stay at zero, so the one-way safe -> risky
    structure survives training.  ``pseudo_count`` is a weak Dirichlet prior on
    the allowed entries of ``pi`` and ``A`` that keeps them off the boundary;
    the reported objective includes that prior, and it never decreases.
    """
    params = init.copy()
    sequences = [(np.asarray(a, dtype=float), np.asarray(e, dtype=float)) for a, e in sequences]
    allowed = init.A > 0
    pi_allowed = init.pi > 0
    history = []
    converged = False

    for _ in range(max_iter):
        # ---- E-step -------------------------------------------------------
        pi_counts = np.zeros(2)
        trans_counts = np.zeros((2, 2))
        gammas = []
        total = 0.0
        for a, expected in sequences:
            gamma, xi_sum, loglik = _e_step(a, expected, params)
            pi_counts += gamma[0]
            trans_counts += xi_sum
            gammas.append(gamma)
            total += loglik

        log_prior = pseudo_count * (
            np.sum(np.log(params.pi[pi_allowed])) + np.sum(np.log(params.A[allowed]))
        )
        history.append(total + log_prior)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * abs(history[-2]):
            converged = True
            break

        # ---- M-step -------------------------------------------------------
        new = params.copy()
        pi_post = np.where(pi_allowed, pi_counts + pseudo_count, 0.0)
        new.pi = pi_post / pi_post.sum()
        A_post = np.where(allowed, trans_counts + pseudo_count, 0.0)
        new.A = A_post / A_post.sum(axis=1, keepdims=True)

        if learn_emissions:
            all_a = np.concatenate([a for a, _ in sequences])
            all_e = np.concatenate([e for _, e in sequences])
            all_g = np.concatenate(gammas)
            for j in (SAFE, RISKY):
                new.rho[j], new.k[j] = _fit_emission(
                    all_a, all_e, all_g[:, j], params.rho[j], params.k[j], rho_bounds, k_bounds
                )
        params = new

    return FitResult(params=params, log_likelihood=history, converged=converged)


# --------------------------------------------------------------------------- #
# Simulation (used by the tests and for sanity checks)
# --------------------------------------------------------------------------- #

def sample(params, expected, rng):
    """Draw a hidden path and observed counts for a given expected-count series."""
    expected = np.asarray(expected, dtype=float)
    T = len(expected)
    z = np.empty(T, dtype=int)
    z[0] = rng.choice(2, p=params.pi)
    for t in range(1, T):
        z[t] = rng.choice(2, p=params.A[z[t - 1]])
    rate = rng.gamma(shape=params.k[z], scale=params.rho[z] / params.k[z])
    return z, rng.poisson(rate * expected)

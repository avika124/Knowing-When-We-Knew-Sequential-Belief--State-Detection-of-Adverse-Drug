"""Raising an alarm as a decision problem on the belief state.

The HMM gives a belief p = P(risky).  Turning it into an alarm needs a rule,
and the usual one, "alarm when p >= tau", hides a value judgement in tau.
This module makes the judgement explicit.

Each quarter the monitor chooses between two actions:

    ALARM   stop.  If the pair is in fact safe this is a false alarm,
            with expected cost  c_false_alarm * (1 - p).
    WAIT    collect one more quarter.  If the pair is already risky, the
            harm goes unflagged for another quarter, with expected cost
            c_delay * p.

The hidden state is not observed, so this is a partially observable MDP.
Its belief state is the single number p, which turns it into an ordinary MDP
on [0, 1] that value iteration can solve:

    V(p) = min{ c_false_alarm * (1 - p),
                c_delay * p + E_a[ V(p') ] }

where a is next quarter's count and p' the belief after seeing it.  With a
one-way safe -> risky chain this is Shiryaev's quickest-detection problem,
and the optimal policy is a threshold on p.  Solving it gives the threshold
that a stated cost ratio implies, and read backwards, the cost ratio that a
chosen threshold implies.
"""

from __future__ import annotations

import numpy as np

from .hmm import RISKY, SAFE, nb_logpmf


def _observation_model(params, expected, mass=1e-10):
    """P(a | state) for a = 0..a_max, truncated where the risky tail is negligible."""
    mean_hi = max(params.rho) * expected
    spread = np.sqrt(mean_hi + mean_hi**2 / min(params.k))
    a_max = int(np.ceil(mean_hi + 12 * spread + 20))
    counts = np.arange(a_max + 1)
    lik = np.exp(np.stack([nb_logpmf(counts, params.rho[j] * expected, params.k[j])
                           for j in (SAFE, RISKY)]))
    assert lik.sum(axis=1).min() > 1 - mass * 1e4, "count grid truncates the distribution"
    return lik / lik.sum(axis=1, keepdims=True)


def belief_transitions(params, expected, grid=401):
    """Belief grid and the matrix M with  E_a[V(p')] = M @ V  under WAIT.

    Row i of M spreads the probability of next quarter's count over the
    beliefs it leads to, using linear interpolation between grid points.
    M depends on the model and on ``expected`` but not on the costs, so one
    matrix serves every cost ratio.
    """
    p = np.linspace(0.0, 1.0, grid)
    lik = _observation_model(params, expected)

    # belief after the hidden state has had a chance to move, before the new count
    ahead = p * params.A[RISKY, RISKY] + (1 - p) * params.A[SAFE, RISKY]
    joint_risky = ahead[:, None] * lik[RISKY][None, :]
    marginal = joint_risky + (1 - ahead)[:, None] * lik[SAFE][None, :]   # P(a | p)
    posterior = joint_risky / np.maximum(marginal, 1e-300)               # p' for each (p, a)

    position = posterior * (grid - 1)
    lower = np.clip(np.floor(position).astype(int), 0, grid - 2)
    weight = position - lower
    rows = np.repeat(np.arange(grid), marginal.shape[1]).reshape(marginal.shape)
    M = np.zeros((grid, grid))
    np.add.at(M, (rows, lower), marginal * (1 - weight))
    np.add.at(M, (rows, lower + 1), marginal * weight)
    return p, M


def solve(p, M, cost_false_alarm, cost_delay=1.0, tol=1e-10, max_iter=200_000):
    """Value iteration on the belief MDP.

    Returns the value function and the optimal threshold: the smallest belief
    at which alarming is at least as good as waiting.
    """
    stop = cost_false_alarm * (1 - p)
    value = stop.copy()
    for _ in range(max_iter):
        new = np.minimum(stop, cost_delay * p + M @ value)
        if np.max(np.abs(new - value)) < tol:
            value = new
            break
        value = new
    alarm = stop <= cost_delay * p + M @ value + 1e-12
    threshold = float(p[np.argmax(alarm)]) if alarm.any() else 1.0
    return value, threshold


def optimal_threshold(params, expected, cost_ratio, grid=401):
    """Belief threshold that is optimal when a false alarm costs ``cost_ratio``
    quarters of undetected risk."""
    p, M = belief_transitions(params, expected, grid)
    return solve(p, M, cost_false_alarm=cost_ratio)[1]


def implied_cost_ratio(params, expected, threshold, low=1e-3, high=1e6, iters=30, grid=401):
    """Cost ratio under which ``threshold`` is the optimal alarm rule.

    The optimal threshold rises with the cost of a false alarm, so the ratio
    is found by bisection on a log scale.  Returns infinity when even the
    largest ratio tried does not justify waiting that long.
    """
    p, M = belief_transitions(params, expected, grid)
    if solve(p, M, high)[1] < threshold:
        return float("inf")
    lo, hi = np.log(low), np.log(high)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if solve(p, M, float(np.exp(mid)))[1] < threshold:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))

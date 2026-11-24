import itertools

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from kwwk import hmm
from kwwk.hmm import HMMParams, RISKY, SAFE


def general_params():
    # deliberately not one-way, to exercise the general recursions
    return HMMParams(pi=[0.7, 0.3], A=[[0.9, 0.1], [0.2, 0.8]], rho=[1.0, 3.0], k=[5.0, 2.0])


def brute_force(a, expected, params):
    """Enumerate every hidden path; returns joint probabilities keyed by path."""
    loglik = hmm.emission_loglik(a, expected, params)
    joint = {}
    for path in itertools.product((0, 1), repeat=len(a)):
        p = params.pi[path[0]] * np.exp(loglik[0, path[0]])
        for t in range(1, len(a)):
            p *= params.A[path[t - 1], path[t]] * np.exp(loglik[t, path[t]])
        joint[path] = p
    return joint


def test_logsumexp_matches_scipy_and_handles_extremes():
    x = np.array([[-1000.0, -1001.0], [-np.inf, 0.0], [700.0, 710.0]])
    assert np.allclose(hmm.logsumexp(x, axis=1), scipy_logsumexp(x, axis=1))
    assert np.isclose(hmm.logsumexp(x[0]), scipy_logsumexp(x[0]))
    assert hmm.logsumexp(np.array([-np.inf, -np.inf])) == -np.inf


def test_nb_emission_is_a_distribution_and_tends_to_poisson():
    from scipy.stats import poisson
    counts = np.arange(0, 400)
    assert np.isclose(np.exp(hmm.nb_logpmf(counts, 7.5, 3.0)).sum(), 1.0, atol=1e-8)
    assert np.allclose(hmm.nb_logpmf(counts[:30], 4.0, 1e7), poisson.logpmf(counts[:30], 4.0), atol=1e-4)


def test_forward_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    params = general_params()
    expected = rng.uniform(0.5, 6, size=8)
    _, a = hmm.sample(params, expected, rng)
    joint = brute_force(a, expected, params)

    loglik = hmm.emission_loglik(a, expected, params)
    log_alpha, total = hmm.forward(loglik, params.pi, params.A)
    assert np.isclose(np.exp(total), sum(joint.values()))

    # filtered posterior at t uses only the first t+1 observations
    filtered = hmm.filtered_posterior(a, expected, params)
    for t in (0, 3, 7):
        prefix = brute_force(a[: t + 1], expected[: t + 1], params)
        risky = sum(p for path, p in prefix.items() if path[-1] == RISKY)
        assert np.isclose(filtered[t], risky / sum(prefix.values()))

    smoothed = hmm.smoothed_posterior(a, expected, params)
    for t in (0, 4, 7):
        risky = sum(p for path, p in joint.items() if path[t] == RISKY)
        assert np.isclose(smoothed[t], risky / sum(joint.values()))
    assert np.isclose(smoothed[-1], filtered[-1])

    best = max(joint, key=joint.get)
    assert tuple(hmm.viterbi(a, expected, params)) == best


def test_log_space_forward_survives_where_probabilities_underflow():
    rng = np.random.default_rng(1)
    params = hmm.default_params()
    expected = np.full(3000, 40.0)
    _, a = hmm.sample(params, expected, rng)
    loglik = hmm.emission_loglik(a, expected, params)
    assert np.exp(loglik.max(axis=1)).prod() == 0.0   # naive product underflows
    _, total = hmm.forward(loglik, params.pi, params.A)
    assert np.isfinite(total)
    belief = hmm.filtered_posterior(a, expected, params)
    assert np.all((belief >= 0) & (belief <= 1))


def test_belief_rises_after_a_shift_and_stays_low_without_one():
    rng = np.random.default_rng(2)
    params = hmm.default_params()
    expected = np.full(28, 10.0)
    quiet = rng.poisson(1.0 * expected)
    shifted = quiet.copy()
    shifted[14:] = rng.poisson(3.0 * expected[14:])
    assert hmm.filtered_posterior(quiet, expected, params).max() < 0.5
    belief = hmm.filtered_posterior(shifted, expected, params)
    assert belief[13] < 0.5 and belief[-1] > 0.99


def test_one_way_transition_makes_risky_absorbing():
    params = hmm.default_params()
    assert params.A[RISKY, SAFE] == 0.0
    a = np.array([0, 0, 30, 30, 0, 0, 0], dtype=float)
    expected = np.full(7, 5.0)
    path = hmm.viterbi(a, expected, params)
    switched = np.argmax(path == RISKY)
    assert np.all(path[switched:] == RISKY) or not path.any()


def test_quarters_without_exposure_do_not_move_the_belief():
    params = hmm.default_params()
    a = np.array([8, 0, 0, 9], dtype=float)
    expected = np.array([4.0, 0.0, 0.0, 4.0])
    loglik = hmm.emission_loglik(a, expected, params)
    assert np.all(loglik[1:3] == 0.0)


def test_expected_count_uses_background_rate_of_other_drugs():
    # 100 drug reports; among the 9900 others, 99 mention the event (1%)
    e = hmm.expected_count(a=5, n_drug=100, n_event=104, n_total=10_000)
    assert np.isclose(e, 100 * (99 + 0.5) / (9900 + 1))
    assert hmm.expected_count(0, 0, 50, 1000) == 0.0


def simulate(params, n_seq, rng, T=28):
    out = []
    for _ in range(n_seq):
        expected = rng.uniform(1, 15, size=T)
        _, a = hmm.sample(params, expected, rng)
        out.append((a, expected))
    return out


def test_baum_welch_never_decreases_the_objective_and_recovers_parameters():
    rng = np.random.default_rng(3)
    truth = HMMParams(pi=[0.8, 0.2], A=[[0.95, 0.05], [0.0, 1.0]], rho=[0.9, 3.0], k=[6.0, 3.0])
    data = simulate(truth, 400, rng)
    start = hmm.default_params(pi_risky=0.5, p_shift=0.2, rho_safe=0.5, rho_risky=6.0, k=1.0)
    fit = hmm.baum_welch(data, start)

    assert fit.converged
    assert np.all(np.diff(fit.log_likelihood) > -1e-6)
    p = fit.params
    assert abs(p.rho[SAFE] - 0.9) < 0.1
    assert abs(p.rho[RISKY] - 3.0) < 0.3
    assert abs(p.A[SAFE, RISKY] - 0.05) < 0.02
    assert abs(p.pi[RISKY] - 0.2) < 0.07
    assert p.A[RISKY, SAFE] == 0.0          # structural zero preserved
    assert np.allclose(p.A.sum(axis=1), 1.0) and np.isclose(p.pi.sum(), 1.0)


def test_baum_welch_can_learn_a_two_way_chain():
    rng = np.random.default_rng(4)
    truth = HMMParams(pi=[0.6, 0.4], A=[[0.9, 0.1], [0.15, 0.85]], rho=[1.0, 4.0], k=[8.0, 8.0])
    data = simulate(truth, 300, rng, T=40)
    start = HMMParams(pi=[0.5, 0.5], A=[[0.7, 0.3], [0.3, 0.7]], rho=[0.7, 2.0], k=[3.0, 3.0])
    fit = hmm.baum_welch(data, start)
    assert np.allclose(fit.params.A, truth.A, atol=0.03)
    assert np.allclose(fit.params.rho, truth.rho, rtol=0.1)


def test_baum_welch_with_fixed_emissions_only_updates_the_chain():
    rng = np.random.default_rng(5)
    truth = hmm.default_params(p_shift=0.1, rho_risky=3.0)
    data = simulate(truth, 200, rng)
    start = hmm.default_params(p_shift=0.01, rho_risky=3.0)
    fit = hmm.baum_welch(data, start, learn_emissions=False)
    assert np.array_equal(fit.params.rho, start.rho)
    assert abs(fit.params.A[SAFE, RISKY] - 0.1) < 0.03

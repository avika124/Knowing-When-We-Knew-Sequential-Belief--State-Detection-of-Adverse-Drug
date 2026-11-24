import numpy as np

from kwwk.gps import DEFAULT_PRIOR, fit_prior, gps_scores


def test_bounds_are_ordered_and_small_counts_are_shrunk():
    n = np.array([1.0, 3.0, 30.0, 300.0])
    e = np.array([0.1, 0.3, 3.0, 30.0])  # every pair has observed/expected = 10
    s = gps_scores(n, e, DEFAULT_PRIOR)
    assert np.all(s["eb05"] < s["ebgm"]) and np.all(s["ebgm"] < s["eb95"])
    assert np.all(np.diff(s["ebgm"]) > 0)  # more data, less shrinkage
    assert s["ebgm"][0] < 5 < s["ebgm"][-1] < 10


def test_zero_expected_is_nan():
    s = gps_scores([3], [0.0], DEFAULT_PRIOR)
    assert np.isnan(s["ebgm"][0]) and np.isnan(s["eb05"][0])


def test_prior_fit_recovers_simulated_mixture():
    rng = np.random.default_rng(1)
    m = 4000
    expected = rng.lognormal(1.0, 1.5, m)
    first = rng.random(m) < 0.3
    lam = np.where(first, rng.gamma(0.5, 1 / 0.2, m), rng.gamma(3.0, 1 / 3.5, m))
    n = rng.poisson(lam * expected)
    a1, b1, a2, b2, p = fit_prior(n, expected)
    assert abs(a1 / b1 - 2.5) < 0.8     # mean of the heavy-tailed component
    assert abs(a2 / b2 - 3.0 / 3.5) < 0.15
    assert abs(p - 0.3) < 0.1


def test_prior_fit_falls_back_on_tiny_pool():
    assert fit_prior([0, 1, 0], [1.0, 1.0, 1.0]) == DEFAULT_PRIOR

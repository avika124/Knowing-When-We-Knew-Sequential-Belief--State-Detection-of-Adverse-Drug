import numpy as np

from kwwk.bcpnn import information_component
from kwwk.disproportionality import evans_signal, prr, ror


def test_prr_and_ror_match_hand_calculation():
    # drug: 20 of 100 reports have the event; others: 100 of 9900
    p = prr(20, 80, 100, 9800)
    assert np.isclose(p["prr"], (20 / 100) / (100 / 9900))
    assert p["prr_lower"] < p["prr"] < p["prr_upper"]
    r = ror(20, 80, 100, 9800)
    assert np.isclose(r["ror"], (20 * 9800) / (80 * 100))
    assert r["ror_lower"] < r["ror"] < r["ror_upper"]


def test_no_association_gives_ratio_one():
    p = prr(10, 90, 1000, 9000)
    assert np.isclose(p["prr"], 1.0)
    assert p["prr_lower"] < 1.0 < p["prr_upper"]


def test_zero_cells_are_undefined_not_errors():
    p = prr([0, 5], [10, 5], [5, 0], [100, 100])
    assert np.isnan(p["prr"]).all()
    assert np.isnan(ror(0, 10, 5, 100)["ror"])
    assert not evans_signal(0, np.nan, np.nan)


def test_evans_rule_needs_all_three_conditions():
    p = prr(20, 80, 100, 9800)
    assert evans_signal(20, p["prr"], p["chi2"])
    assert not evans_signal(2, 5.0, 10.0)      # too few cases
    assert not evans_signal(10, 1.5, 10.0)     # ratio too small
    assert not evans_signal(10, 5.0, 2.0)      # not significant


def test_ic_is_zero_without_data_and_shrinks_small_counts():
    empty = information_component(0, 0, 0, 0)
    assert np.isclose(empty["ic"], 0.0)
    # one report where 0.0025 were expected: raw log2 ratio would be ~8.6
    sparse = information_component(1, 5, 50, 100_000)
    assert 0 < sparse["ic"] < 2
    assert sparse["ic025"] < 0  # far too uncertain to signal


def test_ic_approaches_log2_observed_over_expected_with_large_counts():
    a, n_drug, n_event, n = 2000, 10_000, 12_000, 1_000_000
    out = information_component(a, n_drug, n_event, n)
    raw = np.log2(a / (n_drug * n_event / n))
    assert abs(out["ic"] - raw) < 0.02
    assert out["ic025"] < out["ic"] < out["ic975"]
    assert out["ic025"] > 0

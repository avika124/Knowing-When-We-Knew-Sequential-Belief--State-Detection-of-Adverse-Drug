"""Tests for the decision analysis, the simulator and the bootstrap interval."""

import numpy as np
import pandas as pd

from kwwk import decision, hmm, simulate
from kwwk.contingency import quarter_range
from kwwk.evaluate import cluster_interval
from kwwk.experiment import select_pairs, split_ids


# ---------------------------------------------------------------- decision

def test_belief_transition_matrix_is_stochastic():
    p, M = decision.belief_transitions(hmm.default_params(), expected=10.0, grid=201)
    assert np.allclose(M.sum(axis=1), 1.0, atol=1e-8)
    assert (M >= 0).all()
    # once risky is certain it stays certain (absorbing state)
    assert np.isclose(M[-1, -1], 1.0)


def test_value_function_never_exceeds_the_cost_of_stopping():
    p, M = decision.belief_transitions(hmm.default_params(), expected=10.0, grid=201)
    value, threshold = decision.solve(p, M, cost_false_alarm=8.0)
    assert np.all(value <= 8.0 * (1 - p) + 1e-9)
    assert np.isclose(value[-1], 0.0)          # certain risk: alarm at once, no cost
    assert 0 < threshold < 1
    # the policy is a threshold: wait below it, alarm above it
    stop = 8.0 * (1 - p)
    assert np.all(value[p >= threshold] >= stop[p >= threshold] - 1e-9)
    assert np.all(value[p < threshold] < stop[p < threshold])


def test_costlier_false_alarms_demand_more_certainty():
    params = hmm.default_params()
    thresholds = [decision.optimal_threshold(params, 10.0, c, grid=201) for c in (1, 4, 16, 64)]
    assert all(a < b for a, b in zip(thresholds, thresholds[1:]))
    assert decision.optimal_threshold(params, 10.0, 1e-6, grid=201) < 0.02   # nearly free to alarm


def test_implied_cost_ratio_inverts_the_optimal_threshold():
    params = hmm.default_params()
    ratio = decision.implied_cost_ratio(params, 10.0, threshold=0.5, grid=201)
    assert decision.optimal_threshold(params, 10.0, ratio * 1.05, grid=201) >= 0.5
    assert decision.optimal_threshold(params, 10.0, ratio * 0.95, grid=201) < 0.5


# --------------------------------------------------------------- simulation

def real_like_counts(n_pairs=6, T=28):
    quarters = quarter_range("2004Q1", "2010Q4")[:T]
    frames = []
    for i in range(n_pairs):
        frames.append(pd.DataFrame({
            "pair_id": f"p{i}", "drug": f"d{i}", "event": "e", "group": "omop", "label": 0,
            "split": "tune" if i % 2 else "test", "quarter": quarters,
            "a": 5, "n_drug": 1000 + 100 * i, "n_event": 2000, "n_total": 100_000,
        }))
    return pd.concat(frames, ignore_index=True)


def test_simulated_pairs_keep_real_volumes_and_respect_the_onset():
    profiles = simulate.exposure_profiles(real_like_counts())
    rng = np.random.default_rng(0)
    counts, truth = simulate.simulate(profiles, 200, rng, effect=5.0, onset=14, label=1)
    assert (counts["a"] <= counts["n_drug"]).all() and (counts["a"] >= 0).all()
    assert counts.groupby("pair_id").size().eq(28).all()
    assert (truth["onset"] == 14).all() and (truth["baseline"] == 1.0).all()

    wide = counts.pivot(index="pair_id", columns="quarter", values="a").to_numpy()
    before, after = wide[:, :14].mean(), wide[:, 14:].mean()
    assert 4.0 < after / before < 6.0          # the rate was multiplied by 5


def test_null_pairs_have_no_onset_and_confounding_spreads_the_baseline():
    profiles = simulate.exposure_profiles(real_like_counts())
    rng = np.random.default_rng(1)
    _, truth = simulate.simulate(profiles, 300, rng, baseline_sd=0.5)
    assert (truth["onset"] == -1).all()
    assert 0.4 < np.log(truth["baseline"]).std() < 0.6


def test_detection_delay_outcomes():
    delay, premature, missed = simulate.detection_delay(
        alarm_index=[12, 5, -1, 10], onset=[10, 10, 10, 10], horizon=28)
    assert delay[0] == 2 and delay[3] == 0
    assert premature.tolist() == [False, True, False, False] and np.isnan(delay[1])
    assert missed.tolist() == [False, False, True, False] and delay[2] == 18


# ------------------------------------------------------- bootstrap, selection

def test_cluster_interval_brackets_the_rate_and_is_reproducible():
    rng = np.random.default_rng(0)
    alarmed = rng.random(120) < 0.3
    clusters = np.repeat(np.arange(40), 3)
    low, high = cluster_interval(alarmed, clusters)
    assert low < alarmed.mean() < high
    assert (low, high) == cluster_interval(alarmed, clusters)
    # perfectly correlated pairs within a drug make the interval wider
    grouped = np.repeat(rng.random(40) < 0.3, 3)
    wide = cluster_interval(grouped, clusters)
    independent = cluster_interval(grouped, np.arange(120))
    assert wide[1] - wide[0] > independent[1] - independent[0]


def test_select_pairs_keeps_cases_and_drops_thin_controls():
    counts = real_like_counts()
    counts.loc[counts["pair_id"] == "p0", "n_drug"] = 1          # 28 reports in total
    counts.loc[counts["pair_id"] == "p1", ["group", "n_drug"]] = ["case", 1]
    kept, dropped = select_pairs(counts, "2004Q1", "2010Q4", min_drug_reports=100)
    assert dropped == 1
    assert "p0" not in set(kept["pair_id"]) and "p1" in set(kept["pair_id"])
    ids = split_ids(kept)
    assert set(ids) == {"tune_neg", "tune_pos", "test_neg", "test_pos"}
    assert not set(ids["tune_neg"]) & set(ids["test_neg"])

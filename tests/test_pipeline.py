import numpy as np
import pandas as pd

from kwwk import alarm, hmm
from kwwk.contingency import add_cells, cumulative, quarter_of, quarter_range, to_quarterly
from kwwk.evaluate import roc_auc
from kwwk.queries import both_clause, drug_clause, event_clause
from kwwk.reference import load_cases, load_events, load_pairs, omop_split
from kwwk.replay import run_replay


def test_quarter_aggregation():
    assert quarter_of("20070521") == "2007Q2"
    quarters = quarter_range("2004Q1", "2004Q4")
    daily = {"20040101": 2, "20040331": 3, "20040401": 5, "20031231": 99, "20041231": 1}
    assert to_quarterly(daily, quarters) == [5, 5, 0, 1]


def test_cells_and_cumulative_counts():
    df = pd.DataFrame({
        "pair_id": ["p"] * 3, "quarter": ["2004Q1", "2004Q2", "2004Q3"],
        "a": [1, 2, 3], "n_drug": [10, 10, 10], "n_event": [5, 5, 5], "n_total": [100, 100, 100],
    })
    cells = add_cells(df)
    assert (cells[["a", "b", "c", "d"]].sum(axis=1) == cells["n_total"]).all()
    assert cumulative(df)["a"].tolist() == [1, 3, 6]
    assert cumulative(df)["n_total"].tolist() == [100, 200, 300]


def test_query_builders():
    q = drug_clause(["rofecoxib", "vioxx"])
    assert 'patient.drug.medicinalproduct:"rofecoxib"' in q and '"vioxx"' in q
    assert event_clause(["Rhabdomyolysis"]) == '(patient.reaction.reactionmeddrapt.exact:"RHABDOMYOLYSIS")'
    assert " AND " in both_clause(["x"], ["y"])


def test_reference_data_is_consistent():
    events = load_events()
    pairs = load_pairs()
    assert len({p.pair_id for p in pairs}) == len(pairs)
    assert all(p.event in events for p in pairs)
    cases = [p for p in pairs if p.group == "case"]
    assert len(cases) == len(load_cases()) and all(p.split == "eval" for p in cases)
    # the split is by drug, so no drug is on both sides
    tune = {p.drug for p in pairs if p.split == "tune"}
    test = {p.drug for p in pairs if p.split == "test"}
    assert not tune & test
    assert omop_split("ibuprofen") == omop_split("Ibuprofen")


def test_first_alarm_and_lead_time():
    scores = np.array([[0.1, 0.6, 0.9], [0.1, 0.2, 0.3], [0.7, 0.1, 0.1]])
    assert alarm.first_alarm(scores, 0.5).tolist() == [1, -1, 0]
    assert alarm.first_alarm(scores, 0.9, strict=True).tolist() == [-1, -1, -1]
    # alarm after 2006Q4 fires on 2007-01-01; the action came on 2007-05-21
    lead = alarm.lead_time_months("2006Q4", "2007-05-21")
    assert 4.5 < lead < 4.7
    assert alarm.lead_time_months("2007Q2", "2007-05-21") < 0


def test_tuned_threshold_respects_the_false_alarm_target():
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.normal(size=(200, 28)))
    for target in (0.0, 0.05, 0.1, 0.25):
        tau = alarm.tune_threshold(scores, target)
        far = alarm.false_alarm_rate(scores, tau)
        assert far <= target and far > target - 0.01


def test_roc_auc():
    assert roc_auc([3, 4], [1, 2]) == 1.0
    assert roc_auc([1, 2], [1, 2]) == 0.5


def make_counts(rng, pair_id, rate, split, label, group="omop", T=12):
    quarters = quarter_range("2004Q1", "2006Q4")[:T]
    n_total = np.full(T, 100_000)
    n_drug = np.full(T, 2_000)
    n_event = np.full(T, 1_000)
    rate = np.broadcast_to(rate, (T,))
    a = rng.poisson(rate * n_drug * n_event / n_total)
    return pd.DataFrame({
        "pair_id": pair_id, "drug": pair_id, "event": "e", "group": group, "label": label,
        "split": split, "quarter": quarters, "a": a, "n_drug": n_drug,
        "n_event": n_event, "n_total": n_total,
    })


def test_replay_scores_only_use_the_past():
    rng = np.random.default_rng(0)
    frames = [make_counts(rng, f"neg{i}", 1.0, "tune", 0) for i in range(30)]
    late_signal = np.r_[np.ones(6), np.full(6, 4.0)]
    frames.append(make_counts(rng, "case", late_signal, "eval", 1, group="case"))
    counts = pd.concat(frames, ignore_index=True)
    models = {"hmm_prior": hmm.default_params()}
    pool = [f"neg{i}" for i in range(30)]

    full, _ = run_replay(counts, models, pool)
    truncated, _ = run_replay(counts[counts["quarter"] <= "2005Q2"], models, pool)

    # scores for early quarters must not change when later data is removed
    cols = ["prr", "ic025", "eb05", "hmm_prior_belief"]
    early = full[full["quarter"] <= "2005Q2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(early[cols], truncated.reset_index(drop=True)[cols])

    case = full[full["pair_id"] == "case"].set_index("quarter")
    assert case.loc["2005Q2", "hmm_prior_belief"] < 0.5
    assert case.loc["2006Q4", "hmm_prior_belief"] > 0.95
    assert case.loc["2006Q4", "ic025"] > 0


def test_case_alarm_separates_sustained_from_faded_signals():
    from kwwk.evaluate import case_alarm
    quarters = quarter_range("2006Q1", "2007Q4")
    scores = pd.DataFrame(
        [[2, 0, 0, 0, 0, 0, 3, 3],      # early blip that fades, then a late rise
         [0, 0, 2, 2, 2, 2, 2, 2],      # crosses and stays
         [0, 0, 0, 0, 0, 0, 0, 0]],     # never crosses
        index=["faded", "sustained", "silent"], columns=quarters, dtype=float,
    )
    action = "2007-05-21"
    faded = case_alarm(scores, "faded", 1.0, False, action, "2006Q1")
    assert faded["alarm_quarter"] == "2006Q1" and faded["before_action"]
    assert faded["at_window_start"] and not faded["standing_at_action"]
    sustained = case_alarm(scores, "sustained", 1.0, False, action, "2006Q1")
    assert sustained["alarm_quarter"] == "2006Q3" and sustained["standing_at_action"]
    assert 7.5 < sustained["lead_months"] < 7.8
    silent = case_alarm(scores, "silent", 1.0, False, action, "2006Q1")
    assert silent["alarm_quarter"] is None and np.isnan(silent["lead_months"])


def test_detection_level_agrees_with_the_belief_rule():
    rng = np.random.default_rng(7)
    params = hmm.default_params()
    for rate in (0.6, 1.0, 2.5, 5.0):
        expected = rng.uniform(2, 20, size=20)
        a = rng.poisson(rate * expected).astype(float)
        level = hmm.detection_level(a, expected, params)
        belief = hmm.filtered_posterior(a, expected, params)
        assert np.array_equal(level >= 0, belief >= 0.5)
    # a stronger signal supports a higher level
    expected = np.full(12, 10.0)
    weak = hmm.detection_level(np.full(12, 25.0), expected, params)[-1]
    strong = hmm.detection_level(np.full(12, 60.0), expected, params)[-1]
    assert strong > weak > 0

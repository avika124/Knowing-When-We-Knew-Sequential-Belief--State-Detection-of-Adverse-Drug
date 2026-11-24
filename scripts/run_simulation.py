"""Simulation study: how fast does each method detect a signal with a known onset?

The real cases cannot answer this, because nobody knows when their true risk
began.  Here each synthetic pair borrows the real quarterly report volumes of
a control pair and gets a simulated drug-event count (see kwwk.simulate).

Scenarios, each run with a clean and a confounded baseline:

    null         no change; used to tune thresholds and measure false alarms
    emerging     reporting rate multiplies by `effect` at a random quarter
    from start   elevated from the first quarter

Every method is tuned to 10% false alarms on null tuning pairs, then scored
on separate test pairs.

    python scripts/run_simulation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwwk import plots, simulate  # noqa: E402
from kwwk.alarm import eligible_scores, false_alarm_rate, first_alarm, methods, tune_threshold  # noqa: E402
from kwwk.contingency import load_contingency  # noqa: E402
from kwwk.experiment import fit_hmms, select_pairs  # noqa: E402
from kwwk.replay import run_replay  # noqa: E402

FIRST, LAST = "2004Q1", "2010Q4"
N_PAIRS = 400            # per group
ONSET = (8, 20)          # emerging signals start in quarter 9-21 of 28
TARGET_FAR = 0.10
BASELINES = {"clean": 0.0, "confounded": 0.5}   # sd of the log baseline ratio
EFFECTS = (2.0, 3.0)


def run_setting(profiles, baseline_sd, effect, seed):
    rng = np.random.default_rng(seed)
    groups = {
        "null_tune": dict(split="tune"),
        "null_test": dict(split="test"),
        "emerging_tune": dict(split="tune", effect=effect, onset=ONSET, label=1),
        "emerging_test": dict(split="test", effect=effect, onset=ONSET, label=1),
        "start_test": dict(split="test", effect=effect, onset=0, label=1),
    }
    counts, truth = [], []
    for name, kwargs in groups.items():
        c, t = simulate.simulate(profiles, N_PAIRS, rng, baseline_sd=baseline_sd,
                                 prefix=f"{name}_", **kwargs)
        counts.append(c)
        truth.append(t)
    counts, truth = pd.concat(counts, ignore_index=True), pd.concat(truth)
    horizon = counts["quarter"].nunique()

    ids = counts["pair_id"].unique()
    pool = [p for p in ids if p.startswith(("null_tune", "emerging_tune"))]
    models, fit, _ = fit_hmms(counts, pool)   # Baum-Welch sees unlabelled null + emerging pairs
    replay, _ = run_replay(counts, models, pool)

    rows = []
    for m in methods():
        scores = eligible_scores(replay, m.score)
        group = lambda name: scores[scores.index.str.startswith(name)]  # noqa: E731
        tau = tune_threshold(group("null_tune"), TARGET_FAR)

        emerging = group("emerging_test")
        delay, premature, missed = simulate.detection_delay(
            first_alarm(emerging.to_numpy(), tau),
            truth.loc[emerging.index, "onset"].to_numpy(), horizon)
        start = group("start_test")
        start_alarm = first_alarm(start.to_numpy(), tau)
        rows.append({
            "method": m.label,
            "false_alarm_rate": false_alarm_rate(group("null_test"), tau),
            "emerging_detected": float(np.mean(~missed & ~premature)),
            "emerging_premature": float(np.mean(premature)),
            "emerging_mean_delay": float(np.nanmean(delay)),
            "from_start_detected": float(np.mean(start_alarm >= 0)),
            "from_start_mean_delay": float(np.where(start_alarm >= 0, start_alarm, horizon).mean()),
        })
    return pd.DataFrame(rows), fit.params.as_dict()


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    history = load_contingency(ROOT / "data" / "processed" / "contingency.csv")
    counts, _ = select_pairs(history, FIRST, LAST, 100)
    profiles = simulate.exposure_profiles(counts[counts["group"] == "omop"])

    tables, learned = [], []
    for b, (baseline, sd) in enumerate(BASELINES.items()):
        for e, effect in enumerate(EFFECTS):
            table, params = run_setting(profiles, sd, effect, seed=1000 + 10 * b + e)
            table.insert(0, "effect", effect)
            table.insert(0, "baseline", baseline)
            tables.append(table)
            learned.append({"baseline": baseline, "effect": effect, **params})
            print(f"baseline={baseline} effect={effect} done", flush=True)

    out = ROOT / "results"
    result = pd.concat(tables, ignore_index=True)
    result.round(4).to_csv(out / "tables" / "simulation.csv", index=False)
    pd.DataFrame(learned).round(4).to_csv(out / "tables" / "simulation_baum_welch.csv", index=False)
    plots.setup()
    plots.simulation(result, methods(), out / "figures" / "simulation.png", TARGET_FAR)
    pd.set_option("display.width", 200)
    print(result.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

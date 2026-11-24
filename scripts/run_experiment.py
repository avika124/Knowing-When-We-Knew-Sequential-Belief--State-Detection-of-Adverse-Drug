"""Run the retrospective replay and write every table and figure in results/.

    python scripts/run_experiment.py

Steps
  1. Load the quarterly counts and keep the replay window.
  2. Learn HMM parameters with Baum-Welch on the tuning pairs (unlabelled).
  3. Replay history: score every pair in every quarter with every method.
  4. Tune alarm thresholds on the tuning negative controls.
  5. Measure alarm timing on the historical cases and false alarms on the
     held-out negative controls.
  6. Failure-mode analyses: masking and stimulated reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwwk import hmm, plots  # noqa: E402
from kwwk.alarm import MIN_CASES, methods  # noqa: E402
from kwwk.contingency import load_contingency, quarter_end  # noqa: E402
from kwwk.evaluate import (  # noqa: E402
    case_table, discrimination, operating_points, tradeoff_curve,
)
from kwwk.reference import load_cases, load_events, pair_id  # noqa: E402
from kwwk.replay import run_replay, sequences  # noqa: E402

MASKING_ID = "rosiglitazone__acute_mi__no_rofecoxib"


def masking_counts(path, first, last):
    df = pd.read_csv(path)
    df = df[(df["quarter"] >= first) & (df["quarter"] <= last)].copy()
    df["pair_id"] = MASKING_ID
    df["drug"] = "rosiglitazone (rofecoxib reports removed)"
    df["event"], df["group"], df["label"], df["split"] = "acute_mi", "sensitivity", 1, "eval"
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--first", default="2004Q1", help="first quarter of the replay")
    parser.add_argument("--last", default="2010Q4", help="last quarter of the replay")
    parser.add_argument("--target-far", type=float, default=0.10,
                        help="false-alarm rate the matched thresholds are tuned to")
    parser.add_argument("--min-drug-reports", type=int, default=100,
                        help="control pairs need this many drug reports in the window")
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plots.setup()

    # ------------------------------------------------------------------ data
    history = load_contingency(data / "contingency.csv")
    counts = history[(history["quarter"] >= args.first) & (history["quarter"] <= args.last)]
    exposure = counts.groupby("pair_id")["n_drug"].sum()
    enough = counts["pair_id"].map(exposure >= args.min_drug_reports)
    n_dropped = counts.loc[~enough & (counts["group"] == "omop"), "pair_id"].nunique()
    counts = counts[(counts["group"] == "case") | enough]
    counts = pd.concat(
        [counts, masking_counts(data / "rosiglitazone_mi_excluding_rofecoxib.csv",
                                args.first, args.last)],
        ignore_index=True,
    )

    meta = counts.drop_duplicates("pair_id").set_index("pair_id")
    omop = meta[meta["group"] == "omop"]
    ids = {
        f"{split}_{'pos' if label else 'neg'}":
            list(omop.index[(omop["split"] == split) & (omop["label"] == label)])
        for split in ("tune", "test") for label in (0, 1)
    }
    tune_pool = ids["tune_neg"] + ids["tune_pos"]

    # ---------------------------------------------------------- Baum-Welch
    seqs = sequences(counts)
    prior = hmm.default_params()
    fit = hmm.baum_welch([(seqs[p][1], seqs[p][2]) for p in tune_pool], prior)
    models = {"hmm_prior": prior, "hmm_bw": fit.params}
    (tables / "hmm_parameters.json").write_text(json.dumps({
        "prior": prior.as_dict(),
        "baum_welch": fit.params.as_dict(),
        "baum_welch_training_pairs": len(tune_pool),
        "baum_welch_iterations": len(fit.log_likelihood),
        "baum_welch_converged": fit.converged,
        "baum_welch_objective": fit.log_likelihood,
    }, indent=2) + "\n")
    plots.convergence(fit.log_likelihood, figures / "baum_welch_convergence.png")

    # ---------------------------------------------------------------- replay
    replay, _ = run_replay(counts, models, tune_pool)
    replay.round(5).to_csv(tables / "replay_scores.csv.gz", index=False,
                           compression={"method": "gzip", "mtime": 0})
    method_list = methods()

    # ----------------------------------------------------------- evaluation
    ops, tuned = operating_points(replay, method_list, ids, args.target_far)
    ops.round(4).to_csv(tables / "operating_points.csv", index=False)

    events = load_events()
    cases = []
    for case in load_cases():
        case = dict(case, pair_id=pair_id(case["drug"], case["event"]))
        # an alarm needs at least one complete quarter before the action
        case["evaluable"] = pd.Timestamp(case["action_date"]) > quarter_end(args.first)
        cases.append(case)
    rosi = next(c for c in cases if c["drug"] == "rosiglitazone")
    sensitivity = dict(rosi, pair_id=MASKING_ID, drug="rosiglitazone (rofecoxib reports removed)")
    evaluable = [c for c in cases if c["evaluable"]]

    alarms = case_table(replay, method_list, cases + [sensitivity], tuned, args.first)
    alarms["evaluable"] = alarms["drug"].map(
        {c["drug"]: c["evaluable"] for c in cases + [sensitivity]})
    alarms.to_csv(tables / "case_alarms.csv", index=False)

    curve = tradeoff_curve(replay, method_list, evaluable, ids["test_neg"])
    curve.round(4).to_csv(tables / "tradeoff_curve.csv", index=False)

    auc = discrimination(replay, method_list, ids["test_pos"], ids["test_neg"], args.last)
    auc.round(4).to_csv(tables / "discrimination_auc.csv", index=False)

    case_ids = [c["pair_id"] for c in cases] + [MASKING_ID]
    trajectories = replay[replay["pair_id"].isin(case_ids)]
    trajectories.round(5).to_csv(tables / "case_trajectories.csv", index=False)

    # ---------------------------------------------------------------- figures
    for case in cases:
        traj = replay[replay["pair_id"] == case["pair_id"]].sort_values("quarter")
        title = (f"{case['drug'].capitalize()} ({case['brand']}) and "
                 f"{events[case['event']]['label'].lower()}")
        in_window = pd.Timestamp(case["action_date"]) <= quarter_end(args.last)
        action = case["action_date"] if case["evaluable"] and in_window else None
        if action is None:
            title += f"  (action on {case['action_date']} predates the data)"
        plots.case_trajectory(traj, title, action or quarter_end(args.first),
                              "FDA action" if action else "start of data",
                              figures / f"case_{case['drug']}.png", MIN_CASES)

    shown = alarms[alarms["drug"].isin([c["drug"] for c in evaluable])]
    plots.lead_times(shown, method_list, figures / "lead_times.png", args.target_far)
    plots.tradeoff(curve, method_list, figures / "tradeoff.png", len(evaluable))

    # ------------------------------------------------- failure mode: masking
    until = "2007Q2"
    with_all = replay[(replay["pair_id"] == rosi["pair_id"]) & (replay["quarter"] <= until)]
    without = replay[(replay["pair_id"] == MASKING_ID) & (replay["quarter"] <= until)]
    rofe = counts[(counts["pair_id"] == "rofecoxib__acute_mi") & (counts["quarter"] <= until)]
    share = (rofe["a"] / rofe["n_event"]).to_numpy()
    plots.masking(with_all.sort_values("quarter"), without.sort_values("quarter"), share,
                  rosi["action_date"], figures / "masking_rosiglitazone.png")

    # --------------------------------- failure mode: stimulated reporting
    full = history[history["pair_id"] == rosi["pair_id"]].sort_values("quarter").copy()
    full["expected"] = hmm.expected_count(full["a"], full["n_drug"], full["n_event"],
                                          full["n_total"])
    markers = [("2007-05-21", "FDA alert"), ("2010-09-23", "FDA restricts access"),
               ("2013-11-25", "restrictions lifted")]
    plots.stimulated_reporting(full, markers, figures / "stimulated_reporting_rosiglitazone.png")
    periods = [("before the FDA alert", "2004Q1", "2007Q1"),
               ("alert to access restriction", "2007Q2", "2010Q3"),
               ("restricted access", "2010Q4", "2013Q4"),
               ("2014", "2014Q1", "2014Q4"),
               ("2015 onward", "2015Q1", full["quarter"].max())]
    rows = []
    for name, start, end in periods:
        block = full[(full["quarter"] >= start) & (full["quarter"] <= end)]
        rows.append({
            "period": name, "first_quarter": start, "last_quarter": end,
            "drug_reports": int(block["n_drug"].sum()),
            "drug_reports_with_mi": int(block["a"].sum()),
            "share_with_mi": round(block["a"].sum() / max(block["n_drug"].sum(), 1), 4),
            "observed_over_expected": round(block["a"].sum() / block["expected"].sum(), 2),
        })
    pd.DataFrame(rows).to_csv(tables / "stimulated_reporting_rosiglitazone.csv", index=False)

    # ---------------------------------------------------------------- summary
    summary = {
        "replay_window": [args.first, args.last],
        "target_false_alarm_rate": args.target_far,
        "min_drug_reports": args.min_drug_reports,
        "control_pairs_dropped_for_low_exposure": int(n_dropped),
        "n_pairs": {k: len(v) for k, v in ids.items()},
        "n_cases": len(cases),
        "n_evaluable_cases": len(evaluable),
        "matched_thresholds": {m.label: {"score": m.score, "threshold": tuned[m.key]}
                               for m in method_list},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print(json.dumps(summary["n_pairs"]))
    print(json.dumps(fit.params.as_dict(), indent=1))
    print(ops.round(3).to_string(index=False))
    wide = alarms.pivot_table(index="drug", columns=["operating_point", "method"],
                              values="alarm_quarter", aggfunc="first", sort=False)
    print(wide.to_string())
    print(auc.round(3).to_string(index=False))
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()

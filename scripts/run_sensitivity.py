"""Check that the conclusions survive different analysis choices.

Two choices in the main experiment are judgement calls: how many drug reports
a control pair needs to be kept, and the false-alarm rate the thresholds are
matched to.  This script reruns the replay over a grid of both and records,
for every method, the held-out error rates and how many historical cases had
an alarm before the FDA action.

    python scripts/run_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwwk.alarm import methods  # noqa: E402
from kwwk.contingency import load_contingency, quarter_end  # noqa: E402
from kwwk.evaluate import case_table, operating_points  # noqa: E402
from kwwk.experiment import fit_hmms, select_pairs, split_ids  # noqa: E402
from kwwk.reference import load_cases, pair_id  # noqa: E402
from kwwk.replay import run_replay  # noqa: E402

FIRST, LAST = "2004Q1", "2010Q4"
MIN_REPORTS = (50, 100, 200)
TARGET_FAR = (0.05, 0.10, 0.20)


def main():
    history = load_contingency(ROOT / "data" / "processed" / "contingency.csv")
    cases = [dict(c, pair_id=pair_id(c["drug"], c["event"])) for c in load_cases()]
    cases = [c for c in cases if pd.Timestamp(c["action_date"]) > quarter_end(FIRST)]
    method_list = methods()

    rows = []
    for min_reports in MIN_REPORTS:
        counts, _ = select_pairs(history, FIRST, LAST, min_reports)
        ids = split_ids(counts)
        pool = ids["tune_neg"] + ids["tune_pos"]
        models, _, _ = fit_hmms(counts, pool)
        replay, _ = run_replay(counts, models, pool)
        for target in TARGET_FAR:
            ops, tuned = operating_points(replay, method_list, ids, target)
            alarms = case_table(replay, method_list, cases, tuned, FIRST)
            for kind in ("conventional", "matched"):
                for m in method_list:
                    op = ops[(ops["method"] == m.label) & (ops["operating_point"] == kind)]
                    if op.empty:
                        continue
                    hits = alarms[(alarms["method"] == m.label)
                                  & (alarms["operating_point"] == kind)]
                    rows.append({
                        "min_drug_reports": min_reports,
                        "target_far": target,
                        "operating_point": kind,
                        "method": m.label,
                        "n_test_neg": len(ids["test_neg"]),
                        "far_test_neg": round(float(op["far_test_neg"].iloc[0]), 4),
                        "sens_test_pos": round(float(op["sens_test_pos"].iloc[0]), 4),
                        "cases_alarm_before_action": int(hits["before_action"].sum()),
                        "cases_standing_at_action": int(hits["standing_at_action"].sum()),
                        "n_cases": len(cases),
                    })
        print(f"min_drug_reports={min_reports} done", flush=True)

    out = ROOT / "results" / "tables" / "sensitivity.csv"
    table = pd.DataFrame(rows)
    # the conventional rules do not depend on the target, so keep them once
    conventional = table[table["operating_point"] == "conventional"].drop_duplicates(
        ["min_drug_reports", "method"]).assign(target_far=float("nan"))
    table = pd.concat([conventional, table[table["operating_point"] == "matched"]])
    table.to_csv(out, index=False)

    matched = table[table["operating_point"] == "matched"]
    print(matched.groupby(["min_drug_reports", "target_far"])[
        ["far_test_neg", "cases_alarm_before_action", "cases_standing_at_action"]
    ].agg(["min", "max"]).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

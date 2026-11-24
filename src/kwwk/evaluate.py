"""Evaluation: alarm timing on historical cases and false alarms on controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alarm import (
    alarm_date,
    detection_rate,
    eligible_scores,
    false_alarm_rate,
    first_alarm,
    lead_time_months,
    tune_threshold,
)


def subset(scores, ids):
    return scores.loc[scores.index.intersection(list(ids))]


def roc_auc(positive, negative):
    """Area under the ROC curve via the rank-sum statistic (ties count half)."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def case_alarm(scores, pair_id, threshold, strict, action_date, first_quarter):
    """Alarm quarter and lead time for one historical case at one threshold.

    The alarm is the first quarter the rule is met.  ``standing_at_action``
    records whether the rule was still met in the last complete quarter before
    the regulatory action, which separates a sustained signal from a
    transient crossing that later faded.
    """
    row = scores.loc[[pair_id]].to_numpy()
    idx = int(first_alarm(row, threshold, strict)[0])
    if idx < 0:
        return {"alarm_quarter": None, "alarm_date": None, "lead_months": np.nan,
                "before_action": False, "standing_at_action": False, "at_window_start": False}
    quarter = scores.columns[idx]
    lead = lead_time_months(quarter, action_date)
    known = [i for i, q in enumerate(scores.columns)
             if alarm_date(q) <= pd.Timestamp(action_date)]
    standing = False
    if lead > 0 and known:
        last = row[0, known[-1]]
        standing = bool(last > threshold if strict else last >= threshold)
    return {
        "alarm_quarter": quarter,
        "alarm_date": alarm_date(quarter).date().isoformat(),
        "lead_months": round(lead, 1),
        "before_action": bool(lead > 0),
        "standing_at_action": standing,
        "at_window_start": bool(quarter == first_quarter),
    }


def operating_points(replay, method_list, ids, target_far):
    """Conventional and FAR-matched thresholds with their error rates.

    ``ids`` maps a group name to pair ids: tune_neg, test_neg, tune_pos, test_pos.
    The matched threshold is tuned on ``tune_neg`` only; everything measured on
    the test groups is therefore out of sample.
    """
    rows, tuned = [], {}
    for m in method_list:
        score = eligible_scores(replay, m.score)
        conv = eligible_scores(replay, m.conv_score)
        tau = tune_threshold(subset(score, ids["tune_neg"]), target_far)
        tuned[m.key] = tau
        for kind, matrix, threshold, strict in (
            ("conventional", conv, m.conv_threshold, m.conv_strict),
            ("matched", score, tau, False),
        ):
            if kind == "conventional" and not m.has_conventional:
                continue
            rows.append({
                "method": m.label,
                "operating_point": kind,
                "rule": m.rule if kind == "conventional" else f"{m.score} >= {threshold:.4g}",
                "far_tune_neg": false_alarm_rate(subset(matrix, ids["tune_neg"]), threshold, strict),
                "far_test_neg": false_alarm_rate(subset(matrix, ids["test_neg"]), threshold, strict),
                "sens_test_pos": detection_rate(subset(matrix, ids["test_pos"]), threshold, strict),
            })
    return pd.DataFrame(rows), tuned


def case_table(replay, method_list, cases, tuned, first_quarter):
    """One row per (case, method, operating point) with the alarm and lead time."""
    rows = []
    for m in method_list:
        score = eligible_scores(replay, m.score)
        conv = eligible_scores(replay, m.conv_score)
        for case in cases:
            for kind, matrix, threshold, strict in (
                ("conventional", conv, m.conv_threshold, m.conv_strict),
                ("matched", score, tuned[m.key], False),
            ):
                if kind == "conventional" and not m.has_conventional:
                    continue
                result = case_alarm(matrix, case["pair_id"], threshold, strict,
                                    case["action_date"], first_quarter)
                rows.append({"drug": case["drug"], "event": case["event"],
                             "action_date": case["action_date"], "method": m.label,
                             "operating_point": kind, **result})
    return pd.DataFrame(rows)


def tradeoff_curve(replay, method_list, cases, negative_ids, max_far=0.5):
    """Lead time against false-alarm rate as the alarm threshold is swept.

    For each candidate threshold: the false-alarm rate on the negative controls
    and, on the historical cases, the mean lead time (an alarm that comes after
    the regulatory action, or never, earns zero) and the number caught early.
    """
    rows = []
    for m in method_list:
        score = eligible_scores(replay, m.score)
        neg = subset(score, negative_ids)
        peaks = np.unique(neg.to_numpy().max(axis=1))
        peaks = peaks[np.isfinite(peaks)]
        case_peaks = np.concatenate(
            [score.loc[c["pair_id"]].to_numpy() for c in cases]
        )
        candidates = np.unique(np.concatenate([peaks, case_peaks[np.isfinite(case_peaks)]]))
        for tau in candidates:
            far = false_alarm_rate(neg, tau)
            if far > max_far:
                continue
            leads = []
            for c in cases:
                idx = int(first_alarm(score.loc[[c["pair_id"]]].to_numpy(), tau)[0])
                lead = lead_time_months(score.columns[idx], c["action_date"]) if idx >= 0 else 0.0
                leads.append(max(lead, 0.0))
            rows.append({"method": m.label, "threshold": tau, "far": far,
                         "mean_lead_months": float(np.mean(leads)),
                         "n_early": int(np.sum(np.array(leads) > 0))})
    curve = pd.DataFrame(rows)
    return curve.sort_values(["method", "far", "mean_lead_months"]).reset_index(drop=True)


def discrimination(replay, method_list, positive_ids, negative_ids, quarter):
    """ROC AUC of each score at one quarter, known positives against negatives."""
    snap = replay[replay["quarter"] == quarter].set_index("pair_id")
    rows = []
    for m in method_list:
        col = snap[m.score].astype(float)
        col = col.where(snap["cum_a"] >= 1, -np.inf).fillna(-np.inf)
        pos = col.reindex(list(positive_ids)).dropna()
        neg = col.reindex(list(negative_ids)).dropna()
        rows.append({"method": m.label, "auc": roc_auc(pos, neg),
                     "n_pos": len(pos), "n_neg": len(neg)})
    return pd.DataFrame(rows)

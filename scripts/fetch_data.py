"""Download quarterly 2x2 counts for every (drug, event) pair from openFDA.

Four daily series are needed per pair: all reports, reports with the drug,
reports with the event, and reports with both.  The first is shared by every
pair and the next two are shared across pairs, so the request count is roughly
(drugs + events + pairs).  Raw responses are cached under data/raw/.

    python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwwk.contingency import quarter_range, to_quarterly  # noqa: E402
from kwwk.openfda import OpenFDAClient  # noqa: E402
from kwwk.queries import both_clause, drug_clause, event_clause  # noqa: E402
from kwwk.reference import load_cases, load_events, load_pairs  # noqa: E402


def counts_excluding(client, events, quarters, target, masker):
    """2x2 counts for ``target`` after removing every report that mentions ``masker``.

    Used for the masking analysis: a flood of reports for one drug inflates the
    background event rate that every other drug is compared against.  The
    removal is done by inclusion-exclusion on the report counts.
    """
    pts = events[target["event"]]["preferred_terms"]
    t_drug, m_drug = drug_clause(target["drug_terms"]), drug_clause(masker["drug_terms"])
    event = event_clause(pts)

    def series(search):
        return to_quarterly(client.count_by_receivedate(search), quarters)

    total, n_target, n_masker = series(None), series(t_drug), series(m_drug)
    n_event, a = series(event), series(f"{t_drug} AND {event}")
    masker_event = series(f"{m_drug} AND {event}")
    both_drugs = series(f"{t_drug} AND {m_drug}")
    both_drugs_event = series(f"{t_drug} AND {m_drug} AND {event}")
    return pd.DataFrame({
        "quarter": quarters,
        "a": [x - y for x, y in zip(a, both_drugs_event)],
        "n_drug": [x - y for x, y in zip(n_target, both_drugs)],
        "n_event": [x - y for x, y in zip(n_event, masker_event)],
        "n_total": [x - y for x, y in zip(total, n_masker)],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", default="2004Q1", help="first quarter to keep")
    parser.add_argument("--last", default="2026Q3", help="last complete quarter to keep")
    parser.add_argument("--out", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()

    client = OpenFDAClient(args.cache)
    events = load_events()
    pairs = load_pairs()
    quarters = quarter_range(args.first, args.last)

    total = to_quarterly(client.count_by_receivedate(None), quarters)

    drug_series, event_series = {}, {}
    rows = []
    for i, pair in enumerate(pairs, 1):
        pts = events[pair.event]["preferred_terms"]
        drug_key = (pair.drug, pair.drug_terms)
        if drug_key not in drug_series:
            drug_series[drug_key] = to_quarterly(
                client.count_by_receivedate(drug_clause(pair.drug_terms)), quarters
            )
        if pair.event not in event_series:
            event_series[pair.event] = to_quarterly(
                client.count_by_receivedate(event_clause(pts)), quarters
            )
        both = to_quarterly(
            client.count_by_receivedate(both_clause(pair.drug_terms, pts)), quarters
        )
        for q, a, n_drug, n_event, n_total in zip(
            quarters, both, drug_series[drug_key], event_series[pair.event], total
        ):
            rows.append((pair.pair_id, pair.drug, pair.event, pair.group, pair.label,
                         pair.split, q, a, n_drug, n_event, n_total))
        if i % 25 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)} pairs  ({client.n_requests} API requests)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["pair_id", "drug", "event", "group", "label", "split", "quarter",
            "a", "n_drug", "n_event", "n_total"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "contingency.csv", index=False)

    # Masking analysis: rosiglitazone -> MI with rofecoxib (Vioxx) reports removed.
    cases = {c["drug"]: c for c in load_cases()}
    counts_excluding(client, events, quarters, cases["rosiglitazone"], cases["rofecoxib"]).to_csv(
        out / "rosiglitazone_mi_excluding_rofecoxib.csv", index=False
    )

    (out / "snapshot.json").write_text(json.dumps({
        "source": "openFDA drug adverse event API (FAERS)",
        "openfda_last_updated": client.last_updated,
        "fetched_on": date.today().isoformat(),
        "first_quarter": args.first,
        "last_quarter": args.last,
        "n_pairs": len(pairs),
    }, indent=2) + "\n")
    print(f"wrote {out / 'contingency.csv'}")


if __name__ == "__main__":
    main()

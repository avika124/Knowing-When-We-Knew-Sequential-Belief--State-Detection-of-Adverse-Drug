"""Load the curated reference data: event definitions, historical cases, OMOP controls."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


@dataclass(frozen=True)
class Pair:
    pair_id: str
    drug: str
    drug_terms: tuple
    event: str
    group: str  # "case" (historical positive) or "omop" (reference set)
    label: int  # 1 = known association, 0 = negative control
    split: str  # "eval" for cases; "tune" or "test" for OMOP pairs


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def pair_id(drug, event):
    return f"{slug(drug)}__{event}"


def omop_split(drug):
    """Deterministic drug-level split so one drug never sits on both sides."""
    digest = hashlib.md5(slug(drug).encode()).hexdigest()
    return "tune" if int(digest, 16) % 2 == 0 else "test"


def load_events(reference_dir=REFERENCE_DIR):
    return json.loads((Path(reference_dir) / "events.json").read_text())


def load_cases(reference_dir=REFERENCE_DIR):
    return json.loads((Path(reference_dir) / "cases.json").read_text())


def load_pairs(reference_dir=REFERENCE_DIR):
    """All (drug, event) pairs: historical cases first, then the OMOP reference set."""
    pairs = []
    case_ids = set()
    for case in load_cases(reference_dir):
        pid = pair_id(case["drug"], case["event"])
        case_ids.add(pid)
        pairs.append(
            Pair(pid, case["drug"], tuple(case["drug_terms"]), case["event"], "case", 1, "eval")
        )
    case_drugs = {p.drug for p in pairs}
    omop = pd.read_csv(Path(reference_dir) / "omop_reference_set.csv")
    for row in omop.itertuples():
        pid = pair_id(row.drug, row.event)
        if pid in case_ids:
            continue
        # A drug that is one of the evaluation cases is kept out of tuning.
        split = "test" if row.drug in case_drugs else omop_split(row.drug)
        pairs.append(
            Pair(pid, row.drug, tuple(row.drug_terms.split("|")), row.event, "omop",
                 int(row.ground_truth), split)
        )
    return pairs

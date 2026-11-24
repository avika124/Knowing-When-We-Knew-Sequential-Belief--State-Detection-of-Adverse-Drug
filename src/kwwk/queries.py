"""Build openFDA search expressions for drugs and events."""

from __future__ import annotations

DRUG_FIELDS = (
    "patient.drug.medicinalproduct",
    "patient.drug.openfda.generic_name",
    "patient.drug.openfda.brand_name",
    "patient.drug.openfda.substance_name",
)
EVENT_FIELD = "patient.reaction.reactionmeddrapt.exact"


def _quote(term):
    return '"' + term.replace('"', "") + '"'


def drug_clause(terms):
    """Reports that mention any of the given drug names (any role)."""
    parts = [f"{field}:{_quote(t)}" for t in terms for field in DRUG_FIELDS]
    return "(" + " OR ".join(parts) + ")"


def event_clause(preferred_terms):
    """Reports coded with any of the given MedDRA preferred terms."""
    parts = [f"{EVENT_FIELD}:{_quote(pt.upper())}" for pt in preferred_terms]
    return "(" + " OR ".join(parts) + ")"


def both_clause(terms, preferred_terms):
    return drug_clause(terms) + " AND " + event_clause(preferred_terms)

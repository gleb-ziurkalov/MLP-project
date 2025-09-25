"""Regulation keyword based labeler."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from . import BaseLabeler


class RegulationLabeler(BaseLabeler):
    """Label records that reference regulatory terms."""

    DEFAULT_PATTERNS: Sequence[str] = (
        "TSCA",
        "REACH",
        "SVHC",
        "EPA",
        "OSHA",
        "RoHS",
        "Toxic Substances Control Act",
        "Section 6(h)",
        "PBT substances",
        "Prop65",
        "Prop 65",
        "Proposition 65",
        "Conflict minerals",
        "WEEE",
        "PFAS",
        "ECHA",
        "Toxic Substance Control Act",
    )

    def __init__(self, patterns: Iterable[str] | None = None):
        terms = list(patterns or self.DEFAULT_PATTERNS)
        self._pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
            flags=re.IGNORECASE,
        )

    def label(self, record):  # type: ignore[override]
        labeled_record = record
        text = str(labeled_record.get("text", ""))
        if self._pattern.search(text):
            labeled_record["regulation"] = 1
            labeled_record["compliance_data"] = 1
        else:
            labeled_record["regulation"] = 0
        return labeled_record


__all__ = ["RegulationLabeler"]

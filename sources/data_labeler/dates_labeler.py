"""Date extraction labeler."""

from __future__ import annotations

import datetime
import re
from typing import Iterable, Sequence

from . import BaseLabeler


class DateLabeler(BaseLabeler):
    """Label sentences that contain valid date expressions."""

    DEFAULT_PATTERNS: Sequence[str] = (
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{2}-\d{2}-\d{2}",
        r"\d{4}\.\d{2}\.\d{2}",
        r"\d{2}\.\d{2}\.\d{4}",
        r"\d{2}\.\d{2}\.\d{2}",
        r"\d{4}/\d{2}/\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}/\d{2}/\d{2}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}",
    )
    DEFAULT_FORMATS: Sequence[str] = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%Y.%m.%d",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%b %d, %Y",
        "%B %d, %Y",
    )

    def __init__(
        self,
        patterns: Iterable[str] | None = None,
        date_formats: Sequence[str] | None = None,
    ):
        pattern_list = list(patterns or self.DEFAULT_PATTERNS)
        self._pattern = re.compile("|".join(pattern_list))
        self._date_formats = list(date_formats or self.DEFAULT_FORMATS)

    def label(self, record):  # type: ignore[override]
        labeled_record = record
        text = str(labeled_record.get("text", ""))
        matches = self._pattern.findall(text)

        labeled_record["date"] = 0
        if matches:
            for match in matches:
                if self._is_valid_date(match):
                    labeled_record["date"] = 1
                    labeled_record["compliance_data"] = 1
                    break
        return labeled_record

    def _is_valid_date(self, date_str: str) -> bool:
        for fmt in self._date_formats:
            try:
                datetime.datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False


__all__ = ["DateLabeler"]

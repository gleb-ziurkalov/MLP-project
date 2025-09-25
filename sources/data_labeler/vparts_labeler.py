"""Placeholder labeler for vehicle parts."""

from __future__ import annotations

from . import BaseLabeler


class PartLabeler(BaseLabeler):
    """Labeler stub for vehicle parts extraction."""

    def label(self, record):  # type: ignore[override]
        return record


__all__ = ["PartLabeler"]

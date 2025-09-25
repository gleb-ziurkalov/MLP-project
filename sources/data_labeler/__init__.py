"""Labeling utilities and pipeline abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, MutableMapping, Sequence

from .dates_labeler import DateLabeler
from .entities_labeler import EntityLabeler
from .regulation_labeler import RegulationLabeler
from .vparts_labeler import PartLabeler


class BaseLabeler(ABC):
    """Abstract base class for compliance data labelers."""

    @abstractmethod
    def label(self, record: MutableMapping[str, object]) -> MutableMapping[str, object]:
        """Return a labeled mapping for ``record``."""


class LabelingPipeline:
    """Execute a sequence of labelers over dataset records."""

    def __init__(self, labelers: Sequence[BaseLabeler]):
        self._labelers: List[BaseLabeler] = list(labelers)

    def label_data(self, dataset: Iterable[MutableMapping[str, object]]) -> List[Dict[str, object]]:
        """Return a newly labeled copy of ``dataset``."""

        labeled_dataset: List[Dict[str, object]] = []
        for record in dataset:
            labeled_record: MutableMapping[str, object] = dict(record)
            for labeler in self._labelers:
                labeled_record = labeler.label(labeled_record)
            if labeled_record.get("compliance_data") is None:
                labeled_record["compliance_data"] = 0
            labeled_dataset.append(dict(labeled_record))
        return labeled_dataset


__all__ = [
    "BaseLabeler",
    "LabelingPipeline",
    "DateLabeler",
    "EntityLabeler",
    "RegulationLabeler",
    "PartLabeler",
]

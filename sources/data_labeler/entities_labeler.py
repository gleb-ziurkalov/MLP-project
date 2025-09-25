"""Named entity recognition based labeler."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from . import BaseLabeler


class EntityLabeler(BaseLabeler):
    """Label business entities using a Hugging Face NER model."""

    DEFAULT_TERMS: Sequence[str] = (
        "Inc",
        "LLC",
        "Corp",
        "Ltd",
        "Corporation",
        "GmbH",
        "Group",
        "Co.",
        "KG",
    )

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        confidence_threshold: float = 0.85,
        business_terms: Iterable[str] | None = None,
        ner_pipeline=None,
    ):
        self.confidence_threshold = confidence_threshold
        terms = list(business_terms or self.DEFAULT_TERMS)
        self._business_term_pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE
        )

        if ner_pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        self._ner_pipeline = ner_pipeline

    def label(self, record):  # type: ignore[override]
        labeled_record = record
        text = str(labeled_record.get("text", ""))

        labeled_record.setdefault("business_entity", 0)
        labeled_record.setdefault("business_entity_confidence", None)

        if not text.strip():
            return labeled_record

        entities = self._ner_pipeline(text)
        has_business_term = bool(self._business_term_pattern.search(text))

        for entity in entities:
            if entity.get("entity") == "B-ORG" and entity.get("score", 0) >= self.confidence_threshold:
                labeled_record["business_entity_confidence"] = entity["score"]
                labeled_record["compliance_data"] = 1
                if has_business_term:
                    labeled_record["business_entity"] = 1
                return labeled_record

        labeled_record["business_entity"] = 0
        return labeled_record


__all__ = ["EntityLabeler"]

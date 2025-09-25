"""Utilities for extracting and classifying sentences from PDF pages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Iterable, List

import numpy as np

from .sentence_processor import extract_bbox, segment_sentences

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .services import ClassifierBundle, ClassifierService, OcrService, SentenceSegmenter


def image_to_text(
    pages: Iterable,
    classifier_source: "ClassifierBundle"
    | Callable[[], "ClassifierBundle"]
    | "ClassifierService",
    *,
    ocr_service: "OcrService",
    sentence_segmenter: "SentenceSegmenter",
    batch_size: int = 32,
) -> List[Dict[str, int]]:
    """Extract and classify sentences from rendered PDF pages."""

    if classifier_source is None:
        raise ValueError("classifier_source must be provided")
    if ocr_service is None:
        raise ValueError("ocr_service must be provided")
    if sentence_segmenter is None:
        raise ValueError("sentence_segmenter must be provided")

    if hasattr(classifier_source, "bundle"):
        classifier_bundle = classifier_source.bundle  # type: ignore[attr-defined]
    elif callable(classifier_source):
        classifier_bundle = classifier_source()
    else:
        classifier_bundle = classifier_source

    if classifier_bundle is None:
        raise ValueError("classifier_bundle could not be resolved")

    sentences: List[str] = []

    for page in pages:
        image = np.array(page)
        text_boxes = extract_bbox(image, ocr_service)
        if not text_boxes:
            continue

        segmented_sentences = segment_sentences(text_boxes, sentence_segmenter)
        for sentence in segmented_sentences:
            text = sentence.get("text") if isinstance(sentence, dict) else None
            if isinstance(text, str) and text.strip():
                sentences.append(text)

    if not sentences:
        return []

    classified = classifier_bundle.classify(sentences, batch_size=batch_size)
    return [{"text": text, "label": int(label)} for text, label in classified]


__all__ = ["image_to_text"]

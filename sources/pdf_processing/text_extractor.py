from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .sentence_processor import extract_bbox, segment_sentences
from .services import ClassifierService, OcrService, SentenceSegmenter


def image_to_text(
    pages,
    model_path: Optional[str] = None,
    *,
    ocr_service: Optional[OcrService] = None,
    sentence_segmenter: Optional[SentenceSegmenter] = None,
    classifier_service: Optional[ClassifierService] = None,
    batch_size: int = 32,
) -> List[Tuple[str, int]]:
    if classifier_service is None:
        if not model_path:
            raise ValueError("model_path is required when classifier_service is not provided")
        classifier_service = ClassifierService(model_name=model_path)

    ocr_service = ocr_service or OcrService()
    sentence_segmenter = sentence_segmenter or SentenceSegmenter()

    sentences: List[str] = []

    for page in pages:
        image = np.array(page)
        text_boxes = extract_bbox(image, ocr_service)
        segmented_sentences = segment_sentences(text_boxes, sentence_segmenter)
        sentences.extend([record["text"] for record in segmented_sentences])

    predictions = classifier_service.classify(sentences, batch_size=batch_size)
    return list(zip(sentences, predictions))

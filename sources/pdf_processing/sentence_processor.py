from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .services import OcrService, SentenceSegmenter

TextBox = Tuple[str, Sequence[Sequence[float]]]
SentenceRecord = dict


def extract_bbox(image: np.ndarray, ocr_service: OcrService) -> Optional[List[TextBox]]:
    """Extract bounding boxes and text using OCR."""
    ocr_results = ocr_service.ocr(image, cls=False)

    extracted_data: List[TextBox] = []
    if not ocr_results:
        return None

    for line in ocr_results[0]:
        if len(line) < 2:
            continue

        bbox, (text, conf) = line
        if text and text.strip():
            extracted_data.append((text, bbox))

    return extracted_data if extracted_data else None


def convert_bbox(paddle_bbox: Sequence[Sequence[float]]) -> List[float]:
    """Convert PaddleOCR bounding box format (4 points) to (x_min, y_min, x_max, y_max)."""
    x_min = min(point[0] for point in paddle_bbox)
    y_min = min(point[1] for point in paddle_bbox)
    x_max = max(point[0] for point in paddle_bbox)
    y_max = max(point[1] for point in paddle_bbox)
    return [x_min, y_min, x_max, y_max]


def segment_sentences(
    text_boxes: Optional[Iterable[TextBox]],
    sentence_segmenter: SentenceSegmenter,
) -> List[SentenceRecord]:
    sentences: List[SentenceRecord] = []

    if not text_boxes:
        return sentences

    words = [item[0] for item in text_boxes]
    bounding_boxes = [item[1] for item in text_boxes]

    full_text = " ".join(words)
    doc = sentence_segmenter.segment(full_text)

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        matched_boxes = [bbox for word, bbox in zip(words, bounding_boxes) if word in sentence_text]

        if matched_boxes:
            x_min = min(box[0][0] for box in matched_boxes)
            y_min = min(box[0][1] for box in matched_boxes)
            x_max = max(box[2][0] for box in matched_boxes)
            y_max = max(box[2][1] for box in matched_boxes)
            sentence_bbox = (x_min, y_min, x_max, y_max)
        else:
            sentence_bbox = None

        sentences.append({"text": sentence_text, "bounding_box": sentence_bbox})

    return sentences


def find_sentence_box(sentence: str, text_boxes: Iterable[SentenceRecord]):
    """Find a bounding box that covers a given sentence based on text box locations."""
    matching_boxes = [box["bounding_box"] for box in text_boxes if sentence in box["text"]]
    if not matching_boxes:
        return None
    x_min = min(box[0] for box in matching_boxes)
    y_min = min(box[1] for box in matching_boxes)
    x_max = max(box[2] for box in matching_boxes)
    y_max = max(box[3] for box in matching_boxes)
    return [x_min, y_min, x_max, y_max]

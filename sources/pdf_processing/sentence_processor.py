"""Utilities for working with OCR output and sentences."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .services import OcrService, SentenceSegmenter

TextBox = Tuple[str, Sequence[Sequence[float]]]


def extract_bbox(image, ocr_service: "OcrService") -> List[TextBox]:  # type: ignore[no-untyped-def]
    """Extract bounding boxes and text using the provided OCR service."""

    ocr_results = ocr_service.extract(image, cls=False)
    if not ocr_results:
        return []

    lines = ocr_results[0] if isinstance(ocr_results, list) else ocr_results
    extracted_data: List[TextBox] = []
    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        bbox, info = line[0], line[1]
        text = info[0] if isinstance(info, (list, tuple)) else ""
        if isinstance(text, str) and text.strip():
            extracted_data.append((text, bbox))

    return extracted_data


def convert_bbox(paddle_bbox: Sequence[Sequence[float]]):
    """Convert PaddleOCR bounding box format to ``(x_min, y_min, x_max, y_max)``."""

    x_min = min(point[0] for point in paddle_bbox)
    y_min = min(point[1] for point in paddle_bbox)
    x_max = max(point[0] for point in paddle_bbox)
    y_max = max(point[1] for point in paddle_bbox)
    return [x_min, y_min, x_max, y_max]


def segment_sentences(text_boxes: Sequence[TextBox], segmenter: "SentenceSegmenter"):
    """Segment OCR text boxes into sentences using the provided segmenter."""

    return segmenter.segment(text_boxes)


def find_sentence_box(sentence: str, text_boxes: Sequence[dict]):
    """Find a bounding box that covers ``sentence`` based on text box locations."""

    matching_boxes = [box["bounding_box"] for box in text_boxes if sentence in box.get("text", "")]
    if not matching_boxes:
        return None
    x_min = min(box[0] for box in matching_boxes)
    y_min = min(box[1] for box in matching_boxes)
    x_max = max(box[2] for box in matching_boxes)
    y_max = max(box[3] for box in matching_boxes)
    return [x_min, y_min, x_max, y_max]


__all__ = ["TextBox", "convert_bbox", "extract_bbox", "find_sentence_box", "segment_sentences"]

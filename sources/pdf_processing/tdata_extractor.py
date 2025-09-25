"""Utilities for turning PDF pages into labeled training data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence

import cv2
import numpy as np

from .sentence_processor import extract_bbox, segment_sentences

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .services import OcrService, SentenceSegmenter


def detect_highlighted_regions(image: np.ndarray) -> List[List[int]]:
    """Detect highlighted regions in the image using color detection."""

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_highlight = np.array([90, 50, 200])
    upper_highlight = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        [x, y, x + w, y + h]
        for x, y, w, h in (cv2.boundingRect(contour) for contour in contours)
    ]


def label_compliance_sentences(
    sentences: List[dict],
    highlighted_boxes: Sequence[Sequence[int]],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """Label sentences based on overlap with highlighted regions."""

    for sentence in sentences:
        bbox = sentence.get("bounding_box")
        if bbox and len(bbox) == 4:
            overlaps = [
                calculate_iou(bbox, highlight_box) > iou_threshold
                for highlight_box in highlighted_boxes
            ]
            sentence["compliance_statement"] = int(any(overlaps))
        else:
            sentence["compliance_statement"] = 0
    return sentences


def calculate_iou(box1: Sequence[int], box2: Sequence[int]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes."""

    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def image_to_tdata(
    pages: Iterable,
    ocr_service: "OcrService",
    sentence_segmenter: "SentenceSegmenter",
) -> List[dict]:
    """Convert PDF pages into labeled training data records."""

    consolidated_results: List[dict] = []

    for page in pages:
        image = np.array(page)
        text_boxes = extract_bbox(image, ocr_service)
        if not text_boxes:
            continue

        sentences = segment_sentences(text_boxes, sentence_segmenter)
        if not sentences:
            continue

        highlighted_boxes = detect_highlighted_regions(image)
        labeled_text = label_compliance_sentences(sentences, highlighted_boxes)
        consolidated_results.extend(labeled_text)

    return consolidated_results


__all__ = [
    "calculate_iou",
    "detect_highlighted_regions",
    "image_to_tdata",
    "label_compliance_sentences",
]

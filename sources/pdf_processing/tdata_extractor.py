from __future__ import annotations

from typing import Iterable, List, Optional

import cv2
import numpy as np

from .sentence_processor import extract_bbox, segment_sentences
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
    sentences: Iterable[dict],
    highlighted_boxes: Iterable[Iterable[int]],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """Label sentences as compliance_statement=1 or compliance_statement=0 based on IoU."""
    highlighted_boxes = list(highlighted_boxes)

    for sentence in sentences:
        bbox = sentence.get("bounding_box")
        if bbox:
            x, y, x2, y2 = bbox
            sentence["compliance_statement"] = int(
                any(
                    calculate_iou([x, y, x2, y2], [hx, hy, hx + hw, hy + hh]) > iou_threshold
                    for hx, hy, hw, hh in highlighted_boxes
                )
            )
        else:
            sentence["compliance_statement"] = 0
    return list(sentences)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def image_to_tdata(
    pages,
    *,
    ocr_service: Optional[OcrService] = None,
    sentence_segmenter: Optional[SentenceSegmenter] = None,
) -> List[dict]:
    ocr_service = ocr_service or OcrService()
    sentence_segmenter = sentence_segmenter or SentenceSegmenter()

    consolidated_results: List[dict] = []

    for page in pages:
        image = np.array(page)

        text_boxes = extract_bbox(image, ocr_service)
        sentences = segment_sentences(text_boxes, sentence_segmenter)

        highlighted_boxes = detect_highlighted_regions(image)
        labeled_text = label_compliance_sentences(sentences, highlighted_boxes)

        consolidated_results.extend(labeled_text)

    return consolidated_results

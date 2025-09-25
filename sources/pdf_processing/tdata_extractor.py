"""Training data extraction utilities."""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

from .image_processor import ImageProcessor
from .sentence_processor import SentenceSegmenter


class TrainingDataExtractor:
    """Generate labeled training examples from PDF pages."""

    def __init__(self, image_processor: ImageProcessor, sentence_segmenter: SentenceSegmenter):
        self.image_processor = image_processor
        self.sentence_segmenter = sentence_segmenter

    def extract_pages(self, pdf_path: str) -> List[dict]:
        """Extract and label sentences from a PDF file."""
        pages = self.image_processor.to_images(pdf_path)
        consolidated_results: List[dict] = []

        for page in pages:
            image = np.array(page)
            ocr_results = self.image_processor.ocr(image, cls=False)
            text_boxes = self.sentence_segmenter.extract_text_boxes(ocr_results)
            sentences = self.sentence_segmenter.segment_sentences(text_boxes)

            highlighted_boxes = self._detect_highlighted_regions(image)
            labeled_text = self.label_sentences(sentences, highlighted_boxes)
            consolidated_results.extend(labeled_text)

        return consolidated_results

    def label_sentences(
        self,
        sentences: Sequence[dict],
        highlighted_boxes: Sequence[Sequence[float]],
        iou_threshold: float = 0.5,
    ) -> List[dict]:
        """Apply compliance labels to sentences based on highlight overlap."""
        labeled_sentences: List[dict] = []
        for sentence in sentences:
            bbox = sentence.get("bounding_box")
            if bbox:
                sentence_box = [bbox[0], bbox[1], bbox[2], bbox[3]]
                compliance = int(
                    any(
                        self._calculate_iou(sentence_box, highlight_box) > iou_threshold
                        for highlight_box in highlighted_boxes
                    )
                )
            else:
                compliance = 0

            labeled_sentence = dict(sentence)
            labeled_sentence["compliance_statement"] = compliance
            labeled_sentences.append(labeled_sentence)

        return labeled_sentences

    @staticmethod
    def _detect_highlighted_regions(image: np.ndarray) -> List[List[int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_highlight = np.array([90, 50, 200])
        upper_highlight = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        return boxes

    @staticmethod
    def _calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
        if not box1 or not box2:
            return 0.0

        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union


__all__ = ["TrainingDataExtractor"]

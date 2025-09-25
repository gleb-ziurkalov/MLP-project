"""Sentence segmentation utilities for OCR output."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import spacy


class SentenceSegmenter:
    """Convert OCR output into sentence-level annotations."""

    def __init__(self, model: str = "en_core_web_sm"):
        self._nlp = spacy.load(model)

    @staticmethod
    def _convert_bbox(paddle_bbox: Sequence[Sequence[float]]) -> List[float]:
        x_min = min(point[0] for point in paddle_bbox)
        y_min = min(point[1] for point in paddle_bbox)
        x_max = max(point[0] for point in paddle_bbox)
        y_max = max(point[1] for point in paddle_bbox)
        return [x_min, y_min, x_max, y_max]

    def extract_text_boxes(self, ocr_results: Sequence) -> List[Tuple[str, List[float]]]:
        """Return text and bounding boxes from PaddleOCR results."""
        if not ocr_results:
            return []

        extracted_data: List[Tuple[str, List[float]]] = []
        for line in ocr_results[0]:
            if len(line) < 2:
                continue
            bbox, (text, _conf) = line
            if text.strip():
                extracted_data.append((text, self._convert_bbox(bbox)))
        return extracted_data

    def segment_sentences(self, text_boxes: Iterable[Tuple[str, List[float]]]) -> List[dict]:
        """Segment OCR words into sentences with merged bounding boxes."""
        words: List[str] = []
        bounding_boxes: List[List[float]] = []
        for text, bbox in text_boxes:
            words.append(text)
            bounding_boxes.append(bbox)

        if not words:
            return []

        full_text = " ".join(words)
        doc = self._nlp(full_text)

        sentences: List[dict] = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            matched_boxes: List[List[float]] = []
            for word, bbox in zip(words, bounding_boxes):
                if word in sentence_text:
                    matched_boxes.append(bbox)

            if matched_boxes:
                x_min = min(box[0] for box in matched_boxes)
                y_min = min(box[1] for box in matched_boxes)
                x_max = max(box[2] for box in matched_boxes)
                y_max = max(box[3] for box in matched_boxes)
                sentence_bbox: Optional[List[float]] = [x_min, y_min, x_max, y_max]
            else:
                sentence_bbox = None

            sentences.append({"text": sentence_text, "bounding_box": sentence_bbox})

        return sentences


__all__ = ["SentenceSegmenter"]

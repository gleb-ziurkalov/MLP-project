"""PDF processing facade coordinating extraction and classification."""

from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
import torch

from .image_processor import ImageProcessor
from .sentence_processor import SentenceSegmenter
from .tdata_extractor import TrainingDataExtractor
from .text_extractor import SentenceClassifier


class PDFProcessor:
    """Coordinate PDF image conversion, OCR, and sentence processing."""

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        sentence_segmenter: Optional[SentenceSegmenter] = None,
        training_extractor: Optional[TrainingDataExtractor] = None,
        sentence_classifier: Optional[SentenceClassifier] = None,
    ):
        self.image_processor = image_processor or ImageProcessor()
        self.sentence_segmenter = sentence_segmenter or SentenceSegmenter()
        self.training_extractor = training_extractor or TrainingDataExtractor(
            self.image_processor, self.sentence_segmenter
        )
        self.sentence_classifier = sentence_classifier or SentenceClassifier()

    def to_training_data(self, pdf_path: str) -> List[dict]:
        """Process a PDF into labeled training data."""
        results = self.training_extractor.extract_pages(pdf_path)
        self._clear_memory()
        return results

    def to_compliant_sentences(self, pdf_path: str, model_dir: Optional[str] = None) -> List[str]:
        """Extract sentences classified as compliant from a PDF."""
        classifier = self.sentence_classifier
        if model_dir and model_dir != classifier.model_dir:
            classifier = SentenceClassifier(model_dir=model_dir)
            self.sentence_classifier = classifier

        sentences = self._extract_sentence_text(pdf_path)
        classified = classifier.classify(sentences)
        self._clear_memory()
        return [text for text, label in classified if label == 1]

    def _extract_sentence_text(self, pdf_path: str) -> List[str]:
        sentences: List[str] = []
        pages = self.image_processor.to_images(pdf_path)
        for page in pages:
            image = np.array(page)
            ocr_results = self.image_processor.ocr(image, cls=False)
            text_boxes = self.sentence_segmenter.extract_text_boxes(ocr_results)
            segmented = self.sentence_segmenter.segment_sentences(text_boxes)
            sentences.extend(
                sentence["text"]
                for sentence in segmented
                if sentence.get("text")
            )
        return sentences

    @staticmethod
    def _clear_memory() -> None:
        torch.cuda.empty_cache()
        gc.collect()


__all__ = ["PDFProcessor"]

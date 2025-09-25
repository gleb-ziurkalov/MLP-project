"""Service objects for PDF processing dependencies.

This module exposes light-weight wrappers around heavyweight third-party
libraries used throughout the PDF processing pipeline. Creating explicit
services keeps resource-heavy objects (PaddleOCR, spaCy models, transformer
models) out of module scope so they can be constructed lazily and injected
where needed.  The :class:`PdfProcessingFactory` helper offers a convenient
way to configure and cache shared service instances for orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ClassifierBundle:
    """Container for a tokenizer/model pair used during inference."""

    tokenizer: Any
    model: Any
    device: str

    def classify(self, sentences: Sequence[str], batch_size: int = 32) -> List[Tuple[str, int]]:
        """Predict labels for ``sentences`` using the stored transformer bundle."""

        import torch

        valid_sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
        if not valid_sentences:
            return []

        predictions: List[int] = []
        for start in range(0, len(valid_sentences), batch_size):
            batch = valid_sentences[start : start + batch_size]
            tokenized = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized)
                logits = outputs.logits
                predictions.extend(logits.argmax(dim=-1).tolist())

        return list(zip(valid_sentences, predictions))


def _freeze_config(config: Any) -> Any:
    """Recursively convert configuration objects into hashable structures."""

    if isinstance(config, dict):
        return tuple(sorted((key, _freeze_config(value)) for key, value in config.items()))
    if isinstance(config, (list, tuple, set)):
        return tuple(_freeze_config(value) for value in config)
    return config


class OcrService:
    """Wrapper around :class:`paddleocr.PaddleOCR` for dependency injection."""

    def __init__(self, *, lang: str = "en", **ocr_kwargs: Any) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(lang=lang, **ocr_kwargs)

    def extract(self, image, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Run OCR on ``image`` and return PaddleOCR results."""

        return self._ocr.ocr(image, **kwargs)


class SentenceSegmenter:
    """Sentence segmentation service backed by spaCy."""

    def __init__(self, model: str = "en_core_web_sm", **load_kwargs: Any) -> None:
        import spacy

        self._nlp = spacy.load(model, **load_kwargs)

    def segment(self, text_boxes: Sequence[Tuple[str, Sequence[Sequence[float]]]]):
        """Convert OCR word boxes into sentence dictionaries with bounding boxes."""

        words = [item[0] for item in text_boxes]
        if not words:
            return []
        bounding_boxes = [item[1] for item in text_boxes]
        full_text = " ".join(words)
        doc = self._nlp(full_text)

        sentences = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue

            matched_boxes = [
                bbox
                for word, bbox in zip(words, bounding_boxes)
                if word and word in sentence_text
            ]

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


class ClassifierService:
    """Transformer-based classifier wrapper used for inference."""

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        from torch import cuda
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        tokenizer_source = tokenizer_path or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)

        if device is None:
            device = "cuda" if cuda.is_available() else "cpu"

        model = model.to(device)
        self._bundle = ClassifierBundle(tokenizer=tokenizer, model=model, device=device)

    @property
    def device(self) -> str:
        return self._bundle.device

    @property
    def bundle(self) -> ClassifierBundle:
        return self._bundle

    def classify(self, sentences: Sequence[str], batch_size: int = 32) -> List[Tuple[str, int]]:
        """Predict labels for ``sentences`` using the loaded transformer model."""

        return self._bundle.classify(sentences, batch_size=batch_size)


@dataclass
class PdfProcessingFactory:
    """Factory for creating and caching PDF processing services."""

    ocr_kwargs: Dict[str, Any] = field(default_factory=dict)
    segmenter_kwargs: Dict[str, Any] = field(default_factory=dict)
    classifier_kwargs: Dict[str, Any] = field(default_factory=dict)

    _ocr_service: Optional[OcrService] = field(init=False, default=None)
    _sentence_segmenter: Optional[SentenceSegmenter] = field(init=False, default=None)
    _classifier_cache: Dict[Any, ClassifierService] = field(init=False, default_factory=dict)

    def get_ocr_service(self) -> OcrService:
        if self._ocr_service is None:
            self._ocr_service = OcrService(**self.ocr_kwargs)
        return self._ocr_service

    def get_sentence_segmenter(self) -> SentenceSegmenter:
        if self._sentence_segmenter is None:
            self._sentence_segmenter = SentenceSegmenter(**self.segmenter_kwargs)
        return self._sentence_segmenter

    def get_classifier_service(self, model_path: str, **overrides: Any) -> ClassifierService:
        merged = {**self.classifier_kwargs, **overrides}
        cache_key = (model_path, _freeze_config(merged))
        if cache_key not in self._classifier_cache:
            self._classifier_cache[cache_key] = ClassifierService(model_path, **merged)
        return self._classifier_cache[cache_key]

    def get_classifier_bundle(self, model_path: str, **overrides: Any) -> ClassifierBundle:
        """Return a cached classifier bundle for ``model_path``."""

        service = self.get_classifier_service(model_path, **overrides)
        return service.bundle


def create_ocr_service(**kwargs: Any) -> OcrService:
    return OcrService(**kwargs)


def create_sentence_segmenter(**kwargs: Any) -> SentenceSegmenter:
    return SentenceSegmenter(**kwargs)


def create_classifier_service(model_path: str, **kwargs: Any) -> ClassifierService:
    return ClassifierService(model_path, **kwargs)


__all__ = [
    "ClassifierBundle",
    "ClassifierService",
    "OcrService",
    "PdfProcessingFactory",
    "SentenceSegmenter",
    "create_classifier_service",
    "create_ocr_service",
    "create_sentence_segmenter",
]

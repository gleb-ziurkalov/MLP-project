"""Service classes encapsulating NLP and OCR dependencies."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from paddleocr import PaddleOCR
import spacy
from torch import cuda
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class OcrService:
    """Wraps :class:`paddleocr.PaddleOCR` lifecycle management."""

    def __init__(
        self,
        ocr: Optional[PaddleOCR] = None,
        *,
        lang: str = "en",
        ocr_factory: Optional[Callable[..., PaddleOCR]] = None,
        **ocr_kwargs,
    ) -> None:
        factory = ocr_factory or PaddleOCR
        self._ocr = ocr or factory(lang=lang, **ocr_kwargs)

    def ocr(self, image: np.ndarray, *, cls: bool = False):
        return self._ocr.ocr(image, cls=cls)


class SentenceSegmenter:
    """Provides spaCy-backed sentence segmentation."""

    def __init__(
        self,
        nlp=None,
        *,
        model_name: str = "en_core_web_sm",
        nlp_loader: Optional[Callable[[str], spacy.language.Language]] = None,
    ) -> None:
        loader = nlp_loader or spacy.load
        self._nlp = nlp or loader(model_name)

    def segment(self, text: str) -> spacy.tokens.Doc:
        return self._nlp(text)


class ClassifierService:
    """Handles loading and inference for transformer-based classifiers."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_loader: Optional[
            Callable[[str, str], Tuple[torch.nn.Module, AutoTokenizer]]
        ] = None,
    ) -> None:
        self._device = device or ("cuda" if cuda.is_available() else "cpu")
        loader = model_loader or self._default_loader

        if model is None or tokenizer is None:
            if not model_name:
                raise ValueError(
                    "Either an existing model/tokenizer or model_name must be provided."
                )
            model, tokenizer = loader(model_name, self._device)
        else:
            model = model.to(self._device)

        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def _default_loader(model_name: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model.to(device), tokenizer

    @property
    def device(self) -> str:
        return self._device

    def classify(self, sentences: Sequence[str], batch_size: int = 32) -> List[int]:
        predictions: List[int] = [0] * len(sentences)
        valid_entries: List[Tuple[int, str]] = [
            (idx, sentence)
            for idx, sentence in enumerate(sentences)
            if isinstance(sentence, str) and sentence.strip()
        ]

        if not valid_entries:
            return predictions

        model = self._model
        tokenizer = self._tokenizer
        device = self._device

        batch: List[str] = []
        batch_indices: List[int] = []

        for idx, sentence in valid_entries:
            batch.append(sentence)
            batch_indices.append(idx)

            if len(batch) == batch_size:
                self._classify_batch(batch, batch_indices, predictions, model, tokenizer, device)
                batch, batch_indices = [], []

        if batch:
            self._classify_batch(batch, batch_indices, predictions, model, tokenizer, device)

        return predictions

    @staticmethod
    def _classify_batch(
        batch: Iterable[str],
        indices: Sequence[int],
        predictions: List[int],
        model,
        tokenizer,
        device: str,
    ) -> None:
        tokenized_batch = tokenizer(
            list(batch), truncation=True, padding="max_length", return_tensors="pt"
        )
        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**tokenized_batch)
            logits = outputs.logits
            batch_predictions = logits.argmax(dim=-1).tolist()

        for idx, prediction in zip(indices, batch_predictions):
            predictions[idx] = prediction

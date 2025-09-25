"""Sentence classification utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import cuda
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


class SentenceClassifier:
    """Classify sentences for compliance using a transformer model."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device_resolver: Optional[Callable[[], str]] = None,
    ):
        self.model_dir = model_dir or config.USE_MODEL_DIR
        self.device_resolver = device_resolver or self._default_device_resolver
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._device: Optional[torch.device] = None

    @staticmethod
    def _default_device_resolver() -> str:
        return "cuda" if cuda.is_available() else "cpu"

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._device is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        device = torch.device(self.device_resolver())
        self._device = device
        self._model = self._model.to(device)

    def classify(self, sentences: Iterable[str], batch_size: int = 32) -> List[Tuple[str, int]]:
        """Classify an iterable of sentences as compliant or not."""
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        valid_sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
        predictions: List[int] = []

        for start in range(0, len(valid_sentences), batch_size):
            batch = valid_sentences[start : start + batch_size]
            tokenized = self._tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = outputs.logits
                batch_predictions = logits.argmax(dim=-1).tolist()

            predictions.extend(batch_predictions)

        return list(zip(valid_sentences, predictions))


__all__ = ["SentenceClassifier"]

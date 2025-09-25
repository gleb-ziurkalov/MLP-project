"""Utilities for converting PDFs to images and running OCR."""

from __future__ import annotations

from typing import List, Sequence

from paddleocr import PaddleOCR
from pdf2image import convert_from_path


class ImageProcessor:
    """Handle PDF page conversion and OCR extraction."""

    def __init__(self, dpi: int = 300, ocr_lang: str = "en", **ocr_kwargs):
        self.dpi = dpi
        self._ocr = PaddleOCR(lang=ocr_lang, **ocr_kwargs)

    def to_images(self, pdf_path: str) -> List:  # returns list of PIL Images
        """Convert a PDF document into page images."""
        try:
            return convert_from_path(pdf_path, dpi=self.dpi)
        except Exception as exc:  # pragma: no cover - logging side effect only
            print(f"Error converting PDF to images: {exc}")
            return []

    def ocr(self, image, **ocr_kwargs) -> Sequence:
        """Run OCR on an image and return raw PaddleOCR results."""
        return self._ocr.ocr(image, **ocr_kwargs)


__all__ = ["ImageProcessor"]

"""Utilities for converting PDF documents into image pages."""

from __future__ import annotations

from pdf2image import convert_from_path

DEFAULT_DPI = 300


def pdf_to_image(pdf_path: str, *, dpi: int = DEFAULT_DPI):
    """Render a PDF into a list of PIL.Image pages."""

    try:
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error converting PDF to images: {exc}")
        return []


__all__ = ["DEFAULT_DPI", "pdf_to_image"]

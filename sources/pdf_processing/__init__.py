"""High-level helpers for PDF processing workflows."""

from __future__ import annotations

import gc
from typing import Callable, Optional, Union

import torch

from .image_processor import DEFAULT_DPI, pdf_to_image
from .services import (
    ClassifierBundle,
    ClassifierService,
    OcrService,
    PdfProcessingFactory,
    SentenceSegmenter,
)
from .tdata_extractor import image_to_tdata
from .text_extractor import image_to_text

DEFAULT_FACTORY = PdfProcessingFactory()


def pdf_to_tdata(
    pdf_path: str,
    *,
    ocr_service: Optional[OcrService] = None,
    sentence_segmenter: Optional[SentenceSegmenter] = None,
    factory: Optional[PdfProcessingFactory] = None,
    dpi: int = DEFAULT_DPI,
):
    """Process a PDF into training data dictionaries."""

    pages = pdf_to_image(pdf_path, dpi=dpi)
    active_factory = factory or DEFAULT_FACTORY
    ocr = ocr_service or active_factory.get_ocr_service()
    segmenter = sentence_segmenter or active_factory.get_sentence_segmenter()

    processed_text = image_to_tdata(pages, ocr, segmenter)

    clear_memory()
    return processed_text


def pdf_to_text(
    pdf_path: str,
    model_path: Optional[str] = None,
    *,
    classifier_bundle: Optional[ClassifierBundle] = None,
    classifier_factory: Optional[Callable[[], ClassifierBundle]] = None,
    classifier_service: Optional[ClassifierService] = None,
    ocr_service: Optional[OcrService] = None,
    sentence_segmenter: Optional[SentenceSegmenter] = None,
    factory: Optional[PdfProcessingFactory] = None,
    dpi: int = DEFAULT_DPI,
    batch_size: int = 32,
):
    """Process a PDF into classified text lines."""

    active_factory = factory or DEFAULT_FACTORY

    classifier_source: Union[
        ClassifierBundle,
        Callable[[], ClassifierBundle],
    ]

    if classifier_factory is not None and any(
        item is not None for item in (classifier_bundle, classifier_service)
    ):
        raise ValueError(
            "Provide only one of classifier_factory, classifier_bundle, or classifier_service",
        )

    if classifier_factory is not None:
        classifier_source = classifier_factory
    else:
        if classifier_bundle is None:
            if classifier_service is not None:
                classifier_bundle = classifier_service.bundle
            else:
                if model_path is None:
                    raise ValueError(
                        "model_path must be provided when classifier bundle is not supplied",
                    )
                classifier_bundle = active_factory.get_classifier_bundle(model_path)

        classifier_source = classifier_bundle

    ocr = ocr_service or active_factory.get_ocr_service()
    segmenter = sentence_segmenter or active_factory.get_sentence_segmenter()

    pages = pdf_to_image(pdf_path, dpi=dpi)
    processed_text = image_to_text(
        pages,
        classifier_source,
        ocr_service=ocr,
        sentence_segmenter=segmenter,
        batch_size=batch_size,
    )

    clear_memory()
    return processed_text


def clear_memory():
    """Release GPU memory and clean up resources."""

    torch.cuda.empty_cache()
    gc.collect()


__all__ = [
    "DEFAULT_FACTORY",
    "ClassifierBundle",
    "ClassifierService",
    "OcrService",
    "PdfProcessingFactory",
    "SentenceSegmenter",
    "clear_memory",
    "pdf_to_tdata",
    "pdf_to_text",
]

from .image_processor import pdf_to_image
from .services import ClassifierService, OcrService, SentenceSegmenter
from .tdata_extractor import image_to_tdata
from .text_extractor import image_to_text

import gc
import torch


def pdf_to_tdata(
    pdf_path,
    *,
    ocr_service: OcrService | None = None,
    sentence_segmenter: SentenceSegmenter | None = None,
):
    pages = pdf_to_image(pdf_path)
    processed_text = image_to_tdata(
        pages,
        ocr_service=ocr_service,
        sentence_segmenter=sentence_segmenter,
    )

    clear_memory()
    return processed_text


def pdf_to_text(
    pdf_path,
    model_path,
    *,
    ocr_service: OcrService | None = None,
    sentence_segmenter: SentenceSegmenter | None = None,
    classifier_service: ClassifierService | None = None,
    batch_size: int = 32,
):
    pages = pdf_to_image(pdf_path)
    processed_text = image_to_text(
        pages,
        model_path,
        ocr_service=ocr_service,
        sentence_segmenter=sentence_segmenter,
        classifier_service=classifier_service,
        batch_size=batch_size,
    )

    clear_memory()
    return [line for line, label in processed_text if label == 1]


def clear_memory():
    """Release GPU memory and clean up resources."""
    torch.cuda.empty_cache()
    gc.collect()

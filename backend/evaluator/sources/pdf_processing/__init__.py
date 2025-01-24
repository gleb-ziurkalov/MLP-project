from .image_processor import pdf_to_image
from .tdata_extractor import image_to_tdata
from .text_extractor import image_to_text
from .text_extractor import classify_lines


import gc
import torch


def pdf_to_tdata(pdf_path):
    # Process all pages and consolidate results for training dataset
    pages = pdf_to_image(pdf_path)
    processed_text = image_to_tdata(pages)

    clear_memory()
    return processed_text


def pdf_to_text(pdf_path, model_path):
    # Process all pages and consolidate results for processing
    pages = pdf_to_image(pdf_path)
    processed_text = image_to_text(pages, model_path)

    clear_memory()
    return [line for line, label in processed_text if label == 1]


def clear_memory():
    """Release GPU memory and clean up resources."""
    torch.cuda.empty_cache()
    gc.collect()
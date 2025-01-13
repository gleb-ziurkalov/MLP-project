from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .image_processor import pdf_to_image
from .tdata_extractor import image_to_tdata
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Process all pages and consolidate results for processing
    pages = pdf_to_image(pdf_path)
    classified_lines = classify_lines(model, tokenizer, pages)

    clear_memory()
    return [line for line, label in classified_lines if label == 1]


def clear_memory():
    """Release GPU memory and clean up resources."""
    torch.cuda.empty_cache()
    gc.collect()
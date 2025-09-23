from functools import lru_cache

from .image_processor import OCR
from .sentence_processor import segment_sentences

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import cuda
import torch

import numpy as np


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """Load and cache the classifier model and tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer


def classify_sentences(model, tokenizer, sentences, batch_size=32):
    """Classify sentences as compliant or not compliant in batches."""
    device = model.device  # Automatically get the model's device
    valid_sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
    predictions = []

    for i in range(0, len(valid_sentences), batch_size):
        batch = valid_sentences[i:i + batch_size]

        # Tokenize the batch
        tokenized_batch = tokenizer(batch, truncation=True, padding="max_length", return_tensors="pt")

        # Move input tensors to the same device as the model
        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}

        # Perform inference
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**tokenized_batch)
            logits = outputs.logits
            batch_predictions = logits.argmax(dim=-1).tolist()

        predictions.extend(batch_predictions)

    # Pair sentences with predictions
    return [(sentence, label) for sentence, label in zip(valid_sentences, predictions)]


def image_to_text(pages, model_path):
    sentences = []

    for page in pages:
        print(f"Processing page: {page}")
        image = np.array(page)
        ocr_results = OCR.ocr(image, cls=False)

        # Extract text content and bounding boxes
        text_boxes = [(line[1][0], line[0]) for line in ocr_results[0]]

        # Segment text into sentences
        segmented_sentences = segment_sentences(text_boxes)

        # Collect sentences for classification
        sentences.extend([s[0] for s in segmented_sentences])  # Extract text content only
    
    model, tokenizer = _load_model(model_path)
    classified_sentences = classify_sentences(model, tokenizer, sentences)
    print(classified_sentences)

    return classified_sentences

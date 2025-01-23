from .image_processor import OCR


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import cuda
import torch

import numpy as np

def classify_lines(model, tokenizer, text_lines, batch_size=32):
    """Classify lines of text as compliant or not compliant in batches."""
    device = model.device  # Automatically get the model's device
    valid_lines = [line for line in text_lines if isinstance(line, str) and line.strip()]
    predictions = []

    for i in range(0, len(valid_lines), batch_size):
        # Take a batch of lines
        batch = valid_lines[i:i + batch_size]

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

    # Pair lines with predictions
    return [(line, label) for line, label in zip(valid_lines, predictions)]


def image_to_text(pages, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Check if a GPU is available and move the model to the GPU
    device = "cuda" if cuda.is_available() else "cpu"
    model = model.to(device)

    text_lines = []

    for page in pages:
        print(f"page: {page}")
        image = np.array(page)
        ocr_results = OCR.ocr(image, cls=False)
        for line in ocr_results[0]:
            text_lines.append(line[1][0])  # Extract the text content
    
    classified_lines = classify_lines(model, tokenizer, text_lines)
    print(classified_lines)

    return classified_lines
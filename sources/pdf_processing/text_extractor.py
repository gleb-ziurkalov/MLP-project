from .image_processor import OCR

from datasets import Dataset
import numpy as np

def image_to_text(pages):
    text_lines = []

    for page in pages:
        image = np.array(page)
        ocr_results = OCR.ocr(image, cls=False)
        for line in ocr_results[0]:
            text_lines.append(line[1][0])  # Extract the text content

    return text_lines

def classify_lines(model, tokenizer, pages):
    """Classify lines of text as compliant or not compliant."""
    dataset = Dataset.from_dict({"text": pages})
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    predictions = model(**{
        "input_ids": tokenized_dataset["input_ids"],
        "attention_mask": tokenized_dataset["attention_mask"]
    })

    logits = predictions.logits
    predicted_labels = logits.argmax(dim=-1).tolist()

    # Pair lines with their predicted labels
    return [(line, label) for line, label in zip(pages, predicted_labels)]
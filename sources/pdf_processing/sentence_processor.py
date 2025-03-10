from .image_processor import OCR

import spacy

nlp = spacy.load("en_core_web_sm")  # Used for sentence segmentation


def extract_bbox(image):
    """Extract bounding boxes and text using OCR."""
    ocr_results = OCR.ocr(image, cls=False)
    
    print(ocr_results)

    extracted_data = []
    for line in ocr_results[0]:
        if len(line) < 2:
            continue  # Skip malformed OCR results

        bbox, (text, conf) = line
        if text.strip():  # Ensure non-empty text
            extracted_data.append((text, bbox))
    
    return extracted_data if extracted_data else None  # Avoid returning NULL

def convert_bbox(paddle_bbox):
    """Convert PaddleOCR bounding box format (4 points) to (x_min, y_min, x_max, y_max)."""
    x_min = min(point[0] for point in paddle_bbox)
    y_min = min(point[1] for point in paddle_bbox)
    x_max = max(point[0] for point in paddle_bbox)
    y_max = max(point[1] for point in paddle_bbox)
    return [x_min, y_min, x_max, y_max]


def segment_sentences(text_boxes):
    sentences = []
    words = [item[0] for item in text_boxes]  # Extract words only
    bounding_boxes = [item[1] for item in text_boxes]  # Extract bounding boxes

    # Join words into a full text passage
    full_text = " ".join(words)

    # Use spaCy to split into sentences
    doc = nlp(full_text)
    for sent in doc.sents:
        sentence_text = sent.text.strip()

        # Find words that belong to this sentence
        matched_boxes = []
        for word, bbox in zip(words, bounding_boxes):
            if word in sentence_text:
                matched_boxes.append(bbox)

        # Merge bounding boxes for sentence
        if matched_boxes:
            x_min = min(box[0][0] for box in matched_boxes)  # Left-most x
            y_min = min(box[0][1] for box in matched_boxes)  # Top-most y
            x_max = max(box[2][0] for box in matched_boxes)  # Right-most x
            y_max = max(box[2][1] for box in matched_boxes)  # Bottom-most y

            sentence_bbox = (x_min, y_min, x_max, y_max)
        else:
            sentence_bbox = None  # Avoid NULL values

        # **Use dictionary instead of tuple**
        sentences.append({
            "text": sentence_text,
            "bounding_box": sentence_bbox
        })

    return sentences


def find_sentence_box(sentence, text_boxes):
    """Find a bounding box that covers a given sentence based on text box locations."""
    matching_boxes = [box["bounding_box"] for box in text_boxes if sentence in box["text"]]
    if not matching_boxes:
        return None
    x_min = min(box[0] for box in matching_boxes)
    y_min = min(box[1] for box in matching_boxes)
    x_max = max(box[2] for box in matching_boxes)
    y_max = max(box[3] for box in matching_boxes)
    return [x_min, y_min, x_max, y_max]
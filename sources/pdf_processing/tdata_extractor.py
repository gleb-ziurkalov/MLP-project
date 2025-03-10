from .image_processor import OCR

import numpy as np
import cv2
import spacy

nlp = spacy.load("en_core_web_sm")  # Used for sentence segmentation

def detect_highlighted_regions(image):
    """Detect highlighted regions in the image using color detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_highlight = np.array([90, 50, 200])
    upper_highlight = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_highlight, upper_highlight)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[x, y, x + w, y + h] for x, y, w, h in (cv2.boundingRect(contour) for contour in contours)]


def extract_text_with_bounding_boxes(image):
    """Perform OCR and return text along with converted bounding boxes."""
    results = OCR.ocr(image)[0]
    return [
        {
            "text": line[1][0],
            "confidence": line[1][1],
            "bounding_box": convert_bbox(line[0]),  # Convert PaddleOCR format
        }
        for line in results
    ]

def convert_bbox(paddle_bbox):
    """Convert PaddleOCR bounding box format (4 points) to (x_min, y_min, x_max, y_max)."""
    x_min = min(point[0] for point in paddle_bbox)
    y_min = min(point[1] for point in paddle_bbox)
    x_max = max(point[0] for point in paddle_bbox)
    y_max = max(point[1] for point in paddle_bbox)
    return [x_min, y_min, x_max, y_max]


def segment_sentences(text_boxes):
    """Segment extracted OCR text into individual sentences."""
    full_text = " ".join([box["text"] for box in text_boxes])
    doc = nlp(full_text)
    return [{"text": sent.text, "bounding_box": find_sentence_box(sent.text, text_boxes)} for sent in doc.sents]


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


def label_compliance_sentences(sentences, highlighted_boxes, iou_threshold=0.1):
    """Label sentences as compliance_statement=1 or compliance_statement=0 based on IoU with highlighted regions."""
    for sentence in sentences:
        if sentence["bounding_box"]:
            x, y, x2, y2 = sentence["bounding_box"]
            sentence["compliance_statement"] = int(any(
                calculate_iou([x, y, x2, y2], [hx, hy, hx + hw, hy + hh]) > iou_threshold
                for hx, hy, hw, hh in highlighted_boxes
            ))
        else:
            sentence["compliance_statement"] = 0
    return sentences


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def image_to_tdata(pages):
    consolidated_results = []
    
    for page in pages:
        # Convert page to NumPy array
        image = np.array(page)

        # Detect highlighted regions
        highlighted_boxes = detect_highlighted_regions(image)
        text_boxes = extract_text_with_bounding_boxes(image)
        sentences = segment_sentences(text_boxes)
        labeled_text = label_compliance_sentences(sentences, highlighted_boxes)

        # Add labeled text to consolidated results
        consolidated_results.extend(labeled_text)

    return consolidated_results 
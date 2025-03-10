from .image_processor import OCR
from .sentence_processor import extract_bbox, segment_sentences

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


def label_compliance_sentences(sentences, highlighted_boxes, iou_threshold=0.5):
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

        # Extracts text and converts it into senteces
        text_boxes = extract_bbox(image)
        sentences = segment_sentences(text_boxes)

        highlighted_boxes = detect_highlighted_regions(image)
        # Detect highlighted regions 
        labeled_text = label_compliance_sentences(sentences, highlighted_boxes)

        #print("test")

        # Add labeled text to consolidated results
        consolidated_results.extend(labeled_text)

    return consolidated_results 
from .image_processor import OCR

import numpy as np
import cv2
# Initialize PaddleOCR with English language model

def detect_highlighted_regions(image):
    """Detect highlighted regions in the image using color detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_highlight = np.array([90, 50, 200])  # Light blue range
    upper_highlight = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_highlight, upper_highlight)

    # Convert contours to bounding boxes
    return [cv2.boundingRect(contour) for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]


def extract_text_with_bounding_boxes(image):
    """Perform OCR and return text along with bounding boxes."""
    return [
        {
            "text": line[1][0],
            "confidence": line[1][1],
            "bounding_box": line[0],
        }
        for line in OCR.ocr(image)[0]
    ]


def label_compliance_statements(text_boxes, highlighted_boxes, iou_threshold=0.1):
    """Label text as compliance_statement=1 or compliance_statement=0 based on IoU with highlighted regions."""
    for text_box in text_boxes:
        box_coords = [
            min(point[0] for point in text_box["bounding_box"]),
            min(point[1] for point in text_box["bounding_box"]),
            max(point[0] for point in text_box["bounding_box"]),
            max(point[1] for point in text_box["bounding_box"]),
        ]
        text_box["compliance_statement"] = int(any(
            calculate_iou(box_coords, [x, y, x + w, y + h]) > iou_threshold
            for x, y, w, h in highlighted_boxes
        ))
    return text_boxes


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
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

        # Perform OCR and extract text with bounding boxes
        text_boxes = extract_text_with_bounding_boxes(image)

        # Label text as compliance statements
        labeled_text = label_compliance_statements(text_boxes, highlighted_boxes)

        # Add labeled text to consolidated results
        consolidated_results.extend(labeled_text)

    return consolidated_results 
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR with English language and default model
OCR = PaddleOCR(use_angle_cls=True, lang='en')  # Use 'multilingual' if the document has multiple languages

def pdf_to_text_via_paddleocr(pdf_path):
    # Convert PDF pages to images with a high DPI and grayscale for better OCR accuracy
    pages = [page.convert("L") for page in convert_from_path(pdf_path, dpi=400)]  # Convert to grayscale
    extracted_text = []

    # Process each page with PaddleOCR
    for page_num, page in enumerate(pages):
        # Convert PIL image to NumPy array
        page_np = np.array(page)
        
        # Perform OCR on the page image
        result = OCR.ocr(page_np)
        
        # Iterate over each line, collect and format the text output from the OCR result
        for word_info in result[0]:  
            if isinstance(word_info[1], tuple) and isinstance(word_info[1][0], str):  # Check if text is present
                text = word_info[1]  # Extract only the recognized text part
                extracted_text.append(text)

    preprocessed_text = preprocess_ocr_text(extracted_text)
    return preprocessed_text


def preprocess_ocr_text(text_lines, min_confidence=0.8):
    cleaned_lines = []
    for line in text_lines:
        
        # Add lines if they are not too short or not low confidence
        if len(line[0]) > 3 or line[1] >= min_confidence:
            # Remove extra spaces
            cleaned_line = line[0].strip()
            cleaned_lines.append(cleaned_line)        
    
    return cleaned_lines

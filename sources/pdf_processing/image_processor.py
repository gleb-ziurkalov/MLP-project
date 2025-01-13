from pdf2image import convert_from_path
from paddleocr import PaddleOCR

DPI = 300
OCR = PaddleOCR(lang='en')  # Use 'multilingual' for multi-language PDFs

def pdf_to_image(pdf_path):
    """Extract and consolidate text from a PDF, labeling compliance statements."""
    try:
        # Convert PDF to images (keep colors for highlight detection)
        pages = convert_from_path(pdf_path, dpi=DPI)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

    return pages
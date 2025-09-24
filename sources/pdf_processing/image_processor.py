from pdf2image import convert_from_path

DPI = 300


def pdf_to_image(pdf_path):
    """Extract and consolidate text from a PDF, labeling compliance statements."""
    try:
        pages = convert_from_path(pdf_path, dpi=DPI)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

    return pages

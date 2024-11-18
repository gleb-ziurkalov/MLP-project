from text_extraction._t_extractor import pdf_to_text_via_paddleocr
from metadata_extraction import extract_metadata

# from _workspace.compliance_extraction import extract_compliance

# Substantio. Metadata
'''
Initialize Named Entity Recognition (NER) model (e.g., SpaCy) for business entities and regulatory references
Define regex patterns for dates, CAS numbers, vendor part numbers, etc.
'''

def extract_data_from_document(text):
    metadata = extract_metadata(text)
    # compliance_data = extract_compliance(text)
    
    # Structure extracted data for further processing
    document_data = {
        "metadata": metadata,
        # "compliance_data": compliance_data
    }
    return document_data


def main():
    pdf_path = r'/mnt/c/Users/btj-6/Desktop/ai50/_workspace/_data/Compliance_Statements/_REACH_SVHC-233_SCIP_Compliance_Rittal_20.03.2023.pdf'
    raw_text = pdf_to_text_via_paddleocr(pdf_path)
    
    raw_data = extract_data_from_document(raw_text)


if __name__ == "__main__":
    main()
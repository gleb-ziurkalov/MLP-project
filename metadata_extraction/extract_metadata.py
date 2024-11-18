# Metadata Extraction
def extract_metadata(text):
    metadata = {}
    metadata["dates"] = extract_dates(text)
    metadata["business_entities"] = extract_entities(text)
    metadata["regulatory_documents"] = extract_regulations(text)
    metadata["vendor_parts"] = extract_vparts(text)
    return metadata
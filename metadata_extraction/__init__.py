# utils/__init__.py
from .extract_dates import extract_dates
from .extract_entities import extract_entities
from .extract_regulations import extract_regulations
# from .extract_vparts import extract_vparts

# __all__ to expose these functions when using `from metadata_extraction import *`
__all__ = ["extract_dates", "extract_entities", "extract_regulations", "extract_vparts"]

# Metadata Extraction
def extract_metadata(text):
    metadata = {}
    
    metadata["dates"] = extract_dates(text)
    print(metadata["dates"])
    
    # metadata["business_entities"] = extract_entities(text)
    # print(metadata["business_entities"])
    
    # metadata["regulatory_documents"] = extract_regulations(text)
    # print(metadata["regulatory_documents"])
    # metadata["vendor_parts"] = extract_vparts(text)
    return metadata
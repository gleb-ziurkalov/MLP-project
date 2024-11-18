import re

def extract_vendor_parts(text):
    # Example regex pattern for alphanumeric vendor part numbers (customize as needed)
    vendor_part_pattern = r'\b[A-Z0-9]{4,10}\b'  # Matches codes with 4-10 alphanumeric characters
    
    vendor_parts = re.findall(vendor_part_pattern, text)
    
    return vendor_parts
import re

REGULATION_PATTERNS = [
    "TSCA", "REACH", "SVHC", "EPA", "OSHA", "RoHS",
    "Toxic Substances Control Act", "Section 6(h)", "PBT substances"
]
PATTERN = r'\b(?:' + '|'.join(re.escape(term) for term in REGULATION_PATTERNS) + r')\b'

def label_regulations(dataset):
    # Build a single regex pattern for all terms
    
    for line in dataset:
        print(line)
        matches = re.findall(PATTERN, line["text"], re.IGNORECASE)
        if matches:
            line["regulation"] = True
            line["useful_data"] = True
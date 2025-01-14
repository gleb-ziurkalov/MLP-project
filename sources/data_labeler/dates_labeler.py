import re
import datetime


DATE_PATTERNS = '|'.join([
    r'\d{4}-\d{2}-\d{2}',                                                          # YYYY-MM-DD
    r'\d{2}-\d{2}-\d{4}',                                                          # DD-MM-YYYY
    r'\d{2}-\d{2}-\d{2}',                                                          # DD-MM-YY        
    r'\d{4}\.\d{2}\.\d{2}',                                                        # YYYY.MM.DD
    r'\d{2}\.\d{2}\.\d{4}',                                                        # DD.MM.YYYY
    r'\d{2}\.\d{2}\.\d{2}',                                                        # DD.MM.YY   
    r'\d{4}/\d{2}/\d{2}',                                                          # YYYY/MM/DD
    r'\d{2}/\d{2}/\d{4}',                                                          # DD/MM/YYYY
    r'\d{2}/\d{2}/\d{2}',                                                          # DD/MM/YY
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}'  # Month DD, YYYY
])
DATE_FORMATS = [                            # Valid date formats for validation
    "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y",
    "%Y.%m.%d", "%d.%m.%Y", "%d.%m.%y",
    "%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y",
    "%b %d, %Y", "%B %d, %Y"                # Short and full month names
]

def label_dates(dataset):
    for line in dataset:
        # Find matches using the regex
        matches = re.findall(DATE_PATTERNS, line["text"])
        
        # Validate each match
        if matches:
            for match in matches:
                if is_valid_date(match, DATE_FORMATS):
                    line["date"] = 1
                    line["compliance_data"] = 1
                    break  # No need to check further matches for this line
        else:
            line["date"] = 0


def is_valid_date(date_str, formats):
    for fmt in formats:
        try:
            datetime.datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False
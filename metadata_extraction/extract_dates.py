import re

DATE_PATTERNS = '|'.join([
    r'\d{4}-\d{2}-\d{2}\r',               # YYYY-MM-DD
    r'\d{2}-\d{2}-\d{4}\r',               # DD-MM-YYYY
    r'\d{2}-\d{2}-\d{2}\r',               # DD-MM-YY        
    r'\d{4}.\d{2}.\d{2}\r',               # YYYY.MM.DD
    r'\d{2}.\d{2}.\d{4}\r',               # DD.MM.YYYY
    r'\d{2}.\d{2}.\d{2}\r',               # DD.MM.YY   
    r'\d{4}/\d{2}/\d{2}\r',               # YYYY/MM/DD
    r'\d{2}/\d{2}/\d{4}\r',               # DD/MM/YYYY
    r'\d{2}/\d{2}/\d{2}\r',               # DD/MM/YYYY
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}\r'  # Month DD, YYYY
])

def extract_dates(text):
    dates = []
    # Find all date matches in the text
    for line in text:
        date = re.findall(DATE_PATTERNS, line)
        print(date)
        if date:
            dates.append(date[0])
            line = line + "$date"
            print(line)

    return dates
import spacy
import re

ABBREVIATIONS = [
    "incl",  # Including
    "Dir",   # Directive
    "Mr",    # Mister
    "Dr",    # Doctor
    "Del",   # Delegated
]

# Precompile regex for abbreviations with a dot
ABBREVIATIONS_PATTERN = re.compile(rf'\b({"|".join(ABBREVIATIONS)})\.')

def preprocess_doc(text):

    # combine OCR test into a single string
    text_consolidated = ""
    for line in text:
        if line.strip():
            text_consolidated += " " + line.strip()
    
    # split consolidated text into sentences
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(remove_abbrevs(text_consolidated))

    text_sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    return text_sentences

def remove_abbrevs(text):
    # Remove dots from defined abbreviations
    text = ABBREVIATIONS_PATTERN.sub(lambda match: match.group(1), text)
    return text
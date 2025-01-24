import re

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

BUSINESS_TERMS = ["Inc", "LLC", "Corp", "Ltd", "Corporation", "GmbH", "Group","Co.","KG"]
PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, BUSINESS_TERMS)) + r')\b', re.IGNORECASE) # from BUSINESS_TERMS matches whole words only


# Load Hugging Face NER pipeline
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def label_entities_huggingface(dataset, confidence_threshold=0.85):
    for line in dataset:
        entities = ner_pipeline(line["text"])
        if entities:
            for entity in entities:
                if entity["entity"] == "B-ORG" and entity["score"] > confidence_threshold:
                    line["business_entity_confidence"] = entity["score"] # Store confidence
                    if PATTERN.search(line["text"]):
                        # Label as business entity
                        line["business_entity"] = 1
                        line["compliance_data"] = 1
                        line["business_entity_confidence"] = entity["score"] # Store confidence
                        break # no need to iterate through line anymore
                    else:
                        # Label as potentially useful data but not a business entity
                        line["compliance_data"] = 1
                        line["business_entity_confidence"] = entity["score"] # Store confidence
                        
                line["business_entity"] = 0
        else:
            line["business_entity"] = 0
                    
'''
Not as important as the rest

def extract_contact_info(text):
    # Regex patterns for email and phone numbers
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\+?\d[\d -]{8,}\d'  # Matches international and local phone numbers
    
    # Find all email and phone matches in the text
    emails = re.findall(email_pattern, text)
    phone_numbers = re.findall(phone_pattern, text)
    
    contact_info = {
        "emails": emails,
        "phone_numbers": phone_numbers
    }
    
    return contact_info
'''
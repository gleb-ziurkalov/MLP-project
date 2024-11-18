import spacy
import re


# Load SpaCy's pre-trained NER model (e.g., 'en_core_web_sm' for English)
NLP = spacy.load('en_core_web_sm')

def extract_entities(text):
    business_entities = []
    for line in text:
        doc = NLP(line)
        for entity in doc.ents:
            if entity.label_ == "ORG":  # "ORG" is the label for organizations
                
                # Call extract_contact_info for each entity (if needed)
                # contact_info = extract_contact_info(text)  # Optional: Get contact info associated with the entity
                business_entities.append({
                    "name": entity.text,
                    #"contact_info": contact_info  # This could be linked to specific entities if context is available
                })
                line = line + "$b_entity"
                print(line)

    return business_entities

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
def extract_regulations(text):
    # Keywords related to regulatory documents
    regulatory_keywords = ["TSCA", "EPA", "PBT substances", "Section 6(h)", "Toxic Substances Control Act"]
    
    regulatory_documents = []
    sentences = text.split('.')  # Split text into sentences for better matching
    
    for sentence in sentences:
        if any(keyword in sentence for keyword in regulatory_keywords):
            regulatory_documents.append(sentence.strip())  # Store sentence mentioning regulation
    
    return regulatory_documents
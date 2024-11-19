import json
import numpy as np

from text_extraction._t_extractor import pdf_to_text_via_paddleocr
from data_labeler import label_data


# from _workspace.compliance_extraction import extract_compliance

# Substantio. Metadata
'''
Initialize Named Entity Recognition (NER) model (e.g., SpaCy) for business entities and regulatory references
Define regex patterns for dates, CAS numbers, vendor part numbers, etc.
'''


'''
def save_to_csv(labeled_lines, output_file):
    """
    Saves labeled lines to a CSV file.

    Args:
        labeled_lines (list of dict): Labeled data to save.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(labeled_lines)

# Example usage
if __name__ == "__main__":
    lines = [
        "Today's date is 2023-11-18.",
        "We had an event on 12-01-2023.",
        "The deadline is Jan 1, 2024.",
        "Page 1 of 10"
    ]
    
    # Label the lines
    labeled_lines = label_dates(lines)
    print("Labeled Data:", labeled_lines)

    # Save to CSV
    save_to_csv(labeled_lines, "labeled_dates.csv")
'''
def default_handler(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def label_document(text):
    unlabeled_data = []
    labeled_data = []

    for line in text:
        unlabeled_data.append({"useful_data":None, "date":None, "business_entity": None, "regulation": None, "text": line}) 

    labeled_data = label_data(unlabeled_data)    

    return labeled_data


def main():
    pdf_path = r'/mnt/c/Users/btj-6/Desktop/ai50/_workspace/_data/Compliance_Statements/_REACH_SVHC-233_SCIP_Compliance_Rittal_20.03.2023.pdf'
    dataset_raw = pdf_to_text_via_paddleocr(pdf_path)
    dataset_labeled = label_document(dataset_raw)

    with open("labeled_dataset.json", "w") as f:
        json.dump(dataset_labeled, f, indent=4, default=default_handler)


if __name__ == "__main__":
    main()
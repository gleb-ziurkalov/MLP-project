import os
import json
import numpy as np

from text_extraction._t_extractor import pdf_to_text_via_paddleocr
from data_labeler import label_data

TRAINING_DATASET_DIRECTORY = r'/mnt/c/Users/btj-6/Desktop/ai50/_data/t_dataset/'
DATA_DIRECTORY = r'/mnt/c/Users/btj-6/Desktop/ai50/_data/Compliance_Statements/'
FILES = os.listdir(DATA_DIRECTORY)


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
    for entry in FILES:
        file_path = os.path.join(DATA_DIRECTORY, entry)
        dataset_raw = pdf_to_text_via_paddleocr(file_path)
        dataset_labeled = label_document(dataset_raw)

        training_dataset_path = os.path.join(TRAINING_DATASET_DIRECTORY, f"{entry}.json")
        with open(training_dataset_path, "w") as f:
            json.dump(dataset_labeled, f, indent=4, default=default_handler)


if __name__ == "__main__":
    main()
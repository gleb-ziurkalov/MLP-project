import os
import json
import numpy as np

from sources.pdf_processing import pdf_to_tdata, pdf_to_text
# from sources.text_preprocessor import preprocess_doc
from sources.data_labeler import label_data
# from sources.model_trainer import train_model

TRAINING_PDF_DIR   = "./data/training_data/"
LABELED_DATA_DIR   = "./data/labeled_data/"
INPUT_PDF_DIR      = "./staging/input_pdf/"
OUTPUT_JSON_DIR    = "./staging/output_JSON/"

TRAINED_MODEL_DIR  = "./models/trained_model/"
FINAL_MODEL_DIR    = "./models/block_classification_oversampled/"


TRAIN_FILES = os.listdir(TRAINING_PDF_DIR)
INPUT_FILES = os.listdir(INPUT_PDF_DIR)

def default_handler(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def label_doc(text):
    unlabeled_data = []
    labeled_data = []

    for element in text:
        unlabeled_data.append({"compliance_statement":element["compliance_statement"], "compliance_data":None, "date":None, "business_entity": None, "regulation": None, "text": element["text"]}) 
    
    labeled_data = label_data(unlabeled_data)    

    return labeled_data

def process_tdata(files, input_path=TRAINING_PDF_DIR, output_path=LABELED_DATA_DIR):
    for entry in files:
        file_path = os.path.join(input_path, entry)
        dataset_raw = pdf_to_tdata(file_path)

        dataset_labeled = label_doc(dataset_raw)

        training_dataset_path = os.path.join(output_path, f"{entry}.json")
        with open(training_dataset_path, "w") as f:
            json.dump(dataset_labeled, f, indent=4, default=default_handler)

def process_statement(files, model_path=FINAL_MODEL_DIR):
    # rewrite to take a single file, separate files
    for entry in files:

        file_path = os.path.join(INPUT_PDF_DIR, entry)
        compliance_processed = pdf_to_text(file_path, model_path)

        output_path = os.path.join(OUTPUT_JSON_DIR, f"{entry}.json")
        with open(output_path, "w") as f:
            json.dump(compliance_processed, f, indent=4, default=default_handler)

def main():
    process_tdata(TRAIN_FILES)
    # train_model(LABELED_DATA_DIR, FINAL_MODEL_DIR)
    # process_statement(INPUT_FILES)


if __name__ == "__main__":
    main()
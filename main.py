import config
import os
import json
import numpy as np

from sources.pdf_processing import pdf_to_tdata, pdf_to_text
from sources.data_labeler import label_data
from sources.model_trainer import train_model

TRAIN_FILES = os.listdir(config.LABELED_PDF_DIR)
INPUT_FILES = os.listdir(config.INPUT_PDF_DIR)

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

def process_tdata(files, input_path=config.LABELED_PDF_DIR, output_path=config.TRAINING_DATA_DIR):
    for entry in files:
        file_path = os.path.join(input_path, entry)
        dataset_labeled = pdf_to_tdata(file_path)

        # dataset_labeled = label_doc(dataset_raw)

        training_dataset_path = os.path.join(output_path, f"{entry}.json")
        with open(training_dataset_path, "w") as f:
            json.dump(dataset_labeled, f, indent=4, default=default_handler)

def process_statement(files, model_path=config.USE_MODEL_DIR):
    # rewrite to take a single file, separate files
    for entry in files:

        file_path = os.path.join(config.INPUT_PDF_DIR, entry)
        compliance_processed = pdf_to_text(file_path, model_path)

        output_path = os.path.join(config.OUTPUT_JSON_DIR, f"{entry}.json")
        with open(output_path, "w") as f:
            json.dump(compliance_processed, f, indent=4, default=default_handler)

def main():
    process_tdata(TRAIN_FILES)
    # train_model(config.TRAINING_DATA_DIR, config.USE_MODEL_DIR)
    # process_statement(INPUT_FILES)


if __name__ == "__main__":
    main()
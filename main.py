import os
import json
import numpy as np

from sources.pdf_processing import pdf_to_tdata, pdf_to_text
from sources.text_preprocessor import preprocess_doc
from sources.data_labeler import label_data
from sources.model_trainer import train_model

PDF_STATEMENTS_DIR = "./data/pdf_statements/"
LABELED_DATA_DIR   = "./data/labeled_data/"
INPUT_PDF_DIR      = "./data/input_pdf/"

TRAINED_MODEL_DIR  = "./data/trained_model/"
FINAL_MODEL_DIR  = "./data/trained_model/checkpoint-1518/"


FILES = os.listdir(PDF_STATEMENTS_DIR)

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

def process_data(files):
    for entry in files:
        
        file_path = os.path.join(PDF_STATEMENTS_DIR, entry)
        dataset_raw = pdf_to_tdata(file_path)

        # dataset_prep = preprocess_doc(dataset_raw)
        
        dataset_labeled = label_doc(dataset_raw)
        # dataset_labeled = label_doc(dataset_prep)

        training_dataset_path = os.path.join(LABELED_DATA_DIR, f"{entry}.json")
        with open(training_dataset_path, "w") as f:
            json.dump(dataset_labeled, f, indent=4, default=default_handler)

def main():

    # process_data(FILES)
    train_model(LABELED_DATA_DIR, FINAL_MODEL_DIR)

    #compliant_statements = pdf_to_text(INPUT_PDF_DIR, TRAINED_MODEL_DIR)

    #print("Compliant Statements:")
    #for statement in compliant_statements:
    #    print(statement)

if __name__ == "__main__":
    main()
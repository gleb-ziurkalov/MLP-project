import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evaluator.sources.pdf_processing import pdf_to_tdata, pdf_to_text, pdf_to_image, image_to_text, classify_lines
# from sourcesr.text_preprocessor import preprocess_doc
from evaluator.sources.data_labeler import label_data
# from sources.model_trainer import train_model

TRAINING_PDF_DIR   = "./data/pdf_statements/"
LABELED_DATA_DIR   = "./data/labeled_data/"
INPUT_PDF_DIR      = "./evaluator/data/input_pdf/"
OUTPUT_JSON_DIR    = "./data/output_JSON/"

TRAINED_MODEL_DIR  = "./data/trained_model/"
FINAL_MODEL_DIR  = "./evaluator/data/models/block_classification_oversampled/"


#TRAIN_FILES = os.listdir(TRAINING_PDF_DIR)
INPUT_FILES = os.listdir(INPUT_PDF_DIR)


MODEL = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
TOKENIZER = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

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

def default_handler(obj):
    # Define a default handler for serializing unsupported types (if needed)
    return str(obj)

def json_print(json_path, file_name, data, mod):
    # Ensure the directory exists
    os.makedirs(json_path, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(f"{json_path}{file_name}_{mod}.json")

    # Write the JSON data to the file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, default=default_handler)
    
    return output_path

def read_json_as_array(file_path):
    # Open the JSON file and load the data
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Ensure the data is an array of strings
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    else:
        raise ValueError("The JSON file does not contain an array of strings.")


def process_tdata(files):
    for entry in files:
        
        file_path = os.path.join(TRAINING_PDF_DIR, entry)
        dataset_raw = pdf_to_tdata(file_path)

        # dataset_prep = preprocess_doc(dataset_raw)
        
        dataset_labeled = label_doc(dataset_raw)
        # dataset_labeled = label_doc(dataset_prep)

        training_dataset_path = os.path.join(LABELED_DATA_DIR, f"{entry}.json")
        with open(training_dataset_path, "w") as f:
            json.dump(dataset_labeled, f, indent=4, default=default_handler)

def process_statement(entry, model, tokenizer):
    file_path = os.path.join(INPUT_PDF_DIR, entry)

    # extracting text from pdf
    pages = pdf_to_image(file_path)
    text_lines = image_to_text(pages, model)
    json_print(OUTPUT_JSON_DIR, text_lines, mod="text")
    print("Text extracted")
        
    # evaluating extracted text
    print("Evaluating")
    classified_lines = classify_lines(model, tokenizer, text_lines)
    print("Evaluation complete")
    compliance_processed = [line for line, label in classified_lines if label == 1]
    
    # printing the evaluated JSON
    json_print(OUTPUT_JSON_DIR, compliance_processed,mod="eval")    
    print("Report Printed")

def process_statement_batch(files):
    for entry in files:
        process_statement(entry, model=MODEL, tokenizer=TOKENIZER)

def main():
    # model init
    model=MODEL
    tokenizer=TOKENIZER

    # getting first (and only) file from dir
    file_name = INPUT_FILES[0]
    file_path = os.path.join(INPUT_PDF_DIR, file_name)

    # extracting text from pdf file
    pages = pdf_to_image(file_path)
    text_lines = image_to_text(pages, model)
    json_print(OUTPUT_JSON_DIR, file_name, text_lines, mod="text")
    print("Text extracted")
        
    # evaluating extracted text
    print("Evaluating")
    classified_lines = classify_lines(model, tokenizer, text_lines)
    compliance_processed = [line for line, label in classified_lines if label == 1]
    print("Evaluation complete")
    # printing the evaluated JSON
    json_print(OUTPUT_JSON_DIR, file_name, compliance_processed, mod="eval")    
    print("Report Printed")

    
    # process_tdata(TRAIN_FILES)
    # train_model(LABELED_DATA_DIR, FINAL_MODEL_DIR)
    # process_statement_batch(INPUT_FILES)

if __name__ == "__main__":
    main()
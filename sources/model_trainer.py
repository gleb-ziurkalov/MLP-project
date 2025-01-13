import os
import json
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          Trainer, TrainingArguments)
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prepare dataset
def prepare_dataset(data_dir):
    texts, labels = [], []
    for file_name in filter(lambda f: f.endswith(".json"), os.listdir(data_dir)):
        with open(os.path.join(data_dir, file_name), "r") as f:
            data = json.load(f)
            texts.extend([item["text"] for item in data])
            labels.extend([item.get("compliance_statement", 0) for item in data])
    return texts, labels

# Tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Train model
def train_model(data_dir, output_dir, model_name="nlpaueb/legal-bert-base-uncased", epochs=3, batch_size=32):
    # Prepare data
    texts, labels = prepare_dataset(data_dir)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    datasets = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "validation": Dataset.from_dict({"text": val_texts, "label": val_labels})
    })

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    tokenized_datasets = tokenize_dataset(datasets, tokenizer)
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,  # Mixed precision
        gradient_accumulation_steps=2,
        dataloader_num_workers=4
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    return trainer
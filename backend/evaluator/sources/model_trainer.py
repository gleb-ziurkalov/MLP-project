import os
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Prepare dataset with oversampling
def prepare_oversampled_dataset(data_dir):
    texts, labels = [], []

    # Load data
    for file_name in filter(lambda f: f.endswith(".json"), os.listdir(data_dir)):
        with open(os.path.join(data_dir, file_name), "r") as f:
            data = json.load(f)
            for item in data:
                texts.append(item["text"])
                labels.append(item.get("compliance_statement", 0))

    # Separate majority and minority classes
    majority_class = [(t, l) for t, l in zip(texts, labels) if l == 0]
    minority_class = [(t, l) for t, l in zip(texts, labels) if l == 1]

    # Oversample the minority class to match the majority class size
    oversampled_minority_class = random.choices(minority_class, k=len(majority_class))

    # Combine the oversampled minority class with the majority class
    balanced_data = majority_class + oversampled_minority_class
    random.shuffle(balanced_data)

    balanced_texts, balanced_labels = zip(*balanced_data)
    return list(balanced_texts), list(balanced_labels)

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
    # Prepare data with oversampling
    texts, labels = prepare_oversampled_dataset(data_dir)
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
        dataloader_num_workers=4,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate the model on the validation set
    eval_metrics = trainer.evaluate()
    
    # Save metrics to a JSON file
    metrics_file_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_file_path, "w") as metrics_file:
        json.dump(eval_metrics, metrics_file, indent=4)

    return trainer
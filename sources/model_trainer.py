"""Utilities for training and evaluating sequence classification models."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class ModelTrainer:
    """Train and evaluate transformer-based sequence classification models."""

    training_data_dir: str
    output_dir: str
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    tokenizer_name: Optional[str] = None
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    logging_steps: int = 10
    gradient_accumulation_steps: int = 2
    dataloader_num_workers: int = 4
    fp16: bool = True
    test_size: float = 0.2
    random_state: int = 42
    training_args_overrides: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tokenizer_name = self.tokenizer_name or self.model_name
        self._tokenizer = None
        self._model = None
        self._tokenized_datasets: Optional[DatasetDict] = None
        self._trainer: Optional[Trainer] = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
        return self._model

    def prepare_dataset(self) -> DatasetDict:
        """Load raw data, apply oversampling, split, and tokenize."""

        texts, labels = self._prepare_oversampled_dataset(self.training_data_dir)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        datasets = DatasetDict(
            {
                "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
                "validation": Dataset.from_dict(
                    {"text": val_texts, "label": val_labels}
                ),
            }
        )

        tokenized = self._tokenize_dataset(datasets)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        self._tokenized_datasets = tokenized
        return tokenized

    def train(self, **training_overrides) -> Trainer:
        """Train the configured model and persist artifacts."""

        tokenized_datasets = self._tokenized_datasets or self.prepare_dataset()
        os.makedirs(self.output_dir, exist_ok=True)

        training_arguments = self._build_training_arguments(training_overrides)

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        self._trainer = trainer
        return trainer

    def evaluate(self, save: bool = True) -> Dict[str, float]:
        """Evaluate the trained model on the validation split."""

        if self._trainer is None:
            raise RuntimeError("The model must be trained before evaluation.")

        metrics = self._trainer.evaluate()
        if save:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, "w") as metrics_file:
                json.dump(metrics, metrics_file, indent=4)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _prepare_oversampled_dataset(self, data_dir: str):
        texts, labels = [], []

        for file_name in filter(lambda f: f.endswith(".json"), os.listdir(data_dir)):
            with open(os.path.join(data_dir, file_name), "r") as f:
                data = json.load(f)
                for item in data:
                    texts.append(item["text"])
                    labels.append(item.get("compliance_statement", 0))

        majority_class = [(t, l) for t, l in zip(texts, labels) if l == 0]
        minority_class = [(t, l) for t, l in zip(texts, labels) if l == 1]

        if not minority_class:
            raise ValueError("No positive class examples found for oversampling.")

        oversampled_minority_class = random.choices(minority_class, k=len(majority_class))

        balanced_data = majority_class + oversampled_minority_class
        random.shuffle(balanced_data)

        balanced_texts, balanced_labels = zip(*balanced_data)
        return list(balanced_texts), list(balanced_labels)

    def _tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        return dataset.map(
            lambda x: self.tokenizer(
                x["text"], truncation=True, padding="max_length"
            ),
            batched=True,
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def _build_training_arguments(self, overrides: Dict[str, object]) -> TrainingArguments:
        training_kwargs = {
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "num_train_epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "fp16": self.fp16,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dataloader_num_workers": self.dataloader_num_workers,
        }
        training_kwargs.update(self.training_args_overrides)
        training_kwargs.update(overrides)
        return TrainingArguments(output_dir=self.output_dir, **training_kwargs)

"""Compliance processing pipeline utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .data_labeler import LabelingPipeline
from .model_trainer import ModelTrainer


@dataclass
class CompliancePipelineConfig:
    """Configuration values used by :class:`CompliancePipeline`."""

    labeled_pdf_dir: str
    training_data_dir: str
    input_pdf_dir: str
    output_json_dir: str
    inference_model_dir: str
    trained_model_dir: Optional[str] = None


class CompliancePipeline:
    """Coordinate data extraction, training, and inference workflows."""

    def __init__(
        self,
        pdf_processor,
        labeling_pipeline: LabelingPipeline,
        trainer: ModelTrainer,
        config: CompliancePipelineConfig,
    ):
        self.pdf_processor = pdf_processor
        self.labeling_pipeline = labeling_pipeline
        self.trainer = trainer
        self.config = config

    @staticmethod
    def default_handler(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def label_doc(self, text: Iterable[dict]):
        unlabeled_data = []
        for element in text:
            unlabeled_data.append(
                {
                    "compliance_statement": element.get("compliance_statement"),
                    "compliance_data": None,
                    "date": None,
                    "business_entity": None,
                    "regulation": None,
                    "text": element.get("text"),
                }
            )
        return self.labeling_pipeline.label_data(unlabeled_data)

    def process_tdata(
        self,
        files: Sequence[str],
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        input_dir = input_path or self.config.labeled_pdf_dir
        output_dir = output_path or self.config.training_data_dir

        os.makedirs(output_dir, exist_ok=True)

        for entry in files:
            file_path = os.path.join(input_dir, entry)
            dataset_labeled = self.pdf_processor.to_training_data(file_path)

            training_dataset_path = os.path.join(output_dir, f"{entry}.json")
            with open(training_dataset_path, "w") as f:
                json.dump(dataset_labeled, f, indent=4, default=self.default_handler)

    def process_statement(
        self,
        files: Sequence[str],
        model_path: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        input_dir = input_path or self.config.input_pdf_dir
        output_dir = output_path or self.config.output_json_dir
        model_dir = model_path or self.config.inference_model_dir

        os.makedirs(output_dir, exist_ok=True)

        for entry in files:
            file_path = os.path.join(input_dir, entry)
            compliance_processed = self.pdf_processor.to_compliant_sentences(file_path, model_dir)

            output_file_path = os.path.join(output_dir, f"{entry}.json")
            with open(output_file_path, "w") as f:
                json.dump(compliance_processed, f, indent=4, default=self.default_handler)

    def build_training_data(self, files: Optional[Sequence[str]] = None):
        training_files = files or os.listdir(self.config.labeled_pdf_dir)
        self.process_tdata(training_files)

    def train(self, evaluate: bool = True, **trainer_kwargs):
        self.trainer.prepare_dataset()
        trainer = self.trainer.train(**trainer_kwargs)
        if evaluate:
            return self.trainer.evaluate()
        return trainer

    def run_inference(self, files: Optional[Sequence[str]] = None):
        inference_files = files or os.listdir(self.config.input_pdf_dir)
        self.process_statement(inference_files)

"""Pipeline orchestration utilities.

This module provides high-level helpers for running the end-to-end
processing pipeline. Each function requires explicit input and output paths
so that callers can control where data is read from and written to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from sources.pdf_processing import PdfProcessingFactory, pdf_to_tdata, pdf_to_text
from sources.data_labeler import label_data
from sources.model_trainer import train_model


def _json_serializer(value):
    """Serialize NumPy scalar values for JSON dumping."""

    if isinstance(value, np.generic):  # covers floating, integer, bool scalars
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iter_files(directory: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    lowered = tuple(ext.lower() for ext in suffixes)
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in lowered:
            yield item


def generate_training_data(
    input_pdf_dir: str,
    output_dir: str,
    *,
    processing_factory: Optional[PdfProcessingFactory] = None,
) -> List[Path]:
    """Generate labeled training data from PDFs.

    Args:
        input_pdf_dir: Directory containing labeled PDF documents.
        output_dir: Directory where generated training JSON files will be saved.
        processing_factory: Optional factory used to construct shared services.

    Returns:
        A list of paths to the generated JSON files.
    """

    input_dir = Path(input_pdf_dir)
    output_path = _ensure_directory(Path(output_dir))

    generated_files: List[Path] = []
    for pdf_path in _iter_files(input_dir, (".pdf",)):
        dataset_labeled = pdf_to_tdata(str(pdf_path), factory=processing_factory)
        target = output_path / f"{pdf_path.stem}.json"
        with target.open("w", encoding="utf-8") as fp:
            json.dump(dataset_labeled, fp, indent=4, default=_json_serializer)
        generated_files.append(target)

    return generated_files


def label_metadata(input_data_dir: str, output_dir: str) -> List[Path]:
    """Apply metadata labeling to generated training datasets.

    Args:
        input_data_dir: Directory containing JSON files to label.
        output_dir: Directory where labeled JSON files will be written.

    Returns:
        A list of paths to the labeled JSON files.
    """

    input_dir = Path(input_data_dir)
    output_path = _ensure_directory(Path(output_dir))

    labeled_files: List[Path] = []
    for json_path in _iter_files(input_dir, (".json",)):
        with json_path.open("r", encoding="utf-8") as fp:
            dataset = json.load(fp)

        unlabeled_data = [
            {
                "text": item.get("text", ""),
                "bounding_box": item.get("bounding_box"),
                "compliance_statement": item.get("compliance_statement"),
                "compliance_data": item.get("compliance_data"),
                "date": item.get("date"),
                "business_entity": item.get("business_entity"),
                "regulation": item.get("regulation"),
            }
            for item in dataset
        ]

        labeled_dataset = label_data(unlabeled_data)
        target = output_path / json_path.name
        with target.open("w", encoding="utf-8") as fp:
            json.dump(labeled_dataset, fp, indent=4, default=_json_serializer)
        labeled_files.append(target)

    return labeled_files


def train_classifier(
    training_data_dir: str,
    model_output_dir: str,
    *,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 32,
):
    """Train the classification model using the prepared dataset."""

    _ensure_directory(Path(model_output_dir))
    return train_model(
        training_data_dir,
        model_output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
    )


def run_inference(
    input_pdf_dir: str,
    output_dir: str,
    model_dir: str,
    *,
    processing_factory: Optional[PdfProcessingFactory] = None,
    batch_size: int = 32,
) -> List[Path]:
    """Run inference on PDFs and write classified statements to JSON files.

    Args:
        processing_factory: Optional factory used to configure shared services.
        batch_size: Batch size forwarded to the classifier service.
    """

    input_dir = Path(input_pdf_dir)
    output_path = _ensure_directory(Path(output_dir))

    inference_outputs: List[Path] = []
    for pdf_path in _iter_files(input_dir, (".pdf",)):
        compliance_processed = pdf_to_text(
            str(pdf_path),
            str(model_dir),
            factory=processing_factory,
            batch_size=batch_size,
        )
        target = output_path / f"{pdf_path.stem}.json"
        with target.open("w", encoding="utf-8") as fp:
            json.dump(compliance_processed, fp, indent=4, default=_json_serializer)
        inference_outputs.append(target)

    return inference_outputs


__all__ = [
    "generate_training_data",
    "label_metadata",
    "train_classifier",
    "run_inference",
]

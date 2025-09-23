"""Command-line interface for the compliance processing pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

import config
from sources.pipeline.orchestrator import (
    generate_training_data,
    label_metadata,
    run_inference,
    train_classifier,
)


def _print_paths(description: str, paths):
    if not paths:
        print(f"No files processed for {description}.")
        return

    print(f"{description}:")
    for path in paths:
        print(f" - {path}")


def _handle_generate_training_data(args):
    outputs = generate_training_data(args.input_dir, args.output_dir)
    _print_paths("Generated training datasets", outputs)


def _handle_label_metadata(args):
    outputs = label_metadata(args.input_dir, args.output_dir)
    _print_paths("Labeled datasets", outputs)


def _handle_train(args):
    train_classifier(
        args.data_dir,
        args.model_output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Model training complete. Artifacts saved to {args.model_output_dir}.")


def _handle_infer(args):
    outputs = run_inference(args.input_dir, args.output_dir, args.model_dir)
    _print_paths("Inference outputs", outputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run individual stages of the compliance processing pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate-training-data",
        help="Convert labeled PDFs into JSON training datasets.",
    )
    generate_parser.add_argument(
        "--input-dir",
        default=config.LABELED_PDF_DIR,
        help="Directory containing labeled PDF documents.",
    )
    generate_parser.add_argument(
        "--output-dir",
        default=config.TRAINING_DATA_DIR,
        help="Directory where generated training JSON files will be stored.",
    )
    generate_parser.set_defaults(func=_handle_generate_training_data)

    label_parser = subparsers.add_parser(
        "label-metadata",
        help="Augment training datasets with metadata labels.",
    )
    label_parser.add_argument(
        "--input-dir",
        default=config.TRAINING_DATA_DIR,
        help="Directory containing JSON training datasets to label.",
    )
    label_parser.add_argument(
        "--output-dir",
        default=config.TRAINING_DATA_DIR,
        help="Directory where labeled JSON files will be written.",
    )
    label_parser.set_defaults(func=_handle_label_metadata)

    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune the classifier using labeled training data.",
    )
    train_parser.add_argument(
        "--data-dir",
        default=config.TRAINING_DATA_DIR,
        help="Directory containing labeled training data JSON files.",
    )
    train_parser.add_argument(
        "--model-output-dir",
        default=config.TRAINED_MODEL_DIR,
        help="Directory where the trained model will be saved.",
    )
    train_parser.add_argument(
        "--model-name",
        default="nlpaueb/legal-bert-base-uncased",
        help="Base transformer model to fine-tune.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    train_parser.set_defaults(func=_handle_train)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on PDFs and export compliance statements to JSON.",
    )
    infer_parser.add_argument(
        "--input-dir",
        default=config.INPUT_PDF_DIR,
        help="Directory containing PDFs for inference.",
    )
    infer_parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_JSON_DIR,
        help="Directory where inference JSON files will be stored.",
    )
    infer_parser.add_argument(
        "--model-dir",
        default=config.USE_MODEL_DIR,
        help="Directory containing the trained model used for inference.",
    )
    infer_parser.set_defaults(func=_handle_infer)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

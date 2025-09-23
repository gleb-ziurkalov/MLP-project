An automated “Substance Compliance Processor” that ingests PDFs, extracts and classifies compliance-relevant information, optionally validates it against rules, and outputs structured reports—improving efficiency, accuracy, and scalability of regulatory workflows.

What it must do: extract/classify compliance statements and produce stakeholder-ready reports, with a path to automated rule/KB validation.
How well it should perform: ~95% accuracy, scale to ~1,000 docs/month, and meet GDPR/security expectations.
How it’s organized: a modular pipeline (Input → Processing → Data → Output) with OCR + transformer-based classification (e.g., PaddleOCR, LegalBERT) and reporting.
How it will be delivered: phased MLOps plan from system engineering through core development, integration/testing, deployment, and maintenance.
Optionally, the project also explores an interpretable alternative: learning regex extractors via genetic programming (with separate-and-conquer and context lookarounds) to target specific compliance snippets.

## Running the pipeline

The pipeline is now exposed through an argparse-based CLI, so you no longer need to edit `main.py` to toggle different stages. Each command accepts explicit input and output paths (defaults come from `config.py`).

```bash
# Generate training JSON datasets from labeled PDFs
python main.py generate-training-data \
  --input-dir ./data/labeled_pdfs \
  --output-dir ./data/training_data

# Add metadata labels to the generated datasets
python main.py label-metadata \
  --input-dir ./data/training_data \
  --output-dir ./data/training_data

# Fine-tune the classifier
python main.py train \
  --data-dir ./data/training_data \
  --model-output-dir ./models/trained_model

# Run inference on incoming PDFs
python main.py infer \
  --input-dir ./staging/input_pdf \
  --output-dir ./staging/output_JSON \
  --model-dir ./models/block_classification_oversampled
```

Override any of the paths as needed to point to alternative locations or experiments.

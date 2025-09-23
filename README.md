An automated “Substance Compliance Processor” that ingests PDFs, extracts and classifies compliance-relevant information, optionally validates it against rules, and outputs structured reports—improving efficiency, accuracy, and scalability of regulatory workflows.
- What it must do: extract/classify compliance statements and produce stakeholder-ready reports, with a path to automated rule/KB validation.
- How well it should perform: ~95% accuracy, scale to ~1,000 docs/month, and meet GDPR/security expectations.
- How it’s organized: a modular pipeline (Input → Processing → Data → Output) with OCR + transformer-based classification (e.g., PaddleOCR, LegalBERT) and reporting.
- How it will be delivered: phased MLOps plan from system engineering through core development, integration/testing, deployment, and maintenance.

Optionally, the project also explores an interpretable alternative: learning regex extractors via genetic programming (with separate-and-conquer and context lookarounds) to target specific compliance snippets.

import os

import config
from sources.compliance_processor import CompliancePipeline, CompliancePipelineConfig
from sources import pdf_processing
from sources import data_labeler
from sources import model_trainer


def main():
    pipeline_config = CompliancePipelineConfig(
        labeled_pdf_dir=config.LABELED_PDF_DIR,
        training_data_dir=config.TRAINING_DATA_DIR,
        input_pdf_dir=config.INPUT_PDF_DIR,
        output_json_dir=config.OUTPUT_JSON_DIR,
        inference_model_dir=config.USE_MODEL_DIR,
        trained_model_dir=config.TRAINED_MODEL_DIR,
    )

    pipeline = CompliancePipeline(pdf_processing, data_labeler, model_trainer, pipeline_config)

    training_files = [
        entry for entry in os.listdir(pipeline_config.labeled_pdf_dir)
        if os.path.isfile(os.path.join(pipeline_config.labeled_pdf_dir, entry))
    ]
    pipeline.build_training_data(training_files)
    # pipeline.train()

    input_files = [
        entry for entry in os.listdir(pipeline_config.input_pdf_dir)
        if os.path.isfile(os.path.join(pipeline_config.input_pdf_dir, entry))
    ]
    # pipeline.run_inference(input_files)


if __name__ == "__main__":
    main()

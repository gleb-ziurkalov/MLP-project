import os

import config
from sources.compliance_processor import CompliancePipeline, CompliancePipelineConfig
from sources.data_labeler import DateLabeler, EntityLabeler, LabelingPipeline, PartLabeler, RegulationLabeler
from sources.pdf_processing import PDFProcessor
from sources.model_trainer import ModelTrainer


def build_labeling_pipeline() -> LabelingPipeline:
    labelers = [
        DateLabeler(),
        EntityLabeler(confidence_threshold=0.85),
        RegulationLabeler(),
        PartLabeler(),
    ]
    return LabelingPipeline(labelers)


def main():
    pipeline_config = CompliancePipelineConfig(
        labeled_pdf_dir=config.LABELED_PDF_DIR,
        training_data_dir=config.TRAINING_DATA_DIR,
        input_pdf_dir=config.INPUT_PDF_DIR,
        output_json_dir=config.OUTPUT_JSON_DIR,
        inference_model_dir=config.USE_MODEL_DIR,
        trained_model_dir=config.TRAINED_MODEL_DIR,
    )

    pdf_processor = PDFProcessor()
    labeling_pipeline = build_labeling_pipeline()
    trainer = ModelTrainer(
        training_data_dir=pipeline_config.training_data_dir,
        output_dir=pipeline_config.trained_model_dir or pipeline_config.inference_model_dir,
    )

    pipeline = CompliancePipeline(pdf_processor, labeling_pipeline, trainer, pipeline_config)

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

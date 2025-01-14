# utils/__init__.py
from .dates_labeler import label_dates
from .entities_labeler import label_entities_huggingface as label_entities
from .regulation_labeler import label_regulations
from .vparts_labeler import label_vparts


# from .extract_regulations import extract_regulations
# from .extract_vparts import extract_vparts

# Metadata Extraction
def label_data(dataset):
    dataset_labeled = []

    label_dates(dataset)
    label_entities(dataset)
    label_regulations(dataset)
    # label_parts(dataset)

    for line in dataset:
        # print(line)
        if line["compliance_data"] == None:
            line["compliance_data"] = 0
        dataset_labeled.append(line)

    return dataset_labeled 
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "cv2" not in sys.modules:
    cv2_stub = types.SimpleNamespace(
        cvtColor=lambda *args, **kwargs: None,
        COLOR_RGB2HSV=0,
        inRange=lambda *args, **kwargs: None,
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
        findContours=lambda *args, **kwargs: ([], None),
        boundingRect=lambda contour: (0, 0, 0, 0),
    )
    sys.modules["cv2"] = cv2_stub

from sources.pdf_processing import pdf_to_text
from sources.pdf_processing.text_extractor import image_to_text


class DummyBundle:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def classify(self, sentences, batch_size=32):
        self.calls.append((list(sentences), batch_size))
        return list(zip(sentences, range(len(sentences))))


def test_image_to_text_returns_structured_results(monkeypatch):
    pages = [np.zeros((1, 1))]

    def fake_extract_bbox(image, ocr_service):
        return [("sentence one", []), ("sentence two", [])]

    def fake_segment_sentences(text_boxes, segmenter):
        return [{"text": text} for text, _ in text_boxes]

    monkeypatch.setattr(
        "sources.pdf_processing.text_extractor.extract_bbox",
        fake_extract_bbox,
    )
    monkeypatch.setattr(
        "sources.pdf_processing.text_extractor.segment_sentences",
        fake_segment_sentences,
    )

    bundle = DummyBundle("model-a")
    results = image_to_text(
        pages,
        bundle,
        ocr_service=object(),
        sentence_segmenter=object(),
        batch_size=4,
    )

    assert results == [
        {"text": "sentence one", "label": 0},
        {"text": "sentence two", "label": 1},
    ]
    assert bundle.calls == [(["sentence one", "sentence two"], 4)]

    # verify callable sources are supported
    bundle_callable = DummyBundle("model-b")

    def factory():
        return bundle_callable

    results_callable = image_to_text(
        pages,
        factory,
        ocr_service=object(),
        sentence_segmenter=object(),
        batch_size=2,
    )

    assert results_callable == [
        {"text": "sentence one", "label": 0},
        {"text": "sentence two", "label": 1},
    ]
    assert bundle_callable.calls == [(["sentence one", "sentence two"], 2)]


def test_pdf_to_text_uses_requested_model_path(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class DummyFactory:
        def __init__(self):
            self.requests = []

        def get_classifier_bundle(self, model_path, **overrides):
            bundle = DummyBundle(model_path)
            self.requests.append((model_path, overrides))
            return bundle

        def get_ocr_service(self):
            return object()

        def get_sentence_segmenter(self):
            return object()

    factory = DummyFactory()

    def fake_pdf_to_image(path, dpi=300):  # noqa: D401 - simple stub
        assert Path(path) == pdf_path
        return [np.zeros((1, 1))]

    def fake_image_to_text(pages, classifier_source, **kwargs):
        if callable(classifier_source):
            bundle = classifier_source()
        else:
            bundle = classifier_source
        return [{"text": f"{bundle.name}-text", "label": 1}]

    import sources.pdf_processing as pdf_module

    monkeypatch.setattr(pdf_module, "pdf_to_image", fake_pdf_to_image)
    monkeypatch.setattr(pdf_module, "image_to_text", fake_image_to_text)

    result_a = pdf_to_text(str(pdf_path), "model-a", factory=factory, batch_size=8)
    result_b = pdf_to_text(str(pdf_path), "model-b", factory=factory)

    assert factory.requests == [("model-a", {}), ("model-b", {})]
    assert result_a == [{"text": "model-a-text", "label": 1}]
    assert result_b == [{"text": "model-b-text", "label": 1}]

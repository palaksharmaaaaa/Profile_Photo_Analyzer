from PIL import Image
from unittest.mock import patch
from app.services.classifier_service import (
    friendly_image_type,
    classify_image_type,
)


def test_friendly_image_type():
    assert friendly_image_type("a real photograph of a real human person") == "Real Photo"
    assert friendly_image_type("an anime or Ghibli-style illustration") == "Anime / Ghibli-style Image"
    assert friendly_image_type("a cartoon drawing of a person") == "Cartoon Image"
    assert friendly_image_type("a digital painting of a fictional character") == "Digital Painting (Fictional Character)"
    assert friendly_image_type("a computer-generated 3D character") == "3D / Computer-generated Character"
    assert friendly_image_type("a photograph or picture of an object with no person") == "Object / No Person"
    assert friendly_image_type("unknown random label") == "Uncertain"
    assert friendly_image_type("") == "Uncertain"


def test_classify_image_type_fallback():
    # When CLIP pipeline returns None, fallback defaults to PHOTO_LIKE gracefully
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=None):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "PHOTO_LIKE"
        assert result["top_class_confidence"] > 0.5


def test_classify_image_type_clip_photo():
    mock_predictions = [
        {"label": "a real photograph of a real human person", "score": 0.92},
        {"label": "a cartoon drawing of a person", "score": 0.08}
    ]
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: mock_predictions):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "PHOTO_LIKE"
        assert result["photo_score"] == 0.92
        assert result["display_label"] == "Real Photo"


def test_classify_image_type_clip_cartoon():
    mock_predictions = [
        {"label": "a cartoon drawing of a person", "score": 0.88},
        {"label": "a real photograph of a real human person", "score": 0.12}
    ]
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: mock_predictions):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["cartoon_score"] == 0.88
        assert result["display_label"] == "Cartoon Image"


def test_classify_image_type_clip_object():
    mock_predictions = [
        {"label": "a photograph or picture of an object with no person", "score": 0.95},
        {"label": "a real photograph of a real human person", "score": 0.05}
    ]
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: mock_predictions):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["display_label"] == "Object / No Person"


def test_classify_image_type_error_handling():
    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("Inference engine failure")

    with patch("app.services.classifier_service.get_clip_pipeline", return_value=failing_pipeline):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"
        assert "failed" in result["reason"].lower()


def test_get_clip_pipeline_success():
    import app.services.classifier_service as cs
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=MagicMock(return_value=mock_pipeline))}):
        cs._clip_pipeline = None
        pipeline = cs.get_clip_pipeline()
        assert pipeline == mock_pipeline
        cs._clip_pipeline = None


def test_classify_image_type_empty_predictions():
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: []):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"


def test_classify_image_type_invalid_format():
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: [{"score": 0.5, "invalid": 1}]):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"


def test_classify_image_type_needs_review():
    mock_preds = [
        {"label": "an abstract background pattern", "score": 0.99}
    ]
    with patch("app.services.classifier_service.get_clip_pipeline", return_value=lambda img, candidate_labels: mock_preds):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NEEDS_REVIEW"
        assert "confidently" in result["reason"].lower()


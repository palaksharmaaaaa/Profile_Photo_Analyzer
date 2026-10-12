from unittest.mock import patch

import numpy as np
from PIL import Image

from app.services.classifier_service import (
    classify_image_type,
    friendly_image_type,
)


def test_friendly_image_type():
    assert friendly_image_type("a real photograph of a real human person") == "Real Photo"
    assert friendly_image_type("an anime or Ghibli-style illustration") == "Anime / Ghibli-style Image"
    assert friendly_image_type("a cartoon drawing of a person") == "Cartoon Image"
    assert (
        friendly_image_type("a digital painting of a fictional character") == "Digital Painting (Fictional Character)"
    )
    assert friendly_image_type("a computer-generated 3D character") == "3D / Computer-generated Character"
    assert friendly_image_type("a photograph or picture of an object with no person") == "Object / No Person"
    assert friendly_image_type("unknown random label") == "Uncertain"
    assert friendly_image_type("") == "Uncertain"


def test_classify_image_type_fallback():
    # When CLIP pipeline returns None, unverified fallback defaults to NEEDS_REVIEW
    with (
        patch("app.services.classifier_service.get_clip_pipeline", return_value=None),
        patch("app.services.classifier_service.is_biological_diagram_or_drawing", return_value=False),
        patch("app.services.classifier_service.is_cartoon_or_illustration", return_value=False),
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NEEDS_REVIEW"
        assert result["display_label"] == "Heuristic Screening (Review Required)"


def test_classify_image_type_clip_photo():
    mock_predictions = [
        {"label": "a real photograph of a real human person", "score": 0.92},
        {"label": "a cartoon drawing of a person", "score": 0.08},
    ]
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: mock_predictions,
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "PHOTO_LIKE"
        assert result["photo_score"] == 0.92
        assert result["display_label"] == "Real Photo"


def test_classify_image_type_clip_cartoon():
    mock_predictions = [
        {"label": "a cartoon drawing of a person", "score": 0.88},
        {"label": "a real photograph of a real human person", "score": 0.12},
    ]
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: mock_predictions,
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["cartoon_score"] == 0.88
        assert result["display_label"] == "Cartoon Image"


def test_classify_image_type_clip_object():
    mock_predictions = [
        {"label": "a photograph or picture of an object with no person", "score": 0.95},
        {"label": "a real photograph of a real human person", "score": 0.05},
    ]
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: mock_predictions,
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["display_label"] == "Object / No Person"


def test_classify_image_type_error_handling():
    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("Inference engine failure")

    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=failing_pipeline,
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"
        assert "failed" in result["reason"].lower()


def test_get_clip_pipeline_success():
    from unittest.mock import MagicMock

    import app.services.classifier_service as cs

    mock_pipeline = MagicMock()
    with patch.dict(
        "sys.modules",
        {"transformers": MagicMock(pipeline=MagicMock(return_value=mock_pipeline))},
    ):
        cs._clip_pipeline = None
        pipeline = cs.get_clip_pipeline()
        assert pipeline == mock_pipeline
        cs._clip_pipeline = None


def test_classify_image_type_empty_predictions():
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: [],
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"


def test_classify_image_type_invalid_format():
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: [{"score": 0.5, "invalid": 1}],
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "UNCERTAIN"


def test_classify_image_type_needs_review():
    mock_preds = [{"label": "an abstract background pattern", "score": 0.99}]
    with patch(
        "app.services.classifier_service.get_clip_pipeline",
        return_value=lambda img, candidate_labels: mock_preds,
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NEEDS_REVIEW"
        assert "confidently" in result["reason"].lower()


def test_friendly_image_type_diagram():
    assert friendly_image_type("a biological diagram of human anatomy") == "Biological / Scientific Diagram"


def test_is_biological_diagram_or_drawing():

    from app.services.classifier_service import is_biological_diagram_or_drawing

    assert is_biological_diagram_or_drawing(None) is False
    assert is_biological_diagram_or_drawing(np.zeros((5, 5, 3), dtype=np.uint8)) is False

    # White background with sparse black lines
    diagram = np.full((100, 100, 3), 250, dtype=np.uint8)
    diagram[40:60, 40:60] = 10  # Dark square line in center
    assert is_biological_diagram_or_drawing(diagram) is True

    # Real skin-colored patch (YCrCb skin tones)
    skin_patch = np.full((100, 100, 3), (120, 150, 200), dtype=np.uint8)
    assert is_biological_diagram_or_drawing(skin_patch) is False


def test_decode_yolo_predictions_edges():

    from app.services.classifier_service import _decode_yolo_predictions

    assert _decode_yolo_predictions(None, 640, 640, 0.35) == []
    assert _decode_yolo_predictions(np.zeros((640, 640), dtype=np.float32), 640, 640, 0.35) == []

    # Below threshold -> no detections
    tensor_empty = np.zeros((1, 84, 8400), dtype=np.float32)
    assert _decode_yolo_predictions(tensor_empty, 640, 640, 0.35) == []

    # NMS returns empty array
    with patch("app.services.classifier_service._apply_nms", return_value=np.array([])):
        tensor = np.zeros((1, 5, 8400), dtype=np.float32)
        tensor[0, 4, 0] = 0.90
        assert _decode_yolo_predictions(tensor, 640, 640, 0.35, is_anime=True) == []


def test_is_biological_diagram_edge_ratio_branch():

    from app.services.classifier_service import is_biological_diagram_or_drawing

    diag = np.zeros((100, 100, 3), dtype=np.uint8)
    diag[:68, :] = 250
    diag[:68:8, :] = 0
    assert is_biological_diagram_or_drawing(diag) is True


def test_decode_yolo_predictions_anime():

    from app.services.classifier_service import _decode_yolo_predictions

    # Anime output: shape (1, 5, 8400)
    tensor = np.zeros((1, 5, 8400), dtype=np.float32)
    # Put 1 anchor box with high confidence
    tensor[0, 0, 0] = 320.0  # cx
    tensor[0, 1, 0] = 320.0  # cy
    tensor[0, 2, 0] = 100.0  # w
    tensor[0, 3, 0] = 100.0  # h
    tensor[0, 4, 0] = 0.90  # conf

    res = _decode_yolo_predictions(tensor, 640, 640, 0.35, is_anime=True)
    assert len(res) == 1
    assert res[0]["face_type"] == "cartoon_or_illustration"
    assert res[0]["confidence"] == 0.90


def test_decode_yolo_predictions_animal():

    from app.services.classifier_service import _decode_yolo_predictions

    # COCO output: shape (1, 84, 8400)
    tensor = np.zeros((1, 84, 8400), dtype=np.float32)
    tensor[0, 0, 0] = 200.0  # cx
    tensor[0, 1, 0] = 200.0  # cy
    tensor[0, 2, 0] = 80.0  # w
    tensor[0, 3, 0] = 80.0  # h
    # Class 16 = dog (index 4 + 16 = 20)
    tensor[0, 20, 0] = 0.88

    res = _decode_yolo_predictions(tensor, 640, 640, 0.35, is_anime=False)
    assert len(res) == 1
    assert res[0]["face_type"] == "animal"
    assert res[0]["label"] == "Dog"


def test_detect_non_human_entities():
    from unittest.mock import MagicMock

    from app.services.classifier_service import detect_non_human_entities

    assert detect_non_human_entities(None) == []
    assert detect_non_human_entities(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    img = np.ones((100, 100, 3), dtype=np.uint8) * 100

    # With sessions returning None
    with (
        patch("app.services.classifier_service.get_yolo_session", return_value=None),
        patch("app.services.classifier_service.get_anime_face_session", return_value=None),
    ):
        assert detect_non_human_entities(img) == []

    # With sessions raising exception
    mock_failing_session = MagicMock()
    mock_failing_session.run.side_effect = RuntimeError("YOLO error")
    with (
        patch(
            "app.services.classifier_service.get_yolo_session",
            return_value=mock_failing_session,
        ),
        patch(
            "app.services.classifier_service.get_anime_face_session",
            return_value=mock_failing_session,
        ),
    ):
        assert detect_non_human_entities(img) == []


def test_check_is_non_human_face():

    from app.services.classifier_service import check_is_non_human_face

    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    entities = [
        {
            "face_type": "animal",
            "label": "Cat",
            "confidence": 0.85,
            "box_xyxy": [50, 50, 150, 150],
        }
    ]

    # Face overlapping cat box
    face_box = [60, 60, 140, 140]
    match = check_is_non_human_face(face_box, img, entities)
    assert match is not None
    assert match["face_type"] == "animal"

    # Face not overlapping
    face_box_away = [10, 10, 30, 30]
    no_match = check_is_non_human_face(face_box_away, img, entities)
    assert no_match is None

    # Face on diagram crop
    diagram_img = np.full((200, 200, 3), 250, dtype=np.uint8)
    diagram_img[40:60, 40:60] = 5
    diag_box = [30, 30, 70, 70]
    match_diag = check_is_non_human_face(diag_box, diagram_img, [])
    assert match_diag is not None
    assert match_diag["face_type"] == "biological_diagram"


def test_classify_image_type_diagram_fallback():
    # When CLIP pipeline returns None and image is a diagram
    with (
        patch("app.services.classifier_service.get_clip_pipeline", return_value=None),
        patch(
            "app.services.classifier_service.is_biological_diagram_or_drawing",
            return_value=True,
        ),
    ):
        img = Image.new("RGB", (100, 100), color=(250, 250, 250))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["top_class"] == "a biological or scientific diagram"
        assert result["display_label"] == "Biological / Scientific Diagram"


def test_classify_image_type_cartoon_fallback():
    # When CLIP pipeline returns None and image has cartoon traits
    with (
        patch("app.services.classifier_service.get_clip_pipeline", return_value=None),
        patch("app.services.classifier_service.is_biological_diagram_or_drawing", return_value=False),
        patch("app.services.classifier_service.is_cartoon_or_illustration", return_value=True),
    ):
        img = Image.new("RGB", (100, 100))
        result = classify_image_type(img)
        assert result["status"] == "NON_PHOTO"
        assert result["top_class"] == "a cartoon drawing of a person"
        assert result["display_label"] == "Cartoon / Illustrated Image"


def test_is_cartoon_or_illustration():

    from app.services.classifier_service import is_cartoon_or_illustration

    # Empty image
    assert is_cartoon_or_illustration(np.zeros((0, 0, 3), dtype=np.uint8)) is False

    # Flat solid color (cartoon posterized palette)
    flat = np.full((128, 128, 3), 180, dtype=np.uint8)
    assert is_cartoon_or_illustration(flat) is True

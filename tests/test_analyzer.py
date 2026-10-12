from unittest.mock import patch
from PIL import Image
import numpy as np

from app.services.analyzer import analyze_profile_photo_image
from tests.conftest import MockFace


def test_analyzer_no_faces(test_image_rgb):
    with patch("app.services.analyzer.detect_faces", return_value=[]), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["face_detected"] is False
        assert result["face_count"] == 0
        assert "no human face was detected" in result["reason"].lower()
        assert result["posting_status"] == "REJECTED"


def test_analyzer_no_information():
    # Pure blank white image
    blank_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    result = analyze_profile_photo_image(blank_img)
    assert result["accepted"] is False
    assert result["has_information"] is False
    assert result["posting_status"] == "NO_INFORMATION"
    assert "no information" in result["reason"].lower()


def test_analyzer_other_faces_only(test_image_rgb):
    # Zero human faces, but 1 animal detected
    mock_animal = [{
        "face_type": "animal",
        "label": "Dog",
        "confidence": 0.92,
        "box_xyxy": [50, 50, 200, 200]
    }]
    with patch("app.services.analyzer.detect_faces", return_value=[]), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=mock_animal):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["real_human_face_count"] == 0
        assert result["other_face_count"] == 1
        assert result["posting_status"] == "NON_PHOTO_WARNING"
        assert "dog" in result["reason"].lower()


def test_analyzer_multiple_faces(test_image_rgb):
    faces = [
        MockFace(bbox=[50, 50, 150, 150]),
        MockFace(bbox=[200, 200, 300, 300])
    ]
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["face_count"] == 2
        assert "exactly one" in result["reason"].lower()


def test_analyzer_single_face_approved(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350], age=27.0, gender=0)]
    mock_vis = {
        "status": "LIKELY_COMPLETE",
        "reason": "Complete face",
        "face_box_touches_image_edge": False,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is True
        assert result["posting_status"] == "INITIAL_CHECKS_PASSED"
        assert result["face_count"] == 1
        assert result["estimated_age_group"] == "27 to 30"
        assert result["gender_presentation_prediction"] == "female-presentation prediction"
        assert result["original_image"].startswith("data:image/jpeg;base64,")
        assert result["annotated_image"].startswith("data:image/jpeg;base64,")


def test_analyzer_human_and_other_faces(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350], age=27.0, gender=0)]
    mock_animal = [{
        "face_type": "animal",
        "label": "Cat",
        "confidence": 0.88,
        "box_xyxy": [10, 10, 80, 80]
    }]
    mock_vis = {
        "status": "LIKELY_COMPLETE",
        "reason": "Complete face",
        "face_box_touches_image_edge": False,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=mock_animal), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "NON_PHOTO_WARNING"
        assert result["other_face_count"] == 1
        assert "cat" in result["reason"].lower()


def test_analyzer_non_photo_warning(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350])]
    mock_type = {
        "status": "NON_PHOTO",
        "top_class": "a cartoon drawing of a person",
        "display_label": "Cartoon Image"
    }
    with patch("app.services.analyzer.classify_image_type", return_value=mock_type), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.detect_faces", return_value=faces):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "NON_PHOTO_WARNING"
        assert "cartoon" in result["posting_reason"].lower()


def test_analyzer_minor_risk(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350], age=15.0)]
    mock_vis = {
        "status": "LIKELY_COMPLETE",
        "reason": "Complete face",
        "face_box_touches_image_edge": False,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "AGE_VERIFICATION_REQUIRED"
        assert result["possible_minor_risk"] is True
        assert result["age_verification_required"] is True


def test_analyzer_poor_visibility(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350])]
    mock_vis = {
        "status": "PARTIAL_OR_CROPPED",
        "reason": "The face box is too close to an image edge.",
        "face_box_touches_image_edge": True,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "REJECTED_OR_REVIEW"


def test_analyzer_attribute_extraction_exception(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350])]
    mock_vis = {
        "status": "LIKELY_COMPLETE",
        "reason": "Complete face",
        "face_box_touches_image_edge": False,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis), \
         patch("app.services.analyzer.predict_face_demographics", side_effect=RuntimeError("Extraction failed")):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "AGE_ESTIMATION_UNAVAILABLE"


def test_analyzer_age_not_available(test_image_rgb):
    faces = [MockFace(bbox=[150, 150, 350, 350])]
    mock_vis = {
        "status": "LIKELY_COMPLETE",
        "reason": "Complete face",
        "face_box_touches_image_edge": False,
        "sharpness_score": 85.0,
        "landmarks_available": True,
        "landmarks_inside_image": True,
    }
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=[]), \
         patch("app.services.analyzer.assess_face_quality", return_value=mock_vis), \
         patch("app.services.analyzer.predict_face_demographics", return_value=("Not available", None, None, "male-presentation prediction", None)):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["posting_status"] == "AGE_ESTIMATION_UNAVAILABLE"


def test_analyzer_face_matches_non_human_entity(test_image_rgb):
    faces = [MockFace(bbox=[50, 50, 150, 150])]
    mock_anime = [{
        "face_type": "cartoon_or_illustration",
        "label": "Anime Character",
        "confidence": 0.89,
        "box_xyxy": [50, 50, 150, 150]
    }]
    with patch("app.services.analyzer.detect_faces", return_value=faces), \
         patch("app.services.analyzer.detect_non_human_entities", return_value=mock_anime):
        result = analyze_profile_photo_image(test_image_rgb)
        assert result["accepted"] is False
        assert result["real_human_face_count"] == 0
        assert result["other_face_count"] == 1
        assert result["posting_status"] == "NON_PHOTO_WARNING"


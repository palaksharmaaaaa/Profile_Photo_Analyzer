import numpy as np
from unittest.mock import patch, MagicMock

from app.services.face_service import (
    age_to_group,
    age_to_three_year_group,
    possible_minor_age_group,
    extract_face_attributes,
    detect_faces,
    predict_face_demographics,
    get_face_app,
)
from tests.conftest import MockFace


def test_age_to_group():
    assert age_to_group(1.5) == "0 to 3"
    assert age_to_group(4.0) == "3 to 6"
    assert age_to_group(7.0) == "6 to 9"
    assert age_to_group(10.0) == "9 to 12"
    assert age_to_group(13.0) == "12 to 15"
    assert age_to_group(16.0) == "15 to 18"
    assert age_to_group(19.0) == "18 to 21"
    assert age_to_group(25.0) == "24 to 27"
    assert age_to_group(35.0) == "33 to 36"
    assert age_to_group(45.0) == "45 to 48"
    assert age_to_group(55.0) == "54 to 57"
    assert age_to_group(65.0) == "63 to 66"
    assert age_to_group(75.0) == "75 to 78"
    assert age_to_group(-5.0) == "0 to 3"


def test_possible_minor_age_group():
    assert possible_minor_age_group("0 to 3") is True
    assert possible_minor_age_group("3 to 6") is True
    assert possible_minor_age_group("15 to 18") is True
    assert possible_minor_age_group("minor") is True
    assert possible_minor_age_group("child") is True
    assert possible_minor_age_group("under 18") is True
    assert possible_minor_age_group("12-16") is True
    assert possible_minor_age_group("14") is True

    # Adults
    assert possible_minor_age_group("18 to 21") is False
    assert possible_minor_age_group("21 to 24") is False
    assert possible_minor_age_group("30 to 33") is False
    assert possible_minor_age_group("25") is False
    assert possible_minor_age_group("Adult") is False
    assert possible_minor_age_group("Not Detected") is False


def test_extract_face_attributes():
    face_female = MockFace(age=24.0, gender=0)
    age_grp, conf, gender_str = extract_face_attributes(face_female)
    assert age_grp == "24 to 27"
    assert conf == 0.85
    assert gender_str == "female-presentation prediction"

    face_male = MockFace(age=42.0, gender=1)
    age_grp, conf, gender_str = extract_face_attributes(face_male)
    assert age_grp == "42 to 45"
    assert gender_str == "male-presentation prediction"

    # Face without attributes
    class BareFace:
        pass

    bare_face = BareFace()
    age_grp, conf, gender_str = extract_face_attributes(bare_face)
    assert age_grp == "Not available"
    assert conf is None
    assert gender_str == "Not available"


def test_detect_faces_empty_image():
    assert detect_faces(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert detect_faces(None) == []


def test_detect_faces_yunet_none():
    with patch("app.services.face_service.get_face_app", return_value=None), \
         patch("app.services.face_service.get_yunet_detector", return_value=None):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detect_faces(arr) == []


def test_detect_faces_yunet_execution():
    mock_detector = MagicMock()
    # Mock YuNet output: 1 face row with 15 values
    mock_face_row = np.array([
        [10.0, 20.0, 50.0, 60.0, 25.0, 35.0, 45.0, 35.0, 35.0, 50.0, 25.0, 65.0, 45.0, 65.0, 0.95]
    ], dtype=np.float32)
    mock_detector.detect.return_value = (1, mock_face_row)

    with patch("app.services.face_service.get_face_app", return_value=None), \
         patch("app.services.face_service.get_yunet_detector", return_value=mock_detector):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        faces = detect_faces(img)
        assert len(faces) == 1
        assert round(faces[0].det_score, 2) == 0.95
        assert len(faces[0].bbox) == 4


def test_get_face_app_failure():
    import app.services.face_service as fs
    fs._face_app = None
    with patch.dict("sys.modules", {"insightface": MagicMock(), "insightface.app": MagicMock(FaceAnalysis=MagicMock(side_effect=RuntimeError("Init error")))}):
        assert fs.get_face_app() is None
        fs._face_app = None


def test_detect_faces_yunet_no_faces():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = (0, None)

    with patch("app.services.face_service.get_face_app", return_value=None), \
         patch("app.services.face_service.get_yunet_detector", return_value=mock_detector):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        assert detect_faces(img) == []


def test_detect_faces_yunet_exception():
    mock_detector = MagicMock()
    mock_detector.detect.side_effect = RuntimeError("Detector failed")

    with patch("app.services.face_service.get_face_app", return_value=None), \
         patch("app.services.face_service.get_yunet_detector", return_value=mock_detector):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        assert detect_faces(img) == []


def test_detect_faces_face_app_exception():
    mock_app = MagicMock()
    mock_app.get.side_effect = RuntimeError("Face app failed")
    with patch("app.services.face_service.get_face_app", return_value=mock_app), \
         patch("app.services.face_service.get_yunet_detector", return_value=None):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        assert detect_faces(img) == []


def test_get_face_app_success():
    import app.services.face_service as fs
    mock_fa_cls = MagicMock()
    mock_fa_instance = MagicMock()
    mock_fa_cls.return_value = mock_fa_instance
    with patch.dict("sys.modules", {"insightface": MagicMock(), "insightface.app": MagicMock(FaceAnalysis=mock_fa_cls)}):
        fs._face_app = None
        app_inst = fs.get_face_app()
        assert app_inst == mock_fa_instance
        mock_fa_instance.prepare.assert_called_once()
        fs._face_app = None


def test_predict_face_demographics_with_face_obj():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face_obj = MockFace(age=28.0, gender=1)
    age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 50, 50], face_obj)
    assert age_grp == "27 to 30"
    assert est_age == 28.0
    assert gender_str == "male-presentation prediction"


def test_predict_face_demographics_bare_face_obj():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    class Bare:
        det_score = 0.90
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "data"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[2.5, -1.0, 0.24]], dtype=np.float32)]

    with patch("app.services.face_service.get_genderage_session", return_value=mock_session):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 50, 50], Bare())
        assert age_grp == "24 to 27"
        assert est_age == 24.0
        assert gender_str == "female-presentation prediction"
        assert age_conf is not None
        assert gender_conf is not None


def test_predict_face_demographics_precomputed_face_obj():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    class Precomputed:
        age = 22.0
        gender = 0
    age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 50, 50], Precomputed())
    assert age_grp == "21 to 24"
    assert est_age == 22.0
    assert gender_str == "female-presentation prediction"


def test_predict_face_demographics_invalid_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Zero width/height box
    age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 10, 10], None)
    assert age_grp == "Not Detected"


def test_predict_face_demographics_empty_crop():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Coordinates completely out of bounds
    age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [200, 200, 300, 300], None)
    assert age_grp == "Not Detected"


def test_predict_face_demographics_session_none():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    with patch("app.services.face_service.get_genderage_session", return_value=None):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 80, 80], None)
        assert age_grp == "Not Detected"


def test_predict_face_demographics_onnx_success_male():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "data"
    mock_session.get_inputs.return_value = [mock_input]
    # gender logits [female, male, age_normalized]
    # male logit higher, age 0.28 (28 years old)
    mock_session.run.return_value = [np.array([[-1.5, 2.5, 0.28]], dtype=np.float32)]

    with patch("app.services.face_service.get_genderage_session", return_value=mock_session):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 80, 80], None)
        assert age_grp == "27 to 30"
        assert est_age == 28.0
        assert gender_str == "male-presentation prediction"


def test_predict_face_demographics_onnx_success_female():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "data"
    mock_session.get_inputs.return_value = [mock_input]
    # female logit higher, age 0.22 (22 years old)
    mock_session.run.return_value = [np.array([[3.0, -1.0, 0.22]], dtype=np.float32)]

    with patch("app.services.face_service.get_genderage_session", return_value=mock_session):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 80, 80], None)
        assert age_grp == "21 to 24"
        assert est_age == 22.0
        assert gender_str == "female-presentation prediction"


def test_predict_face_demographics_onnx_unsure_gender_and_age():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "data"
    mock_session.get_inputs.return_value = [mock_input]
    # Equal logits (p=0.5 < 0.6 threshold), age out of range (150 years)
    mock_session.run.return_value = [np.array([[0.0, 0.0, 1.50]], dtype=np.float32)]

    with patch("app.services.face_service.get_genderage_session", return_value=mock_session):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 80, 80], None)
        assert age_grp == "Unsure"
        assert gender_str == "Unsure"


def test_predict_face_demographics_onnx_exception():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("Session crash")

    with patch("app.services.face_service.get_genderage_session", return_value=mock_session):
        age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(img, [10, 10, 80, 80], None)
        assert age_grp == "Not Detected"


def test_sort_faces_reading_order():
    from app.services.face_service import sort_faces_reading_order, DetectedFace

    assert sort_faces_reading_order([]) == []

    # Three faces:
    # Face A: at y=200, x=50 (second row)
    # Face B: at y=30, x=150 (first row, right)
    # Face C: at y=25, x=40  (first row, left)
    face_a = DetectedFace(bbox=[50, 200, 90, 240])
    face_b = DetectedFace(bbox=[150, 30, 190, 70])
    face_c = DetectedFace(bbox=[40, 25, 80, 65])

    sorted_faces = sort_faces_reading_order([face_a, face_b, face_c], row_threshold_px=50)
    # In reading order: row 1 left (C), row 1 right (B), then row 2 (A)
    assert sorted_faces == [face_c, face_b, face_a]

    # Test with dict-like face objects
    dict_faces = [
        {"bbox": [100, 120, 150, 170]},
        {"bbox": [20, 20, 60, 60]}
    ]
    sorted_dicts = sort_faces_reading_order(dict_faces, row_threshold_px=50)
    assert sorted_dicts[0]["bbox"] == [20, 20, 60, 60]

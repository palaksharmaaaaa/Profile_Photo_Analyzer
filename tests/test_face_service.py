from app.services.face_service import (
    age_to_group,
    possible_minor_age_group,
    extract_face_attributes,
    detect_faces,
)
from tests.conftest import MockFace


def test_age_to_group():
    assert age_to_group(1.5) == "0-2"
    assert age_to_group(6.0) == "3-9"
    assert age_to_group(15.0) == "10-19"
    assert age_to_group(25.0) == "20-29"
    assert age_to_group(35.0) == "30-39"
    assert age_to_group(45.0) == "40-49"
    assert age_to_group(55.0) == "50-59"
    assert age_to_group(65.0) == "60-69"
    assert age_to_group(75.0) == "70+"


def test_possible_minor_age_group():
    assert possible_minor_age_group("0-2") is True
    assert possible_minor_age_group("3-9") is True
    assert possible_minor_age_group("10-19") is True
    assert possible_minor_age_group("minor") is True
    assert possible_minor_age_group("child") is True
    assert possible_minor_age_group("under 18") is True
    assert possible_minor_age_group("12-16") is True
    assert possible_minor_age_group("14") is True

    # Adults
    assert possible_minor_age_group("20-29") is False
    assert possible_minor_age_group("30-39") is False
    assert possible_minor_age_group("40-49") is False
    assert possible_minor_age_group("25") is False
    assert possible_minor_age_group("Adult") is False


def test_extract_face_attributes():
    face_female = MockFace(age=24.0, gender=0)
    age_grp, conf, gender_str = extract_face_attributes(face_female)
    assert age_grp == "20-29"
    assert conf == 0.85
    assert gender_str == "female-presentation prediction"

    face_male = MockFace(age=42.0, gender=1)
    age_grp, conf, gender_str = extract_face_attributes(face_male)
    assert age_grp == "40-49"
    assert gender_str == "male-presentation prediction"

    # Face without attributes
    class BareFace:
        pass

    bare_face = BareFace()
    age_grp, conf, gender_str = extract_face_attributes(bare_face)
    assert age_grp == "Not available"
    assert conf is None
    assert gender_str == "Not available"


def test_detect_faces_uninitialized():
    import numpy as np
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_faces(arr) == []


def test_get_face_app_success():
    import app.services.face_service as fs
    from unittest.mock import patch, MagicMock
    mock_fa_cls = MagicMock()
    mock_fa_instance = MagicMock()
    mock_fa_cls.return_value = mock_fa_instance
    with patch.dict("sys.modules", {"insightface": MagicMock(), "insightface.app": MagicMock(FaceAnalysis=mock_fa_cls)}):
        fs._face_app = None
        app_inst = fs.get_face_app()
        assert app_inst == mock_fa_instance
        mock_fa_instance.prepare.assert_called_once()
        fs._face_app = None


def test_detect_faces_with_loaded_model():
    import app.services.face_service as fs
    from unittest.mock import patch, MagicMock
    import numpy as np
    mock_app = MagicMock()
    mock_app.get.return_value = ["mock_face"]
    with patch("app.services.face_service.get_face_app", return_value=mock_app):
        res = fs.detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        assert res == ["mock_face"]


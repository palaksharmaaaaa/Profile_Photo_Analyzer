import numpy as np
import pytest
from app.services.quality_service import assess_face_quality


def test_quality_service_complete_face():
    # 500x500 image with centered texture
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    # High frequency texture in center for sharpness
    image[150:350, 150:350] = np.random.randint(40, 200, (200, 200, 3), dtype=np.uint8)

    face_box = [150, 150, 350, 350]
    landmarks = np.array([
        [200, 200],
        [300, 200],
        [250, 250],
        [220, 300],
        [280, 300]
    ])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "LIKELY_COMPLETE"
    assert result["face_box_touches_image_edge"] is False
    assert result["landmarks_available"] is True
    assert result["landmarks_inside_image"] is True
    assert result["sharpness_score"] > 30.0


def test_quality_service_touches_edge():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    # Face touches top-left border
    face_box = [2, 2, 200, 200]
    landmarks = np.array([[50, 50], [100, 50], [75, 75], [60, 100], [90, 100]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "PARTIAL_OR_CROPPED"
    assert result["face_box_touches_image_edge"] is True


def test_quality_service_blurry():
    # Uniform smooth image (Laplacian variance near 0)
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    face_box = [100, 100, 300, 300]
    landmarks = np.array([[150, 150], [250, 150], [200, 200], [170, 250], [230, 250]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "UNCERTAIN"
    assert "blurry" in result["reason"].lower()


def test_quality_service_dark_lighting():
    # Nearly black image with slight pixel variation
    image = np.ones((500, 500, 3), dtype=np.uint8) * 15
    image[100:300, 100:300] = np.random.randint(5, 25, (200, 200, 3), dtype=np.uint8)
    face_box = [100, 100, 300, 300]
    landmarks = np.array([[150, 150], [250, 150], [200, 200], [170, 250], [230, 250]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "UNCERTAIN"
    assert "poor lighting" in result["reason"].lower() or "underexposed" in result["reason"].lower()


def test_quality_service_too_small_face():
    # 1000x1000 image with tiny 20x20 face box
    image = np.ones((1000, 1000, 3), dtype=np.uint8) * 128
    image[100:120, 100:120] = np.random.randint(50, 200, (20, 20, 3), dtype=np.uint8)
    face_box = [100, 100, 120, 120]
    landmarks = np.array([[105, 105], [115, 105], [110, 110], [107, 115], [113, 115]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "UNCERTAIN"
    assert "too small" in result["reason"].lower()


def test_quality_service_overexposed():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 250
    face_box = [100, 100, 300, 300]
    landmarks = np.array([[150, 150], [250, 150], [200, 200], [170, 250], [230, 250]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "UNCERTAIN"
    assert "overexposed" in result["reason"].lower() or "washed out" in result["reason"].lower()


def test_quality_service_landmarks_outside():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    image[100:300, 100:300] = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    face_box = [100, 100, 300, 300]
    # Landmark coordinates outside image boundary
    landmarks = np.array([[150, 150], [600, 150], [200, 200], [170, 250], [230, 250]])

    result = assess_face_quality(image, face_box, landmarks)
    assert result["status"] == "UNCERTAIN"
    assert "landmarks" in result["reason"].lower()


def test_quality_service_invalid_box():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    face_box = [100, 100, 100, 100]  # Zero width and height

    result = assess_face_quality(image, face_box, None)
    assert result["status"] == "UNCERTAIN"
    assert "invalid" in result["reason"].lower()


def test_quality_service_landmarks_none():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    image[100:300, 100:300] = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    face_box = [100, 100, 300, 300]
    result = assess_face_quality(image, face_box, landmarks=None)
    assert result["status"] == "UNCERTAIN"
    assert result["landmarks_available"] is False


def test_quality_service_landmarks_malformed_shape():
    image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    image[100:300, 100:300] = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    face_box = [100, 100, 300, 300]
    landmarks = np.array([1, 2, 3])  # 1D array
    result = assess_face_quality(image, face_box, landmarks=landmarks)
    assert result["status"] == "UNCERTAIN"
    assert result["landmarks_inside_image"] is False


def test_check_image_information_valid():
    from app.services.quality_service import check_image_information
    img = np.random.randint(20, 220, (100, 100, 3), dtype=np.uint8)
    res = check_image_information(img)
    assert res["has_information"] is True
    assert res["entropy"] > 0.5


def test_check_image_information_blank_or_solid():
    from app.services.quality_service import check_image_information
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    res = check_image_information(blank)
    assert res["has_information"] is False
    assert "no visual information" in res["reason"]

    solid = np.full((100, 100, 3), (120, 120, 120), dtype=np.uint8)
    res_solid = check_image_information(solid)
    assert res_solid["has_information"] is False


def test_check_image_information_empty():
    from app.services.quality_service import check_image_information
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res = check_image_information(empty)
    assert res["has_information"] is False
    assert res["entropy"] == 0.0

    none_res = check_image_information(None)
    assert none_res["has_information"] is False



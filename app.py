

import io
import re
import base64
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from insightface.app import FaceAnalysis
from transformers import pipeline


# =========================================================
# CONFIGURATION
# =========================================================

APP_TITLE = "Profile Photo Analyzer"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PHOTO_SCORE_THRESHOLD = 0.35
MIN_SHARPNESS_SCORE = 30.0

app = FastAPI(title=APP_TITLE, version="2.0.0")


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading face detection model...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

print("Loading image-type screening model...")

image_type_pipeline = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=-1
)

IMAGE_LABELS = [
    "a real photograph of a real human person",
    "a cartoon drawing of a person",
    "an anime or Ghibli-style illustration",
    "a digital painting of a fictional character",
    "a computer-generated 3D character",
    "a photograph or picture of an object with no person"
]

print("Loading approximate age-group model...")

age_pipeline = pipeline(
    "image-classification",
    model="dima806/fairface_age_image_detection",
    device=-1
)

print("All models loaded.")


# =========================================================
# IMAGE TYPE SCREENING
# =========================================================

def classify_image_type(image):
    """Classify the uploaded image using the existing CLIP model."""

    try:
        predictions = image_type_pipeline(
            image.convert("RGB"),
            candidate_labels=IMAGE_LABELS
        )

        if not isinstance(predictions, list) or not predictions:
            raise ValueError("No predictions returned by CLIP.")

        predictions = sorted(
            predictions,
            key=lambda item: float(item["score"]),
            reverse=True
        )

        all_predictions = [
            {
                "label": str(item["label"]),
                "confidence": round(float(item["score"]), 4)
            }
            for item in predictions
            if isinstance(item, dict)
            and "label" in item
            and "score" in item
        ]

        if not all_predictions:
            raise ValueError("Invalid CLIP prediction format.")

        top = all_predictions[0]
        top_label = top["label"]
        normalized = top_label.lower()

        if "real photograph of a real human person" in normalized:
            status = "PHOTO_LIKE"
            reason = "The highest-scoring category is a real human photograph."

        elif "photograph or picture of an object with no person" in normalized:
            status = "NON_PHOTO"
            reason = "The image was classified as an object or scene without a person."

        elif any(term in normalized for term in (
            "cartoon",
            "anime",
            "illustration",
            "digital painting",
            "computer-generated"
        )):
            status = "NON_PHOTO"
            reason = "The highest-scoring category is an illustrated or computer-generated image."

        else:
            status = "NEEDS_REVIEW"
            reason = "The image type could not be determined confidently."

        photo_score = max(
            (
                item["confidence"]
                for item in all_predictions
                if "real photograph of a real human person"
                in item["label"].lower()
            ),
            default=0.0
        )

        synthetic_score = max(
            (
                item["confidence"]
                for item in all_predictions
                if any(term in item["label"].lower() for term in (
                    "cartoon",
                    "anime",
                    "illustration",
                    "digital painting",
                    "computer-generated"
                ))
            ),
            default=0.0
        )

        return {
            "status": status,
            "top_class": top_label,
            "top_class_confidence": top["confidence"],
            "photo_score": round(photo_score, 4),
            "cartoon_score": round(synthetic_score, 4),
            "all_predictions": all_predictions,
            "scores": {
                item["label"]: item["confidence"]
                for item in all_predictions
            },
            "reason": reason
        }

    except Exception as exc:
        print("Image type classification error:", repr(exc))

        return {
            "status": "UNCERTAIN",
            "top_class": "Uncertain",
            "top_class_confidence": 0.0,
            "photo_score": 0.0,
            "cartoon_score": 0.0,
            "all_predictions": [],
            "reason": f"Image classification failed: {exc}"
        }


def friendly_image_type(top_class: str) -> str:
    """Convert the CLIP label into a short, readable image type."""

    n = (top_class or "").lower()

    if "real photograph of a real human person" in n:
        return "Real Photo"
    if "anime" in n:
        return "Anime / Ghibli-style Image"
    if "cartoon" in n:
        return "Cartoon Image"
    if "digital painting" in n:
        return "Digital Painting (Fictional Character)"
    if "computer-generated" in n:
        return "3D / Computer-generated Character"
    if "object with no person" in n:
        return "Object / No Person"
    return "Uncertain"


# =========================================================
# FACE VISIBILITY / QUALITY CHECK
# =========================================================

def assess_visibility(image_bgr: np.ndarray, face) -> dict:
    height, width = image_bgr.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    if box_width == 0 or box_height == 0:
        return {
            "status": "UNCERTAIN",
            "reason": "Invalid face bounding box."
        }

    margin_x = max(5, int(width * 0.01))
    margin_y = max(5, int(height * 0.01))

    touches_edge = (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= width - margin_x
        or y2 >= height - margin_y
    )

    face_crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    landmarks = getattr(face, "kps", None)
    landmarks_available = landmarks is not None

    landmarks_inside = False

    if landmarks_available:
        points = np.asarray(landmarks)
        landmarks_inside = bool(
            np.all(points[:, 0] >= 0)
            and np.all(points[:, 0] < width)
            and np.all(points[:, 1] >= 0)
            and np.all(points[:, 1] < height)
        )

    if touches_edge:
        status = "PARTIAL_OR_CROPPED"
        reason = "The face box is too close to an image edge."

    elif sharpness < MIN_SHARPNESS_SCORE:
        status = "UNCERTAIN"
        reason = "The face image may be blurry."

    elif not landmarks_available or not landmarks_inside:
        status = "UNCERTAIN"
        reason = "Facial landmarks could not be verified reliably."

    else:
        status = "LIKELY_COMPLETE"
        reason = (
            "The face box is not near an image edge and "
            "facial landmarks are available."
        )

    return {
        "status": status,
        "reason": reason,
        "face_box_touches_image_edge": touches_edge,
        "sharpness_score": round(sharpness, 2),
        "landmarks_available": landmarks_available,
        "landmarks_inside_image": landmarks_inside
    }


# =========================================================
# AGE-GROUP ESTIMATION
# =========================================================

def estimate_age_group(image: Image.Image, face_box: list) -> dict:
    x1, y1, x2, y2 = face_box

    face_crop = image.crop((x1, y1, x2, y2))
    predictions = age_pipeline(face_crop)

    best = predictions[0]
    label = str(best["label"]).strip()

    return {
        "estimated_age_group": label,
        "confidence": round(float(best["score"]), 4),
        "model": "dima806/fairface_age_image_detection",
        "note": (
            "Approximate appearance-based estimate only. "
            "It cannot confirm whether someone is 18 or older."
        )
    }


def possible_minor_age_group(age_label: str) -> bool:
    label = age_label.lower().replace(" ", "")

    if label in {"0-2", "3-9", "10-19"}:
        return True

    if "under18" in label or "child" in label or "minor" in label:
        return True

    match = re.search(r"(\d+)\s*[-–]\s*(\d+)", label)

    if match:
        lower = int(match.group(1))
        upper = int(match.group(2))
        return lower < 18

    return False


# =========================================================
# MAIN PROFILE PHOTO ANALYSIS
# =========================================================


def analyze_image(image: Image.Image) -> dict:
    image = image.convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    image_type = classify_image_type(image)
    image_type["display_label"] = friendly_image_type(
        image_type.get("top_class", "")
    )

    non_photo = image_type.get("status") == "NON_PHOTO"
    type_warning = (
        f"Warning: this looks like a non-photo image "
        f"({image_type['display_label']}), not a real photograph."
        if non_photo else ""
    )

    faces = face_app.get(image_bgr)

    if len(faces) == 0:
        return {
            "accepted": False,
            "posting_status": (
                "NON_PHOTO_WARNING" if non_photo else "REJECTED"
            ),
            "reason": (
                (type_warning + " ") if type_warning else ""
            ) + "No human face was detected.",
            "image_type_check": image_type,
            "face_detected": False,
            "face_count": 0,
            "age_gender_prediction_performed": False,
            "estimated_age_group": "Not available",
            "gender_presentation_prediction": "Not available",
            "age_verification_required": True
        }

    if len(faces) != 1:
        return {
            "accepted": False,
            "posting_status": (
                "NON_PHOTO_WARNING" if non_photo else "REJECTED"
            ),
            "reason": (
                (type_warning + " ") if type_warning else ""
            ) + "A profile photo must contain exactly one detected face.",
            "image_type_check": image_type,
            "face_detected": True,
            "face_count": len(faces),
            "age_gender_prediction_performed": False,
            "estimated_age_group": "Not available",
            "gender_presentation_prediction": "Not available",
            "age_verification_required": True
        }

    face = faces[0]
    height, width = image_bgr.shape[:2]

    # Keep the face coordinates inside the image.
    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    face_box = [x1, y1, x2, y2]
    visibility = assess_visibility(image_bgr, face)
        

    
    age_result = None
    age_error = None
    age_label = "Not performed"
    age_confidence = None

    gender_prediction = "Not performed"
    gender_error = None

    # Do not run age/gender prediction on cartoons, illustrations,
    # anime/Ghibli images, or computer-generated characters.
    if non_photo:
        age_error = "Skipped because the image appears to be a non-photo image."
        gender_error = "Skipped because the image appears to be a non-photo image."

    else:
        # Run age prediction even if the face visibility check reports blur.
        try:
            if x2 > x1 and y2 > y1:
                age_result = estimate_age_group(image, face_box)
                age_label = age_result.get(
                    "estimated_age_group", "Not available"
                )
                age_confidence = age_result.get("confidence")
            else:
                age_error = "The detected face crop is invalid."

        except Exception as exc:
            print("Age prediction error:", repr(exc))
            age_error = "The age model could not estimate the age group."
            age_label = "Not available"

        # Run gender prediction only for images classified as photo-like.
        try:
            gender_value = getattr(face, "gender", None)

            if gender_value is not None and int(gender_value) in (0, 1):
                gender_prediction = (
                    "female-presentation prediction"
                    if int(gender_value) == 0
                    else "male-presentation prediction"
                )
            else:
                gender_prediction = "Not available"
                gender_error = "Gender prediction was unavailable."

        except Exception as exc:
            print("Gender prediction error:", repr(exc))
            gender_prediction = "Not available"
            gender_error = "The gender prediction could not be read."

    minor_risk = (
        possible_minor_age_group(age_label)
        if age_label != "Not available"
        and age_label != "Not performed"
        else True
    )

    # IMPORTANT:
    # A blurry-face warning must not automatically stop age prediction.
    # Attempt the age model independently.
    try:
        if x2 > x1 and y2 > y1:
            age_result = estimate_age_group(image, face_box)
        else:
            age_error = "The detected face crop is invalid."
    except Exception as exc:
        print("Age prediction error:", repr(exc))
        age_error = "The age model could not estimate the age group."

    age_label = (
        age_result.get("estimated_age_group", "Not available")
        if age_result else "Not available"
    )

    age_confidence = (
        age_result.get("confidence")
        if age_result else None
    )

    # InsightFace gender output:
    # 0 = female presentation, 1 = male presentation.
    gender_prediction = "Not available"
    gender_error = None

    try:
        gender_value = getattr(face, "gender", None)

        if gender_value is not None and int(gender_value) in (0, 1):
            gender_prediction = (
                "female-presentation prediction"
                if int(gender_value) == 0
                else "male-presentation prediction"
            )
        else:
            gender_error = "Gender prediction was unavailable."
    except Exception as exc:
        print("Gender prediction error:", repr(exc))
        gender_error = "The gender prediction could not be read."

    minor_risk = (
        possible_minor_age_group(age_label)
        if age_label != "Not available"
        else True
    )

    # A non-photo image gets a warning, not an adult-age conclusion.
    if non_photo:
        accepted = False
        posting_status = "NON_PHOTO_WARNING"
        posting_reason = (
            f"{type_warning} Face detection found one face. "
            "Age and gender estimates on illustrations or "
            "computer-generated characters may be unreliable."
        )

    elif visibility["status"] != "LIKELY_COMPLETE":
        accepted = False
        posting_status = "REJECTED_OR_REVIEW"
        posting_reason = (
            f"{visibility['reason']} "
            "Age and gender prediction were attempted independently "
            "of the visibility check."
        )

    elif age_label == "Not available":
        accepted = False
        posting_status = "AGE_ESTIMATION_UNAVAILABLE"
        posting_reason = (
            "The face was detected, but the age group could not be "
            "estimated. Please try a clear, front-facing photograph."
        )

    elif minor_risk:
        accepted = False
        posting_status = "AGE_VERIFICATION_REQUIRED"
        posting_reason = (
            f"The estimated age group is {age_label}. "
            "This range may include people under 18. "
            "Verify age before permitting posting."
        )

    else:
        accepted = True
        posting_status = "INITIAL_CHECKS_PASSED"
        posting_reason = (
            f"The estimated age group is {age_label}. "
            "This is an appearance-based estimate, not proof of age."
        )

    return {
        "accepted": accepted,
        "posting_status": posting_status,
        "posting_reason": posting_reason,
        "reason": posting_reason,
        "image_type_check": image_type,
        "image_type_review_required": non_photo,
        "face_detected": True,
        "face_count": 1,
        "face_visibility": visibility,
        "estimated_age_group": age_label,
        "age_model_confidence": age_confidence,
        "age_note": (
            age_result.get("note", "")
            if age_result else age_error
        ),
        "age_prediction_error": age_error,
        "possible_minor_risk": minor_risk,
        "age_verification_required": True,
        "gender_presentation_prediction": gender_prediction,
        "gender_note": (
            "Appearance-based model prediction only; "
            "not a confirmed gender identity."
        ),
        "gender_prediction_error": gender_error,
        "face_detection_confidence": round(
            float(face.det_score), 4
        ),
        "face_box_xyxy": face_box,
        "age_gender_prediction_performed": (
            not non_photo
            and (
                age_result is not None
                or gender_prediction != "Not available"
            )
),
        "note": (
            "Age estimates may be incorrect. Age estimation cannot "
            "confirm whether a person is 18 or older. Use a separate "
            "age-verification process for an 18+ posting policy."
        )
    }



# =========================================================
# IMAGE PREVIEW AND FACE REGION
# =========================================================

def encode_image(image_bgr: np.ndarray) -> str:
    """Encode an image as a JPEG data URL for the browser."""

    success, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        raise ValueError("Could not encode preview image.")

    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


def build_image_previews(image: Image.Image, result: dict) -> dict:
    """
    Add original and annotated image previews to the API response.
    Does not modify the existing model analysis.
    """

    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]

    boxes = []
    existing_box = result.get("face_box_xyxy")

    if existing_box and len(existing_box) == 4:
        boxes.append(existing_box)
    else:
        # Find face regions even when an earlier screening step stopped
        # the analysis or multiple faces were found.
        try:
            detected_faces = face_app.get(image_bgr)


            for face in detected_faces:
                boxes.append([int(v) for v in face.bbox])

        except Exception as exc:
            print("Preview face detection error:", repr(exc))

    possible_minor = result.get("possible_minor_risk", False)
    age_group = result.get("estimated_age_group", "")
    status = result.get("posting_status", "")

    ORANGE = (0, 140, 255)   # BGR

    if status == "NON_PHOTO_WARNING":
        label = "WARNING: NON-PHOTO IMAGE"
        box_color = ORANGE
    elif possible_minor:
        label = f"AGE GROUP: {age_group}"
        box_color = ORANGE
    elif not result.get("accepted", False):
        label = "REVIEW REQUIRED"
        box_color = ORANGE
    else:
        label = "FACE DETECTED"
        box_color = (0, 180, 0)

    valid_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        valid_boxes.append([x1, y1, x2, y2])

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(width, height) / 1000)
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = min(
            height - baseline - 2,
            max(text_height + 10, y1)
        )

        rect_top = max(0, label_y - text_height - 8)
        rect_bottom = min(height - 1, label_y + baseline)
        rect_right = min(width - 1, x1 + text_width + 8)

        cv2.rectangle(
            annotated,
            (x1, rect_top),
            (rect_right, rect_bottom),
            box_color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    result["face_regions_xyxy"] = valid_boxes
    result["original_image"] = encode_image(image_bgr)
    result["annotated_image"] = encode_image(annotated)

    if not valid_boxes:
        result["region_note"] = (
            "No face region could be located. "
            "The original uploaded photo is shown for review."
        )
    else:
        result["region_note"] = (
            "The outlined area indicates the detected face region. "
            "It does not independently prove age."
        )

    return result


# =========================================================
# SIMPLE FRONTEND
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Profile Photo Analyzer</title>

        <style>
            * { box-sizing: border-box; }

            body {
                margin: 0;
                padding: 35px 16px;
                background: #f3f5fa;
                color: #202938;
                font-family: Arial, sans-serif;
            }

            .container {
                max-width: 850px;
                margin: auto;
            }

            .header, .panel {
                background: white;
                padding: 25px;
                border-radius: 16px;
                margin-bottom: 20px;
                box-shadow: 0 4px 18px #18233a0d;
            }

            h1 { margin-top: 0; font-size: 28px; }
            h2 { font-size: 19px; margin-top: 0; }

            .subtitle {
                color: #647084;
                line-height: 1.6;
            }

            .upload-box {
                border: 2px dashed #c6d2e5;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                background: #fafbfe;
            }

            input[type=file] {
                width: 100%;
                margin: 15px 0;
            }

            button {
                width: 100%;
                padding: 14px;
                background: #315ee7;
                color: white;
                border: 0;
                border-radius: 9px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:disabled {
                opacity: 0.6;
                cursor: wait;
            }

            .status {
                padding: 14px 16px;
                border-radius: 10px;
                margin-bottom: 18px;
                font-weight: bold;
            }

            .success { background: #e5f8ec; color: #16743a; }
            .warning { background: #fff4d8; color: #885800; }
            .danger { background: #fde9e9; color: #ad2929; }
            .neutral { background: #edf1f7; color: #34445e; }

            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 14px;
            }

            .card {
                border: 1px solid #e4e9f1;
                border-radius: 12px;
                padding: 17px;
                min-width: 0;
            }

            .card-label {
                color: #69758a;
                font-size: 13px;
                margin-bottom: 9px;
            }

            .card-value {
                font-size: 19px;
                font-weight: bold;
                overflow-wrap: anywhere;
            }

            .small {
                color: #69758a;
                font-size: 13px;
                line-height: 1.6;
                margin-top: 8px;
            }

            .hidden { display: none; }

            .error {
                color: #ad2929;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }

            .footer-note {
                font-size: 13px;
                color: #69758a;
                line-height: 1.6;
            }

            .photo-grid {
                margin-bottom: 20px;
            }

            .photo-preview {
                display: block;
                width: 100%;
                max-height: 440px;
                object-fit: contain;
                background: #f3f5fa;
                border-radius: 10px;
                border: 1px solid #e4e9f1;
            }

            #uploadedFilename {
                overflow-wrap: anywhere;
            }

            @media (max-width: 560px) {
                .grid { grid-template-columns: 1fr; }
                .header, .panel { padding: 19px; }
                h1 { font-size: 24px; }
            }
        </style>
    </head>

    <body>
    <div class="container">

        <div class="header">
            <h1>Profile Photo Analyzer</h1>
            <p class="subtitle">
                Validate a profile photo, check face count and visibility,
                and view an approximate age-group prediction.
            </p>

            <form id="uploadForm">
                <div class="upload-box">
                    <strong>Upload Profile Photo</strong>
                    <p class="subtitle">
                        Select a JPG, JPEG, PNG or WEBP image (max 8 MB).
                    </p>
                    <input
                        type="file"
                        name="file"
                        id="photo"
                        accept=".jpg,.jpeg,.png,.webp"
                        required
                    >
                    <button id="analyzeBtn" type="submit">
                        Analyze Profile Photo
                    </button>
                </div>
            </form>
        </div>

        <div id="loading" class="panel hidden">
            <h2>Analyzing your photo...</h2>
            <p class="subtitle">
                Checking image type, face count and face visibility.
                The first analysis may take longer while models initialize.
            </p>
        </div>

        <div id="resultPanel" class="panel hidden">
            <h2>Analysis Result</h2>
            <div id="mainStatus" class="status neutral"></div>

            <div class="grid photo-grid">
                <div class="card">
                    <h2>Original Uploaded Photo</h2>
                    <img
                        id="uploadedPreview"
                        class="photo-preview"
                        alt="Original uploaded profile photo"
                    >
                    <p id="uploadedFilename" class="small"></p>
                </div>

                <div class="card">
                    <h2>Detected Face Region</h2>
                    <img
                        id="annotatedPreview"
                        class="photo-preview"
                        alt="Annotated photo with detected face region"
                    >
                    <p id="regionNote" class="small"></p>
                </div>
            </div>

            <p id="reason" class="subtitle"></p>

            <div class="grid">
                <div class="card">
                    <div class="card-label">Image Type Check</div>
                    <div id="imageType" class="card-value">—</div>
                    <div id="imageConfidence" class="small"></div>
                </div>

                <div class="card">
                    <div class="card-label">Face Detection</div>
                    <div id="faceCount" class="card-value">—</div>
                    <div id="faceConfidence" class="small"></div>
                </div>

                <div class="card">
                    <div class="card-label">Face Visibility</div>
                    <div id="visibility" class="card-value">—</div>
                    <div id="visibilityReason" class="small"></div>
                </div>

                <div class="card">
                    <div class="card-label">Estimated Age Group</div>
                    <div id="age" class="card-value">Not performed</div>
                    <div id="ageConfidence" class="small"></div>
                </div>

                <div class="card">
                    <div class="card-label">Gender Presentation</div>
                    <div id="gender" class="card-value">Not performed</div>
                    <div id="genderNote" class="small"></div>
                </div>

                <div class="card">
                    <div class="card-label">Age Verification</div>
                    <div id="ageVerification" class="card-value">
                        Required
                    </div>
                    <div class="small">
                        Appearance-based age estimation does not verify
                        that a user is 18 or older.
                    </div>
                </div>
            </div>

            <div id="imagePredictions" class="card hidden"
                 style="margin-top:14px">
                <h2>Image Classification Details</h2>
                <div id="predictionList"></div>
            </div>

            <p id="extraNote" class="footer-note"></p>
        </div>

        <div id="errorPanel" class="panel hidden">
            <h2>Analysis Error</h2>
            <p id="errorText" class="error"></p>
        </div>

        <div class="panel">
            <p class="footer-note">
                Note: Image classification and face visibility checks can
                make mistakes. Age estimation is not age verification.
                A separate age-assurance process is required before enforcing
                an 18+ posting policy.
            </p>
        </div>

    </div>

    <script>
        const form = document.getElementById("uploadForm");
        const button = document.getElementById("analyzeBtn");

        const hide = id =>
            document.getElementById(id).classList.add("hidden");

        const show = id =>
            document.getElementById(id).classList.remove("hidden");

        function setText(id, value) {
            document.getElementById(id).textContent =
                value === undefined || value === null || value === ""
                ? "—" : String(value);
        }

        function setStatus(text, type) {
            const el = document.getElementById("mainStatus");
            el.textContent = text;
            el.className = "status " + type;
        }

        form.addEventListener("submit", async event => {
            event.preventDefault();

            const file = document.getElementById("photo").files[0];

            if (!file) {
                alert("Please select a profile photo.");
                return;
            }

            hide("resultPanel");
            hide("errorPanel");
            show("loading");

            button.disabled = true;
            button.textContent = "Analyzing...";

            const controller = new AbortController();
            const timeout = setTimeout(
                () => controller.abort(), 180000
            );

            try {
                const formData = new FormData(form);

                const response = await fetch(
                    "/analyze-profile-photo",
                    {
                        method: "POST",
                        body: formData,
                        signal: controller.signal
                    }
                );

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.reason || data.detail ||
                        "Request failed: " + response.status
                    );
                }

                renderResult(data);
                show("resultPanel");

            } catch (error) {
                setText(
                    "errorText",
                    error.name === "AbortError"
                        ? "Analysis timed out. Check your server terminal."
                        : error.message
                );
                show("errorPanel");
            } finally {
                clearTimeout(timeout);
                hide("loading");
                button.disabled = false;
                button.textContent = "Analyze Profile Photo";
            }
        });

        function renderResult(data) {
            const originalPreview =
                document.getElementById("uploadedPreview");
            const annotatedPreview =
                document.getElementById("annotatedPreview");

            originalPreview.removeAttribute("src");
            annotatedPreview.removeAttribute("src");

            if (data.original_image) {
                originalPreview.src = data.original_image;
            }

            if (data.annotated_image) {
                annotatedPreview.src = data.annotated_image;
            }

            setText("uploadedFilename", data.filename || "Uploaded photo");
            setText(
                "regionNote",
                data.region_note || "No region information available."
            );

            const status = data.posting_status || "UNKNOWN";
            const accepted = data.accepted === true;
            const imageCheck = data.image_type_check || {};

            if (status === "NON_PHOTO_WARNING") {
                setStatus(
                    "WARNING — " +
                    (imageCheck.display_label || "NON-PHOTO IMAGE").toUpperCase() +
                    " DETECTED",
                    "warning"
                );
            } else if (status === "AGE_VERIFICATION_REQUIRED") {
                setStatus(
                    "PHOTO CHECKS PASSED — AGE VERIFICATION REQUIRED",
                    "warning"
                );
            } else if (accepted) {
                setStatus("PHOTO CHECKS PASSED", "success");
            } else if (
                status.includes("REVIEW") ||
                status.includes("UNCERTAIN")
            ) {
                setStatus("REVIEW REQUIRED", "warning");
            } else {
                setStatus("PHOTO REJECTED", "danger");
            }

            setText("reason", data.reason || data.posting_reason);

            setText(
                "imageType",
                imageCheck.display_label || imageCheck.top_class || imageCheck.status
            );
            setText(
                "imageConfidence",
                imageCheck.top_class_confidence !== undefined
                    ? "Top-class confidence: " +
                      (imageCheck.top_class_confidence * 100).toFixed(1) + "%"
                    : imageCheck.confidence !== undefined
                    ? "Confidence: " +
                      (imageCheck.confidence * 100).toFixed(1) + "%"
                    : imageCheck.note
            );

            setText(
                "faceCount",
                data.face_count === undefined
                    ? (data.face_detected ? "Face detected" : "No face")
                    : data.face_count +
                      (data.face_count === 1 ? " face" : " faces")
            );

            setText(
                "faceConfidence",
                data.face_detection_confidence !== undefined
                    ? "Detection confidence: " +
                      (data.face_detection_confidence * 100).toFixed(1) + "%"
                    : ""
            );

            const visibility = data.face_visibility || {};
            setText("visibility", visibility.status || "Not checked");
            setText("visibilityReason", visibility.reason || "");

            setText(
                "age",
                data.estimated_age_group || "Not performed"
            );

            setText(
                "ageConfidence",
                data.age_model_confidence !== undefined
                    ? "Model score: " +
                      (data.age_model_confidence * 100).toFixed(1) + "%"
                    : data.age_note || ""
            );

            setText(
                "gender",
                data.gender_presentation_prediction || "Not performed"
            );
            setText("genderNote", data.gender_note || "");

            setText(
                "ageVerification",
                data.age_verification_required === false
                    ? "Not required"
                    : "Required"
            );

            const predictions = imageCheck.all_predictions || [];
            const predictionList = document.getElementById("predictionList");

            predictionList.replaceChildren();

            if (predictions.length) {
                predictions.forEach(item => {
                    const line = document.createElement("p");
                    const score = (item.confidence * 100).toFixed(1);
                    line.textContent = item.label + " — " + score + "%";
                    predictionList.appendChild(line);
                });
                show("imagePredictions");
            } else {
                hide("imagePredictions");
            }

            setText("extraNote", data.note || imageCheck.note || "");
        }
    </script>
    </body>
    </html>
    """


# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/analyze-profile-photo")
async def analyze_profile_photo(file: UploadFile = File(...)):
    filename = file.filename or "upload.jpg"
    extension = "." + filename.rsplit(".", 1)[-1].lower() \
        if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "Only JPG, JPEG, PNG and WEBP files are allowed."
            }
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "The uploaded file is empty."
            }
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "accepted": False,
                "reason": "Maximum upload size is 8 MB."
            }
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "The uploaded file is not a valid supported image."
            }
        )

    try:
        result = analyze_image(image)
        result["filename"] = filename

        # Add image previews while keeping analysis logic unchanged.
        result = build_image_previews(image, result)

        return JSONResponse(content=result)

    except Exception as exc:
        print("Profile photo analysis error:", repr(exc))
        return JSONResponse(
            status_code=500,
            content={
                "accepted": False,
                "reason": "Analysis failed. Check server logs."
            }
        )


@app.get("/health")
def health():
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


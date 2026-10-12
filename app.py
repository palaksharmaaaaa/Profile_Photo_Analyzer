
# import io
# import re
# import base64

# import cv2
# import numpy as np
# from PIL import Image, ImageOps, UnidentifiedImageError

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse, JSONResponse
# from insightface.app import FaceAnalysis
# from transformers import pipeline


# # =========================================================
# # CONFIGURATION
# # =========================================================

# APP_TITLE = "Profile Photo Analyzer"
# MAX_UPLOAD_BYTES = 8 * 1024 * 1024

# ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# PHOTO_SCORE_THRESHOLD = 0.35
# MIN_SHARPNESS_SCORE = 30.0

# app = FastAPI(title=APP_TITLE, version="2.1.0")


# # =========================================================
# # LOAD MODELS
# # =========================================================

# print("Loading face detection model...")

# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CPUExecutionProvider"]
# )
# face_app.prepare(ctx_id=-1, det_size=(640, 640))

# print("Loading image-type screening model...")

# image_type_pipeline = pipeline(
#     "zero-shot-image-classification",
#     model="openai/clip-vit-base-patch32",
#     device=-1
# )

# IMAGE_LABELS = [
#     "a real photograph of a real human person",
#     "a cartoon drawing of a person",
#     "an anime or Ghibli-style illustration",
#     "a digital painting of a fictional character",
#     "a computer-generated 3D character",
#     "a photograph or picture of an object with no person"
# ]

# print("Loading approximate age-group model...")

# age_pipeline = pipeline(
#     "image-classification",
#     model="dima806/fairface_age_image_detection",
#     device=-1
# )

# print("All models loaded.")


# # =========================================================
# # IMAGE TYPE SCREENING
# # =========================================================

# def classify_image_type(image):
#     """Classify the uploaded image using the existing CLIP model."""

#     try:
#         predictions = image_type_pipeline(
#             image.convert("RGB"),
#             candidate_labels=IMAGE_LABELS
#         )

#         if not isinstance(predictions, list) or not predictions:
#             raise ValueError("No predictions returned by CLIP.")

#         predictions = sorted(
#             predictions,
#             key=lambda item: float(item["score"]),
#             reverse=True
#         )

#         all_predictions = [
#             {
#                 "label": str(item["label"]),
#                 "confidence": round(float(item["score"]), 4)
#             }
#             for item in predictions
#             if isinstance(item, dict)
#             and "label" in item
#             and "score" in item
#         ]

#         if not all_predictions:
#             raise ValueError("Invalid CLIP prediction format.")

#         top = all_predictions[0]
#         top_label = top["label"]
#         normalized = top_label.lower()

#         if "real photograph of a real human person" in normalized:
#             status = "PHOTO_LIKE"
#             reason = "The highest-scoring category is a real human photograph."

#         elif "photograph or picture of an object with no person" in normalized:
#             status = "NON_PHOTO"
#             reason = "The image was classified as an object or scene without a person."

#         elif any(term in normalized for term in (
#             "cartoon",
#             "anime",
#             "illustration",
#             "digital painting",
#             "computer-generated"
#         )):
#             status = "NON_PHOTO"
#             reason = "The highest-scoring category is an illustrated or computer-generated image."

#         else:
#             status = "NEEDS_REVIEW"
#             reason = "The image type could not be determined confidently."

#         photo_score = max(
#             (
#                 item["confidence"]
#                 for item in all_predictions
#                 if "real photograph of a real human person"
#                 in item["label"].lower()
#             ),
#             default=0.0
#         )

#         synthetic_score = max(
#             (
#                 item["confidence"]
#                 for item in all_predictions
#                 if any(term in item["label"].lower() for term in (
#                     "cartoon",
#                     "anime",
#                     "illustration",
#                     "digital painting",
#                     "computer-generated"
#                 ))
#             ),
#             default=0.0
#         )

#         return {
#             "status": status,
#             "top_class": top_label,
#             "top_class_confidence": top["confidence"],
#             "photo_score": round(photo_score, 4),
#             "cartoon_score": round(synthetic_score, 4),
#             "all_predictions": all_predictions,
#             "scores": {
#                 item["label"]: item["confidence"]
#                 for item in all_predictions
#             },
#             "reason": reason
#         }

#     except Exception as exc:
#         print("Image type classification error:", repr(exc))

#         return {
#             "status": "UNCERTAIN",
#             "top_class": "Uncertain",
#             "top_class_confidence": 0.0,
#             "photo_score": 0.0,
#             "cartoon_score": 0.0,
#             "all_predictions": [],
#             "reason": "Image classification could not be completed."
#         }


# def friendly_image_type(top_class: str) -> str:
#     """Convert the CLIP label into a short, readable image type."""

#     n = (top_class or "").lower()

#     if "real photograph of a real human person" in n:
#         return "Real Photo"
#     if "anime" in n:
#         return "Anime / Ghibli-style Image"
#     if "cartoon" in n:
#         return "Cartoon Image"
#     if "digital painting" in n:
#         return "Digital Painting (Fictional Character)"
#     if "computer-generated" in n:
#         return "3D / Computer-generated Character"
#     if "object with no person" in n:
#         return "Object / No Person"

#     return "Uncertain"


# # =========================================================
# # FACE VISIBILITY / QUALITY CHECK
# # =========================================================

# def assess_visibility(image_bgr: np.ndarray, face) -> dict:
#     height, width = image_bgr.shape[:2]

#     x1, y1, x2, y2 = [int(v) for v in face.bbox]

#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(width, x2)
#     y2 = min(height, y2)

#     box_width = max(0, x2 - x1)
#     box_height = max(0, y2 - y1)

#     if box_width == 0 or box_height == 0:
#         return {
#             "status": "UNCERTAIN",
#             "reason": "Invalid face bounding box."
#         }

#     margin_x = max(5, int(width * 0.01))
#     margin_y = max(5, int(height * 0.01))

#     touches_edge = (
#         x1 <= margin_x
#         or y1 <= margin_y
#         or x2 >= width - margin_x
#         or y2 >= height - margin_y
#     )

#     face_crop = image_bgr[y1:y2, x1:x2]
#     gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

#     sharpness = float(
#         cv2.Laplacian(gray, cv2.CV_64F).var()
#     )

#     landmarks = getattr(face, "kps", None)
#     landmarks_available = landmarks is not None

#     landmarks_inside = False

#     if landmarks_available:
#         points = np.asarray(landmarks)
#         landmarks_inside = bool(
#             np.all(points[:, 0] >= 0)
#             and np.all(points[:, 0] < width)
#             and np.all(points[:, 1] >= 0)
#             and np.all(points[:, 1] < height)
#         )

#     if touches_edge:
#         status = "PARTIAL_OR_CROPPED"
#         reason = "The face box is too close to an image edge."

#     elif sharpness < MIN_SHARPNESS_SCORE:
#         status = "UNCERTAIN"
#         reason = "The face image may be blurry."

#     elif not landmarks_available or not landmarks_inside:
#         status = "UNCERTAIN"
#         reason = "Facial landmarks could not be verified reliably."

#     else:
#         status = "LIKELY_COMPLETE"
#         reason = (
#             "The face box is not near an image edge and "
#             "facial landmarks are available."
#         )

#     return {
#         "status": status,
#         "reason": reason,
#         "face_box_touches_image_edge": touches_edge,
#         "sharpness_score": round(sharpness, 2),
#         "landmarks_available": landmarks_available,
#         "landmarks_inside_image": landmarks_inside
#     }


# # =========================================================
# # AGE-GROUP ESTIMATION
# # =========================================================

# def estimate_age_group(image: Image.Image, face_box: list) -> dict:
#     x1, y1, x2, y2 = face_box

#     width, height = image.size

#     x1 = max(0, min(int(x1), width))
#     y1 = max(0, min(int(y1), height))
#     x2 = max(0, min(int(x2), width))
#     y2 = max(0, min(int(y2), height))

#     if x2 <= x1 or y2 <= y1:
#         raise ValueError("Invalid face crop coordinates.")

#     face_crop = image.crop((x1, y1, x2, y2))
#     predictions = age_pipeline(face_crop)

#     if not predictions:
#         raise ValueError("The age model returned no predictions.")

#     best = max(
#         predictions,
#         key=lambda item: float(item.get("score", 0))
#     )

#     label = str(best["label"]).strip()

#     return {
#         "estimated_age_group": label,
#         "confidence": round(float(best["score"]), 4),
#         "model": "dima806/fairface_age_image_detection",
#         "note": (
#             "Approximate appearance-based estimate only. "
#             "It cannot confirm whether someone is 18 or older."
#         )
#     }


# def possible_minor_age_group(age_label: str) -> bool:
#     label = (age_label or "").lower().replace(" ", "")

#     if label in {"0-2", "3-9", "10-19"}:
#         return True

#     if "under18" in label or "child" in label or "minor" in label:
#         return True

#     match = re.search(r"(\d+)\s*[-–]\s*(\d+)", label)

#     if match:
#         lower = int(match.group(1))
#         upper = int(match.group(2))

#         # Flag any age range that includes ages below 18.
#         return lower < 18

#     # An unrecognized label cannot establish that the person is an adult.
#     return True


# # =========================================================
# # MAIN PROFILE PHOTO ANALYSIS
# # =========================================================

# def analyze_image(image: Image.Image) -> dict:
#     image = image.convert("RGB")
#     image_rgb = np.array(image)
#     image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#     image_type = classify_image_type(image)
#     image_type["display_label"] = friendly_image_type(
#         image_type["top_class"]
#     )

#     # Non-photo images receive a warning, not an automatic rejection.
#     non_photo = image_type["status"] == "NON_PHOTO"

#     type_warning = (
#         f"Warning: this looks like a non-photo image "
#         f"({image_type['display_label']}), not a real photograph."
#         if non_photo else ""
#     )

#     faces = face_app.get(image_bgr)

#     # No face found: ask for review instead of showing PHOTO REJECTED.
#     if len(faces) == 0:
#         return {
#             "accepted": None,
#             "posting_status": "PHOTO_REVIEW_REQUIRED",
#             "reason": (
#                 (type_warning + " " if type_warning else "")
#                 + "No human face was detected. Please review the photo."
#             ),
#             "image_type_check": image_type,
#             "face_detected": False,
#             "face_count": 0,
#             "age_gender_prediction_performed": False,
#             "age_verification_required": True
#         }

#     # Multiple faces: no individual age/gender prediction.
#     if len(faces) != 1:
#         return {
#             "accepted": None,
#             "posting_status": "PHOTO_REVIEW_REQUIRED",
#             "reason": (
#                 (type_warning + " " if type_warning else "")
#                 + "Multiple faces were detected. Please review the image; "
#                 "individual age and gender predictions were skipped."
#             ),
#             "image_type_check": image_type,
#             "image_type_review_required": non_photo,
#             "face_detected": True,
#             "face_count": len(faces),
#             "age_gender_prediction_performed": False,
#             "age_verification_required": True
#         }

#     face = faces[0]
#     visibility = assess_visibility(image_bgr, face)
#     x1, y1, x2, y2 = [int(v) for v in face.bbox]

#     # -----------------------------------------------------
#     # CHANGE: CARTOON / GHIBLI / ILLUSTRATION
#     # Keep face detection, but skip age/gender predictions.
#     # -----------------------------------------------------

#     if non_photo:
#         return {
#             "accepted": None,
#             "posting_status": "NON_PHOTO_WARNING",
#             "reason": (
#                 f"{type_warning} Face detection was attempted, "
#                 "but age and gender predictions were skipped "
#                 "because the image is illustration-like."
#             ),
#             "image_type_check": image_type,
#             "image_type_review_required": True,
#             "face_detected": True,
#             "face_count": 1,
#             "face_visibility": visibility,
#             "face_box_xyxy": [x1, y1, x2, y2],
#             "estimated_age_group": "Not analyzed",
#             "age_model_confidence": None,
#             "possible_minor_risk": None,
#             "age_verification_required": True,
#             "gender_presentation_prediction": "Not analyzed",
#             "gender_note": (
#                 "Gender prediction was skipped for cartoon or "
#                 "illustration-style images."
#             ),
#             "age_gender_prediction_performed": False,
#             "face_detection_confidence": round(
#                 float(face.det_score), 4
#             )
#         }

#     # -----------------------------------------------------
#     # REAL-PHOTO-LIKE IMAGE
#     # Continue analysis even when visibility needs review.
#     # -----------------------------------------------------

#     try:
#         age_result = estimate_age_group(
#             image,
#             [x1, y1, x2, y2]
#         )
#         age_label = age_result["estimated_age_group"]
#         minor_risk = possible_minor_age_group(age_label)
#     except Exception as exc:
#         print("Age prediction error:", repr(exc))

#         age_result = {
#             "estimated_age_group": "Uncertain",
#             "confidence": None,
#             "note": "Age estimation could not be completed."
#         }
#         age_label = "Uncertain"
#         minor_risk = True

#     gender_value = getattr(face, "gender", None)

#     if gender_value is None:
#         gender_prediction = "Uncertain"
#     elif int(gender_value) == 0:
#         gender_prediction = "Female-presentation prediction"
#     elif int(gender_value) == 1:
#         gender_prediction = "Male-presentation prediction"
#     else:
#         gender_prediction = "Uncertain"

#     # -----------------------------------------------------
#     # CHANGE: NO UNDERAGE REJECTION
#     # Underage estimates lead to review / age verification.
#     # They do not set a rejected status.
#     # -----------------------------------------------------

#     if minor_risk:
#         accepted = None
#         posting_status = "AGE_VERIFICATION_REQUIRED"
#         posting_reason = (
#             f"The estimated age group is {age_label}. "
#             "Age verification is required because this estimate may "
#             "include someone under 18. The photo has not been "
#             "automatically rejected."
#         )

#     elif visibility["status"] != "LIKELY_COMPLETE":
#         accepted = None
#         posting_status = "PHOTO_REVIEW_REQUIRED"
#         posting_reason = (
#             visibility["reason"]
#             + " Please review the face region before making a decision."
#         )

#     else:
#         accepted = True
#         posting_status = "INITIAL_CHECKS_PASSED"
#         posting_reason = (
#             f"The estimated age group is {age_label}. Initial photo "
#             "checks passed, but this estimate does not prove that "
#             "the user is 18 or older."
#         )

#     return {
#         "accepted": accepted,
#         "posting_status": posting_status,
#         "posting_reason": posting_reason,
#         "reason": posting_reason,
#         "image_type_check": image_type,
#         "image_type_review_required": (
#             image_type.get("status") != "PHOTO_LIKE"
#         ),
#         "face_detected": True,
#         "face_count": 1,
#         "face_visibility": visibility,
#         "estimated_age_group": age_label,
#         "age_model_confidence": age_result["confidence"],
#         "age_note": age_result["note"],
#         "possible_minor_risk": minor_risk,
#         "age_verification_required": True,
#         "gender_presentation_prediction": gender_prediction,
#         "gender_note": (
#             "Appearance-based prediction only; it may be inaccurate "
#             "and does not identify a person's gender identity."
#         ),
#         "face_detection_confidence": round(
#             float(face.det_score), 4
#         ),
#         "face_box_xyxy": [x1, y1, x2, y2],
#         "age_gender_prediction_performed": True,
#         "note": (
#             "Age estimates can be wrong. An estimated age group "
#             "cannot verify adulthood. Use an appropriate age-assurance "
#             "process when required."
#         )
#     }


# # =========================================================
# # IMAGE PREVIEW AND FACE REGION
# # =========================================================

# def encode_image(image_bgr: np.ndarray) -> str:
#     success, buffer = cv2.imencode(
#         ".jpg",
#         image_bgr,
#         [cv2.IMWRITE_JPEG_QUALITY, 90]
#     )

#     if not success:
#         raise ValueError("Could not encode preview image.")

#     encoded = base64.b64encode(
#         buffer.tobytes()
#     ).decode("utf-8")

#     return "data:image/jpeg;base64," + encoded


# def build_image_previews(image: Image.Image, result: dict) -> dict:
#     image_rgb = np.array(image.convert("RGB"))
#     image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#     annotated = image_bgr.copy()
#     height, width = annotated.shape[:2]

#     boxes = []
#     existing_box = result.get("face_box_xyxy")

#     if existing_box and len(existing_box) == 4:
#         boxes.append(existing_box)
#     else:
#         try:
#             detected_faces = face_app.get(image_bgr)

#             for detected_face in detected_faces:
#                 boxes.append([
#                     int(v) for v in detected_face.bbox
#                 ])

#         except Exception as exc:
#             print("Preview face detection error:", repr(exc))

#     image_check = result.get("image_type_check", {})
#     non_photo = image_check.get("status") == "NON_PHOTO"
#     possible_minor = result.get("possible_minor_risk", False)
#     age_group = result.get("estimated_age_group", "")
#     status = result.get("posting_status", "")

#     ORANGE = (0, 140, 255)
#     GREEN = (0, 180, 0)

#     if non_photo:
#         label = "WARNING: NON-PHOTO IMAGE"
#         box_color = ORANGE
#     elif status == "AGE_VERIFICATION_REQUIRED" or possible_minor:
#         label = "AGE VERIFICATION REQUIRED"
#         box_color = ORANGE
#     elif status == "PHOTO_REVIEW_REQUIRED":
#         label = "PHOTO REVIEW"
#         box_color = ORANGE
#     else:
#         label = "FACE DETECTED"
#         box_color = GREEN

#     valid_boxes = []

#     for box in boxes:
#         x1, y1, x2, y2 = [int(v) for v in box]

#         x1 = max(0, min(x1, width - 1))
#         y1 = max(0, min(y1, height - 1))
#         x2 = max(0, min(x2, width - 1))
#         y2 = max(0, min(y2, height - 1))

#         if x2 <= x1 or y2 <= y1:
#             continue

#         valid_boxes.append([x1, y1, x2, y2])

#         cv2.rectangle(
#             annotated,
#             (x1, y1),
#             (x2, y2),
#             box_color,
#             3
#         )

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = max(0.45, min(width, height) / 1000)
#         thickness = 2

#         (text_width, text_height), baseline = cv2.getTextSize(
#             label, font, font_scale, thickness
#         )

#         label_y = min(
#             height - baseline - 2,
#             max(text_height + 10, y1)
#         )

#         rect_top = max(0, label_y - text_height - 8)
#         rect_bottom = min(height - 1, label_y + baseline)
#         rect_right = min(width - 1, x1 + text_width + 8)

#         cv2.rectangle(
#             annotated,
#             (x1, rect_top),
#             (rect_right, rect_bottom),
#             box_color,
#             -1
#         )

#         cv2.putText(
#             annotated,
#             label,
#             (x1 + 4, label_y - 4),
#             font,
#             font_scale,
#             (255, 255, 255),
#             thickness,
#             cv2.LINE_AA
#         )

#     result["face_regions_xyxy"] = valid_boxes
#     result["original_image"] = encode_image(image_bgr)
#     result["annotated_image"] = encode_image(annotated)

#     if not valid_boxes:
#         result["region_note"] = (
#             "No face region could be located. "
#             "The original uploaded photo is shown for review."
#         )
#     else:
#         result["region_note"] = (
#             "The outlined area indicates the detected face region. "
#             "It does not independently prove age or face completeness."
#         )

#     return result


# # =========================================================
# # FRONTEND
# # =========================================================

# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Profile Photo Analyzer</title>

#         <style>
#             * { box-sizing: border-box; }

#             body {
#                 margin: 0;
#                 padding: 35px 16px;
#                 background: #f3f5fa;
#                 color: #202938;
#                 font-family: Arial, sans-serif;
#             }

#             .container { max-width: 850px; margin: auto; }

#             .header, .panel {
#                 background: white;
#                 padding: 25px;
#                 border-radius: 16px;
#                 margin-bottom: 20px;
#                 box-shadow: 0 4px 18px #18233a0d;
#             }

#             h1 { margin-top: 0; font-size: 28px; }
#             h2 { font-size: 19px; margin-top: 0; }

#             .subtitle {
#                 color: #647084;
#                 line-height: 1.6;
#             }

#             .upload-box {
#                 border: 2px dashed #c6d2e5;
#                 padding: 25px;
#                 border-radius: 12px;
#                 text-align: center;
#                 background: #fafbfe;
#             }

#             input[type=file] {
#                 width: 100%;
#                 margin: 15px 0;
#             }

#             button {
#                 width: 100%;
#                 padding: 14px;
#                 background: #315ee7;
#                 color: white;
#                 border: 0;
#                 border-radius: 9px;
#                 font-size: 16px;
#                 font-weight: bold;
#                 cursor: pointer;
#             }

#             button:disabled {
#                 opacity: 0.6;
#                 cursor: wait;
#             }

#             .status {
#                 padding: 14px 16px;
#                 border-radius: 10px;
#                 margin-bottom: 18px;
#                 font-weight: bold;
#                 overflow-wrap: anywhere;
#             }

#             .success { background: #e5f8ec; color: #16743a; }
#             .warning { background: #fff4d8; color: #885800; }
#             .danger { background: #fde9e9; color: #ad2929; }
#             .neutral { background: #edf1f7; color: #34445e; }

#             .grid {
#                 display: grid;
#                 grid-template-columns: repeat(2, minmax(0, 1fr));
#                 gap: 14px;
#             }

#             .card {
#                 border: 1px solid #e4e9f1;
#                 border-radius: 12px;
#                 padding: 17px;
#                 min-width: 0;
#             }

#             .card-label {
#                 color: #69758a;
#                 font-size: 13px;
#                 margin-bottom: 9px;
#             }

#             .card-value {
#                 font-size: 19px;
#                 font-weight: bold;
#                 overflow-wrap: anywhere;
#             }

#             .small {
#                 color: #69758a;
#                 font-size: 13px;
#                 line-height: 1.6;
#                 margin-top: 8px;
#             }

#             .hidden { display: none; }

#             .error {
#                 color: #ad2929;
#                 white-space: pre-wrap;
#                 overflow-wrap: anywhere;
#             }

#             .footer-note {
#                 font-size: 13px;
#                 color: #69758a;
#                 line-height: 1.6;
#             }

#             .photo-preview {
#                 display: block;
#                 width: 100%;
#                 max-height: 440px;
#                 object-fit: contain;
#                 background: #f3f5fa;
#                 border-radius: 10px;
#                 border: 1px solid #e4e9f1;
#             }

#             #uploadedFilename { overflow-wrap: anywhere; }

#             @media (max-width: 560px) {
#                 .grid { grid-template-columns: 1fr; }
#                 .header, .panel { padding: 19px; }
#                 h1 { font-size: 24px; }
#             }
#         </style>
#     </head>

#     <body>
#     <div class="container">

#         <div class="header">
#             <h1>Profile Photo Analyzer</h1>
#             <p class="subtitle">
#                 Validate a profile photo, check face count and visibility,
#                 and view an approximate age-group prediction.
#             </p>

#             <form id="uploadForm">
#                 <div class="upload-box">
#                     <strong>Upload Profile Photo</strong>
#                     <p class="subtitle">
#                         Select a JPG, JPEG, PNG or WEBP image (max 8 MB).
#                     </p>
#                     <input
#                         type="file"
#                         name="file"
#                         id="photo"
#                         accept=".jpg,.jpeg,.png,.webp"
#                         required
#                     >
#                     <button id="analyzeBtn" type="submit">
#                         Analyze Profile Photo
#                     </button>
#                 </div>
#             </form>
#         </div>

#         <div id="loading" class="panel hidden">
#             <h2>Analyzing your photo...</h2>
#             <p class="subtitle">
#                 Checking image type, face count and face visibility.
#                 The first analysis may take longer while models initialize.
#             </p>
#         </div>

#         <div id="resultPanel" class="panel hidden">
#             <h2>Analysis Result</h2>
#             <div id="mainStatus" class="status neutral"></div>

#             <div class="grid">
#                 <div class="card">
#                     <h2>Original Uploaded Photo</h2>
#                     <img id="uploadedPreview" class="photo-preview"
#                          alt="Original uploaded profile photo">
#                     <p id="uploadedFilename" class="small"></p>
#                 </div>

#                 <div class="card">
#                     <h2>Detected Face Region</h2>
#                     <img id="annotatedPreview" class="photo-preview"
#                          alt="Annotated photo with detected face region">
#                     <p id="regionNote" class="small"></p>
#                 </div>
#             </div>

#             <p id="reason" class="subtitle"></p>

#             <div class="grid">
#                 <div class="card">
#                     <div class="card-label">Image Type Check</div>
#                     <div id="imageType" class="card-value">—</div>
#                     <div id="imageConfidence" class="small"></div>
#                 </div>

#                 <div class="card">
#                     <div class="card-label">Face Detection</div>
#                     <div id="faceCount" class="card-value">—</div>
#                     <div id="faceConfidence" class="small"></div>
#                 </div>

#                 <div class="card">
#                     <div class="card-label">Face Visibility</div>
#                     <div id="visibility" class="card-value">—</div>
#                     <div id="visibilityReason" class="small"></div>
#                 </div>

#                 <div class="card">
#                     <div class="card-label">Estimated Age Group</div>
#                     <div id="age" class="card-value">Not performed</div>
#                     <div id="ageConfidence" class="small"></div>
#                 </div>

#                 <div class="card">
#                     <div class="card-label">Gender Presentation</div>
#                     <div id="gender" class="card-value">Not performed</div>
#                     <div id="genderNote" class="small"></div>
#                 </div>

#                 <div class="card">
#                     <div class="card-label">Age Verification</div>
#                     <div id="ageVerification" class="card-value">Required</div>
#                     <div class="small">
#                         Appearance-based age estimation does not verify
#                         that a user is 18 or older.
#                     </div>
#                 </div>
#             </div>

#             <div id="imagePredictions" class="card hidden"
#                  style="margin-top:14px">
#                 <h2>Image Classification Details</h2>
#                 <div id="predictionList"></div>
#             </div>

#             <p id="extraNote" class="footer-note"></p>
#         </div>

#         <div id="errorPanel" class="panel hidden">
#             <h2>Analysis Error</h2>
#             <p id="errorText" class="error"></p>
#         </div>

#         <div class="panel">
#             <p class="footer-note">
#                 Note: Image classification and face visibility checks can
#                 make mistakes. Age estimation is not age verification.
#                 A separate age-assurance process is required before enforcing
#                 an 18+ posting policy.
#             </p>
#         </div>

#     </div>

#     <script>
#         const form = document.getElementById("uploadForm");
#         const button = document.getElementById("analyzeBtn");

#         const hide = id =>
#             document.getElementById(id).classList.add("hidden");

#         const show = id =>
#             document.getElementById(id).classList.remove("hidden");

#         function setText(id, value) {
#             document.getElementById(id).textContent =
#                 value === undefined || value === null || value === ""
#                 ? "—" : String(value);
#         }

#         function setStatus(text, type) {
#             const el = document.getElementById("mainStatus");
#             el.textContent = text;
#             el.className = "status " + type;
#         }

#         form.addEventListener("submit", async event => {
#             event.preventDefault();

#             const file = document.getElementById("photo").files[0];

#             if (!file) {
#                 alert("Please select a profile photo.");
#                 return;
#             }

#             if (file.size > 8 * 1024 * 1024) {
#                 alert("Maximum upload size is 8 MB.");
#                 return;
#             }

#             hide("resultPanel");
#             hide("errorPanel");
#             show("loading");

#             button.disabled = true;
#             button.textContent = "Analyzing...";

#             const controller = new AbortController();
#             const timeout = setTimeout(
#                 () => controller.abort(), 180000
#             );

#             try {
#                 const formData = new FormData(form);

#                 const response = await fetch(
#                     "/analyze-profile-photo",
#                     {
#                         method: "POST",
#                         body: formData,
#                         signal: controller.signal
#                     }
#                 );

#                 const data = await response.json();

#                 if (!response.ok) {
#                     throw new Error(
#                         data.reason || data.detail ||
#                         "Request failed: " + response.status
#                     );
#                 }

#                 renderResult(data);
#                 show("resultPanel");

#             } catch (error) {
#                 setText(
#                     "errorText",
#                     error.name === "AbortError"
#                         ? "Analysis timed out. Check your server terminal."
#                         : error.message
#                 );
#                 show("errorPanel");
#             } finally {
#                 clearTimeout(timeout);
#                 hide("loading");
#                 button.disabled = false;
#                 button.textContent = "Analyze Profile Photo";
#             }
#         });

#         function renderResult(data) {
#             const originalPreview =
#                 document.getElementById("uploadedPreview");
#             const annotatedPreview =
#                 document.getElementById("annotatedPreview");

#             originalPreview.removeAttribute("src");
#             annotatedPreview.removeAttribute("src");

#             if (data.original_image) {
#                 originalPreview.src = data.original_image;
#             }

#             if (data.annotated_image) {
#                 annotatedPreview.src = data.annotated_image;
#             }

#             setText("uploadedFilename", data.filename || "Uploaded photo");
#             setText(
#                 "regionNote",
#                 data.region_note || "No region information available."
#             );

#             const status = data.posting_status || "PHOTO_REVIEW_REQUIRED";
#             const accepted = data.accepted === true;
#             const imageCheck = data.image_type_check || {};

#             if (status === "NON_PHOTO_WARNING") {
#                 setStatus(
#                     "WARNING — " +
#                     (imageCheck.display_label || "NON-PHOTO IMAGE").toUpperCase(),
#                     "warning"
#                 );
#             } else if (status === "AGE_VERIFICATION_REQUIRED") {
#                 setStatus(
#                     "PHOTO REVIEW — AGE VERIFICATION REQUIRED",
#                     "warning"
#                 );
#             } else if (status === "PHOTO_REVIEW_REQUIRED") {
#                 setStatus("PHOTO REVIEW REQUIRED", "warning");
#             } else if (accepted) {
#                 setStatus("PHOTO CHECKS PASSED", "success");
#             } else {
#                 // Changed: never display "PHOTO REJECTED".
#                 setStatus("PHOTO REVIEW REQUIRED", "warning");
#             }

#             setText("reason", data.reason || data.posting_reason);

#             setText(
#                 "imageType",
#                 imageCheck.display_label ||
#                 imageCheck.top_class ||
#                 imageCheck.status
#             );

#             setText(
#                 "imageConfidence",
#                 imageCheck.top_class_confidence !== undefined
#                     ? "Top-class confidence: " +
#                       (imageCheck.top_class_confidence * 100).toFixed(1) + "%"
#                     : ""
#             );

#             setText(
#                 "faceCount",
#                 data.face_count === undefined
#                     ? (data.face_detected ? "Face detected" : "No face")
#                     : data.face_count +
#                       (data.face_count === 1 ? " face" : " faces")
#             );

#             setText(
#                 "faceConfidence",
#                 data.face_detection_confidence !== undefined
#                     ? "Detection confidence: " +
#                       (data.face_detection_confidence * 100).toFixed(1) + "%"
#                     : ""
#             );

#             const visibility = data.face_visibility || {};
#             setText("visibility", visibility.status || "Not checked");
#             setText("visibilityReason", visibility.reason || "");

#             setText(
#                 "age",
#                 data.estimated_age_group || "Not performed"
#             );

#             setText(
#                 "ageConfidence",
#                 data.age_model_confidence !== undefined &&
#                 data.age_model_confidence !== null
#                     ? "Model score: " +
#                       (data.age_model_confidence * 100).toFixed(1) + "%"
#                     : data.age_note || ""
#             );

#             setText(
#                 "gender",
#                 data.gender_presentation_prediction || "Not performed"
#             );
#             setText("genderNote", data.gender_note || "");

#             setText(
#                 "ageVerification",
#                 data.age_verification_required === false
#                     ? "Not required"
#                     : "Required / not verified"
#             );

#             const predictions = imageCheck.all_predictions || [];
#             const predictionList = document.getElementById("predictionList");

#             predictionList.replaceChildren();

#             if (predictions.length) {
#                 predictions.forEach(item => {
#                     const line = document.createElement("p");
#                     const score = (item.confidence * 100).toFixed(1);
#                     line.textContent = item.label + " — " + score + "%";
#                     predictionList.appendChild(line);
#                 });
#                 show("imagePredictions");
#             } else {
#                 hide("imagePredictions");
#             }

#             setText("extraNote", data.note || imageCheck.note || "");
#         }
#     </script>
#     </body>
#     </html>
#     """


# # =========================================================
# # API ENDPOINT
# # =========================================================

# @app.post("/analyze-profile-photo")
# async def analyze_profile_photo(file: UploadFile = File(...)):
#     filename = file.filename or "upload.jpg"

#     extension = (
#         "." + filename.rsplit(".", 1)[-1].lower()
#         if "." in filename else ""
#     )

#     if extension not in ALLOWED_EXTENSIONS:
#         return JSONResponse(
#             status_code=400,
#             content={
#                 "reason": "Only JPG, JPEG, PNG and WEBP files are allowed."
#             }
#         )

#     contents = await file.read(MAX_UPLOAD_BYTES + 1)

#     if not contents:
#         return JSONResponse(
#             status_code=400,
#             content={"reason": "The uploaded file is empty."}
#         )

#     if len(contents) > MAX_UPLOAD_BYTES:
#         return JSONResponse(
#             status_code=413,
#             content={"reason": "Maximum upload size is 8 MB."}
#         )

#     try:
#         with Image.open(io.BytesIO(contents)) as source_image:
#             image = ImageOps.exif_transpose(source_image).convert("RGB")
#             image.load()

#     except (UnidentifiedImageError, OSError, ValueError):
#         return JSONResponse(
#             status_code=400,
#             content={
#                 "reason": "The uploaded file is not a valid supported image."
#             }
#         )

#     try:
#         result = analyze_image(image)
#         result["filename"] = filename
#         result = build_image_previews(image, result)

#         return JSONResponse(content=result)

#     except Exception as exc:
#         print("Profile photo analysis error:", repr(exc))

#         return JSONResponse(
#             status_code=500,
#             content={
#                 "reason": "Analysis failed. Check server logs."
#             }
#         )


# @app.get("/health")
# def health():
#     return {"status": "running"}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         "app:app",
#         host="127.0.0.1",
#         port=8000,
#         reload=False
#     )





import io
import re
import base64

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

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

MIN_SHARPNESS_SCORE = 30.0

app = FastAPI(title=APP_TITLE, version="2.1.0")


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
            "reason": "Image classification could not be completed."
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


def is_illustration(image_type: dict) -> bool:
    """
    Return True when the top image category is an illustration,
    cartoon, anime, digital painting, or computer-generated character.
    """

    label = str(image_type.get("top_class", "")).lower()

    return any(term in label for term in (
        "cartoon",
        "anime",
        "illustration",
        "digital painting",
        "computer-generated"
    ))


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
            "reason": "The detected face region is invalid.",
            "face_box_touches_image_edge": True,
            "sharpness_score": 0.0,
            "landmarks_available": False,
            "landmarks_inside_image": False
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

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

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
        reason = "The face box touches or is very close to an image edge."

    elif sharpness < MIN_SHARPNESS_SCORE:
        status = "UNCERTAIN"
        reason = "The detected face may be blurry."

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
    """
    Estimate age group from the detected face crop.

    This function is called for a single detected face in a
    non-illustration image, including when the age estimate may
    fall into a group containing people under 18.
    """

    x1, y1, x2, y2 = [int(v) for v in face_box]

    width, height = image.size

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("The detected face crop is invalid.")

    face_crop = image.crop((x1, y1, x2, y2)).convert("RGB")

    predictions = age_pipeline(face_crop)

    if not predictions:
        raise ValueError("The age model returned no predictions.")

    best = max(predictions, key=lambda item: float(item["score"]))
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
    """
    Flag age groups that may include a person under 18.

    For example, 10-19 is flagged because that group includes
    both minors and adults. This flag does not suppress predictions
    and does not itself produce a rejection.
    """

    label = (age_label or "").lower().replace(" ", "")

    if label in {"0-2", "3-9", "10-19"}:
        return True

    if any(term in label for term in (
        "under18",
        "under-18",
        "child",
        "minor"
    )):
        return True

    match = re.search(r"(\d+)\s*[-–]\s*(\d+)", label)

    if match:
        lower = int(match.group(1))
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

    illustration = is_illustration(image_type)

    # Face detection runs regardless of the image-type result.
    faces = face_app.get(image_bgr)

    base_result = {
        "image_type_check": image_type,
        "image_type_review_required": image_type.get("status") != "PHOTO_LIKE",
        "face_detected": len(faces) > 0,
        "face_count": len(faces),
        "age_gender_prediction_performed": False,
        "estimated_age_group": None,
        "age_model_confidence": None,
        "gender_presentation_prediction": None,
        "possible_minor_risk": False,
        "age_verification_required": True
    }

    # -----------------------------------------------------
    # NO FACE DETECTED
    # -----------------------------------------------------

    if len(faces) == 0:
        if illustration:
            status = "CARTOON_WARNING"
            reason = (
                f"Warning: this image looks like "
                f"{image_type['display_label']}. "
                "No face was detected."
            )
        else:
            status = "PHOTO_REVIEW"
            reason = (
                "No face was detected. Please review the uploaded image."
            )

        return {
            **base_result,
            "posting_status": status,
            "reason": reason,
            "face_visibility": {
                "status": "NOT_AVAILABLE",
                "reason": "No face was detected."
            },
            "note": (
                "Face detection can make mistakes. "
                "Age and gender estimates were not performed because "
                "no face was detected."
            )
        }

    # -----------------------------------------------------
    # GROUP PHOTO
    # Do not assign an individual age or gender to an arbitrary face.
    # -----------------------------------------------------

    if len(faces) > 1:
        return {
            **base_result,
            "posting_status": "GROUP_PHOTO_WARNING",
            "reason": (
                "Multiple faces were detected. Please use a profile "
                "photo containing one clearly visible face."
            ),
            "face_visibility": {
                "status": "MULTIPLE_FACES",
                "reason": "Individual age and gender predictions were skipped."
            },
            "note": (
                "A group photo was detected. Age and gender predictions "
                "were not assigned to an individual."
            )
        }

    # -----------------------------------------------------
    # EXACTLY ONE FACE
    # -----------------------------------------------------

    face = faces[0]

    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    visibility = assess_visibility(image_bgr, face)

    # -----------------------------------------------------
    # CARTOON / GHIBLI / ILLUSTRATION
    # Face detection and visibility are shown, but age and gender
    # are deliberately NOT predicted or displayed.
    # -----------------------------------------------------

    if illustration:
        return {
            **base_result,
            "posting_status": "CARTOON_WARNING",
            "reason": (
                f"Warning: this image may be a "
                f"{image_type['display_label']}. "
                "Face detection was attempted, but age and gender "
                "estimates are not shown for illustration-style images."
            ),
            "face_detected": True,
            "face_count": 1,
            "face_visibility": visibility,
            "face_detection_confidence": round(
                float(face.det_score), 4
            ),
            "face_box_xyxy": [x1, y1, x2, y2],
            "age_gender_prediction_performed": False,
            "estimated_age_group": None,
            "age_model_confidence": None,
            "gender_presentation_prediction": None,
            "age_verification_required": True,
            "note": (
                "Cartoon and Ghibli-style images can trigger a warning. "
                "Age and gender are not displayed for these images because "
                "appearance-based predictions may be unreliable."
            )
        }

    # -----------------------------------------------------
    # AGE AND GENDER
    #
    # IMPORTANT:
    # Do NOT return early when the face is partial/cropped or
    # visibility is uncertain. Still attempt the predictions.
    #
    # Underage age-group estimates do NOT suppress predictions
    # and do NOT cause a rejection.
    # -----------------------------------------------------

    age_result = None
    age_error = None

    try:
        age_result = estimate_age_group(
            image,
            [x1, y1, x2, y2]
        )
    except Exception as exc:
        print("Age estimation error:", repr(exc))
        age_error = "The age model could not estimate an age group."

    # InsightFace gender value: 0 = female, 1 = male.
    gender_prediction = None
    gender_error = None

    try:
        gender_value = int(face.gender)

        if gender_value == 0:
            gender_prediction = "Female-presentation prediction"
        elif gender_value == 1:
            gender_prediction = "Male-presentation prediction"
        else:
            gender_prediction = "Uncertain"
    except Exception as exc:
        print("Gender prediction error:", repr(exc))
        gender_error = "The gender-presentation model could not provide a result."

    age_label = (
        age_result["estimated_age_group"]
        if age_result else None
    )

    minor_risk = (
        possible_minor_age_group(age_label)
        if age_label else False
    )

    # This is an age-assurance flag only. It does not reject the image
    # and does not remove the age/gender results.
    age_verification_required = (
        True if age_result is None else minor_risk
    )

    # -----------------------------------------------------
    # DISPLAY STATUS
    # -----------------------------------------------------

    if minor_risk:
        posting_status = "PHOTO_REVIEW"
        posting_reason = (
            f"The estimated age group is {age_label}. "
            "This age range may include people under 18, so age "
            "verification is required. The age and gender predictions "
            "are still displayed below."
        )

    elif visibility["status"] != "LIKELY_COMPLETE":
        posting_status = "PHOTO_REVIEW"
        posting_reason = (
            "The face was detected, but its visibility needs review. "
            "Age and gender estimation was still attempted."
        )

    elif age_result is None or gender_prediction is None:
        posting_status = "PHOTO_REVIEW"
        posting_reason = (
            "The face was detected, but one or more predictions "
            "could not be completed."
        )

    else:
        posting_status = "PHOTO_CHECKS_COMPLETED"
        posting_reason = (
            "The initial photo checks have completed. "
            "The age group is an estimate and does not verify that "
            "the person is 18 or older."
        )

    return {
        **base_result,
        "posting_status": posting_status,
        "reason": posting_reason,
        "posting_reason": posting_reason,
        "face_detected": True,
        "face_count": 1,
        "face_visibility": visibility,
        "estimated_age_group": age_label,
        "age_model_confidence": (
            age_result["confidence"] if age_result else None
        ),
        "age_note": (
            age_result["note"] if age_result else age_error
        ),
        "possible_minor_risk": minor_risk,
        "age_verification_required": age_verification_required,
        "gender_presentation_prediction": gender_prediction,
        "gender_note": (
            "Appearance-based model prediction only. It may be inaccurate "
            "and does not identify a person's gender identity."
        ),
        "face_detection_confidence": round(
            float(face.det_score), 4
        ),
        "face_box_xyxy": [x1, y1, x2, y2],
        "age_gender_prediction_performed": (
            age_result is not None or gender_prediction is not None
        ),
        "age_error": age_error,
        "gender_error": gender_error,
        "note": (
            "Age estimates can be wrong. An age group such as 10-19 "
            "includes both minors and adults. Use a separate age-assurance "
            "process where required. No automatic rejection is made "
            "based solely on the estimated age group."
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
    Add the original and annotated photo previews to the API response.
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
        # Also show detected regions for group photos.
        try:
            detected_faces = face_app.get(image_bgr)

            for detected_face in detected_faces:
                boxes.append([
                    int(v) for v in detected_face.bbox
                ])

        except Exception as exc:
            print("Preview face detection error:", repr(exc))

    status = result.get("posting_status", "")
    image_type = result.get("image_type_check", {})
    possible_minor = result.get("possible_minor_risk", False)
    visibility = result.get("face_visibility", {})

    if status == "CARTOON_WARNING":
        label = "WARNING: CARTOON IMAGE"
        box_color = (0, 140, 255)  # Orange, BGR

    elif status == "GROUP_PHOTO_WARNING":
        label = "WARNING: MULTIPLE FACES"
        box_color = (0, 140, 255)

    elif possible_minor:
        label = "AGE VERIFICATION REQUIRED"
        box_color = (0, 140, 255)

    elif visibility.get("status") not in (
        None,
        "LIKELY_COMPLETE",
        "NOT_AVAILABLE",
        "MULTIPLE_FACES"
    ):
        label = "PHOTO REVIEW"
        box_color = (0, 140, 255)

    elif status == "PHOTO_REVIEW":
        label = "PHOTO REVIEW"
        box_color = (0, 140, 255)

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
            label,
            font,
            font_scale,
            thickness
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
            "A bounding box does not prove face completeness, identity, "
            "or age."
        )

    return result


# =========================================================
# FRONTEND
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
                line-height: 1.5;
            }

            .success { background: #e5f8ec; color: #16743a; }
            .warning { background: #fff4d8; color: #885800; }
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
                overflow-wrap: anywhere;
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
                Check image type, face count, face visibility and approximate
                age-group prediction.
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
                    <div class="card-label">Image Type</div>
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
                        Not verified
                    </div>
                    <div id="ageVerificationNote" class="small">
                        An estimated age group cannot confirm whether a
                        person is 18 or older.
                    </div>
                </div>
            </div>

            <div
                id="imagePredictions"
                class="card hidden"
                style="margin-top:14px"
            >
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
                Note: Image classification, face detection and age estimation
                can make mistakes. Underage estimates do not suppress the
                displayed age/gender predictions. A separate age-assurance
                process is required where an 18+ policy applies.
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

            const status = data.posting_status || "PHOTO_REVIEW";
            const imageCheck = data.image_type_check || {};

            if (status === "CARTOON_WARNING") {
                setStatus(
                    "WARNING — " +
                    (imageCheck.display_label || "CARTOON IMAGE").toUpperCase(),
                    "warning"
                );
            } else if (status === "GROUP_PHOTO_WARNING") {
                setStatus(
                    "PHOTO REVIEW — MULTIPLE FACES DETECTED",
                    "warning"
                );
            } else if (status === "PHOTO_REVIEW") {
                if (data.possible_minor_risk) {
                    setStatus(
                        "PHOTO REVIEW — AGE VERIFICATION REQUIRED",
                        "warning"
                    );
                } else {
                    setStatus("PHOTO REVIEW", "warning");
                }
            } else if (status === "PHOTO_CHECKS_COMPLETED") {
                setStatus("PHOTO CHECKS COMPLETED", "success");
            } else {
                setStatus("PHOTO REVIEW", "warning");
            }

            setText("reason", data.reason || data.posting_reason);

            setText(
                "imageType",
                imageCheck.display_label ||
                imageCheck.top_class ||
                imageCheck.status
            );

            setText(
                "imageConfidence",
                imageCheck.top_class_confidence !== undefined
                    ? "Top image-type score: " +
                      (imageCheck.top_class_confidence * 100).toFixed(1) + "%"
                    : ""
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
                    ? "Detection score: " +
                      (data.face_detection_confidence * 100).toFixed(1) + "%"
                    : ""
            );

            const visibility = data.face_visibility || {};

            setText("visibility", visibility.status || "Not checked");
            setText("visibilityReason", visibility.reason || "");

            const ageGenderPerformed =
                data.age_gender_prediction_performed === true;

            // Cartoon/illustration: do not show age or gender.
            // Underage estimate: show age and gender normally.
            if (status === "CARTOON_WARNING") {
                setText("age", "Not shown for illustration");
                setText("ageConfidence", "");
                setText("gender", "Not shown for illustration");
                setText(
                    "genderNote",
                    "Age and gender estimates are skipped for cartoon/Ghibli-style images."
                );
            } else {
                setText(
                    "age",
                    data.estimated_age_group || "Could not estimate"
                );

                setText(
                    "ageConfidence",
                    data.age_model_confidence !== undefined &&
                    data.age_model_confidence !== null
                        ? "Model score: " +
                          (data.age_model_confidence * 100).toFixed(1) + "%"
                        : (data.age_note || data.age_error || "")
                );

                setText(
                    "gender",
                    data.gender_presentation_prediction ||
                    "Could not estimate"
                );

                setText(
                    "genderNote",
                    data.gender_note || data.gender_error || ""
                );
            }

            if (data.possible_minor_risk === true) {
                setText(
                    "ageVerification",
                    "Required / not verified"
                );
                setText(
                    "ageVerificationNote",
                    "The estimated age group may include people under 18. "
                    + "The age and gender predictions are still displayed. "
                    + "This estimate does not verify actual age."
                );
            } else {
                setText(
                    "ageVerification",
                    "Not verified"
                );
                setText(
                    "ageVerificationNote",
                    "An estimated age group cannot confirm whether a "
                    + "person is 18 or older."
                );
            }

            if (status === "CARTOON_WARNING") {
                setText(
                    "ageVerificationNote",
                    "Age cannot be verified from this illustration. "
                    + "Age and gender predictions are not shown."
                );
            }

            const predictions = imageCheck.all_predictions || [];
            const predictionList =
                document.getElementById("predictionList");

            predictionList.replaceChildren();

            if (predictions.length) {
                predictions.forEach(item => {
                    const line = document.createElement("p");
                    const score =
                        (Number(item.confidence) * 100).toFixed(2);

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

    extension = (
        "." + filename.rsplit(".", 1)[-1].lower()
        if "." in filename else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "reason": "Only JPG, JPEG, PNG and WEBP files are allowed."
            }
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        return JSONResponse(
            status_code=400,
            content={"reason": "The uploaded file is empty."}
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"reason": "Maximum upload size is 8 MB."}
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "reason": "The uploaded file is not a valid supported image."
            }
        )

    try:
        result = analyze_image(image)
        result["filename"] = filename

        result = build_image_previews(image, result)

        return JSONResponse(content=result)

    except Exception as exc:
        print("Profile photo analysis error:", repr(exc))

        return JSONResponse(
            status_code=500,
            content={
                "reason": (
                    "Analysis failed. Check the server terminal "
                    "for the error details."
                )
            }
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {"status": "running"}


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
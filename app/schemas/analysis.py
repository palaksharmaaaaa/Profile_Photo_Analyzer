from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FaceVisibilityResult(BaseModel):
    status: str
    reason: str
    face_box_touches_image_edge: bool = False
    sharpness_score: float = 0.0
    landmarks_available: bool = False
    landmarks_inside_image: bool = False
    brightness_score: Optional[float] = None
    face_area_ratio: Optional[float] = None


class ImagePredictionItem(BaseModel):
    label: str
    confidence: float


class ImageTypeResult(BaseModel):
    status: str
    top_class: str
    top_class_confidence: float
    photo_score: float
    cartoon_score: float
    display_label: str
    all_predictions: List[ImagePredictionItem] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    reason: str


class ImageInfoResult(BaseModel):
    has_information: bool
    entropy: float
    std_dev: float
    laplacian_var: float
    reason: str


NOT_DETECTED: str = "Not Detected"


class OtherFaceItem(BaseModel):
    face_id: int
    face_type: str  # animal, bird, cartoon_or_illustration, biological_diagram
    label: str
    confidence: float
    box_xyxy: List[int]


class HumanFaceItem(BaseModel):
    face_id: int
    box_xyxy: List[int]
    detection_confidence: float
    gender: str = NOT_DETECTED
    gender_confidence: Optional[float] = None
    age_group: str = NOT_DETECTED
    estimated_age: Optional[float] = None
    age_confidence: Optional[float] = None
    possible_minor_risk: bool = False
    face_visibility: Optional[FaceVisibilityResult] = None
    is_primary: bool = False


class AnalysisResponse(BaseModel):
    accepted: bool
    posting_status: str
    posting_reason: str
    reason: str
    has_information: bool = True
    image_info_check: ImageInfoResult
    image_type_check: ImageTypeResult
    image_type_review_required: bool = False
    face_detected: bool
    total_face_count: int = 0
    real_human_face_count: int = 0
    other_face_count: int = 0
    human_faces: List[HumanFaceItem] = Field(default_factory=list)
    other_faces: List[OtherFaceItem] = Field(default_factory=list)

    # Primary / Legacy fields for single-face callers
    face_count: int = 0
    face_visibility: Optional[FaceVisibilityResult] = None
    estimated_age_group: str = NOT_DETECTED
    age_model_confidence: Optional[float] = None
    age_note: Optional[str] = None
    age_prediction_error: Optional[str] = None
    possible_minor_risk: bool = False
    age_verification_required: bool = True
    gender_presentation_prediction: str = NOT_DETECTED

    gender_note: Optional[str] = None
    gender_prediction_error: Optional[str] = None
    face_detection_confidence: Optional[float] = None
    face_box_xyxy: Optional[List[int]] = None
    face_regions_xyxy: Optional[List[List[int]]] = None
    age_gender_prediction_performed: bool = False
    original_image: Optional[str] = None
    annotated_image: Optional[str] = None
    filename: Optional[str] = None
    region_note: Optional[str] = None
    note: Optional[str] = None


class ErrorResponse(BaseModel):
    accepted: bool = False
    reason: str


from pydantic import BaseModel, Field


class FaceVisibilityResult(BaseModel):
    status: str
    reason: str
    face_box_touches_image_edge: bool = False
    sharpness_score: float = 0.0
    landmarks_available: bool = False
    landmarks_inside_image: bool = False
    brightness_score: float | None = None
    face_area_ratio: float | None = None


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
    all_predictions: list[ImagePredictionItem] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
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
    box_xyxy: list[int]
    box_normalized: list[float] = Field(default_factory=list)  # [ymin%, xmin%, ymax%, xmax%]


class HumanFaceItem(BaseModel):
    face_id: int
    box_xyxy: list[int]
    box_normalized: list[float] = Field(default_factory=list)  # [ymin%, xmin%, ymax%, xmax%]
    detection_confidence: float
    gender: str = NOT_DETECTED
    gender_confidence: float | None = None
    age_group: str = NOT_DETECTED
    estimated_age: float | None = None
    age_confidence: float | None = None
    possible_minor_risk: bool = False
    face_visibility: FaceVisibilityResult | None = None
    is_primary: bool = False


class PipelineStepItem(BaseModel):
    step_id: str
    step_number: int
    title: str
    status: str  # completed, failed, skipped
    detail: str


class TechnicalDiagnostics(BaseModel):
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    file_format_detected: str = "unknown"
    entropy_score: float = 0.0
    std_dev_score: float = 0.0
    laplacian_var: float = 0.0
    models_executed: list[str] = Field(default_factory=list)


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
    human_faces: list[HumanFaceItem] = Field(default_factory=list)
    other_faces: list[OtherFaceItem] = Field(default_factory=list)
    pipeline_steps: list[PipelineStepItem] = Field(default_factory=list)
    technical_diagnostics: TechnicalDiagnostics | None = None
    processing_time_ms: float = 0.0

    # Primary / Legacy fields for single-face callers
    face_count: int = 0
    face_visibility: FaceVisibilityResult | None = None
    estimated_age_group: str = NOT_DETECTED
    age_model_confidence: float | None = None
    age_note: str | None = None
    age_prediction_error: str | None = None
    possible_minor_risk: bool = False
    age_verification_required: bool = True
    gender_presentation_prediction: str = NOT_DETECTED

    gender_note: str | None = None
    gender_prediction_error: str | None = None
    face_detection_confidence: float | None = None
    face_box_xyxy: list[int] | None = None
    face_regions_xyxy: list[list[int]] | None = None
    age_gender_prediction_performed: bool = False
    original_image: str | None = None
    annotated_image: str | None = None
    filename: str | None = None
    region_note: str | None = None
    note: str | None = None


class ErrorResponse(BaseModel):
    accepted: bool = False
    reason: str

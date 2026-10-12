"""The public response contract of ``POST /analyze-profile-photo``."""

from enum import StrEnum

from pydantic import BaseModel, Field

NOT_DETECTED = "Not Detected"
UNSURE = "Unsure"


class Status(StrEnum):
    """Overall outcome. ``PASSED`` is the only status with ``accepted = true``."""

    NO_INFORMATION = "NO_INFORMATION"  # blank, solid colour or uniform image
    NO_HUMAN_FACE = "NO_HUMAN_FACE"  # nothing that looks like a face was found
    ONLY_NON_HUMAN_FACES = "ONLY_NON_HUMAN_FACES"  # animals, birds, cartoons ... but no real human face
    MULTIPLE_HUMAN_FACES = "MULTIPLE_HUMAN_FACES"
    NON_HUMAN_FACE_PRESENT = "NON_HUMAN_FACE_PRESENT"  # one human face, plus an animal/cartoon face
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # a face or the image type could not be verified
    POOR_FACE_QUALITY = "POOR_FACE_QUALITY"  # cropped, blurry, dark, too small or turned away
    AGE_UNSURE = "AGE_UNSURE"
    POSSIBLE_MINOR = "POSSIBLE_MINOR"
    PASSED = "PASSED"


class FaceQuality(BaseModel):
    """Framing, exposure, sharpness and pose of one face."""

    status: str  # LIKELY_COMPLETE, UNCERTAIN or PARTIAL_OR_CROPPED
    reason: str
    touches_image_edge: bool = False
    sharpness: float = 0.0  # Laplacian variance of the face resized to 112x112 (resolution independent)
    brightness: float | None = None  # mean grey level of the face, 0-255
    face_area_ratio: float | None = None  # face area / image area
    min_side_px: int | None = None
    yaw_ratio: float | None = None  # head turn proxy: about 0 when facing the camera
    landmarks_available: bool = False
    landmarks_inside_image: bool = False
    usable_for_demographics: bool | None = None
    demographics_caveat: str | None = None


class ImageInfo(BaseModel):
    """Whether the image carries any visual information."""

    has_information: bool
    entropy: float
    std_dev: float
    laplacian_var: float
    reason: str


class Prediction(BaseModel):
    label: str
    probability: float


class ImageType(BaseModel):
    """What kind of picture the whole upload is (photograph, cartoon, painting, diagram, ...)."""

    status: str  # PHOTO_LIKE, NON_PHOTO, NEEDS_REVIEW, UNCERTAIN or NOT_CHECKED
    label: str
    photo_share: float | None = None  # probability mass on photographic types
    synthetic_share: float | None = None  # probability mass on drawn, painted or rendered types
    predictions: list[Prediction] = Field(default_factory=list)
    model: str | None = None
    reason: str


class HumanFace(BaseModel):
    """A face verified to be a real human face, with its age and gender."""

    face_id: int
    box_xyxy: list[int]  # pixels in the analysed image
    box_percent: list[float] = Field(default_factory=list)  # same box as [x1, y1, x2, y2] in percent of the image
    detection_confidence: float
    verification_confidence: float | None = None  # SigLIP2 share for "real human face"
    quality: FaceQuality | None = None
    gender: str = NOT_DETECTED  # Female, Male, Unsure or Not Detected
    gender_confidence: float | None = None
    age_group: str = NOT_DETECTED  # a 3-year bracket such as "27 to 30", Unsure or Not Detected
    age_status: str = NOT_DETECTED  # Estimated, Unsure or Not Detected
    estimated_age: float | None = None
    age_range: list[float] | None = None  # 80% interval in years
    age_group_probability: float | None = None  # probability that the true age lies in age_group
    p_under_18: float | None = None
    possible_minor: bool = False
    notes: list[str] = Field(default_factory=list)


class OtherFace(BaseModel):
    """A non-human face or figure: animal, bird, cartoon, anime, painting, 3D render, doll/statue/mask, diagram."""

    face_id: int
    face_type: str
    label: str
    confidence: float | None = None
    box_xyxy: list[int]
    box_percent: list[float] = Field(default_factory=list)
    source: str | None = None  # the detector that found it
    verified_by: str | None = None  # "SigLIP2" or "detector" (found by a detector only)


class UnsureFace(BaseModel):
    """A face candidate the models could not classify with enough certainty."""

    face_id: int
    label: str
    reason: str
    detection_confidence: float | None = None
    box_xyxy: list[int]
    box_percent: list[float] = Field(default_factory=list)


class PipelineStep(BaseModel):
    step_id: str
    step_number: int
    title: str
    status: str  # completed, failed or skipped
    detail: str


class Diagnostics(BaseModel):
    processing_time_ms: float = 0.0
    image_width: int = 0  # analysed image
    image_height: int = 0
    original_width: int | None = None  # as uploaded
    original_height: int | None = None
    exif_orientation_applied: bool = False
    downscaled_for_analysis: bool = False
    file_format: str = "unknown"
    rejected_candidate_count: int = 0  # detections discarded as false positives
    models_executed: list[str] = Field(default_factory=list)  # only models that really ran


class AnalysisResponse(BaseModel):
    status: Status
    accepted: bool
    reason: str
    summary: str
    real_human_face_count: int = 0
    other_face_count: int = 0
    unsure_face_count: int = 0
    human_faces: list[HumanFace] = Field(default_factory=list)
    other_faces: list[OtherFace] = Field(default_factory=list)
    unsure_faces: list[UnsureFace] = Field(default_factory=list)
    image_info: ImageInfo
    image_type: ImageType
    pipeline_steps: list[PipelineStep] = Field(default_factory=list)
    diagnostics: Diagnostics
    original_image: str | None = None  # data URL
    annotated_image: str | None = None  # data URL
    filename: str | None = None
    legend: str | None = None  # explains the box colours of annotated_image
    disclaimer: str | None = None


class ErrorResponse(BaseModel):
    accepted: bool = False
    reason: str

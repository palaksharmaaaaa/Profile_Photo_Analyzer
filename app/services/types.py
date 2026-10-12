"""
The typed shapes that move between pipeline stages.

They are ``TypedDict`` so they serialise straight into the public response (``app/schemas/analysis.py``) while a
type checker still catches a misspelt or missing key between stages.
"""

from typing import NotRequired, TypedDict

Box = list[int]  # [x1, y1, x2, y2] in pixels


class Verdict(TypedDict):
    """What SigLIP2 decided one crop shows."""

    verdict: str  # human, non_human, not_face, unsure or unavailable
    face_type: str
    label: str
    confidence: float | None
    margin: float | None
    scores: dict[str, float]


class Entity(TypedDict):
    """A region reported by an object or anime detector."""

    face_type: str
    label: str
    confidence: float | None
    box_xyxy: Box
    source: str


class OtherFaceItem(TypedDict):
    """A non-human face as reported in the response."""

    face_type: str
    label: str
    confidence: float | None
    box_xyxy: Box
    source: str
    verified_by: str
    face_id: NotRequired[int]
    box_percent: NotRequired[list[float]]


class UnsureFaceItem(TypedDict):
    """A candidate that could not be verified either way."""

    label: str
    reason: str
    detection_confidence: float | None
    box_xyxy: Box
    face_id: NotRequired[int]
    box_percent: NotRequired[list[float]]


class ImageInfoItem(TypedDict):
    has_information: bool
    entropy: float
    std_dev: float
    laplacian_var: float
    reason: str


class PredictionItem(TypedDict):
    label: str
    probability: float


class ImageTypeItem(TypedDict):
    status: str  # PHOTO_LIKE, NON_PHOTO, NEEDS_REVIEW, UNCERTAIN or NOT_CHECKED
    label: str
    photo_share: float | None
    synthetic_share: float | None
    predictions: list[PredictionItem]
    model: str | None
    reason: str


class QualityItem(TypedDict):
    status: str  # LIKELY_COMPLETE, UNCERTAIN or PARTIAL_OR_CROPPED
    reason: str
    touches_image_edge: bool
    sharpness: float
    landmarks_available: bool
    landmarks_inside_image: bool
    brightness: float
    face_area_ratio: float
    yaw_ratio: float | None
    min_side_px: int
    usable_for_demographics: bool
    demographics_caveat: str | None


class HumanFaceItem(TypedDict):
    face_id: int
    box_xyxy: Box
    box_percent: list[float]
    detection_confidence: float
    verification_confidence: float | None
    quality: QualityItem
    gender: str
    gender_confidence: float | None
    age_group: str
    age_status: str
    estimated_age: float | None
    age_range: list[float] | None
    age_group_probability: float | None
    p_under_18: float | None
    possible_minor: bool
    notes: list[str]


class PipelineStepItem(TypedDict):
    step_id: str
    step_number: int
    title: str
    status: str  # completed, failed or skipped
    detail: str


class DiagnosticsItem(TypedDict):
    processing_time_ms: float
    image_width: int
    image_height: int
    original_width: int | None
    original_height: int | None
    exif_orientation_applied: bool
    downscaled_for_analysis: bool
    file_format: str
    rejected_candidate_count: int
    models_executed: list[str]


class AnalysisResult(TypedDict):
    """The complete analysis, shaped like ``AnalysisResponse``."""

    status: str
    accepted: bool
    reason: str
    summary: str
    real_human_face_count: int
    other_face_count: int
    unsure_face_count: int
    human_faces: list[HumanFaceItem]
    other_faces: list[OtherFaceItem]
    unsure_faces: list[UnsureFaceItem]
    image_info: ImageInfoItem
    image_type: ImageTypeItem
    pipeline_steps: list[PipelineStepItem]
    diagnostics: DiagnosticsItem
    annotated_image: str | None
    filename: str | None
    legend: str
    disclaimer: str

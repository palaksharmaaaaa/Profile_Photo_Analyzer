"""
Application settings, grouped by pipeline stage so every stage receives only its own slice.

Every field can be overridden with an environment variable ``PPA_<GROUP>__<FIELD>`` (two underscores between the
group and the field), for example ``PPA_AGE__MINOR_RISK_PROBABILITY=0.2`` or ``PPA_SECURITY__API_KEY=secret``.
Fields that belong to no group keep a single underscore: ``PPA_ENVIRONMENT=production``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class UploadSettings(BaseModel):
    """Limits on what is accepted and how it is decoded."""

    max_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 80_000_000  # guards against decompression bombs (checked while decoding)
    max_dimension: int = 20_000  # longest allowed side in pixels
    working_max_side: int = 2048  # larger images are downscaled to this before analysis


class InformationSettings(BaseModel):
    """Step 2: does the image have any visual information? It is "blank" if any measure is below its minimum."""

    min_entropy: float = 0.5  # Shannon entropy of the grey levels, in bits
    min_std_dev: float = 3.0  # standard deviation of the grey levels
    min_laplacian_var: float = 1.5  # variance of the Laplacian (amount of edge detail)


class DetectionSettings(BaseModel):
    """Step 3a: finding face candidates, animals, birds and people."""

    enable_scrfd: bool = True  # SCRFD weights are non-commercial (see NOTICE.md); False leaves the MIT YuNet detector
    scrfd_threshold: float = 0.40  # minimum SCRFD score for a candidate
    yunet_threshold: float = 0.60  # minimum YuNet score for a candidate
    nms_iou: float = 0.40  # candidates overlapping more than this are the same face
    small_image_upscale_side: int = 1000  # YuNet also runs at 2x on images whose longest side is below this
    animal_threshold: float = 0.45  # YOLOX score for an animal or bird
    animal_detector_only_min_conf: float = 0.65  # an unverified animal box below this is "Unsure", not "Dog"/"Cat"
    anime_threshold: float = 0.35
    person_threshold: float = 0.50  # YOLOX score for a person (gives the age model a body crop)
    body_min_height_ratio: float = 2.0  # a person box is a body crop only if this many times taller than the face


class VerificationSettings(BaseModel):
    """
    Step 3b: verifying each candidate with SigLIP2.

    SigLIP2 compares a crop with text prompts ("a photograph of a real human face", ...). The softmax over the prompt
    similarities is a relative score, not a calibrated probability.
    """

    temperature: float = 100.0  # sharpness of that softmax
    crop_margin: float = 0.30  # context around the face box; 0.30 verified 13 more of 36 hard real faces than 0.15
    human_min_prob: float = 0.60  # a crop is a verified real human face if "human" has at least this share...
    human_min_margin: float = 0.25  # ...and leads the runner-up class by at least this much
    nonhuman_max_human_prob: float = 0.25  # a crop is non-human if the human share is below this
    not_face_min_prob: float = 0.50  # share of "not a face" needed to discard a candidate
    image_type_min_share: float = 0.60  # share a group (photo / drawn) needs for a confident image type


class AgeSettings(BaseModel):
    """Step 4: age and gender."""

    # Every face is shown to MiVOLO in several views (crop margin, mirrored). Their disagreement measures uncertainty.
    views: tuple[tuple[float, bool], ...] = ((0.0, False), (0.0, True), (0.15, False))
    # Average age error of MiVOLO over 8 public datasets in an independent benchmark (arXiv 2602.07815). It sets the
    # width of the reported age range.
    model_mae_years: float = 5.1
    sigma_per_mae: float = 1.25  # Gaussian sigma = this x MAE
    interval_z: float = 1.2816  # z-score of the 80% interval
    unsure_view_std: float = 6.0  # the age is Unsure if the views differ by more than this many years (std)
    min_credible: float = 1.0  # estimates outside this range are Unsure
    max_credible: float = 90.0
    adult_years: float = 18.0
    minor_risk_probability: float = 0.10  # a face needs review as a possible minor at P(age < 18) of at least this
    gender_surety: float = 0.80  # below this model score the gender is Unsure
    gender_min_age: float = 13.0  # gender is Unsure for apparent children below this age


class QualitySettings(BaseModel):
    """Framing, exposure, sharpness and pose of a face."""

    min_sharpness: float = 20.0  # Laplacian variance of the face resized to 112x112 (resolution independent)
    min_brightness: float = 35.0  # mean grey level of the face, 0-255
    max_brightness: float = 235.0
    min_face_area_ratio: float = 0.02  # face area / image area
    max_face_area_ratio: float = 0.90
    min_face_side_px: int = 48  # smaller faces are skipped by the age/gender model
    max_yaw_ratio: float = 0.40  # head-turn proxy above this means "not facing the camera"
    edge_margin_ratio: float = 0.01  # a face closer to the image edge than this share counts as cropped
    min_edge_margin_pixels: int = 5


class ModelSettings(BaseModel):
    """Where the models live and how they run."""

    directory: Path = PROJECT_ROOT / "models"
    min_file_size: int = 50_000  # smaller files are treated as corrupt
    force_cpu: bool = True  # False lets onnxruntime use CUDA or DirectML when installed
    ort_threads: int = 0  # threads per model; 0 means min(4, CPU cores)
    download_timeout_seconds: int = 60
    warm_up_on_start: bool = True  # run one tiny inference per model at start-up so the first request is not slow
    parallel_inference: bool = True  # independent detectors and age views run at the same time
    inference_workers: int = 4  # size of the shared pool behind ``parallel_inference``


class JobSettings(BaseModel):
    """The asynchronous job API: a bounded worker pool in front of the models."""

    workers: int = 2  # analyses running at once
    queue_size: int = 8  # analyses waiting; a full queue answers 503 instead of piling up
    result_ttl_seconds: float = 600  # finished jobs are forgotten after this long


class SecuritySettings(BaseModel):
    """Authentication, abuse controls and CORS."""

    api_key: str | None = None  # sent as X-API-Key or "Authorization: Bearer <key>"
    api_key_header: str = "X-API-Key"
    require_api_key_in_production: bool = True
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60  # requests per client IP (process-local; use the edge for several processes)
    max_concurrent_requests: int = 10
    trusted_proxies: list[str] = ["127.0.0.1", "::1"]  # only these may set X-Forwarded-For
    max_tracked_ips: int = 10_000
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
    cors_allow_credentials: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PPA_", env_nested_delimiter="__", case_sensitive=False)

    app_title: str = "Profile Photo Analyzer"
    app_version: str = "4.0.0"
    environment: str = "development"  # "production" requires PPA_SECURITY__API_KEY
    log_level: str = "INFO"
    log_json: bool = False  # one JSON object per log line, for log collectors

    upload: UploadSettings = UploadSettings()
    information: InformationSettings = InformationSettings()
    detection: DetectionSettings = DetectionSettings()
    verification: VerificationSettings = VerificationSettings()
    age: AgeSettings = AgeSettings()
    quality: QualitySettings = QualitySettings()
    models: ModelSettings = ModelSettings()
    jobs: JobSettings = JobSettings()
    security: SecuritySettings = SecuritySettings()

    @model_validator(mode="after")
    def check_api_key_is_set_in_production(self) -> Settings:
        security = self.security
        if self.environment.lower() == "production" and security.require_api_key_in_production and not security.api_key:
            raise ValueError(
                "Production mode (PPA_ENVIRONMENT='production') requires PPA_SECURITY__API_KEY to be configured "
                "for API-key authentication."
            )
        return self


settings = Settings()

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application runtime configuration (override any field with a ``PPA_`` environment variable)."""

    model_config = SettingsConfigDict(env_prefix="PPA_", case_sensitive=False)

    APP_TITLE: str = "Profile Photo Analyzer"
    APP_VERSION: str = "4.0.0"

    # ---- Upload & decoding limits -------------------------------------------------
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_PIXELS: int = 80_000_000  # decompression-bomb guard on the *decoded* image
    MAX_IMAGE_DIMENSION: int = 20_000
    WORKING_MAX_SIDE: int = 2048  # analysis runs on a copy whose longest side is capped here
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    # ---- Image information check --------------------------------------------------
    MIN_IMAGE_ENTROPY: float = 0.5
    MIN_IMAGE_STD_DEV: float = 3.0
    MIN_LAPLACIAN_VAR: float = 1.5

    # ---- Face detection -----------------------------------------------------------
    SCRFD_THRESHOLD: float = 0.40  # SCRFD-10G candidate threshold (candidates are verified later)
    HUMAN_FACE_CONF_THRESHOLD: float = 0.60  # YuNet candidate threshold (candidates are verified later)
    FACE_NMS_IOU: float = 0.40
    SMALL_IMAGE_UPSCALE_SIDE: int = 1000  # extra 2x detection pass when the longest side is below this
    ANIMAL_CONF_THRESHOLD: float = 0.45
    PERSON_CONF_THRESHOLD: float = 0.50  # person boxes give MiVOLO a body crop in addition to the face
    BODY_MIN_HEIGHT_RATIO: float = (
        2.0  # a person box is used as a body crop only if it is this much taller than the face
    )
    ANIMAL_DETECTOR_ONLY_MIN_CONF: float = 0.65  # unverified animal boxes below this are listed as Unsure, not Dog/Cat
    ANIME_FACE_CONF_THRESHOLD: float = 0.35

    # ---- Semantic (SigLIP2) verification ----------------------------------------
    SEMANTIC_TEMPERATURE: float = 100.0  # softmax sharpness over prompt similarities (not a calibrated probability)
    HUMAN_MIN_PROB: float = 0.60  # a crop is a *verified real human face* only above this share...
    HUMAN_MIN_MARGIN: float = 0.25  # ...and when it beats the runner-up class by this much
    NONHUMAN_MAX_HUMAN_PROB: float = 0.25  # below this human share a crop is declared non-human
    NOT_FACE_MIN_PROB: float = 0.50
    # Crop margin around a face box for verification. Measured on 196 real faces: 0.15 verified 160/160 random but
    # 0/36 hard ones; 0.30 verified 160/160 and 13/36; wider margins lost faces. Tuned on real faces only.
    FACE_CROP_MARGIN: float = 0.30
    IMAGE_TYPE_MIN_MASS: float = 0.60  # share of probability mass a group needs for a confident image type

    # ---- Age / gender -------------------------------------------------------------
    # Views per face, as (crop margin, mirrored); the disagreement between views is the uncertainty measure.
    AGE_VIEWS: tuple[tuple[float, bool], ...] = ((0.0, False), (0.0, True), (0.15, False))
    # Average age error of MiVOLO across 8 public datasets in an independent benchmark (arXiv 2602.07815);
    # NOT re-measured here. It sets the width of the reported age range.
    AGE_MODEL_MAE_YEARS: float = 5.1
    AGE_SIGMA_PER_MAE: float = 1.25  # MAE -> Gaussian sigma for a normal error assumption
    AGE_UNSURE_VIEW_STD: float = 6.0  # views of the same face disagree more than this => age is Unsure
    AGE_MIN_CREDIBLE: float = 1.0
    AGE_MAX_CREDIBLE: float = 90.0
    AGE_INTERVAL_Z: float = 1.2816  # 80% two-sided interval
    MINOR_RISK_PROBABILITY: float = 0.10  # P(age < 18) at or above this flags possible minor
    ADULT_AGE_YEARS: float = 18.0
    GENDER_SURETY_THRESHOLD: float = 0.80
    GENDER_MIN_AGE_YEARS: float = 13.0  # the model is documented as unreliable for children

    # ---- Face quality -------------------------------------------------------------
    MIN_SHARPNESS_SCORE: float = 20.0  # Laplacian variance of the face resized to 112x112
    EDGE_MARGIN_RATIO: float = 0.01
    MIN_EDGE_MARGIN_PIXELS: int = 5
    MIN_FACE_AREA_RATIO: float = 0.02
    MAX_FACE_AREA_RATIO: float = 0.90
    MIN_FACE_SIDE_PX: int = 48  # smaller faces are too low-resolution for age/gender
    MAX_YAW_RATIO: float = 0.40  # |nose offset| / eye distance above this => not frontal
    MIN_BRIGHTNESS: float = 35.0
    MAX_BRIGHTNESS: float = 235.0

    # ---- Models -------------------------------------------------------------------
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    MIN_MODEL_FILE_SIZE: int = 50_000
    FORCE_CPU: bool = True
    ORT_INTRA_OP_THREADS: int = 0  # 0 = onnxruntime default
    DOWNLOAD_TIMEOUT_SECONDS: int = 60

    # ---- Environment & production controls ---------------------------------------
    ENVIRONMENT: str = "development"
    REQUIRE_API_KEY_IN_PRODUCTION: bool = True
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_CONCURRENT_REQUESTS: int = 10
    API_KEY: str | None = None
    API_KEY_HEADER: str = "X-API-Key"
    TRUSTED_PROXIES: list[str] = ["127.0.0.1", "::1"]
    MAX_TRACKED_IPS: int = 10_000
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = False

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
        if self.ENVIRONMENT.lower() == "production" and self.REQUIRE_API_KEY_IN_PRODUCTION and not self.API_KEY:
            raise ValueError(
                "Production mode (ENVIRONMENT='production') requires PPA_API_KEY to be configured "
                "for API-key authentication."
            )
        return self


settings = Settings()

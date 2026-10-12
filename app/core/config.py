"""Application settings. Every field can be overridden with an environment variable named ``PPA_<FIELD>``."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PPA_", case_sensitive=False)

    APP_TITLE: str = "Profile Photo Analyzer"
    APP_VERSION: str = "4.0.0"

    # ---- Uploads and decoding -------------------------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 80_000_000  # guards against decompression bombs (checked while decoding)
    MAX_IMAGE_DIMENSION: int = 20_000  # longest allowed side in pixels
    WORKING_MAX_SIDE: int = 2048  # larger images are downscaled to this before analysis
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    # ---- Step 2: does the image have any visual information? ---------------------------------------------
    # An image is "blank" if any of these measures falls below its minimum.
    MIN_IMAGE_ENTROPY: float = 0.5  # Shannon entropy of the grey levels, in bits
    MIN_IMAGE_STD_DEV: float = 3.0  # standard deviation of the grey levels
    MIN_LAPLACIAN_VAR: float = 1.5  # variance of the Laplacian (amount of edge detail)

    # ---- Step 3a: finding face candidates ------------------------------------------------------------------
    SCRFD_THRESHOLD: float = 0.40  # minimum SCRFD score for a candidate
    HUMAN_FACE_CONF_THRESHOLD: float = 0.60  # minimum YuNet score for a candidate
    FACE_NMS_IOU: float = 0.40  # candidates overlapping more than this are the same face
    SMALL_IMAGE_UPSCALE_SIDE: int = 1000  # YuNet also runs at 2x on images whose longest side is below this
    ANIMAL_CONF_THRESHOLD: float = 0.45  # YOLOX score for an animal or bird
    ANIMAL_DETECTOR_ONLY_MIN_CONF: float = 0.65  # an unverified animal box below this is "Unsure", not "Dog"/"Cat"
    ANIME_FACE_CONF_THRESHOLD: float = 0.35
    PERSON_CONF_THRESHOLD: float = 0.50  # YOLOX score for a person (gives the age model a body crop)
    BODY_MIN_HEIGHT_RATIO: float = 2.0  # a person box is a body crop only if this many times taller than the face

    # ---- Step 3b: verifying each candidate with SigLIP2 ------------------------------------------------------
    # SigLIP2 compares a crop with text prompts ("a photograph of a real human face", ...). The softmax over the
    # prompt similarities is a relative score, not a calibrated probability.
    SEMANTIC_TEMPERATURE: float = 100.0  # sharpness of that softmax
    FACE_CROP_MARGIN: float = 0.30  # context around the face box; 0.30 verified 13 more of 36 hard real faces than 0.15
    HUMAN_MIN_PROB: float = 0.60  # a crop is a verified real human face if "human" has at least this share...
    HUMAN_MIN_MARGIN: float = 0.25  # ...and leads the runner-up class by at least this much
    NONHUMAN_MAX_HUMAN_PROB: float = 0.25  # a crop is non-human if the human share is below this
    NOT_FACE_MIN_PROB: float = 0.50  # share of "not a face" needed to discard a candidate
    IMAGE_TYPE_MIN_SHARE: float = 0.60  # share a group (photo / drawn) needs for a confident image type

    # ---- Step 4: age and gender -----------------------------------------------------------------------------
    # Every face is shown to MiVOLO in several views (crop margin, mirrored). Their disagreement measures uncertainty.
    AGE_VIEWS: tuple[tuple[float, bool], ...] = ((0.0, False), (0.0, True), (0.15, False))
    # Average age error of MiVOLO over 8 public datasets in an independent benchmark (arXiv 2602.07815).
    # It is not re-measured here; it sets the width of the reported age range.
    AGE_MODEL_MAE_YEARS: float = 5.1
    AGE_SIGMA_PER_MAE: float = 1.25  # Gaussian sigma = this x MAE
    AGE_INTERVAL_Z: float = 1.2816  # z-score of the 80% interval
    AGE_UNSURE_VIEW_STD: float = 6.0  # the age is Unsure if the views differ by more than this many years (std)
    AGE_MIN_CREDIBLE: float = 1.0  # estimates outside this range are Unsure
    AGE_MAX_CREDIBLE: float = 90.0
    ADULT_AGE_YEARS: float = 18.0
    MINOR_RISK_PROBABILITY: float = 0.10  # a face is a "possible minor" at P(age < 18) of at least this
    GENDER_SURETY_THRESHOLD: float = 0.80  # below this model score the gender is Unsure
    GENDER_MIN_AGE_YEARS: float = 13.0  # gender is Unsure for apparent children below this age

    # ---- Face quality -----------------------------------------------------------------------------------------
    MIN_SHARPNESS_SCORE: float = 20.0  # Laplacian variance of the face resized to 112x112 (resolution independent)
    MIN_BRIGHTNESS: float = 35.0  # mean grey level of the face, 0-255
    MAX_BRIGHTNESS: float = 235.0
    MIN_FACE_AREA_RATIO: float = 0.02  # face area / image area
    MAX_FACE_AREA_RATIO: float = 0.90
    MIN_FACE_SIDE_PX: int = 48  # smaller faces are skipped by the age/gender model
    MAX_YAW_RATIO: float = 0.40  # head-turn proxy above this means "not facing the camera"
    EDGE_MARGIN_RATIO: float = 0.01  # a face closer to the image edge than this share counts as cropped
    MIN_EDGE_MARGIN_PIXELS: int = 5

    # ---- Models -----------------------------------------------------------------------------------------------
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    MIN_MODEL_FILE_SIZE: int = 50_000  # smaller files are treated as corrupt
    FORCE_CPU: bool = True  # False lets onnxruntime use CUDA or DirectML when installed
    ORT_INTRA_OP_THREADS: int = 0  # threads per model; 0 means min(4, CPU cores)
    DOWNLOAD_TIMEOUT_SECONDS: int = 60

    # ---- Security and abuse controls -----------------------------------------------------------------------------
    ENVIRONMENT: str = "development"  # "production" requires PPA_API_KEY
    REQUIRE_API_KEY_IN_PRODUCTION: bool = True
    API_KEY: str | None = None  # sent as X-API-Key or "Authorization: Bearer <key>"
    API_KEY_HEADER: str = "X-API-Key"
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60  # requests per client IP (process-local)
    MAX_CONCURRENT_REQUESTS: int = 10
    TRUSTED_PROXIES: list[str] = ["127.0.0.1", "::1"]  # only these may set X-Forwarded-For
    MAX_TRACKED_IPS: int = 10_000
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = False

    @model_validator(mode="after")
    def require_api_key_in_production(self) -> Settings:
        if self.ENVIRONMENT.lower() == "production" and self.REQUIRE_API_KEY_IN_PRODUCTION and not self.API_KEY:
            raise ValueError(
                "Production mode (ENVIRONMENT='production') requires PPA_API_KEY to be configured "
                "for API-key authentication."
            )
        return self


settings = Settings()

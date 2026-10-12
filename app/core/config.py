from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime configuration."""

    model_config = SettingsConfigDict(env_prefix="PPA_", case_sensitive=False)

    APP_TITLE: str = "Profile Photo Analyzer"
    APP_VERSION: str = "3.0.0"
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_DIMENSION: int = 4096
    MAX_IMAGE_PIXELS: int = 16_000_000  # 16 Megapixels (prevents decompression bombs)
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Detection & Image Quality Parameters
    PHOTO_SCORE_THRESHOLD: float = 0.35
    MIN_SHARPNESS_SCORE: float = 30.0
    EDGE_MARGIN_RATIO: float = 0.01
    MIN_EDGE_MARGIN_PIXELS: int = 5
    MIN_FACE_AREA_RATIO: float = 0.02
    MAX_FACE_AREA_RATIO: float = 0.90
    MIN_BRIGHTNESS: float = 35.0
    MAX_BRIGHTNESS: float = 235.0

    # Image Information Check Thresholds
    MIN_IMAGE_ENTROPY: float = 0.5
    MIN_IMAGE_STD_DEV: float = 3.0
    MIN_LAPLACIAN_VAR: float = 1.5

    # Confidence & Surety Thresholds
    HUMAN_FACE_CONF_THRESHOLD: float = 0.60
    GENDER_SURETY_THRESHOLD: float = 0.60
    ANIME_FACE_CONF_THRESHOLD: float = 0.35
    ANIMAL_CONF_THRESHOLD: float = 0.35

    # Model Directories and Paths
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
    MIN_MODEL_FILE_SIZE: int = 50_000  # 50 KB minimum threshold for model integrity
    YUNET_MODEL_NAME: str = "face_detection_yunet_2023mar.onnx"
    GENDERAGE_MODEL_NAME: str = "genderage.onnx"
    ANIME_FACE_MODEL_NAME: str = "anime_face_v1.4_n.onnx"
    YOLO_MODEL_NAME: str = "yolov8n.onnx"

    # Download URLs for pre-trained lightweight models (<250MB total)
    YUNET_URL: str = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
    GENDERAGE_URL: str = "https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx"
    ANIME_FACE_URL: str = (
        "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/model.onnx"
    )
    YOLO_URL: str = (
        "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx"
    )

    # Legacy compatibility settings
    DET_SIZE: tuple[int, int] = (640, 640)
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    FAIRFACE_MODEL_NAME: str = "dima806/fairface_age_image_detection"
    FORCE_CPU: bool = True

    # Environment & Deployment Mode
    ENVIRONMENT: str = "development"
    REQUIRE_API_KEY_IN_PRODUCTION: bool = True

    # Classifier Fallback Policy
    ALLOW_HEURISTIC_ACCEPTANCE: bool = False
    REQUIRE_NEURAL_CLASSIFIER: bool = False

    # Production Controls: Rate Limiting & Auth & Concurrency
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
        if self.ENVIRONMENT.lower() == "production":
            if self.REQUIRE_API_KEY_IN_PRODUCTION and not self.API_KEY:
                raise ValueError(
                    "Production mode (ENVIRONMENT='production') requires PPA_API_KEY to be configured for API-key authentication."
                )
            self.REQUIRE_NEURAL_CLASSIFIER = True
        return self


settings = Settings()

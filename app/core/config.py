from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set, Tuple, List


class Settings(BaseSettings):
    """Application runtime configuration."""
    model_config = SettingsConfigDict(
        env_prefix="PPA_",
        case_sensitive=False
    )

    APP_TITLE: str = "Profile Photo Analyzer"
    APP_VERSION: str = "2.1.0"
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024  # 8 MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

    # Detection & Image Quality Parameters
    PHOTO_SCORE_THRESHOLD: float = 0.35
    MIN_SHARPNESS_SCORE: float = 30.0
    EDGE_MARGIN_RATIO: float = 0.01
    MIN_EDGE_MARGIN_PIXELS: int = 5
    MIN_FACE_AREA_RATIO: float = 0.02
    MAX_FACE_AREA_RATIO: float = 0.90
    MIN_BRIGHTNESS: float = 35.0
    MAX_BRIGHTNESS: float = 235.0

    # Model settings
    DET_SIZE: Tuple[int, int] = (640, 640)
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    FAIRFACE_MODEL_NAME: str = "dima806/fairface_age_image_detection"
    FORCE_CPU: bool = True

    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()

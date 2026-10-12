"""The operational endpoints: health, readiness and licensing."""

from pydantic import BaseModel, Field


class ModelHealth(BaseModel):
    model_name: str
    file: str
    role: str
    license: str
    ready: bool
    size_bytes: int
    enabled: bool
    commercial_use_cleared: bool


class Licensing(BaseModel):
    """Whether the models in use may be used commercially (several may not; see NOTICE.md)."""

    commercial_use_cleared: bool
    blocking_models: list[str] = Field(default_factory=list)


class Limits(BaseModel):
    """What the service accepts, so a client can check a photo before it uploads it."""

    max_upload_bytes: int
    max_pixels: int
    max_dimension: int
    working_max_side: int
    formats: list[str]


class HealthResponse(BaseModel):
    status: str  # ready or degraded
    version: str
    models: dict[str, ModelHealth]
    licensing: Licensing


class ModelWarmUp(BaseModel):
    loaded: bool
    warm_up_ms: float | None = None
    providers: list[str] = Field(default_factory=list)


class ReadyResponse(BaseModel):
    """Ready means every enabled model is loaded and has answered one inference."""

    status: str  # ready, warming_up or not_ready
    models: dict[str, ModelWarmUp] = Field(default_factory=dict)

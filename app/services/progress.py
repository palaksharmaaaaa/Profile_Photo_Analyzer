"""Progress events: what the analysis reports while it runs, one event per change of a stage."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

STAGES: tuple[tuple[str, str], ...] = (
    ("validate", "File check"),
    ("information", "Visual information"),
    ("detection", "Faces: detection and verification"),
    ("demographics", "Age and gender"),
)
RUNNING, COMPLETED, FAILED, SKIPPED = "running", "completed", "failed", "skipped"


@dataclass(frozen=True)
class StageEvent:
    """One stage of the analysis changed state. ``elapsed_ms`` is the time the stage took (0 while it runs)."""

    step_id: str
    step_number: int
    title: str
    status: str  # running, completed, failed or skipped
    detail: str = ""
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressReporter = Callable[[StageEvent], None]


def stage_event(step_id: str, status: str, detail: str = "", elapsed_ms: float = 0.0) -> StageEvent:
    """Build the event for a known stage."""
    for number, (known_id, title) in enumerate(STAGES, start=1):
        if known_id == step_id:
            return StageEvent(step_id, number, title, status, detail, round(elapsed_ms, 1))
    raise KeyError(step_id)

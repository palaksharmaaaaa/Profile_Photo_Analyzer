"""
The profile-photo policy: what the findings of the pipeline mean for the upload.

Everything here is a pure function of the findings (no models, no I/O, no image), so the rules that make up the
product can be read, tested and changed on their own. The thresholds come from ``settings``.
"""

from dataclasses import dataclass

from app.core.config import settings
from app.schemas.analysis import NOT_DETECTED, UNSURE, Status
from app.services.types import HumanFaceItem, ImageTypeItem, OtherFaceItem, UnsureFaceItem

IMAGE_TYPES_NEEDING_REVIEW = ("NON_PHOTO", "NEEDS_REVIEW", "UNCERTAIN")
GOOD_FRAMING = "LIKELY_COMPLETE"


@dataclass(frozen=True)
class Decision:
    """The overall outcome and the sentence that explains it."""

    status: Status
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status == Status.PASSED


def is_possible_minor(p_under_18: float | None) -> bool:
    """True when the chance of being under 18 reaches the screening threshold (a signal to review, not a finding)."""
    return p_under_18 is not None and p_under_18 >= settings.age.minor_risk_probability


def decide(
    humans: list[HumanFaceItem],
    others: list[OtherFaceItem],
    unsure: list[UnsureFaceItem],
    image_type: ImageTypeItem,
) -> Decision:
    """
    Accept only exactly one verified real human face, nothing doubtful around it, good quality, a determinable
    age and no sign of a minor. The first rule that does not hold decides the outcome.
    """
    other_labels = ", ".join(sorted({other["label"] for other in others}))
    non_photo_note = (
        f"This looks like a non-photo image ({image_type['label']}). " if image_type["status"] == "NON_PHOTO" else ""
    )

    if not humans:
        if others:
            return Decision(
                Status.ONLY_NON_HUMAN_FACES,
                f"{non_photo_note}No real human face was detected. Found {len(others)} other face(s): {other_labels}.",
            )
        if unsure:
            return Decision(
                Status.REVIEW_REQUIRED,
                f"{len(unsure)} face candidate(s) could not be verified as real human faces. Manual review is needed.",
            )
        return Decision(Status.NO_HUMAN_FACE, f"{non_photo_note}No human face was detected.")

    if len(humans) > 1:
        return Decision(
            Status.MULTIPLE_HUMAN_FACES,
            f"{non_photo_note}A profile photo must contain exactly one real human face; found {len(humans)}.",
        )

    face = humans[0]
    if image_type["status"] in IMAGE_TYPES_NEEDING_REVIEW:
        return Decision(
            Status.REVIEW_REQUIRED,
            f"{non_photo_note}Image type screening ({image_type['reason']}) requires a manual review.",
        )
    if others:
        return Decision(
            Status.NON_HUMAN_FACE_PRESENT,
            f"One real human face was found, but the image also contains {len(others)} non-human face(s): "
            f"{other_labels}.",
        )
    if unsure:
        return Decision(
            Status.REVIEW_REQUIRED,
            f"One real human face was found, but {len(unsure)} other candidate(s) could not be verified.",
        )
    if face["quality"]["status"] != GOOD_FRAMING:
        return Decision(
            Status.POOR_FACE_QUALITY,
            f"{face['quality']['reason']} Please provide a clear, centred, uncropped photo.",
        )
    if face["age_group"] in (UNSURE, NOT_DETECTED):
        return Decision(
            Status.AGE_UNSURE, "The face was verified, but its age could not be estimated with enough certainty."
        )
    if face["possible_minor"]:
        return Decision(
            Status.POSSIBLE_MINOR,
            f"There is an estimated {face['p_under_18']:.0%} chance that this person is under 18 "
            f"(estimated age group {face['age_group']}). This is a screening signal, not a determination: "
            "an age check is needed.",
        )
    return Decision(
        Status.PASSED, f"All checks passed. Estimated age group: {face['age_group']} (appearance-based estimate)."
    )


def summarise(humans: list[HumanFaceItem], others: list[OtherFaceItem], unsure: list[UnsureFaceItem]) -> str:
    """One line with the three counts, naming the kinds of non-human faces."""
    counts: dict[str, int] = {}
    for other in others:
        counts[other["label"]] = counts.get(other["label"], 0) + 1
    non_human = (
        f"{len(others)} non-human face(s): " + ", ".join(f"{n} x {label}" for label, n in counts.items())
        if others
        else "0 non-human faces"
    )
    return f"{len(humans)} real human face(s); {non_human}; {len(unsure)} unverified candidate(s)."

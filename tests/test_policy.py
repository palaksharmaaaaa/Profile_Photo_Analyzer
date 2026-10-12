"""The profile-photo policy as pure rules: one case per outcome, then properties that hold for every input."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.core.config import settings
from app.schemas.analysis import NOT_DETECTED, UNSURE, Status
from app.services import policy

PHOTO = {"status": "PHOTO_LIKE", "label": "Real Photo", "reason": "photo"}
NON_PHOTO = {"status": "NON_PHOTO", "label": "Diagram", "reason": "drawn"}


def human(quality="LIKELY_COMPLETE", age_group="30 to 33", possible_minor=False, p_under_18=0.01):
    return {
        "quality": {"status": quality, "reason": "The face is partly cropped."},
        "age_group": age_group,
        "possible_minor": possible_minor,
        "p_under_18": p_under_18,
    }


def other(label="Dog"):
    return {"label": label}


def decide(humans=(), others=(), unsure=(), image_type=PHOTO):
    return policy.decide(list(humans), list(others), list(unsure), image_type)


# ---- one case per outcome ----------------------------------------------------------------------------------------


def test_nothing_found():
    decision = decide()
    assert decision.status == Status.NO_HUMAN_FACE
    assert decision.reason == "No human face was detected."
    assert not decision.accepted


def test_only_non_human_faces_are_named_once_each():
    decision = decide(others=[other("Dog"), other("Cat"), other("Dog")])
    assert decision.status == Status.ONLY_NON_HUMAN_FACES
    assert "Found 3 other face(s): Cat, Dog." in decision.reason


def test_only_unverified_candidates_need_a_review():
    decision = decide(unsure=[{}, {}])
    assert decision.status == Status.REVIEW_REQUIRED
    assert decision.reason.startswith("2 face candidate(s) could not be verified")


def test_a_non_photo_without_faces_says_so():
    decision = decide(image_type=NON_PHOTO)
    assert decision.status == Status.NO_HUMAN_FACE
    assert "non-photo image (Diagram)" in decision.reason


def test_several_human_faces():
    decision = decide(humans=[human(), human()])
    assert decision.status == Status.MULTIPLE_HUMAN_FACES
    assert "found 2" in decision.reason


@pytest.mark.parametrize("status", ["NON_PHOTO", "NEEDS_REVIEW", "UNCERTAIN"])
def test_a_doubtful_image_type_needs_a_review(status):
    decision = decide(humans=[human()], image_type={**PHOTO, "status": status, "reason": "doubt"})
    assert decision.status == Status.REVIEW_REQUIRED
    assert "(doubt)" in decision.reason


def test_a_non_human_face_next_to_a_human_one():
    decision = decide(humans=[human()], others=[other("Cat")])
    assert decision.status == Status.NON_HUMAN_FACE_PRESENT
    assert "Cat" in decision.reason


def test_an_unverified_candidate_next_to_a_human_face():
    decision = decide(humans=[human()], unsure=[{}])
    assert decision.status == Status.REVIEW_REQUIRED
    assert "1 other candidate(s)" in decision.reason


def test_poor_quality_gives_the_quality_reason_and_advice():
    decision = decide(humans=[human(quality="PARTIAL_OR_CROPPED")])
    assert decision.status == Status.POOR_FACE_QUALITY
    assert decision.reason == "The face is partly cropped. Please provide a clear, centred, uncropped photo."


@pytest.mark.parametrize("age_group", [UNSURE, NOT_DETECTED])
def test_an_unknown_age(age_group):
    assert decide(humans=[human(age_group=age_group)]).status == Status.AGE_UNSURE


def test_a_possible_minor_is_described_as_a_screening_signal():
    decision = decide(humans=[human(age_group="12 to 15", possible_minor=True, p_under_18=0.7)])
    assert decision.status == Status.POSSIBLE_MINOR
    assert "70%" in decision.reason
    assert "not a determination" in decision.reason


def test_a_clean_single_face_passes():
    decision = decide(humans=[human()])
    assert decision.status == Status.PASSED
    assert decision.accepted
    assert "30 to 33" in decision.reason


# ---- the screening threshold and the summary -------------------------------------------------------------------------


def test_possible_minor_threshold(monkeypatch):
    monkeypatch.setattr(settings.age, "minor_risk_probability", 0.2)
    assert policy.is_possible_minor(0.2)
    assert not policy.is_possible_minor(0.19)
    assert not policy.is_possible_minor(None)


def test_summary_groups_non_human_labels():
    others = [other("Dog"), other("Dog"), other("Bird")]
    assert policy.summarise([human()], others, [{}, {}]) == (
        "1 real human face(s); 3 non-human face(s): 2 x Dog, 1 x Bird; 2 unverified candidate(s)."
    )
    assert policy.summarise([], [], []) == "0 real human face(s); 0 non-human faces; 0 unverified candidate(s)."


# ---- properties ----------------------------------------------------------------------------------------------------

humans_strategy = st.lists(
    st.builds(
        human,
        quality=st.sampled_from(["LIKELY_COMPLETE", "UNCERTAIN", "PARTIAL_OR_CROPPED"]),
        age_group=st.sampled_from(["30 to 33", UNSURE, NOT_DETECTED]),
        possible_minor=st.booleans(),
        p_under_18=st.floats(0, 1),
    ),
    max_size=3,
)
image_types = st.sampled_from([PHOTO, NON_PHOTO, {**PHOTO, "status": "NEEDS_REVIEW"}, {**PHOTO, "status": "UNCERTAIN"}])


@given(humans_strategy, st.lists(st.just(other()), max_size=2), st.lists(st.just({}), max_size=2), image_types)
def test_a_photo_is_accepted_only_when_every_condition_holds(humans, others, unsure, image_type):
    decision = decide(humans, others, unsure, image_type)
    clean = (
        len(humans) == 1
        and not others
        and not unsure
        and image_type["status"] == "PHOTO_LIKE"
        and humans[0]["quality"]["status"] == "LIKELY_COMPLETE"
        and humans[0]["age_group"] not in (UNSURE, NOT_DETECTED)
        and not humans[0]["possible_minor"]
    )
    assert decision.accepted == clean
    assert (decision.status == Status.PASSED) == clean
    assert decision.reason


@given(humans_strategy, st.lists(st.just(other()), max_size=2), st.lists(st.just({}), max_size=2))
def test_the_summary_always_counts_what_it_is_given(humans, others, unsure):
    summary = policy.summarise(humans, others, unsure)
    assert summary.startswith(f"{len(humans)} real human face(s);")
    assert summary.endswith(f"{len(unsure)} unverified candidate(s).")

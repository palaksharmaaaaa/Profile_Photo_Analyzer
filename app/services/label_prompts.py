"""Zero-shot label taxonomy. Shared by the runtime and ``scripts/build_models.py`` (pure data, no heavy imports)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelClass:
    name: str  # stable machine name
    display: str  # user-facing label
    face_type: str  # category reported in the API ("human" for real human faces)
    prompts: tuple[str, ...]


# Crop-level taxonomy: what is inside a candidate face box?
FACE_CLASSES: tuple[LabelClass, ...] = (
    LabelClass(
        "real_human",
        "Real human face",
        "human",
        (
            "a photograph of a real human face",
            "a close-up photo of a real person's face",
            "a candid photograph of a real person",
            "a studio portrait photograph of a real person",
        ),
    ),
    LabelClass(
        "animal",
        "Animal face",
        "animal",
        (
            "a photograph of an animal's face",
            "a close-up photo of a cat or a dog",
            "a photo of a wild animal",
            "a photo of a monkey or a primate",
        ),
    ),
    LabelClass(
        "bird",
        "Bird",
        "bird",
        (
            "a photograph of a bird",
            "a close-up photo of a bird's head",
        ),
    ),
    LabelClass(
        "cartoon",
        "Cartoon / illustration",
        "cartoon_or_illustration",
        (
            "a cartoon character face",
            "a line-art illustration of a character",
            "a comic book drawing of a face",
            "an engraving or ink illustration of a person",
        ),
    ),
    LabelClass(
        "anime",
        "Anime / manga character",
        "anime",
        (
            "an anime character face",
            "a manga style illustration of a character",
            "a ghibli style animated character",
        ),
    ),
    LabelClass(
        "painting",
        "Painting / sketch",
        "painting",
        (
            "an old master oil painting of a face",
            "a painted portrait on canvas",
            "a pencil sketch or charcoal drawing of a person",
        ),
    ),
    LabelClass(
        "render_3d",
        "3D / CGI character",
        "render_3d",
        (
            "a 3d rendered character face",
            "a video game character render",
            "computer generated imagery of a character",
        ),
    ),
    LabelClass(
        "doll_statue",
        "Doll / statue / mask",
        "doll_statue",
        (
            "a photo of a doll, mannequin or toy face",
            "a photo of a statue or sculpture of a face",
            "a photo of a mask",
        ),
    ),
    LabelClass(
        "diagram",
        "Diagram / anatomical drawing",
        "diagram",
        (
            "a scientific or anatomical diagram",
            "a medical illustration of the human body",
            "a technical line drawing with labels",
        ),
    ),
    LabelClass(
        "not_face",
        "Not a face",
        "not_a_face",
        (
            "a photo of an object, texture or landscape with no face",
            "a photo of text or a pattern",
            "a photo of a hand, clothing or background",
        ),
    ),
)

# Whole-image taxonomy (informational screening of the upload).
IMAGE_CLASSES: tuple[LabelClass, ...] = (
    LabelClass(
        "real_photo",
        "Real Photo",
        "photo",
        (
            "a real photograph of a person",
            "a photograph of people",
            "a portrait photograph",
        ),
    ),
    LabelClass(
        "animal_photo",
        "Animal Photo",
        "photo",
        (
            "a photograph of an animal",
            "a photograph of a bird",
        ),
    ),
    LabelClass(
        "object_scene",
        "Object / Scene (No Person)",
        "photo",
        (
            "a photograph of an object",
            "a photograph of a landscape or a building",
            "a screenshot or a document with text",
        ),
    ),
    LabelClass(
        "cartoon",
        "Cartoon / Illustration",
        "synthetic",
        (
            "a cartoon drawing",
            "a comic or line-art illustration",
        ),
    ),
    LabelClass(
        "anime",
        "Anime / Ghibli-style Image",
        "synthetic",
        (
            "an anime or manga illustration",
            "a ghibli style animated scene",
        ),
    ),
    LabelClass(
        "painting",
        "Painting / Sketch",
        "synthetic",
        (
            "a painting",
            "a pencil or charcoal sketch",
        ),
    ),
    LabelClass(
        "render_3d",
        "3D / Computer-generated Image",
        "synthetic",
        (
            "a 3d render",
            "computer generated imagery",
        ),
    ),
    LabelClass(
        "diagram",
        "Biological / Scientific Diagram",
        "synthetic",
        (
            "a scientific or anatomical diagram",
            "a technical diagram with labels",
        ),
    ),
)


def prompt_templates() -> dict[str, list[str]]:
    """All prompts keyed by taxonomy, used to record provenance in the label bank."""
    return {
        "face": [p for c in FACE_CLASSES for p in c.prompts],
        "image": [p for c in IMAGE_CLASSES for p in c.prompts],
    }

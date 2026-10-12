"""The page and the API must agree: fixtures satisfy the schema, every element the scripts use exists, CSS keeps its promises."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.schemas.analysis import AnalysisResponse, JobStatus

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
PAGE = ROOT / "app" / "templates" / "index.html"
FIXTURES = ROOT / "tests" / "js" / "fixtures"


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.ids += [value for name, value in attrs if name == "id" and value]


def page_ids() -> list[str]:
    collector = IdCollector()
    collector.feed(PAGE.read_text(encoding="utf-8"))
    return collector.ids


def test_the_fixture_is_a_valid_analysis_response():
    result = json.loads((FIXTURES / "result.json").read_text(encoding="utf-8"))
    parsed = AnalysisResponse.model_validate(result)
    assert (parsed.real_human_face_count, parsed.other_face_count, parsed.unsure_face_count) == (1, 1, 1)
    assert "original_image" not in result


def test_the_page_has_no_duplicate_ids():
    ids = page_ids()
    assert len(ids) == len(set(ids))


def test_every_element_the_scripts_look_up_exists_in_the_page():
    ids = set(page_ids())
    used = set()
    for script in (STATIC / "js").glob("*.js"):
        source = script.read_text(encoding="utf-8")
        used |= set(re.findall(r'(?:\$|show|hide|setText)\("([a-z0-9-]+)"', source))
    dynamic_prefixes = ("human-card-", "other-chip-", "unsure-chip-", "box-")
    missing = {name for name in used if name not in ids and not name.startswith(dynamic_prefixes)}
    assert not missing, f"the scripts use elements the page does not have: {sorted(missing)}"
    assert {"photo", "drop-zone", "upload-form", "verdict", "overlay", "original-image"} <= used


def test_the_page_loads_the_module_entry_point_and_stylesheet_from_static():
    page = PAGE.read_text(encoding="utf-8")
    assert '<script type="module" src="/static/js/main.js"></script>' in page
    assert 'href="/static/analyzer.css"' in page
    assert (STATIC / "js" / "main.js").is_file()


def test_the_scripts_do_not_build_html_from_server_text():
    for script in (STATIC / "js").glob("*.js"):
        source = script.read_text(encoding="utf-8")
        assert "innerHTML" not in source, script.name
        assert "insertAdjacentHTML" not in source, script.name


def block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"{selector} is not styled"
    return match.group(1)


def test_the_scanning_line_keeps_moving_for_people_who_prefer_reduced_motion():
    """A regression guard: a blanket reduced-motion rule once froze the only progress indicator."""
    css = (STATIC / "analyzer.css").read_text(encoding="utf-8")
    assert "animation: scan" in block(css, ".scan-laser")
    reduced = css[css.index("prefers-reduced-motion") :]
    assert "animation-duration: 0.01ms" not in reduced
    assert ".scan-laser { animation-duration" in reduced
    keyframes = re.search(r"@keyframes scan \{(.*?)\n\}", css, re.S).group(1)
    assert "top: 0" in keyframes
    assert "calc(100% - 3px)" in keyframes


@pytest.mark.parametrize("selector", [".box:focus-visible", ".hidden", ".sr-only"])
def test_keyboard_and_visibility_helpers_are_defined(selector):
    css = (STATIC / "analyzer.css").read_text(encoding="utf-8")
    assert selector in css


def test_a_job_status_with_a_result_is_a_valid_contract():
    result = json.loads((FIXTURES / "result.json").read_text(encoding="utf-8"))
    status = JobStatus.model_validate(
        {
            "job_id": "abc",
            "status": "succeeded",
            "stages": [
                {"step_id": "validate", "step_number": 1, "title": "File check", "status": "completed"},
            ],
            "result": result,
        }
    )
    assert status.result.status == "NON_HUMAN_FACE_PRESENT"

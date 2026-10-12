"""The build script cannot import the app (it must run without the web dependencies), so keep their names in sync."""

import importlib.util
from pathlib import Path

from app.services.model_manager import LABEL_BANK_NAME, MANIFEST_NAME, MODEL_SPECS

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_models.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_script_writes_the_files_the_app_loads():
    script = load_script()
    assert MODEL_SPECS["siglip_vision"].filename == script.OUT_VISION
    assert MODEL_SPECS["mivolo"].filename == script.OUT_MIVOLO
    assert MODEL_SPECS["scrfd"].filename == script.OUT_SCRFD
    assert script.OUT_BANK == LABEL_BANK_NAME
    assert script.OUT_MANIFEST == MANIFEST_NAME


def test_every_locally_built_model_is_built_by_a_stage_and_nothing_else_is():
    script = load_script()
    built_in_app = {key for key, spec in MODEL_SPECS.items() if spec.url is None}
    assert built_in_app == {"siglip_vision", "mivolo", "scrfd"}
    assert set(script.STAGE_SOURCES) == {"siglip", "mivolo", "scrfd"}
    for names in script.STAGE_SOURCES.values():
        assert all(name in script.SOURCES for name in names)

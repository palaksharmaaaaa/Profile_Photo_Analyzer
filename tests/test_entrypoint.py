"""Tests for application root entrypoint app.py."""
import importlib.util
import runpy
from unittest.mock import patch


def load_root_app_module():
    spec = importlib.util.spec_from_file_location("root_app", "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_root_app_import():
    mod = load_root_app_module()
    assert mod.app is not None


def test_root_app_main():
    with patch("uvicorn.run") as mock_run:
        mod = load_root_app_module()
        mod.main()
        mock_run.assert_called_once_with("app:app", host="127.0.0.1", port=8000, reload=True)


def test_root_app_execution():
    with patch("uvicorn.run") as mock_run:
        runpy.run_path("app.py", run_name="__main__")
        mock_run.assert_called_once()

"""Tests for application root entrypoint app.py."""

import importlib.util
from unittest.mock import patch


def test_root_app_main():
    with patch("uvicorn.run") as mock_run:
        spec = importlib.util.spec_from_file_location("root_entrypoint", "app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.app is not None
        mod.main()
        mock_run.assert_called_with("app:app", host="127.0.0.1", port=8000, reload=True)


def test_root_app_main_block():
    with patch("uvicorn.run") as mock_run:
        spec = importlib.util.spec_from_file_location("__main__", "app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mock_run.assert_called_with("app:app", host="127.0.0.1", port=8000, reload=True)

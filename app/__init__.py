"""Profile Photo Analyzer package."""

from app.core.config import settings
from app.main import app, create_app

__version__ = settings.APP_VERSION
__all__ = ["__version__", "app", "create_app"]

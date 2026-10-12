"""Application factory: logging, error handlers, middleware, the static page assets, the routes and the model life cycle."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.middleware import SecurityMiddleware
from app.api.routes import INDEX_HTML_PATH, router, v1
from app.core.config import PROJECT_ROOT, settings
from app.core.errors import install_error_handlers
from app.core.logging import configure_logging
from app.services.jobs import reset_job_manager
from app.services.model_manager import warm_up
from app.services.parallel import shutdown as shutdown_inference_pool

logger = logging.getLogger(__name__)

STATIC_DIR = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the models up in the background (the server answers at once; ``/ready`` says when they are), then clean up."""
    app.state.warm_up = None
    task: asyncio.Task[None] | None = None
    if settings.models.warm_up_on_start:

        async def run_warm_up() -> None:
            app.state.warm_up = await asyncio.to_thread(warm_up)
            logger.info("Models warmed up: %s", {key: entry["warm_up_ms"] for key, entry in app.state.warm_up.items()})

        task = asyncio.create_task(run_warm_up())
    else:
        app.state.warm_up = {}
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
        reset_job_manager()
        shutdown_inference_pool()


def create_app() -> FastAPI:
    configure_logging(settings.log_level, settings.log_json)
    if not INDEX_HTML_PATH.is_file():
        raise RuntimeError(f"The frontend page is missing: {INDEX_HTML_PATH}")

    application = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description="Checks a profile photo for real human faces and estimates age and gender with open models.",
        lifespan=lifespan,
    )
    install_error_handlers(application)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
    )
    application.add_middleware(SecurityMiddleware)  # added last, so it is the outermost layer

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.include_router(router)
    application.include_router(v1)
    return application


app = create_app()

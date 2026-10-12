"""``python -m app`` entrypoint."""

import logging

import uvicorn

logger = logging.getLogger(__name__)


def main(reload: bool = False) -> None:
    """Run the uvicorn ASGI server on localhost."""
    logger.info("Starting Profile Photo Analyzer on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=reload)


if __name__ == "__main__":
    main()

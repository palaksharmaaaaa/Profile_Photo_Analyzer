"""``python -m app`` entrypoint."""

import logging

import uvicorn

from app.core.config import settings

logger = logging.getLogger(__name__)


def main(reload: bool = False) -> None:
    """Run the uvicorn ASGI server on the configured host and port (``PPA_SERVER__HOST`` / ``PPA_SERVER__PORT``)."""
    server = settings.server
    logger.info("Starting Profile Photo Analyzer on http://%s:%s", server.host, server.port)
    uvicorn.run("app.main:app", host=server.host, port=server.port, reload=reload, log_config=None)


if __name__ == "__main__":
    main()

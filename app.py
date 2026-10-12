"""Application entrypoint for Profile Photo Analyzer."""
import logging
from app.main import app

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Profile Photo Analyzer server on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

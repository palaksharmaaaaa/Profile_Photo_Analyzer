"""
Download the models that can be used as published (YuNet, the anime face detector, YOLOX-S) into ``models/``.

    python scripts/fetch_models.py

Each file is verified against the SHA-256 pinned in ``app/services/model_manager.py`` before it is kept. The other
models are built locally by ``scripts/build_models.py``. The service also downloads a missing file on first use;
running this script (as the Docker image build does) puts every file in place beforehand.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.model_manager import MODEL_SPECS, ensure_model_file, model_path  # noqa: E402


def main() -> int:
    failed = []
    for key, spec in MODEL_SPECS.items():
        if spec.url is None:
            continue
        ready = ensure_model_file(model_path(key), spec.url)
        print(f"{'ok    ' if ready else 'FAILED'} {spec.filename}")
        if not ready:
            failed.append(spec.filename)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# syntax=docker/dockerfile:1
#
# Two stages: the first builds the models once (about 1 GB of pinned downloads, every one verified against a SHA-256),
# the second is the small runtime image that only carries the service and the finished model files.
#
#   docker build -t profile-photo-analyzer .
#   docker run --rm -p 8000:8000 -e PPA_SECURITY__API_KEY=change-me profile-photo-analyzer
#
# NOTICE.md explains the licences of the models: SCRFD (non-commercial) can be left out with
# -e PPA_DETECTION__ENABLE_SCRFD=false, MiVOLO has no permissive replacement here.

# ---- Stage 1: build the models ----------------------------------------------------------------------------
FROM python:3.14-slim AS models

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY requirements.lock requirements-build.txt ./
RUN pip install -r requirements.lock -r requirements-build.txt

COPY app ./app
COPY scripts/build_models.py scripts/fetch_models.py ./scripts/
RUN python scripts/build_models.py --cleanup && python scripts/fetch_models.py

# ---- Stage 2: the runtime image ------------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PPA_SERVER__HOST=0.0.0.0 \
    PPA_LOG_JSON=true

WORKDIR /srv/app

COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY app ./app
COPY --from=models /build/models ./models

RUN useradd --system --uid 10001 --no-create-home ppa
USER ppa

EXPOSE 8000

# Ready means every model has loaded and answered one inference; warming up takes a few seconds after start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]

CMD ["python", "-m", "app"]

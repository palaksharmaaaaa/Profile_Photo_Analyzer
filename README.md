# Profile Photo Analyzer

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=alert_status)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=coverage)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=bugs)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)

A production-ready, modular computer vision web service built with FastAPI to screen and validate profile photos for identity compliance, facial visibility, sharpness, framing, demographic attributes, and synthetic image detection.

---

## Architecture & Project Structure

```text
project-profile-photo-analyzer/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # Synchronous threadpool route handlers
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Pydantic Settings with env var support
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── analysis.py            # Strongly typed Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_utils.py         # Clamping, Base64 encoding, annotations
│   │   ├── quality_service.py     # Sharpness, edge margin, exposure, framing
│   │   ├── classifier_service.py  # Zero-shot CLIP photo/illustration screening
│   │   ├── face_service.py        # InsightFace single-pass detection & attributes
│   │   └── analyzer.py            # End-to-end analysis orchestrator
│   ├── templates/
│   │   └── index.html             # Responsive modern UI
│   ├── __init__.py
│   └── main.py                    # FastAPI application factory
├── tests/
│   ├── conftest.py                # Synthetic image fixtures & mocks
│   ├── test_analyzer.py           # Orchestrator decision tests
│   ├── test_api.py                # HTTP status & schema tests
│   ├── test_classifier_service.py # Image type screening tests
│   ├── test_face_service.py       # Demographic & minor detection tests
│   └── test_quality_service.py    # Blur, exposure, & landmark tests
├── app.py                         # Root entrypoint (`uvicorn app:app`)
├── sonar-project.properties       # SonarCloud configuration
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Key Improvements & Bug Fixes

1. **Eliminated Duplicate Execution**:
   - Fixed the critical bug where age estimation and gender prediction were run twice on every image, and ran on cartoons despite guards.
   - Removed duplicate face detection in the preview generation pipeline.
2. **Non-Blocking Threadpool Execution**:
   - Replaced synchronous operations in `async def` with dedicated thread pool dispatch, ensuring the FastAPI event loop is never starved.
3. **Optimized Model Footprint**:
   - Replaced heavyweight redundant models with single-pass native InsightFace multitask heads (`genderage.onnx`).
4. **Enhanced Quality Analysis**:
   - Added exposure / luminance checks (under/over-exposed detection).
   - Added face framing checks (face area to frame ratio).
   - Normalized edge margin detection to avoid false positives.
5. **Security & Clean Architecture**:
   - Modularized monolithic script into decoupled service layers.
   - Resolved reflected XSS and parameter validation findings.
   - 0 bugs, 0 vulnerabilities, 0 code smells, and 0.0% duplication on SonarCloud.

---

## Installation & Setup

### 1. Create a Virtual Environment

```bash
uv venv --python 3.10 .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Run the Application

```bash
uvicorn app:app --reload
```

Open your browser at `http://127.0.0.1:8000`.

---

## Running Tests & Coverage

```bash
pytest --cov=app --cov-report=term-missing --cov-report=xml:coverage.xml
```

Target test coverage: **>92%** across all application modules.

---

## SonarCloud Analysis

```powershell
$env:SONAR_HOST_URL = "https://sonarcloud.io"
$env:SONAR_TOKEN = "your_sonar_token"
npx sonarqube-scanner
```

- **SonarCloud Project Key**: `hgarg02_project-profile-photo-analyzer`
- **SonarCloud Organization**: `hgarg02`
- **Dashboard**: [SonarCloud Project Dashboard](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)

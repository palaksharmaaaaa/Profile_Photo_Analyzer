# Profile Photo Analyzer

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=alert_status)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=coverage)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=bugs)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=hgarg02_project-profile-photo-analyzer&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=hgarg02_project-profile-photo-analyzer)

A robust, modular computer vision web service built with FastAPI to screen and validate profile photos for identity compliance, facial visibility, sharpness, framing, demographic attributes, and synthetic image detection.

---

## Architecture & Project Structure

```text
project-profile-photo-analyzer/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow (pytest + SonarCloud)
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
│   │   ├── face_service.py        # YuNet & InsightFace single-pass detection & demographics
│   │   ├── model_manager.py       # Pinned cryptographic hash verification & model downloads
│   │   └── analyzer.py            # End-to-end analysis orchestrator
│   ├── templates/
│   │   └── index.html             # Responsive modern UI
│   ├── __init__.py
│   └── main.py                    # FastAPI application factory with security middleware
├── tests/
│   ├── conftest.py                # Synthetic image fixtures & mocks
│   ├── test_analyzer.py           # Orchestrator decision tests
│   ├── test_api.py                # HTTP status & schema tests
│   ├── test_classifier_service.py # Image type screening tests
│   ├── test_entrypoint.py         # Root app.py entrypoint tests
│   ├── test_face_service.py       # Demographic & minor detection tests
│   ├── test_image_utils.py        # Utility tests
│   ├── test_model_manager.py      # Cryptographic hash and integrity tests
│   └── test_quality_service.py    # Blur, exposure, & landmark tests
├── app.py                         # Root entrypoint (`uvicorn app:app`)
├── sonar-project.properties       # SonarCloud configuration
├── requirements.txt               # Direct dependencies
├── requirements.lock              # Fully pinned reproducible dependency lockfile
├── pyproject.toml                 # Ruff and Pytest configuration
└── README.md
```

---

## Key Features & Security Hardening

1. **Production Security & Access Controls**:
   - **Rate Limiting**: Sliding window per client IP with configurable threshold (`RATE_LIMIT_PER_MINUTE`).
   - **Concurrency Guard**: Request semaphore limiting active heavy inference tasks (`MAX_CONCURRENT_REQUESTS`).
   - **Authentication**: Optional API key authentication header (`X-API-Key` or Bearer token).
   - **Resource Limits**: 10 MiB upload cap, 4096px dimension cap, and 16 Megapixel Pillow decompression bomb guards.
2. **Cryptographic Model Verification**:
   - Model files are checked against pinned SHA-256 hashes (`model_manager.py`).
   - Atomic temporary file downloads prevent partial or corrupt model states.
3. **Truthful, Un-fabricated Diagnostics & Demographics**:
   - Continuous age regression honestly reports uncalibrated confidence as `None` rather than borrowing face detector scores.
   - `models_executed` dynamically tracks only the confirmed models that actually performed inference during the request.
   - Heuristic image classification fails closed and flags review-required rather than silently assuming real photograph status when CLIP is absent.
4. **Non-Blocking Execution & Clean Architecture**:
   - Synchronous route operations executed via FastAPI threadpool workers.
   - 100.00% statement test coverage across all application modules.
   - 0 bugs, 0 vulnerabilities, 0 code smells, and 0.0% duplication on SonarCloud.

---

## Production Security & Configuration

| Environment Variable | Default | Purpose / Production Guidance |
| :--- | :--- | :--- |
| `PPA_ENVIRONMENT` | `development` | Set to `production` for live deployments. Enforces authentication fail-closed rules. |
| `PPA_API_KEY` | `None` | **Mandatory** in `production` mode. Rejects unauthenticated callers (`401 Unauthorized`). |
| `PPA_API_KEY_HEADER` | `X-API-Key` | Header checked for API key authentication (Bearer token also supported). |
| `PPA_RATE_LIMIT_PER_MINUTE`| `60` | Sliding-window request allowance per client IP. Returns `429 Too Many Requests`. |
| `PPA_MAX_CONCURRENT_REQUESTS`| `10` | Concurrency guard per worker process. Returns `503 Service Unavailable`. |
| `PPA_TRUSTED_PROXIES` | `["127.0.0.1", "::1"]` | Trusted reverse proxy IPs used to inspect `X-Forwarded-For` for client origin. |
| `PPA_MAX_TRACKED_IPS` | `10000` | Bounded memory ceiling for in-memory rate-limit IP records. |
| `PPA_ALLOW_HEURISTIC_ACCEPTANCE` | `False` | When `False` (default, fail-closed), images without neural CLIP verification require manual review. Set to `True` only in trusted offline environments to allow CV heuristic acceptance. |
| `PPA_REQUIRE_NEURAL_CLASSIFIER` | `False` | When `True` (enforced automatically in `production` mode), routes non-CLIP environments to `MANUAL_REVIEW_REQUIRED`. |

> [!NOTE]
> For multi-worker, horizontally scaled clusters, rate-limit state should be backed by a shared Redis cluster or sticky-session routing at the API gateway layer.
>
> **Production Photo Acceptance**: In the default installation without `transformers`/CLIP, non-cartoon human photos fail closed to `accepted: false` (`MANUAL_REVIEW_REQUIRED`). For automated photo acceptance without human-in-the-loop review, provision the CLIP model by installing `transformers` (`pip install transformers torch`).

---

## Evaluation, Accuracy & Fairness

Empirical latency benchmarks measured via `scripts/benchmark_latency.py` (~1.5 s p50, ~2.5 s p95 on standard CPU), model optimization guidance (INT8 quantization, ONNX Runtime TensorRT/CUDA acceleration for sub-250 ms p95), and target accuracy/fairness governance specifications (FairFace, UTKFace, LFW targets, age regression MAE $\le 4.2$ years, gender parity metrics, fail-closed minor protection) are detailed in the [Evaluation and Fairness Specification](docs/EVALUATION_AND_FAIRNESS.md).

---

## Installation & Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Pinned Dependencies

```bash
uv pip install -r requirements.lock
```

### 3. Run the Application

```bash
uvicorn app:app --reload
```

Open your browser at `http://127.0.0.1:8000`.

---

## Running Tests & Coverage

```bash
pytest --cov=app --cov=app.py --cov-config=.coveragerc --cov-report=term-missing --cov-report=xml
```

Test coverage standard: **100.00% statement coverage** across all source files.

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


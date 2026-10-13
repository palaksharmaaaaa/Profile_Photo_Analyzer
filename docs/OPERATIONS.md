# Operations

How to run, scale, watch and secure the service. Everything here is implemented in this repository; where something is
a limit of the design it says so.

## Running it

| How | Command |
| :--- | :--- |
| Locally | `python -m app` (listens on `PPA_SERVER__HOST` : `PPA_SERVER__PORT`, default `127.0.0.1:8000`) |
| Docker | `docker build -t profile-photo-analyzer .` then `docker run -p 8000:8000 -e PPA_SECURITY__API_KEY=... profile-photo-analyzer` |
| Docker Compose with an edge | `PPA_API_KEY=... docker compose -f deploy/docker-compose.yml up --build` (nginx on port 8080) |

The image builds the models inside it (about 1 GB of downloads, each verified against a pinned SHA-256), so nothing has
to be prepared on the host. It runs as a non-root user and works with a read-only root file system.

### Start-up, liveness and readiness

* The server answers at once; the models are loaded and warmed up in the background (`models.warm_up_on_start`).
* `GET /health` is **liveness**: it reports every model file (present, hash verified, enabled, licence) and the
  licensing position. It never runs a model.
* `GET /ready` is **readiness**: `200` once every enabled model has loaded and answered one inference (with its
  time and execution providers), `503` with `warming_up` before that and `not_ready` if a model failed.
  Point a load balancer or the container `HEALTHCHECK` here. A missing model never produces a guess: faces come back
  `Unsure` and nothing is accepted.

## Configuration

Settings are grouped by stage in `app/core/config.py`. An environment variable is `PPA_<GROUP>__<FIELD>` (two
underscores), for example `PPA_AGE__MINOR_RISK_PROBABILITY=0.2`; ungrouped fields use one underscore
(`PPA_ENVIRONMENT=production`, `PPA_LOG_JSON=true`).

| Group | What it holds |
| :--- | :--- |
| `upload` | size, pixel and dimension limits, working size |
| `information` | the blank-image measures |
| `detection` | detector thresholds, `enable_scrfd` |
| `verification` | SigLIP2 thresholds and crop margin |
| `age` | views, age error by age band, minor threshold, gender thresholds |
| `quality` | framing, exposure, sharpness, pose |
| `models` | model directory, threads, CPU only or not, warm-up, parallel inference |
| `jobs` | workers, queue size, result lifetime |
| `server` | host and port |
| `security` | API key, rate limit, concurrency, trusted proxies, CORS |

`PPA_ENVIRONMENT=production` refuses to start without `PPA_SECURITY__API_KEY` (unless
`PPA_SECURITY__REQUIRE_API_KEY_IN_PRODUCTION=false`).

## Security

* **Authentication.** When `security.api_key` is set every endpoint except the page, its assets, `/health`, `/ready`,
  `/v1/limits` and the API documentation needs `X-API-Key: <key>` (or `Authorization: Bearer <key>`); keys are compared in
  constant time. The page itself cannot hold a key: put the service behind a gateway that adds it, or use it without a key
  on a trusted network.
* **Uploads.** The body is cut off as soon as it exceeds the upload limit plus the multipart framing (a declared
  `Content-Length` over the limit is refused before anything is read). The file type is decided from the file's bytes, never
  from its name; pixel count and dimensions are capped while decoding; the file name is reduced to a safe base name.
* **Responses** carry `X-Content-Type-Options`, `Referrer-Policy` and `X-Frame-Options`; the page carries a strict
  Content-Security-Policy (no inline script or style, no third-party origins).
* **Privacy.** Images exist only in memory while they are analysed. The response does not echo the original image back;
  the annotated copy can be left out with `?annotated_image=false`.

## Scaling

Inference is CPU bound: one analysis takes a few seconds and uses several cores.

* **In one process.** `POST /v1/analyze-profile-photo` is synchronous. `POST /v1/jobs` queues the work on a bounded pool
  (`jobs.workers` running, `jobs.queue_size` waiting); beyond that it answers `503 queue_full` with `Retry-After`, so a
  burst can never pile up. Independent detectors and the three age views can run at the same time (`models.parallel_inference`, off by default because it measured slower on a 14-thread CPU, where the models compete for cores);
  `scripts/benchmark_latency.py --parallel` measures what it does on your machine.
* **Across processes or machines.** Job state and the rate limiter are per process. Run several instances behind a load
  balancer, send a client's follow-up requests (`GET /v1/jobs/{id}`, the event stream) to the instance that created the
  job, and rate limit at the edge: `deploy/nginx.conf` does that (with the in-process limiter switched off via
  `PPA_SECURITY__RATE_LIMIT_ENABLED=false`). A shared queue (for example Redis) is the next step and is not implemented.
* **GPU.** `PPA_MODELS__FORCE_CPU=false` lets ONNX Runtime use CUDA or DirectML when installed; `/ready` lists the
  providers each model really uses. This path has not been exercised on a GPU here.

## Observability

* **Request ids.** Every response has `X-Request-ID` (a sane one sent by the caller is kept); error bodies repeat it and
  every log line carries it.
* **Logs.** `PPA_LOG_JSON=true` writes one JSON object per line (`time`, `level`, `logger`, `request_id`, `message`).
* **Metrics.** `GET /metrics` (Prometheus text format, authenticated when an API key is set, never rate limited):

  | Metric | Meaning |
  | :--- | :--- |
  | `ppa_http_requests_total{method,route,status}` | requests by route template and status |
  | `ppa_http_request_seconds{method,route}` | request duration |
  | `ppa_analyses_total{status}` | finished analyses by outcome (`PASSED`, `NO_HUMAN_FACE`, ...) |
  | `ppa_stage_seconds{stage}` | duration of each pipeline stage |
  | `ppa_rejected_uploads_total{code}` | uploads refused before analysis |
  | `ppa_jobs_active`, `ppa_jobs_total{state}` | jobs queued or running, and finished by state |

## Release checks

Continuous integration (`.github/workflows/ci.yml`) runs on every push and pull request:

| Job | Checks |
| :--- | :--- |
| `python` | ruff (lint and format), strict mypy, the tests with 100 % statement and branch coverage, `pip-audit` of the locked runtime dependencies |
| `frontend` | ESLint, the browser tests with 100 % coverage, `npm audit` |
| `sonar` | SonarQube Cloud analysis of Python, JavaScript, CSS and HTML with both coverage reports, and its quality gate |
| `image` | optional, started by hand only: the Docker image builds (models included) and its container becomes ready |

`.github/workflows/evaluation.yml` (weekly and on demand) builds the models, downloads the labelled images, runs
`scripts/evaluate.py` and fails when a limit in `evaluation/gates.json` is violated. Dependabot proposes updates for pip, npm,
GitHub Actions and Docker.

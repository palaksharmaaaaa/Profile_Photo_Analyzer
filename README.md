# Profile Photo Analyzer

Upload a photo and find out, using models that run locally on your own machine:

1. whether the image has any visual information at all (blank, solid-colour and uniform images are rejected);
2. how many faces it contains and **which are real human faces**, as opposed to animals, birds, cartoons, anime,
   paintings, 3D renders, dolls/statues/masks or diagrams (each kind is counted and reported separately);
3. the **gender** and the **age, in 3-year brackets** (0 to 3, 3 to 6, ... 18 to 21, ...), of every real human face.

Nothing is invented. If a model cannot support an answer the field says **`Unsure`**, and if nothing could be
detected it says **`Not Detected`**. The response lists the models that actually ran
(`diagnostics.models_executed`), and every number comes from one of them.

> **Licensing:** this build uses the strongest models found regardless of licence. The face detector's weights are
> non-commercial and the age model's training data is only partly disclosed, so it is **not cleared for commercial
> use** (`GET /health` says so too). The SCRFD detector can be switched off with `PPA_DETECTION__ENABLE_SCRFD=false`;
> the age model has no permissive replacement here. Read [NOTICE.md](NOTICE.md) first.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows;  source .venv/bin/activate on Linux and macOS
pip install -r requirements.lock            # runtime dependencies (no torch, no transformers)

pip install -r requirements-build.txt       # one-time: builds the models into models/ (about 1 GB of downloads,
python scripts/build_models.py --cleanup    # every file verified by SHA-256, scratch files removed afterwards)
python scripts/fetch_models.py              # the models that are used as published (optional: fetched on first use)

python -m app                               # open http://127.0.0.1:8000
```

or with Docker, which builds the models inside the image (see [docs/OPERATIONS.md](docs/OPERATIONS.md)):

```bash
docker build -t profile-photo-analyzer .
docker run --rm -p 8000:8000 -e PPA_SECURITY__API_KEY=change-me profile-photo-analyzer
```

`GET /health` reports every model file as ready or not (`ready` / `degraded`) and `GET /ready` says when every model has
loaded and warmed up. If the built models are missing the service still starts, but faces cannot be verified: they come
back as **Unsure** and nothing is accepted. It never guesses.

---

## How it works

```text
upload (JPEG, PNG, WebP, GIF, TIFF or BMP, up to 10 MB; the format is read from the file's bytes, not its name)
  |
  v  decode: EXIF rotation applied, transparency flattened onto white, first frame of animations,
  |          working size capped at 2048 px
  v
stage 2  visual information check (entropy, standard deviation, Laplacian)      blank image -> stop, no model runs
  |
  v
stage 3  face candidates: SCRFD-10G and YuNet (merged), animals/birds/people: YOLOX-S, anime faces: DeepGHS
         (optionally at the same time, see PPA_MODELS__PARALLEL_INFERENCE); every candidate crop is verified by SigLIP2 as one of:
         real human | animal | bird | cartoon | anime | painting | 3D | doll/statue/mask | diagram | not a face | unsure
  |
  v
stage 4  for each verified real human face: quality and pose check, then MiVOLO v2 age and gender
         (face crop, plus the body crop when a body is visible; 3 views; the views' disagreement is the uncertainty)
  |
  v
policy   app/services/policy.py turns the findings into one outcome (a pure function of the findings)
  |
  v
answer   counts, per-face results, unverified candidates, the annotated image, the models that ran
```

Why every face is verified on its own crop: face detectors also fire on cat, dog and cartoon faces, and one photo can
contain people and animals together. The decision "is this a real human face" is made per crop, and age and gender are
only produced for faces that pass it.

Stage 2 uses classical measures on purpose. "Is this image blank?" is a measurable property; a neural network would only
add latency and a chance of being wrong.

### What the age and gender fields mean

| Field | Meaning |
| :--- | :--- |
| `age_group` | The 3-year bracket containing the estimated age, `Unsure`, or `Not Detected`. |
| `estimated_age`, `age_range` | Mean over the views, and an 80 % interval whose width is the age error measured here for that age (see *Limits*). |
| `age_group_probability` | Probability that the true age lies inside `age_group`; typically 12-18 %, because a 3-year bracket is narrower than the model's error. |
| `p_under_18`, `possible_minor` | Probability the person is under 18, and whether it reaches `PPA_AGE__MINOR_RISK_PROBABILITY` (0.10). A screening signal that says "an age check is needed", never a finding. |
| `gender` | `Female`, `Male`, `Unsure` or `Not Detected`. Appearance-based; `Unsure` for apparent children, low scores or disagreeing views. |

An `Unsure` age or gender never carries a number. Faces that are too small, too turned or too blurred for the model are
reported `Unsure` without running it.

---

## Models

All run locally on the CPU through ONNX Runtime. Together about 277 MB; each is under 250 MB.

| Task | Model | Size | Licence |
| :--- | :--- | ---: | :--- |
| Face candidates | **SCRFD-10G** (InsightFace `buffalo_l` pack) | 16.9 MB | InsightFace weights: non-commercial research only |
| Face candidates, second opinion | OpenCV **YuNet** 2023mar | 0.2 MB | MIT |
| Animals, birds, people | **YOLOX-S** (COCO) | 35.9 MB | Apache-2.0 |
| Anime / cartoon faces | DeepGHS anime face detector v1.4n | 12.1 MB | MIT (model card) |
| Real human vs. everything else; image type | **SigLIP2** Base/16-224 vision tower (INT8) and pre-computed prompt embeddings | 94.6 MB | Apache-2.0 |
| Age and gender | **MiVOLO v2** (face + body transformer; community ONNX export with the gender output exposed by the build script) | 117.8 MB | Apache-2.0 per its card; training data only partly disclosed |

---

## API

The interactive reference (every field, with types) is at `/docs`; the schemas are in `app/schemas/`.

| Endpoint | Purpose |
| :--- | :--- |
| `POST /v1/analyze-profile-photo` | Multipart form with one `file`; answers with the analysis. `?annotated_image=false` leaves out the largest part of the answer. The old path `POST /analyze-profile-photo` still works and is marked deprecated. |
| `POST /v1/jobs` | Same upload, answers `202` with `status_url` and `events_url`. |
| `GET /v1/jobs/{id}` | State, the stages seen so far and, when finished, the result or the error. |
| `GET /v1/jobs/{id}/events` | Server-sent events: one `stage` event per change, then `done`. The web page uses this for its live progress. |
| `GET /v1/limits` | What the service accepts (upload size, formats), so a client can check before uploading. |
| `GET /health`, `GET /ready`, `GET /metrics` | Liveness with model files and licensing, readiness after warm-up, Prometheus metrics. |

```json
{
  "status": "PASSED",
  "accepted": true,
  "reason": "All checks passed. Estimated age group: 33 to 36 (appearance-based estimate).",
  "summary": "1 real human face(s); 0 non-human faces; 0 unverified candidate(s).",
  "real_human_face_count": 1,
  "other_face_count": 0,
  "unsure_face_count": 0,
  "human_faces": [{
    "face_id": 1, "box_xyxy": [46, 56, 158, 188], "box_percent": [25.8, 25.7, 88.8, 86.2],
    "detection_confidence": 0.91, "verification_confidence": 0.87,
    "gender": "Male", "gender_confidence": 0.997,
    "age_group": "33 to 36", "age_status": "Estimated", "estimated_age": 34.9, "age_range": [26.7, 43.1],
    "age_group_probability": 0.17, "p_under_18": 0.005, "possible_minor": false, "notes": [],
    "quality": { "status": "LIKELY_COMPLETE", "reason": "...", "sharpness": 558.9, "yaw_ratio": -0.12 }
  }],
  "other_faces": [], "unsure_faces": [],
  "image_info": { "has_information": true, "entropy": 7.4, "std_dev": 58.1, "laplacian_var": 812.0, "reason": "..." },
  "image_type": { "status": "PHOTO_LIKE", "label": "Real Photo", "photo_share": 0.98, "predictions": [] },
  "pipeline_steps": [], "diagnostics": { "processing_time_ms": 2300, "models_executed": ["..."] },
  "annotated_image": "data:image/jpeg;base64,..."
}
```

Boxes are `[x1, y1, x2, y2]`, in pixels of the analysed image (`box_xyxy`) or in percent of it (`box_percent`). The
original photo is not sent back: the client already has it.

| `status` | Meaning |
| :--- | :--- |
| `PASSED` | Exactly one verified real human face, nothing doubtful, good quality, no sign of a minor. The only status with `accepted: true`. |
| `NO_INFORMATION` | Blank, solid-colour or uniform image. No model ran. |
| `NO_HUMAN_FACE` | Nothing that looks like a face was found. |
| `ONLY_NON_HUMAN_FACES` | Animals, birds, cartoons, ... but no real human face. |
| `MULTIPLE_HUMAN_FACES` | More than one verified human face (each still gets its own age and gender). |
| `NON_HUMAN_FACE_PRESENT` | One human face, plus an animal or cartoon face. |
| `REVIEW_REQUIRED` | A candidate or the image type could not be verified. |
| `POOR_FACE_QUALITY` | Cropped, blurry, too dark or bright, too small, or turned away. |
| `AGE_UNSURE` | The age could not be determined with enough certainty. |
| `POSSIBLE_MINOR` | `p_under_18` is at or above the threshold; an age check is needed. |

The rules that map findings to a status are in one pure module, `app/services/policy.py`.

**Errors** always have one shape, with a stable `code` and the request id (also in the `X-Request-ID` header):

```json
{ "accepted": false, "code": "file_too_large", "reason": "Maximum upload size is 10 MB.", "request_id": "3f2c..." }
```

Codes: `empty_file`, `file_too_large`, `invalid_image`, `analysis_failed`, `unauthorized`, `rate_limited`, `busy`,
`queue_full`, `job_not_found`, `not_found`, `method_not_allowed`, `validation_error`, `internal_error`.

### Web page

`/` serves a single-page interface with no framework and no build step (`app/templates/index.html`,
`app/static/analyzer.css`, ES modules in `app/static/js/`): upload by clicking, dragging or keyboard (photos over the
size limit are shrunk in the browser first), live progress for the four stages, the verdict in plain words, the photo
with boxes you can hover or tab to (green = human, purple = non-human, amber = unverified), a card per human face with
age bracket, range and probability, gender and quality, and a technical-details panel. It can export the JSON report and the
annotated image, follows the system light or dark setting, and respects reduced motion.

---

## Configuration

Settings are grouped by pipeline stage in `app/core/config.py`. An environment variable is `PPA_<GROUP>__<FIELD>` (two
underscores), for example `PPA_AGE__MINOR_RISK_PROBABILITY=0.2`; ungrouped fields use one underscore. The important ones
(all of them, with their groups, are described in [docs/OPERATIONS.md](docs/OPERATIONS.md)):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PPA_ENVIRONMENT` | `development` | `production` requires `PPA_SECURITY__API_KEY`. |
| `PPA_SECURITY__API_KEY` | none | Sent as `X-API-Key` or `Authorization: Bearer ...`. |
| `PPA_SERVER__HOST`, `PPA_SERVER__PORT` | `127.0.0.1`, `8000` | Where `python -m app` listens (`0.0.0.0` in a container). |
| `PPA_UPLOAD__MAX_BYTES` | 10 MiB | Upload limit. |
| `PPA_VERIFICATION__HUMAN_MIN_PROB`, `PPA_VERIFICATION__HUMAN_MIN_MARGIN` | 0.70, 0.25 | How sure SigLIP2 must be to call a crop a real human face (measured trade-off in the evaluation document). |
| `PPA_DETECTION__ENABLE_SCRFD` | true | `false` leaves out the non-commercial SCRFD detector (YuNet alone). |
| `PPA_AGE__MINOR_RISK_PROBABILITY` | 0.10 | `p_under_18` at or above which `possible_minor` is set. |
| `PPA_AGE__UNSURE_VIEW_STD` | 6.0 | Age is `Unsure` if the views of a face differ by more than this (years). |
| `PPA_AGE__GENDER_SURETY` | 0.80 | Below this model score the gender is `Unsure`. |
| `PPA_MODELS__ORT_THREADS` | 0 (= min(4, cores)) | ONNX Runtime threads per model; "all cores" measured about 4x slower. |
| `PPA_MODELS__PARALLEL_INFERENCE` | false | Run independent detectors and age views at the same time. Measured slower on a 14-thread CPU (they compete for cores), so it is opt-in. |
| `PPA_MODELS__FORCE_CPU` | true | `false` lets ONNX Runtime use CUDA or DirectML if installed (not exercised here). |
| `PPA_JOBS__WORKERS`, `PPA_JOBS__QUEUE_SIZE` | 2, 8 | Analyses running and waiting in the job API; beyond that `503 queue_full`. |
| `PPA_SECURITY__RATE_LIMIT_PER_MINUTE`, `PPA_SECURITY__MAX_CONCURRENT_REQUESTS` | 60, 10 | Per-process abuse limits; limit at the edge when running several processes. |
| `PPA_LOG_JSON` | false | One JSON object per log line. |

---

## Limits

What was measured, how, and what was not is in [docs/EVALUATION_AND_FAIRNESS.md](docs/EVALUATION_AND_FAIRNESS.md).
The short version:

* **Age is approximate.** The error was measured on labelled faces and sets the width of the reported range; a 3-year
  bracket is therefore the most likely bracket, not a certainty, and age estimation is not age verification.
* **Gender is an appearance classification**: binary, hard to judge for children (reported `Unsure` below 13) and for
  atypical presentation. Do not use it for decisions about people.
* **Paintings and pencil sketches of faces are the weakest case** for telling a real face from a depiction; the
  measured rates are in the evaluation document.
* **Animals and birds** are located as whole-animal boxes for ten COCO classes (cat, dog, horse, sheep, cow, elephant,
  bear, zebra, giraffe, bird). Other species are only counted if a face-like candidate is verified as an animal.
* **Speed:** a few seconds per image on a laptop CPU, most of it the age model.
* **Scale:** job state and the rate limiter live in one process; see [docs/OPERATIONS.md](docs/OPERATIONS.md).

---

## Development

```bash
pip install -r requirements-dev.lock ruff    # tests, strict type checking, the build and evaluation tooling
npm ci                                       # lint and tests for the browser code

ruff check . && ruff format --check .
mypy                                         # strict
pytest                                       # 100 % statement and branch coverage; real-model tests run when the models are built
npm run lint && npm test                     # ESLint; browser tests with 100 % coverage

python scripts/fetch_eval_data.py            # about 2,000 labelled images into .eval_data/
python scripts/evaluate.py --workers 3       # measure the real pipeline; add --gates to check evaluation/gates.json
python scripts/benchmark_latency.py --image portrait.jpg [--parallel]
```

Tests that need the built models skip themselves when the models are absent. Set `PPA_TEST_IMAGES_DIR` to also
smoke-test a small folder of your own photos (it analyzes every image in the folder, so keep it small).

```text
app/
  api/            routes.py (endpoints), middleware.py (request id, auth, limits, body cap), limiter.py
  core/           config.py (grouped settings), errors.py, logging.py, metrics.py
  schemas/        analysis.py (results, jobs, errors), ops.py (health, readiness, limits)
  services/
    analyzer.py             orchestrator: runs the stages, reports progress
    policy.py               the profile-photo rules (pure functions)
    types.py                typed shapes passed between stages
    jobs.py                 bounded worker pool and job table
    parallel.py             concurrent model calls on a shared pool
    progress.py             stage events
    face_service.py         SCRFD + YuNet detection; MiVOLO age/gender with uncertainty
    classifier_service.py   image type; YOLOX and anime detectors; verdict on each crop
    semantic_service.py     SigLIP2 embeddings and zero-shot scoring
    label_prompts.py        the zero-shot label taxonomy
    quality_service.py      visual-information check; face quality and pose
    image_utils.py          decoding (EXIF, alpha, limits), crops, drawing
    model_manager.py        model registry, SHA-256 verification, ONNX sessions, warm-up
  static/js/      the browser code (ES modules); static/analyzer.css
  templates/      the page
evaluation/       metrics and runner for the labelled evaluation
scripts/          build_models, fetch_models, fetch_eval_data, evaluate, benchmark_latency
deploy/           nginx edge and Docker Compose
tests/            Python tests; tests/js holds the browser tests (jsdom)
models/           ONNX files (git-ignored), manifest.json, label_embeddings.npz
```

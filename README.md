# Profile Photo Analyzer

A FastAPI web service and single-page UI that analyses an uploaded photo and tells you, using local ONNX models only:

1. whether the image has any visual information at all (blank / solid colour / uniform fill are rejected),
2. how many faces it contains and **which of them are real human faces** versus animals, birds, cartoons, anime,
   paintings, 3D renders, dolls/statues/masks or diagrams (each kind is counted and reported separately),
3. the **gender** and the **age in 3-year brackets** (0 to 3, 3 to 6, ... 18 to 21, ...) of every real human face.

Nothing is invented: when a model cannot support an answer the field says **`Unsure`**, and when nothing could be
detected it says **`Not Detected`**. Every number in a response comes from a model that actually ran, and the response
lists exactly which ones (`technical_diagnostics.models_executed`).

---

## How the pipeline works

```text
upload (<= 10 MB; JPEG, PNG, WebP, GIF, TIFF, BMP - format checked from the file's bytes, not its extension)
  -> decode: EXIF-upright, transparency flattened onto white, first frame of animations, working size capped at 2048 px
  -> 2. information check (entropy / std-dev / Laplacian)           -> "no information" stops here, no model runs
  -> 3. face candidates:  SCRFD-10G + YuNet (merged)  and  YOLOX-S (animals, birds, people)  and  DeepGHS (anime faces)
        every candidate crop is verified by SigLIP2 zero-shot:  real human | animal | bird | cartoon | anime |
        painting | 3D | doll/statue/mask | diagram | not a face | Unsure
  -> 4. for every verified real human face: quality/pose check, then MiVOLO v2 age + gender on the face crop
        (and the body crop when a body is visible), 3 views per face; disagreement between views = uncertainty
  -> 5. response: counts, per-face results, unverified candidates, the models that ran
```

Why per-face verification: the detectors also fire on cat, dog and cartoon faces, and the image as a whole may be a
photo that contains both people and animals. The decision "is this a real human face" is therefore made on each face
crop, and age/gender are only produced for faces that pass it.

Step 2 deliberately uses classical measures (Shannon entropy, standard deviation, Laplacian variance) rather than a
neural network: "is this image blank or a uniform fill" is a measurable property, and a pretrained model would only
add latency and a chance of being wrong.

### What the age and gender fields mean

| Field | Meaning |
| :--- | :--- |
| `age_group` | The 3-year bracket containing the estimated age, `Unsure`, or `Not Detected`. |
| `estimated_age`, `age_range` | Mean over the 6 views, and an 80 % interval. The interval is wide on purpose (see *Limits*). |
| `age_confidence` | Probability that the true age is inside `age_group` (typically 12-18 %: a 3-year bracket is narrower than the model's error). |
| `p_under_18` | Probability the person is under 18. `possible_minor_risk` is true at or above `PPA_MINOR_RISK_PROBABILITY` (0.10). |
| `gender` | `Female`, `Male`, `Unsure` or `Not Detected` (appearance-based; `Unsure` for apparent children, low scores or disagreeing views). |

`Unsure` age/gender fields never carry a number. Faces that are too small, too turned or too blurred for the model are
reported `Unsure` without running it.

---

## Models (all run locally, CPU, ONNX Runtime)

| Task | Model | Size | Licence |
| :--- | :--- | ---: | :--- |
| Face candidates (primary) | **SCRFD-10G** (InsightFace `buffalo_l` pack) | 16.9 MB | InsightFace weights: non-commercial research only |
| Face candidates (second opinion) | OpenCV **YuNet** 2023mar | 0.2 MB | MIT (OpenCV Zoo) |
| Animals, birds, people | **YOLOX-S** (COCO) | 35.9 MB | Apache-2.0 |
| Anime / cartoon face locator | DeepGHS anime face v1.4n | 12.1 MB | MIT (model card) |
| Real human vs. everything else, image type | **SigLIP2** Base/16-224 vision tower, INT8, pre-computed prompt embeddings | 94.6 MB | Apache-2.0 |
| Age + gender | **MiVOLO v2** (face + body transformer; community ONNX export with the gender head exposed) | 117.8 MB | Apache-2.0 per model card; training data only partly disclosed |

Total: about 277 MB (each model is under 250 MB). These were chosen as the strongest pretrained models found, without
regard to licence. **Read [NOTICE.md](NOTICE.md) before any commercial use**: the SCRFD weights are non-commercial,
and the MiVOLO v2 training data is not fully disclosed.

---

## Install and run

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.lock           # runtime dependencies only (no torch, no transformers)

# One-time model build: downloads ~1 GB of pinned upstream files (SHA-256 verified) and builds the SigLIP2 model +
# label embeddings, the MiVOLO age/gender model and the SCRFD detector into models/.
# YuNet, YOLOX and the anime detector download themselves on first use.
pip install -r requirements-build.txt
python scripts/build_models.py --cleanup

python -m app                              # http://127.0.0.1:8000   (or: uvicorn app.main:app)
```

`GET /health` reports every model as ready or not (`degraded`). Without the built models the service stays up but
verification is impossible, so faces come back as **Unsure** and nothing is accepted - it never guesses.

### API

`POST /analyze-profile-photo` with a multipart `file`. Main response fields: `accepted`, `posting_status`,
`reason`, `summary`, `real_human_face_count`, `other_face_count`, `unsure_face_count`, `human_faces[]`,
`other_faces[]` (each with `face_type`, `label`, `verified_by`), `unsure_faces[]`, `pipeline_steps[]`,
`technical_diagnostics` (models that ran, original resolution, EXIF rotation, timing), and `original_image` /
`annotated_image` data URLs. The schema is `app/schemas/analysis.py`.

`posting_status` values: `NO_INFORMATION`, `REJECTED`, `NON_PHOTO_WARNING`, `MANUAL_REVIEW_REQUIRED`,
`REJECTED_OR_REVIEW`, `AGE_ESTIMATION_UNAVAILABLE`, `AGE_VERIFICATION_REQUIRED`, `INITIAL_CHECKS_PASSED`.
The profile policy (exactly one verified real human face, nothing unverified, good quality, adult-looking) lives in
`analyzer._decide`; the per-face data is independent of it.

---

## Configuration

Every setting in `app/core/config.py` can be overridden with a `PPA_` environment variable. The ones that matter most:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PPA_ENVIRONMENT` | `development` | `production` requires `PPA_API_KEY`. |
| `PPA_API_KEY` | none | Required in production; sent as `X-API-Key` or `Authorization: Bearer`. |
| `PPA_MAX_UPLOAD_BYTES` | 10 MiB | Upload limit. |
| `PPA_WORKING_MAX_SIDE` | 2048 | Longest side the analysis runs at (larger uploads are downscaled). |
| `PPA_HUMAN_MIN_PROB` / `PPA_HUMAN_MIN_MARGIN` | 0.60 / 0.25 | How sure SigLIP2 must be to call a face a real human face. |
| `PPA_SCRFD_THRESHOLD` | 0.40 | Candidate score for the SCRFD detector (candidates are verified afterwards). |
| `PPA_AGE_UNSURE_VIEW_STD` | 6.0 | View disagreement (years) above which age is `Unsure`. |
| `PPA_MINOR_RISK_PROBABILITY` | 0.10 | `p_under_18` at/above which `possible_minor_risk` is set. |
| `PPA_GENDER_SURETY_THRESHOLD` | 0.80 | Minimum gender score, otherwise `Unsure`. |
| `PPA_ORT_INTRA_OP_THREADS` | 0 (= min(4, cores)) | ONNX Runtime threads per model; the "all cores" default measured ~4x slower. |
| `PPA_FORCE_CPU` | true | Set false to use CUDA/DirectML providers if installed. |
| `PPA_RATE_LIMIT_PER_MINUTE`, `PPA_MAX_CONCURRENT_REQUESTS`, `PPA_TRUSTED_PROXIES` | 60, 10, loopback | Process-local abuse controls (use a gateway for multi-worker deployments). |

---

## Limits you should know about

* **Age is approximate.** In an independent 8-dataset benchmark MiVOLO's average error was 5.1 years (the figure
  used here to size the age range; not re-measured on our data). A 3-year bracket is narrower than that error, so it is the *most likely* bracket, shown together
  with an 80 % range and its probability. It is not age verification.
* **Gender is a visual appearance classification**, binary, hard to judge for children (this service reports `Unsure` below 13) and for atypical presentation. Do not use it for decisions about people.
* **Animals/birds** are found as whole-animal boxes from the 10 COCO animal classes (cat, dog, horse, sheep, cow,
  elephant, bear, zebra, giraffe, bird); other species are only caught if a face-like candidate is verified as an animal.
* Cartoons/paintings/diagrams are *counted* only when a detector or YuNet localises them; otherwise the image-type
  card reports the type and the face count stays 0.
* Accuracy has **not** been measured on a labelled benchmark yet - see
  [docs/EVALUATION_AND_FAIRNESS.md](docs/EVALUATION_AND_FAIRNESS.md) for what was and was not verified, and
  `scripts/evaluate_fairface.py` to measure it.

---

## Development

```bash
pip install -r requirements.lock && pip install ruff
ruff check app tests scripts && ruff format --check app tests scripts
pytest                                       # unit tests + (if models are built) real-model integration tests
python scripts/benchmark_latency.py --image portrait.jpg
python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <dir>
```

Current state: 239 tests (one needs `PPA_TEST_IMAGES_DIR`); statement and branch coverage of `app/` is 100 % (the real-model integration tests are
included when the models are built, and skipped otherwise). Set `PPA_TEST_IMAGES_DIR` to also smoke-test your own photos.

```text
app/
  api/routes.py            upload validation, endpoints
  core/config.py           settings (PPA_ env vars)
  schemas/analysis.py      response schema
  services/
    image_utils.py         decoding (EXIF, alpha, limits), crops, annotation
    quality_service.py     information check, face quality/pose
    face_service.py        SCRFD + YuNet detection, MiVOLO age/gender with uncertainty
    classifier_service.py  image type, YOLOX/anime detectors, crop verdicts
    semantic_service.py    SigLIP2 embeddings and zero-shot scoring
    label_prompts.py       the zero-shot label taxonomy
    model_manager.py       model registry, SHA-256 verification, sessions
    analyzer.py            orchestration and profile policy
  templates/index.html     UI
scripts/                   build_models.py, benchmark_latency.py, evaluate_fairface.py
models/                    ONNX files (git-ignored), manifest.json, label_embeddings.npz
```

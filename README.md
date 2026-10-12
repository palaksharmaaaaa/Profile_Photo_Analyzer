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
> use**. Read [NOTICE.md](NOTICE.md) first.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows;  source .venv/bin/activate on Linux and macOS
pip install -r requirements.lock            # runtime dependencies (no torch, no transformers)

pip install -r requirements-build.txt       # one-time: builds the models into models/ (about 1 GB of downloads,
python scripts/build_models.py --cleanup    # every file verified by SHA-256, scratch files removed afterwards)

python -m app                               # open http://127.0.0.1:8000
```

YuNet, YOLOX and the anime detector download themselves on first use. `GET /health` reports every model as ready
or not (`ready` / `degraded`). If the built models are missing the service still starts, but faces cannot be
verified: they come back as **Unsure** and nothing is accepted. It never guesses.

---

## How it works

```text
upload (JPEG, PNG, WebP, GIF, TIFF or BMP, up to 10 MB; the format is read from the file's bytes, not its extension)
  |
  v  decode: EXIF rotation applied, transparency flattened onto white, first frame of animations,
  |          working size capped at 2048 px
  v
step 2   visual information check (entropy, standard deviation, Laplacian)      blank image -> stop, no model runs
  |
  v
step 3   face candidates: SCRFD-10G and YuNet (merged), animals/birds/people: YOLOX-S, anime faces: DeepGHS
         every candidate crop is verified by SigLIP2 as one of:
         real human | animal | bird | cartoon | anime | painting | 3D | doll/statue/mask | diagram | not a face | unsure
  |
  v
step 4   for each verified real human face: quality and pose check, then MiVOLO v2 age and gender
         (face crop, plus the body crop when a body is visible; 3 views; the views' disagreement is the uncertainty)
  |
  v
step 5   response: counts, per-face results, unverified candidates, annotated image, the models that ran
```

Why every face is verified on its own crop: face detectors also fire on cat, dog and cartoon faces, and one photo can
contain people and animals together. The decision "is this a real human face" is made per crop, and age and gender are
only produced for faces that pass it.

Step 2 uses classical measures on purpose. "Is this image blank?" is a measurable property; a neural network would only
add latency and a chance of being wrong.

### What the age and gender fields mean

| Field | Meaning |
| :--- | :--- |
| `age_group` | The 3-year bracket containing the estimated age, `Unsure`, or `Not Detected`. |
| `estimated_age`, `age_range` | Mean over the views, and an 80 % interval. The interval is wide on purpose (see *Limits*). |
| `age_group_probability` | Probability that the true age lies inside `age_group`; typically 12-18 %, because a 3-year bracket is narrower than the model's error. |
| `p_under_18`, `possible_minor` | Probability the person is under 18, and whether it is at least `PPA_AGE__MINOR_RISK_PROBABILITY` (0.10). Use this, not the bracket, for policy decisions. |
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

`POST /analyze-profile-photo` takes a multipart form with one `file`. The interactive reference (every field, with
types) is at `/docs`; the schema is `app/schemas/analysis.py`.

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
  "original_image": "data:image/jpeg;base64,...", "annotated_image": "data:image/jpeg;base64,..."
}
```

Boxes are `[x1, y1, x2, y2]`, in pixels (`box_xyxy`) or in percent of the image (`box_percent`).

| `status` | Meaning |
| :--- | :--- |
| `PASSED` | Exactly one verified real human face, nothing doubtful, good quality, adult-looking. The only status with `accepted: true`. |
| `NO_INFORMATION` | Blank, solid-colour or uniform image. No model ran. |
| `NO_HUMAN_FACE` | Nothing that looks like a face was found. |
| `ONLY_NON_HUMAN_FACES` | Animals, birds, cartoons, ... but no real human face. |
| `MULTIPLE_HUMAN_FACES` | More than one verified human face (each still gets its own age and gender). |
| `NON_HUMAN_FACE_PRESENT` | One human face, plus an animal or cartoon face. |
| `REVIEW_REQUIRED` | A candidate or the image type could not be verified. |
| `POOR_FACE_QUALITY` | Cropped, blurry, too dark or bright, too small, or turned away. |
| `AGE_UNSURE` | The age could not be determined with enough certainty. |
| `POSSIBLE_MINOR` | `p_under_18` is at or above the threshold; age verification is required. |

The policy that maps a result to a status is the single function `analyzer._decide`. Errors answer with
`{"accepted": false, "reason": "..."}` and status 400 (bad file), 413 (too large) or 500.

### Web page

`/` serves a single-page interface (`app/templates/index.html`, `app/static/analyzer.css`, `app/static/analyzer.js`):
upload by clicking or dragging, a live four-step progress strip, the verdict in plain words, the photo with hoverable
boxes (green = human, purple = non-human, amber = unverified), a card per human face with age bracket, range and
probability, gender and quality, and a technical-details panel. It can export the JSON report and the annotated image.

---

## Configuration

Every setting in `app/core/config.py` can be overridden with an environment variable named `PPA_<SETTING>`. The
important ones:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PPA_ENVIRONMENT` | `development` | `production` requires `PPA_SECURITY__API_KEY`. |
| `PPA_SECURITY__API_KEY` | none | Sent as `X-API-Key` or `Authorization: Bearer ...`. |
| `PPA_UPLOAD__MAX_BYTES` | 10 MiB | Upload limit. |
| `PPA_UPLOAD__WORKING_MAX_SIDE` | 2048 | Larger images are downscaled to this for analysis. |
| `PPA_VERIFICATION__HUMAN_MIN_PROB`, `PPA_VERIFICATION__HUMAN_MIN_MARGIN` | 0.60, 0.25 | How sure SigLIP2 must be to call a crop a real human face. |
| `PPA_DETECTION__SCRFD_THRESHOLD` | 0.40 | Minimum SCRFD score for a face candidate. |
| `PPA_AGE__MINOR_RISK_PROBABILITY` | 0.10 | `p_under_18` at or above which `possible_minor` is set. |
| `PPA_AGE__MODEL_MAE_YEARS` | 5.1 | Average age error assumed when sizing the age range. |
| `PPA_AGE__UNSURE_VIEW_STD` | 6.0 | Age is `Unsure` if the views of a face differ by more than this (years). |
| `PPA_AGE__GENDER_SURETY` | 0.80 | Below this model score the gender is `Unsure`. |
| `PPA_MODELS__ORT_THREADS` | 0 (= min(4, cores)) | ONNX Runtime threads per model; "all cores" measured about 4x slower. |
| `PPA_MODELS__FORCE_CPU` | true | `false` lets ONNX Runtime use CUDA or DirectML if installed. |
| `PPA_SECURITY__RATE_LIMIT_PER_MINUTE`, `PPA_SECURITY__MAX_CONCURRENT_REQUESTS` | 60, 10 | Per-process abuse limits; use a gateway for multi-worker setups. |

---

## Limits

* **Age is approximate.** In an independent 8-dataset benchmark MiVOLO's average error was 5.1 years (this is the value
  used to size the age range; it was not re-measured here). A 3-year bracket is therefore the most likely bracket, not
  a certainty. It is also not age verification.
* **Gender is an appearance classification**: binary, hard to judge for children (reported `Unsure` below 13) and for
  atypical presentation. Do not use it for decisions about people.
* **Animals and birds** are located as whole-animal boxes for ten COCO classes (cat, dog, horse, sheep, cow, elephant,
  bear, zebra, giraffe, bird). Other species are only counted if a face-like candidate is verified as an animal.
* **Cartoons, paintings and diagrams** are counted when a detector or the face detector finds them; otherwise the
  image-type card names the type and the face count stays 0.
* **Speed:** about 3-4 seconds per image on a 14-thread laptop CPU, most of it the age model.
* **Accuracy has not been measured on a labelled benchmark.** [docs/EVALUATION_AND_FAIRNESS.md](docs/EVALUATION_AND_FAIRNESS.md)
  states what was and was not verified; `scripts/evaluate_fairface.py` measures it on labelled data.

---

## Development

```bash
pip install ruff
ruff check app tests scripts && ruff format --check app tests scripts
pytest                                      # unit tests, plus real-model tests when the models are built
python scripts/benchmark_latency.py --image portrait.jpg
python scripts/analyze_folder.py <folder> --out results.csv
python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <folder>
```

243 tests (242 run by default); statement and branch coverage of `app/` is 100 %. Tests that need the built models
skip themselves when the models are absent. Set `PPA_TEST_IMAGES_DIR` to also smoke-test a small folder of your own photos (it analyzes
every image in the folder, so keep it small).

```text
app/
  api/routes.py             endpoints: upload validation, /analyze-profile-photo, /health, the page
  core/config.py            all settings
  schemas/analysis.py       the response contract (Status, HumanFace, ...)
  services/
    analyzer.py             orchestrator and the profile-photo policy
    face_service.py         SCRFD + YuNet detection; MiVOLO age/gender with uncertainty
    classifier_service.py   image type; YOLOX and anime detectors; verdict on each crop
    semantic_service.py     SigLIP2 embeddings and zero-shot scoring
    label_prompts.py        the zero-shot label taxonomy
    quality_service.py      visual-information check; face quality and pose
    image_utils.py          decoding (EXIF, alpha, limits), crops, drawing
    model_manager.py        model registry, SHA-256 verification, ONNX sessions
  static/ templates/        the web page
  main.py                   application factory, security middleware
scripts/                    build_models.py, benchmark_latency.py, analyze_folder.py, evaluate_fairface.py
models/                     ONNX files (git-ignored), manifest.json, label_embeddings.npz
```

# Evaluation status

This document separates three things: **what has been measured** in this repository, **what is only claimed** by a
model's authors, and **what has not been measured at all**. Every number in the measured tables is generated from a run of
`scripts/evaluate.py` (see section 4); none is typed by hand. Nothing here is a target presented as a result.

## 1. Measured here

### 1.1 Method

The real pipeline (detection, verification, age and gender, policy; no fakes) was run on labelled images that are
downloaded by `scripts/fetch_eval_data.py` into the git-ignored `.eval_data/` folder. A fixed random seed makes the sample
reproducible.

| Set | Source | What the label says |
| :--- | :--- | :--- |
| `fairface` | FairFace validation split, 1.25-padding crops (CC BY 4.0) | Real human faces, each with an age **bin**, a gender and a race group |
| `pets`, `birds` | Oxford pets, Caltech-UCSD Birds-200-2011 | Cats, dogs and birds: must never be reported as a real human face |
| `anime`, `cartoon` | anime-faces-256, cartoon-faces | Drawn faces: same |
| `paintings` | WikiArt, genre "portrait" | Painted portraits of people: not real human faces |
| `sketches` | Face-Sketches40 | Pencil sketches of faces: not real human faces |

How to read what follows:

* **FairFace crops come from group photos**, so neighbours can be in frame. "Exactly one face" is therefore not a ground
  truth for them; "at least one real human face found" is the recall that can be trusted.
* **FairFace gives age as a bin, never as a number.** Anything derived from a bin names its assumption: the distance
  to the bin is a lower bound on the real error (an estimate inside the bin counts as zero), and the "if ages were spread
  evenly" figures assume that, which is not known. The error spread per age band takes the bin's own width out of the
  variance of the estimates, under the same assumption.
* **MiVOLO v2's training data is not fully disclosed**, so overlap with FairFace cannot be ruled out; the accuracy figures
  may be better than on unseen data.
* The non-human sets are small and their images are not photographs of the kind a profile-photo upload contains
  (pets and birds are whole-animal photos, anime and cartoon images are tight crops). They show which kinds of image fool
  the verification, not how often that happens in production.
* A face reported as `Unsure` is counted as such everywhere: it is neither a hit nor a miss.

### 1.2 Results

<!-- measured:start -->

Run of 2026-10-13. Images: anime 120, birds 120, cartoon 40, fairface 1200, paintings 120, pets 120, sketches 40.
Settings: verification threshold 0.7 with margin 0.25, minor screening threshold 0.1, SCRFD on.

#### Real portraits: detection and verification

| Outcome on real portraits | Share |
| :--- | ---: |
| At least one real human face found | 95.8 % |
| Exactly one real human face found | 78.3 % |
| More than one real human face found | 17.4 % |
| No real human face found | 4.2 % |
| A non-human face reported as well | 0.2 % |
| Analysis errors | 0 of 1200 |

#### Gender

| Group | Faces | Accuracy when answered | Left Unsure | Unsure, labelled 0-9 | Unsure, labelled 20+ | Unsure with no score at all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 1149 | 98.4 % | 29.4 % | 96.3 % | 14.9 % | 13.4 % |
| Black | 165 | 97.2 % | 35.1 % | 93.5 % | 16.8 % | 10.9 % |
| East Asian | 149 | 98.0 % | 31.5 % | 100.0 % | 13.3 % | 12.1 % |
| Indian | 170 | 100.0 % | 30.0 % | 100.0 % | 13.1 % | 12.9 % |
| Latino_Hispanic | 168 | 96.7 % | 28.6 % | 90.5 % | 13.6 % | 16.1 % |
| Middle Eastern | 124 | 100.0 % | 25.0 % | 100.0 % | 17.0 % | 14.5 % |
| Southeast Asian | 154 | 100.0 % | 26.0 % | 91.7 % | 8.8 % | 9.1 % |
| White | 219 | 97.4 % | 28.8 % | 100.0 % | 19.6 % | 16.9 % |

#### Age

| Age measure | Value |
| :--- | ---: |
| Faces with an age bin | 1149 |
| Age left Unsure | 13.6 % |
| Estimate inside the labelled bin | 60.8 % |
| Mean distance to the bin, years (a lower bound on the error) | 1.69 |
| Mean error, years, if ages were spread evenly inside each bin | 5.46 |

| Estimated age below | Faces | Bias (years) | Error spread (years) | Spread taken from |
| :--- | ---: | ---: | ---: | ---: |
| 10 | 145 | 1.51 | 7.32 | band |
| 20 | 91 | 1.28 | 7.30 | band |
| 30 | 307 | 0.07 | 4.96 | band |
| 40 | 219 | -1.52 | 6.49 | band |
| 50 | 132 | 0.86 | 8.26 | band |
| 60 | 63 | 0.58 | 6.81 | band |
| any | 23 | 0.21 | 6.63 | pooled |

#### Age by race group

| Group | Faces | Age Unsure | Inside bin | Distance to bin (years) | Labelled 0-9 flagged | Labelled 30+ flagged |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 1149 | 13.6 % | 60.8 % | 1.69 | 98.6 % | 4.2 % |
| Black | 165 | 10.9 % | 58.5 % | 2.34 | 96.5 % | 7.1 % |
| East Asian | 149 | 12.1 % | 67.9 % | 1.23 | 100.0 % | 5.4 % |
| Indian | 170 | 12.9 % | 57.4 % | 1.72 | 100.0 % | 5.8 % |
| Latino_Hispanic | 168 | 16.7 % | 65.0 % | 2.14 | 94.1 % | 3.1 % |
| Middle Eastern | 124 | 14.5 % | 58.5 % | 1.50 | 100.0 % | 1.5 % |
| Southeast Asian | 154 | 9.1 % | 57.1 % | 1.49 | 100.0 % | 7.5 % |
| White | 219 | 17.3 % | 61.3 % | 1.41 | 100.0 % | 1.1 % |

#### Possible-minor screening

| `minor_risk_probability` | Labelled 0-9 flagged | Labelled 10-19 flagged | Labelled 30+ flagged |
| :--- | ---: | ---: | ---: |
| 0.02 | 98.6 % | 97.8 % | 16.4 % |
| 0.05 | 98.6 % | 97.8 % | 6.0 % |
| 0.10 | 98.6 % | 94.5 % | 4.2 % |
| 0.15 | 98.6 % | 90.1 % | 2.2 % |
| 0.20 | 98.6 % | 83.5 % | 2.0 % |
| 0.30 | 98.6 % | 75.8 % | 1.3 % |
| 0.50 | 98.6 % | 59.3 % | 0.0 % |

Faces scored: 147 labelled 0-9, 450 labelled 30 or older.

#### Images that must not be reported as a real human face

| Set | Images | A face reported as real human | Passed the whole policy | Non-human face | Unsure | Nothing |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 560 | 2.5 % | 0.5 % | 73.0 % | 19.1 % | 5.4 % |
| anime | 120 | 0.0 % | 0.0 % | 95.0 % | 0.0 % | 5.0 % |
| birds | 120 | 0.0 % | 0.0 % | 90.8 % | 5.0 % | 4.2 % |
| cartoon | 40 | 7.5 % | 2.5 % | 87.5 % | 2.5 % | 2.5 % |
| paintings | 120 | 6.7 % | 0.8 % | 28.3 % | 50.8 % | 14.2 % |
| pets | 120 | 0.0 % | 0.0 % | 96.7 % | 2.5 % | 0.8 % |
| sketches | 40 | 7.5 % | 2.5 % | 2.5 % | 90.0 % | 0.0 % |

#### The verification threshold

| `human_min_prob` | Portraits verified | Non-human images verified as human |
| :--- | ---: | ---: |
| 0.5 | 98.8 % | 10.9 % |
| 0.6 | 98.1 % | 6.2 % |
| 0.7 (used) | 96.2 % | 2.5 % |
| 0.8 | 85.0 % | 0.9 % |
| 0.9 | 35.0 % | 0.0 % |

#### Speed during this run (3 processes sharing the machine)

Whole analysis, median 7994 ms and 90th percentile 12036 ms.

<!-- measured:end -->

### 1.3 What the tables show, and what they do not

* **Verification.** The threshold trades real portraits that are sent to manual review against depictions that pass as
  real. The table shows the whole trade-off; the value in use was chosen from it (see `verification.human_min_prob` in
  `app/core/config.py`) and is a policy choice, not an optimum.
* **Paintings and pencil sketches of faces** are the hardest case: a realistic depiction of a face looks like a face to
  every model here. The "passed the whole policy" column is the one that matters for safety; it is not zero.
* **Gender** is answered with high accuracy when it is answered, and is left `Unsure` for apparent children by design.
  The accuracy is for a binary appearance label on a dataset that may overlap the model's training data; it says
  nothing about people whose appearance does not fit the two labels.
* **Age error** is several years and differs by age; the per-band table is what sizes the reported 80 % range. The
  differences between race groups in the by-race tables are measured, not explained, and the groups are small enough
  that small gaps are noise.
* **Minor screening** is a trade-off the threshold controls: a lower threshold sends more adults to an age check and
  misses fewer minors. FairFace's 10-19 bin straddles 18, so it can only show how often adolescents are flagged, not
  recall. The screening is a signal that an age check is needed, never a determination of age.
* **Speed** was measured while three analyses ran at once on one machine; it is not the latency of a single request.
  `scripts/benchmark_latency.py` measures that.

### 1.4 Speed of a single request

`python scripts/benchmark_latency.py --image <portrait>` on the development machine (Windows 11, 14 logical CPUs, no GPU),
one stage at a time and with parallel inference (`models.parallel_inference`). On this machine parallel inference makes the whole analysis slower, although the three age views alone get faster, because the models compete for the same cores; it is therefore off by default:

<!-- benchmark:start -->

Python 3.14.5, onnxruntime 1.31.0, 14 logical CPUs, min(4, cores) ONNX threads per model, image 448 x 448 px, 8 iterations.

| Stage (ms) | One at a time p50 | p90 | Parallel p50 | p90 |
| :--- | ---: | ---: | ---: | ---: |
| information check | 3 | 3 | 3 | 4 |
| image type (SigLIP2) | 94 | 109 | 123 | 130 |
| face detection (SCRFD + YuNet) | 243 | 374 | 242 | 252 |
| animal + person + anime detection | 412 | 488 | 403 | 611 |
| face verification (1 crop) | 68 | 133 | 126 | 128 |
| age + gender (MiVOLO, 3 views) | 1896 | 2457 | 1534 | 1618 |
| END-TO-END | 2729 | 2788 | 3619 | 4168 |

<!-- benchmark:end -->

Design choices that came from measurements: ONNX Runtime's default "use all cores" setting was about 4x slower than
4 threads on this CPU, so the default is `min(4, cores)`; INT8 quantisation of MiVOLO was 33 % faster but shifted ages by
about 0.8 years on average, so the exact fp32 model is used; the INT8 SigLIP2 model quantises per batch, so its output
depends on what else is in the batch and crops are embedded one at a time (an integration test guards this).

### 1.5 Automated tests

The Python suite gives 100 % statement and branch coverage of `app/`, `evaluation/` and `scripts/`; the browser code
has 100 % coverage from its own tests. That proves the code paths behave as specified (with fakes, and with the real
models where they are built, including one that checks parallel and sequential inference give the same answer); it says
nothing about model accuracy.

## 2. Claimed by the models' authors (not re-measured)

| Claim | Source | Caveat |
| :--- | :--- | :--- |
| MiVOLO v2: age error 3.65 years, gender accuracy 98 % (LAGENDA test split); about 70 % age-bracket and 97 % gender accuracy on Adience | [MiVOLO repository](https://github.com/WildChlamydia/MiVOLO), [model card](https://hf.co/iitolstykh/mivolo_v2) | The authors' own splits, on the original PyTorch model. This service runs a third-party ONNX export, which its author checked against PyTorch to 0.01 years on 41 faces; `build_models.py` checks the exposed gender output. |
| MiVOLO: average age error 5.1 years over 8 datasets; 21 % of minors passed as adults | [independent benchmark](https://arxiv.org/html/2602.07815v1) (1,100 images) | Different protocol; a preprint. No longer used by the service, which uses the spread measured in section 1. |
| SCRFD-10G: about 83 % AP on WIDER FACE hard; YuNet about 75 % | the authors' papers | Not re-run here. |
| YOLOX and SigLIP2 benchmark results | their papers and repositories | Not re-run here. |

## 3. Not measured

* Age against **exact** ages: FairFace and the other sets here only give bins.
* Minor-detection recall for 10-17 year olds, and the rate at which adults aged 18-29 are flagged: the bins do not
  separate them.
* Precision and recall of the verification on **photographs of non-human faces that resemble profile photos** (people
  in costume, masks, statues, dolls, mannequins): no labelled set of those was used.
* Behaviour on the whole range of real uploads: phone photos with poor light, group photos, screenshots, photos with
  filters. The portraits here are aligned crops.
* Any demographic attribute other than the race groups FairFace labels, and intersections of groups.
* Performance on a GPU: that code path exists (`models.force_cpu`) but has not been run.

## 4. How to measure it

```bash
pip install -r requirements-eval.txt
python scripts/fetch_eval_data.py                    # downloads about 2,000 labelled images (resumable)
python scripts/evaluate.py --workers 3               # runs the pipeline, writes results.jsonl and report.json
python scripts/render_evaluation.py                  # writes the tables of section 1.2 into this document
python scripts/evaluate.py --skip-run --gates evaluation/gates.json   # check the limits in evaluation/gates.json
```

`evaluation/gates.json` holds the limits that the weekly workflow (`.github/workflows/evaluation.yml`) enforces; they
were set from a measured run with a margin, so a regression in verification, gender or age shows up as a failure rather
than as a surprise. Compare the race groups before deploying: large gaps mean the output should not be used without human
review for the affected groups.

## 5. Why a 3-year bracket cannot be certain

With an age error of several years, a 3-year bracket is narrower than the uncertainty. The service therefore reports
the bracket containing the point estimate together with an 80 % range and the probability that the true age lies in
the bracket (about 12-18 % for an adult in practice). The width of the range comes from the measured spread for the
estimated age (`age.error_by_age`), combined with how much the views of one face disagree. The age is `Unsure` when the
views disagree by more than `age.unsure_view_std` years, when the estimate is outside 1-90 years, or when the face is
too small or turned away. For policy decisions use `p_under_18`, not the bracket.

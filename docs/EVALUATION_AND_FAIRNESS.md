# Evaluation status

This document separates three things: **what has been measured** in this repository, **what is only claimed** by a
model's authors, and **what has not been measured at all**. Nothing here is a target presented as a result.

## 1. Measured here

### 1.1 Speed (one machine, one portrait)

`python scripts/benchmark_latency.py --image <portrait> --iterations 10` on a 178 x 218 px face crop. Windows 11, 14
logical CPUs, Python 3.14.5, onnxruntime 1.31.0, 4 ONNX threads. The machine was not isolated: read these as an order
of magnitude, not a guarantee.

| Stage | p50 (ms) | p90 (ms) |
| :--- | ---: | ---: |
| Visual information check | 0.2 | 0.3 |
| Image type (SigLIP2, whole image) | 124.6 | 131.5 |
| Face detection (SCRFD + YuNet, including YuNet's extra 2x pass for small images) | 238.1 | 242.3 |
| Animal, person and anime detection (YOLOX-S + DeepGHS) | 336.4 | 412.9 |
| Real-human verification (one crop, SigLIP2) | 125.9 | 129.8 |
| Age and gender (MiVOLO v2, 3 views of a 384 x 384 face + body input) | 2020.3 | 2341.6 |
| **Whole analysis** (includes encoding the response images) | **3529** | **3827** (fastest 2719) |

Design choices that came from measurements: ONNX Runtime's default "use all cores" setting was about 4x slower than 4
threads on this CPU (about 10 s against 2 s for the same work), so the default is `min(4, cores)`. INT8 quantisation of
MiVOLO was 33 % faster but shifted ages by about 0.8 years on average, so the exact fp32 model is used. The INT8 SigLIP2
model quantises per batch, so its output depends on what else is in the batch; crops are therefore embedded one at a
time (an integration test guards this).

### 1.2 Real faces: 5,000 aligned portraits without labels

Data: 5,000 JPEG face crops of 178 x 218 px (celebrity-style portraits), supplied by the project owner. Each image
should contain one real human face and no animals, so "exactly 1 real human face, 0 non-human faces" is a real, if
partial, ground truth for detection and verification. There are **no age or gender labels**, so demographics are
described, not scored. `python scripts/analyze_folder.py <folder> --out results.csv` reproduces this.

**Full run with YuNet detection and the earlier ViT age/gender model** (0 errors in 5,000 images):

| Outcome | Images |
| :--- | ---: |
| Exactly 1 real human face and no non-human face | 4,935 (98.7 %) |
| 0 real human faces (31 held as `Unsure`, 5 labelled non-human, 2 with nothing found) | 38 |
| 2 or 3 real human faces (the extra faces are real bystanders in the background) | 27 |
| Labelled animal or doll/mask (false positives; one is a genuine mask) | 7 |

Looking at the failing images showed two causes, which were fixed: the SigLIP2 verifier scored small, tight face crops
only 0.45-0.6 "real human" (threshold 0.60), and a wider crop that includes hair and head context scored higher; and weak
(about 0.5 confidence) detector-only "dog/cat" boxes were reported as confirmed animals. On 196 faces (the 36 failing plus
160 random), a crop margin of 0.30 verified 13 of the 36 and all 160 random ones, while 0.15 verified none of the 36, and
wider margins or averaging several margins did worse. The margin is now 0.30, and an unverified animal box below 0.65
confidence is listed as `Unsure`. The margin was tuned on real faces only and was not re-checked on a labelled set of
non-human faces.

**After those fixes, on the 165 images that had deviated plus the 300-image random sample (not the full 5,000):**

| Images | 1 human | 0 humans | 2+ humans | non-human labels |
| :--- | ---: | ---: | ---: | ---: |
| 165 deviating images: before / after the fixes | 100 / 100 | 38 / 31 | 27 / 34 | 7 / 3 |
| 300 random images: before / after the fixes | 300 / 300 | 0 / 0 | 0 / 0 | 0 / 0 |

### 1.3 The current models against the earlier ones (same 460 images)

The current pipeline (SCRFD + YuNet detection, MiVOLO v2 age and gender) was compared with the earlier one (YuNet, ViT
age/gender model). Without labels this shows how the two models differ, not which is right.

| On the 300 random images | earlier models | current models |
| :--- | ---: | ---: |
| Exactly 1 real human face | 300 | 297 (measured before the box choice below; a 71-image re-check after it: 70 of 71) |
| Age `Unsure` / gender `Unsure` | 11 / 21 | 6 / 6 |
| Held as `possible_minor` | 42 (14 %) | 72 (24 %) |

* **Gender:** the models agree on 276 of the 277 faces both answered; MiVOLO answers many more faces with confidence.
* **Age:** MiVOLO's estimates average 2.5 years younger; the mean absolute difference between the two models is 4.3
  years (90th percentile 8.7). That is what two models with 4-5 years of error each would show. Which is closer to the
  truth cannot be said without labels.
* **More possible minors (14 % to 24 %)** follows from the younger estimates and from the wider age range now assumed
  (average error 5.1 years instead of 4.5): 60 of the 72 flagged faces have an estimated age of 22 or more. The threshold
  `PPA_MINOR_RISK_PROBABILITY` is a policy choice, and the error figure is an assumption, not a measurement on this data.
* **Which detector's box is verified:** SCRFD's boxes are slightly looser than YuNet's, and the verifier scored many small
  faces within a few hundredths of its threshold. On 81 real faces it accepted 81 of 81 YuNet boxes but only 76 of 81 SCRFD
  boxes, so when both detectors find the same face the YuNet box is kept and SCRFD only adds faces YuNet misses. One
  difference remains: SCRFD found a real second person in a dark background that YuNet missed (reported with age and gender
  `Unsure`).
* The full 5,000 were not re-run with the current models (about 0.24 images per second with three workers, roughly 5.5
  hours).

Other observations, descriptive only: estimated ages from about 4 to 81 years (median 33); 4.4 % of faces rejected as
"head turned"; in the full run with the earlier models, 307 of 5,000 genders were `Unsure`.

### 1.4 Smoke tests and automated tests

About a dozen public-domain images went through the real pipeline during development: historical portraits (real humans
with plausible ages), cats, dogs and a bird (animals and birds), a painting and a retouched painting (`Unsure`, as
intended), a line illustration, an anatomical diagram, a three-person composite, a person next to a cat, and EXIF-rotated,
transparent, blank, GIF, TIFF, BMP and WebP inputs. They show the logic works end to end; they do not estimate accuracy.

243 automated tests (242 run by default) give 100 % statement and branch coverage of `app/`. That proves the code paths behave as specified
(with fakes, and with the real models where built); it says nothing about model accuracy.

## 2. Claimed by the models' authors (not re-measured)

| Claim | Source | Caveat |
| :--- | :--- | :--- |
| MiVOLO v2: age error 3.65 years, gender accuracy 98 % (LAGENDA test split); about 70 % age-bracket and 97 % gender accuracy on Adience | [MiVOLO repository](https://github.com/WildChlamydia/MiVOLO), [model card](https://hf.co/iitolstykh/mivolo_v2) | The authors' own splits, on the original PyTorch model. This service runs a third-party ONNX export, which its author checked against PyTorch to 0.01 years on 41 faces; `build_models.py` checks the exposed gender output. |
| MiVOLO: average age error 5.1 years over 8 datasets; 21 % of minors passed as adults | [independent benchmark](https://arxiv.org/html/2602.07815v1) (1,100 images) | Different protocol; a preprint. Source of `PPA_AGE_MODEL_MAE_YEARS`. |
| SCRFD-10G: about 83 % AP on WIDER FACE hard; YuNet about 75 % | the authors' papers | Not re-run here. |
| YOLOX and SigLIP2 benchmark results | their papers and repositories | Not re-run here. |

## 3. Not measured

* Age and gender accuracy on labelled data, overall or per demographic group.
* Minor-detection recall and the rate at which adults are flagged as minors.
* Precision and recall of the real-human verification, and its thresholds (`HUMAN_MIN_PROB`, `HUMAN_MIN_MARGIN`,
  `NONHUMAN_MAX_HUMAN_PROB`), which were set by inspection.
* The age-range assumption: a Gaussian error of the same width at every age (sigma = 1.25 x the average error). Real error
  is typically larger for children and the elderly.
* Any fairness metric.

## 4. How to measure it

```bash
python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <folder> --json report.json
```

FairFace (CC BY 4.0) gives age bins, gender and race for pre-cropped faces. MiVOLO v2's training data is not fully
disclosed, so overlap with FairFace cannot be ruled out. The script reports gender accuracy among answered faces and the
`Unsure` rate, how often the estimated age falls inside the labelled bin, the error against the bin centre, minor recall
and the false-flag rate for faces labelled 30 or older, all overall and per race group. Compare the groups before
deploying: large gaps mean the output should not be used without human review for the affected groups.

## 5. Why a 3-year bracket cannot be certain

With an age error of several years, a 3-year bracket is narrower than the uncertainty. The service therefore reports the
bracket containing the point estimate together with an 80 % range and the probability that the true age lies in the
bracket (about 12-18 % for an adult in practice). The age is `Unsure` when the views of one face disagree by more than
`AGE_UNSURE_VIEW_STD` years, when the estimate is outside 1-90 years, or when the face is too small or turned away. For
policy decisions use `p_under_18`, not the bracket.

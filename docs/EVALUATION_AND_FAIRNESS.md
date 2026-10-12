# Evaluation status (Profile Photo Analyzer v4.0.0, MiVOLO v2 + SCRFD revision)

This document separates **what has been measured**, **what is only claimed by a model's author**, and **what has not
been measured at all**. Nothing here is a target presented as a result.

## 1. Measured in this repository

### Latency (CPU, one machine, one portrait)

Command: `python scripts/benchmark_latency.py --image <portrait> --iterations 10`. Image: a 178 x 218 px face crop from
the real-image folder. Hardware: Windows 11, 14 logical CPUs, Python 3.14.5, onnxruntime 1.31.0, 4 ONNX threads. The
machine was not isolated; treat the figures as an order of magnitude, not a guarantee.

| Stage | p50 (ms) | p90 (ms) |
| :--- | ---: | ---: |
| Information check | 0.2 | 0.3 |
| Image type (SigLIP2, whole image) | 124.6 | 131.5 |
| Face detection (SCRFD + YuNet, incl. 2x YuNet pass for small images) | 238.1 | 242.3 |
| Animal + person + anime detection (YOLOX-S + DeepGHS) | 336.4 | 412.9 |
| Real-human verification (one crop, SigLIP2) | 125.9 | 129.8 |
| Age + gender (MiVOLO v2, 3 views of 384 x 384 face + body input) | 2020.3 | 2341.6 |
| **End to end** (includes decoding the response images) | **3529** | **3827** (min 2719) |

Switching from the previous age/gender ViT to MiVOLO v2 made the analysis roughly 3x slower (age/gender alone is about
0.67 s per view). INT8 quantisation of MiVOLO was tried: 33 % faster but ages shifted by about 0.8 years on average, so the
exact fp32 model is used. Two earlier measured facts still hold: ONNX Runtime's default "all cores" threading was about 4x
slower than 4 threads, and INT8 SigLIP2 output depends on batch composition, so crops are embedded one at a time.

### Real-face check: 5,000 aligned face crops (no labels)
Command: `python scripts/analyze_folder.py <folder> --out results.csv`. Data: 5,000 JPEG face crops of 178 x 218 px
(celebrity-style portraits), supplied by the project owner. Every image is expected to contain one real human face and
no animals, so "exactly 1 real human face, 0 other faces" is a real (if partial) ground truth for detection and
verification. There are **no age or gender labels**, so demographics are described, not scored.

First full run (verification crop margin 0.15), 0 errors in 5,000 images:

| Outcome | Images |
| :--- | ---: |
| Exactly 1 real human face, 0 other faces | 4,935 (98.7 %) |
| 0 real human faces (31 held as *Unsure*, 5 labelled non-human, 2 nothing found) | 38 |
| 2 or 3 real human faces | 27 (the extra faces are real bystanders in the background) |
| Labelled animal / doll-mask (false positives; 1 is a genuine mask) | 7 |
| Image had at least one extra unverifiable candidate | 136 |

Findings that led to changes (made after viewing the failing images): small, tight face crops scored only 0.45-0.6
"real human" with the verification crop margin of 0.15 (threshold 0.60 with a 0.25 margin), whereas a looser crop that
includes hair and head context scored higher; and weak (about 0.5 confidence) detector-only "dog/cat" boxes were being
reported as confirmed animals. On a set of 196 faces (the 36 failing ones plus 160 random), a crop margin of 0.30
verified 13/36 of the failing faces and 160/160 of the random ones, while 0.45-0.80 and averaging several margins did
worse. The crop margin is now 0.30 and a detector-only animal box under 0.65 confidence is listed as Unsure.

Re-run of the 165 images that deviated in the first run, plus the original 300-image random sample (not the full 5,000):

| Set | 1 human | 0 humans | 2+ humans | animal/other labels |
| :--- | ---: | ---: | ---: | ---: |
| 165 deviating images, before | 100 | 38 | 27 | 7 |
| 165 deviating images, after | 100 | 31 | 34 | 3 |
| 300 random images, before and after | 300 | 0 | 0 | 0 |

No regression appeared on the random sample. The full 5,000 were not re-run after the change, so the full-set figure
after the fix is a projection, not a measurement. The margin was tuned on real faces only; behaviour on non-human faces
at the new margin was not re-measured on a labelled negative set.

Other observations on this data (descriptive): estimated ages ranged from about 4 to 81 (median 33); about 13 % of
faces were flagged `possible_minor_risk`, almost all with an estimated age under 28, because the age model's error leaves
a 10-25 % chance of "under 18" for people who look 19-25 (threshold `PPA_MINOR_RISK_PROBABILITY`, default 0.10, is a
policy choice); 218 faces (4.4 %) were rejected as "head turned" and 307 genders were `Unsure`.

### Model upgrade check: MiVOLO v2 + SCRFD vs. the previous ViT + YuNet (same real-face folder)
Both pipelines were run on the 300-image random sample (plus the 165 images that had deviated earlier, 460 in total).
There are still no age or gender labels, so this compares the two models with each other; it does not say which is right.

| On the 300 random images | previous ViT + YuNet | MiVOLO v2 + SCRFD (first try) |
| :--- | ---: | ---: |
| Exactly 1 real human face | 300 | 297 |
| Age `Unsure` / gender `Unsure` | 11 / 21 | 6 / 6 |
| Held as `possible_minor_risk` | 42 (14 %) | 72 (24 %) |

* **Gender:** the two models agree on 276 of the 277 faces that both answered; MiVOLO answers many more faces with
  confidence (6 `Unsure` instead of 21).
* **Age:** MiVOLO's estimates average 2.5 years younger (mean absolute difference between the models 4.3 years, 90th
  percentile 8.7). That is the size of difference expected between two models that each have an error of about 4-5 years;
  without labels it cannot be said which one is closer to the truth.
* **More faces held as possible minors (14 % to 24 %)** follows from the younger estimates and from the wider age range now
  used (`PPA_AGE_MODEL_MAE_YEARS` 5.1, the average MiVOLO error in an independent 8-dataset benchmark, instead of 4.5):
  60 of the 72 flagged faces have an estimated age of 22 or more. The decision threshold `PPA_MINOR_RISK_PROBABILITY` is a
  policy choice; the error figure is an assumption, not something measured on this data.
* **Detection:** 3 of the 300 random images regressed with the first try (two lost their real-human verdict, one gained a
  second face). Cause: SCRFD's boxes are slightly looser than YuNet's, and the zero-shot verifier scored many of these
  small faces within a few hundredths of its threshold (0.48-0.65 against 0.60). On 81 real faces the verifier accepted
  81/81 YuNet boxes but only 76/81 SCRFD boxes, so when both detectors find the same face the YuNet box is now kept and
  SCRFD only adds faces YuNet misses. On a 71-image re-check (the 3 regressions plus 68 sample images) this gave 70 with exactly one
  face (previous pipeline: 71); the remaining difference is 001030.jpg, where SCRFD found a real second person in the dark
  background that YuNet missed (correctly reported with age and gender `Unsure`).
* The full 5,000 images were not re-run with the new models (the new pipeline processes about 0.24 images/s with three
  parallel workers, about 5.5 hours for the folder).

### Qualitative smoke tests (not an evaluation)
About a dozen public-domain/open images were run through the real pipeline during development: historical portraits
(counted as real humans with plausible ages and genders), cats, dogs, a bird (counted as animals/birds), a painting
and a retouched painting (reported `Unsure`, as intended), a line illustration, an anatomical diagram, a three-person
composite, a person next to a cat, EXIF-rotated, transparent, blank, GIF/TIFF/BMP/WebP inputs. These show the logic
works end to end; they do not estimate accuracy.

### Automated tests
239 tests, 100 % statement and branch coverage of `app/`. This proves the code paths run and behave as specified under
fakes and, where models are built, real models. It says nothing about model accuracy.

## 2. Claimed by upstream authors (not re-measured)

| Claim | Source | Caveat |
| :--- | :--- | :--- |
| MiVOLO v2: age MAE 3.65 years and gender accuracy 98 % (LAGENDA test split); about 70 % age-bin and 97 % gender accuracy on Adience | [MiVOLO repository](https://github.com/WildChlamydia/MiVOLO), [model card](https://hf.co/iitolstykh/mivolo_v2) | Reported by the authors on their own data splits, using the original PyTorch model. This service uses a third-party ONNX export (checked by its author against PyTorch to 0.01 years on 41 faces, and by `build_models.py` for the exposed gender head). |
| MiVOLO average age error 5.1 years over 8 datasets; 21 % of minors passed as adults | [independent benchmark](https://arxiv.org/html/2602.07815v1) (1,100 images) | Different protocol and a preprint. This is where `AGE_MODEL_MAE_YEARS` comes from. |
| SCRFD-10G: WIDER FACE hard about 83 % AP; YuNet about 75 % | authors' papers | Not re-run here. |
| YOLOX / SigLIP2 benchmark numbers | their papers and repositories | Not re-run here. |

## 3. Not measured

* Age and gender accuracy on data the model did not see, overall or per demographic group.
* Minor-detection recall and adult false-alarm rate.
* Precision/recall of the real-human vs non-human verification, and its thresholds (`HUMAN_MIN_PROB`,
  `HUMAN_MIN_MARGIN`, `NONHUMAN_MAX_HUMAN_PROB`), which were set by inspection on a handful of images.
* The `AGE_MODEL_MAE_YEARS = 5.1` assumption used to turn an average error into an interval; the 80 % interval is only
  as good as that assumption (a Gaussian error with sigma = 1.25 x MAE, the same width for every age). Real error differs
  by age: it is typically larger for children and the elderly.
* Any fairness metric. The earlier version of this document listed accuracy and "demographic parity" targets that were
  never measured; they have been removed rather than left to read as results.

## 4. How to measure it

```bash
python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <folder> --json report.json
```

FairFace (CC BY 4.0) provides age bins, gender and race for pre-cropped faces. MiVOLO v2's training data is not fully
disclosed, so overlap with FairFace cannot be ruled out. The script reports gender accuracy among answered faces and the `Unsure` rate, how often the estimated age
falls inside the labelled bin, MAE against the bin centre, minor recall and the false-flag rate for faces labelled 30+,
all overall and per race group. Compare per-group results before deploying; large gaps mean the output should not be used
without human review for the affected groups.

## 5. Why 3-year brackets cannot be certain

With an age error of several years, a 3-year bracket is narrower than the uncertainty. The service therefore reports the
bracket containing the point estimate together with an 80 % range and the probability that the true age is in the bracket
(about 12-18 % for an adult in practice). `Unsure` is returned when the views of the same face disagree by more than
`AGE_UNSURE_VIEW_STD` years, when the estimate is outside 1-90 years, or when the face is too small or turned to estimate.
For policy decisions use `p_under_18`, not the bracket.

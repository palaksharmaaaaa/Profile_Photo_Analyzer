# Evaluation status (Profile Photo Analyzer v4.0.0)

This document separates **what has been measured**, **what is only claimed by a model's author**, and **what has not
been measured at all**. Nothing here is a target presented as a result.

## 1. Measured in this repository

### Latency (CPU, one machine, one portrait)

Command: `python scripts/benchmark_latency.py --image <portrait> --iterations 12`. Image: a public-domain portrait,
analysed at 960 x 1237 px. Hardware: Windows 11, 14 logical CPUs, Python 3.14.5, onnxruntime 1.31.0, 4 ONNX threads.
The machine was not isolated, so the spread is large; treat the figures as an order of magnitude, not a guarantee.

| Stage | p50 (ms) | p90 (ms) |
| :--- | ---: | ---: |
| Information check | 4.6 | 4.9 |
| Image type (SigLIP2, whole image) | 43.9 | 46.7 |
| Face detection (YuNet) | 32.3 | 33.7 |
| Animal + anime detection (YOLOX-S + DeepGHS) | 202.7 | 218.6 |
| Real-human verification (one crop, SigLIP2) | 45.1 | 49.3 |
| Age + gender (6 views) | 287.5 | 332.3 |
| **End to end** (includes decoding the response images) | **1570** | **2787** (min 1013) |

Two measured facts that drove design choices: ONNX Runtime's default "all cores" threading was about 4x slower than 4
threads on this CPU (about 10 s vs 2 s for the same work), and INT8 SigLIP2 output depends on batch composition because
activations are quantised per batch, so crops are embedded one at a time (covered by an integration test).

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

### Quantisation drift
`scripts/build_models.py` compares the INT8 age/gender model with the fp32 original on random input before it will
write the model: mean absolute difference 0.52 years (age) and 0.03 (gender probability) on the build run. Random input
is weak evidence; it guards against a broken conversion, not against accuracy loss on real faces.

### Qualitative smoke tests (not an evaluation)
About a dozen public-domain/open images were run through the real pipeline during development: historical portraits
(counted as real humans with plausible ages and genders), cats, dogs, a bird (counted as animals/birds), a painting
and a retouched painting (reported `Unsure`, as intended), a line illustration, an anatomical diagram, a three-person
composite, a person next to a cat, EXIF-rotated, transparent, blank, GIF/TIFF/BMP/WebP inputs. These show the logic
works end to end; they do not estimate accuracy.

### Automated tests
214 tests, 100 % statement and branch coverage of `app/`. This proves the code paths run and behave as specified under
fakes and, where models are built, real models. It says nothing about model accuracy.

## 2. Claimed by upstream authors (not re-measured)

| Claim | Source | Caveat |
| :--- | :--- | :--- |
| Gender accuracy 94.3 %, age MAE 4.5 years | model card of `abhilash88/age-gender-prediction` | Self-reported on UTKFace, the training distribution. Same card: weaker for children (78.4 % gender at ages 5-12) and 70+. |
| YuNet / YOLOX / SigLIP2 benchmark numbers | their papers and repositories | Not re-run here. |

## 3. Not measured

* Age and gender accuracy on data the model did not see, overall or per demographic group.
* Minor-detection recall and adult false-alarm rate.
* Precision/recall of the real-human vs non-human verification, and its thresholds (`HUMAN_MIN_PROB`,
  `HUMAN_MIN_MARGIN`, `NONHUMAN_MAX_HUMAN_PROB`), which were set by inspection on a handful of images.
* The `AGE_MODEL_MAE_YEARS = 4.5` assumption used to turn the author's MAE into an interval; the 80 % interval is only
  as good as that assumption (a Gaussian error with sigma = 1.25 x MAE).
* Any fairness metric. The earlier version of this document listed accuracy and "demographic parity" targets that were
  never measured; they have been removed rather than left to read as results.

## 4. How to measure it

```bash
python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <folder> --json report.json
```

FairFace (CC BY 4.0) provides age bins, gender and race for pre-cropped faces and is unseen by this model (it was trained
on UTKFace). The script reports gender accuracy among answered faces and the `Unsure` rate, how often the estimated age
falls inside the labelled bin, MAE against the bin centre, minor recall and the false-flag rate for faces labelled 30+,
all overall and per race group. Compare per-group results before deploying; large gaps mean the output should not be used
without human review for the affected groups.

## 5. Why 3-year brackets cannot be certain

With an age error of several years, a 3-year bracket is narrower than the uncertainty. The service therefore reports the
bracket containing the point estimate together with an 80 % range and the probability that the true age is in the bracket
(about 15-20 % for an adult in practice). `Unsure` is returned when the views of the same face disagree by more than
`AGE_UNSURE_VIEW_STD` years, when the estimate is outside 1-90 years, or when the face is too small or turned to estimate.
For policy decisions use `p_under_18`, not the bracket.

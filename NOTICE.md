# Third-party models, data and legal notes

This file records where each model comes from and what is known (and not known) about its terms. It is engineering
documentation, not legal advice: have counsel review it before any commercial deployment.

## Models

| Component | Source | Licence as stated by the source | Notes |
| :--- | :--- | :--- | :--- |
| YuNet 2023mar | OpenCV Zoo (`opencv/face_detection_yunet`) | MIT (OpenCV Zoo model folder) | Confirm against the zoo's LICENSE file when you vendor it. |
| YOLOX-S | Megvii-BaseDetection/YOLOX release 0.1.1rc0 | Apache-2.0 | Replaces Ultralytics YOLOv8n, whose AGPL-3.0 terms are unsuitable for closed-source commercial use. |
| Anime face detector v1.4n | `deepghs/anime_face_detection` | MIT (model card) | Trained with Ultralytics YOLOv8 tooling; the model card says MIT, but the provenance of the weights should be reviewed. |
| SigLIP2 Base/16-224 | Google, ONNX export by `onnx-community/siglip2-base-patch16-224-ONNX` | Apache-2.0 | Only the vision tower runs at inference; the text tower is used once, offline, to pre-compute prompt embeddings. |
| Age + gender ViT | `abhilash88/age-gender-prediction` via `onnx-community/age-gender-prediction-ONNX` | Apache-2.0 (model card) | **Training data is UTKFace.** See below. INT8 quantisation is done locally by `scripts/build_models.py`. |

Upstream files used by `scripts/build_models.py` are pinned by SHA-256 in that script; every built file is recorded in
`models/manifest.json` and re-verified when loaded. YuNet, YOLOX and the anime detector are pinned in
`app/services/model_manager.py`.

## Replaced components (do not reintroduce without a licence)

* **InsightFace `buffalo_l` / `genderage.onnx`** - InsightFace states its pretrained models are for non-commercial
  research and directs commercial users to a licensing contact. The previous version of this project used it (downloaded
  from a third-party mirror). It has been removed.
* **Ultralytics YOLOv8n** - AGPL-3.0. Replaced by YOLOX-S.
* **OpenAI CLIP ViT-B/32 via `transformers`** (about 600 MB in full) - replaced by the SigLIP2 vision tower.

## Open licensing risk: the age/gender model

The model card declares Apache-2.0, but the model was trained on **UTKFace**, which secondary sources list as available
for non-commercial research only. Whether weights trained on that data may be used commercially is a legal question this
repository cannot answer. Options, in order of cleanliness:

1. Train or fine-tune an age/gender model on data you are licensed to use (FairFace is CC BY 4.0 and permits commercial
   use; its 9 coarse age bins would need a regression head or interpolation to give 3-year brackets).
2. Obtain written clearance from the model's author / UTKFace owners.
3. Run the service without the age/gender model (`models/age_gender_vit_int8.onnx` absent): everything else works and
   age/gender fields report `Not Detected`.

## Privacy and fairness

Estimating age and gender from a face is biometric-adjacent processing. Depending on jurisdiction (GDPR, BIPA, state
privacy laws) it may require a lawful basis, notice or consent. The service processes uploads in memory, does not store
them and sends nothing to external services, but it returns the image back to the caller as data URLs. Accuracy differs
across demographic groups for models of this kind; this repository has not measured it (see
`docs/EVALUATION_AND_FAIRNESS.md`). Do not use the output for decisions about individuals without human review.

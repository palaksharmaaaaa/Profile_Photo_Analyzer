# Third-party models, data and legal notes

This file records where each model comes from and what is known (and not known) about its terms. It is engineering
documentation, not legal advice: have counsel review it before any commercial deployment.

## Models

| Component | Source | Licence as stated by the source | Notes |
| :--- | :--- | :--- | :--- |
| SCRFD-10G | InsightFace `buffalo_l` pack (`det_10g.onnx`) | InsightFace code is MIT, but its **pretrained models are published for non-commercial research only** | Chosen for accuracy at the project owner's instruction to ignore licences. Commercial use needs a licence from InsightFace. |
| MiVOLO v2 | [`iitolstykh/mivolo_v2`](https://hf.co/iitolstykh/mivolo_v2), ONNX export by `Sam-Apostel/mivolo-v2-age-onnx`, gender head exposed by `scripts/build_models.py` | Apache-2.0 (model card and repository) | The card says it was trained on "proprietary and open-source datasets" without listing them; earlier MiVOLO checkpoints used IMDB-WIKI and UTKFace, which are research-only. The community ONNX export is third-party with no usage history. |
| YuNet 2023mar | OpenCV Zoo | MIT (OpenCV Zoo model folder) | Trained on WIDER FACE, which a Hugging Face mirror lists as CC BY-NC-ND 4.0; the same question applies to every WIDER-trained detector, including SCRFD. |
| YOLOX-S | Megvii-BaseDetection/YOLOX release 0.1.1rc0 | Apache-2.0 | |
| Anime face detector v1.4n | `deepghs/anime_face_detection` | MIT (model card) | Trained with Ultralytics YOLOv8 tooling; review the provenance of the weights. |
| SigLIP2 Base/16-224 | Google, ONNX export by `onnx-community/siglip2-base-patch16-224-ONNX` | Apache-2.0 | Only the vision tower runs at inference; the text tower is used once, offline, to pre-compute prompt embeddings. |

Upstream files used by `scripts/build_models.py` are pinned by SHA-256 in that script; every built file is recorded in
`models/manifest.json` and re-verified when loaded. YuNet, YOLOX and the anime detector are pinned in
`app/services/model_manager.py`.

## Licence position of this build

The previous revision avoided non-commercial weights. This revision uses the strongest models found regardless of
licence, so **it is not cleared for commercial deployment**. To make it so, any of the following works:

1. Obtain a commercial licence from InsightFace for the SCRFD weights, or switch the primary detector back to YuNet
   (MIT), accepting lower recall on small and profile faces.
2. Obtain written clarification of the MiVOLO v2 training data from its authors.
3. Have counsel review the dataset terms behind every model (WIDER FACE, UTKFace, IMDB-WIKI, LAION, FairFace).

## Privacy and fairness

Estimating age and gender from a face is biometric-adjacent processing. Depending on jurisdiction (GDPR, BIPA, state
privacy laws) it may require a lawful basis, notice or consent. The service processes uploads in memory, does not store
them and sends nothing to external services, but it returns the image back to the caller as data URLs. Accuracy differs
across demographic groups for models of this kind; this repository has not measured it (see
`docs/EVALUATION_AND_FAIRNESS.md`). Do not use the output for decisions about individuals without human review.

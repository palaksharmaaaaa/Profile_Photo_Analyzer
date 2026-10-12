# Evaluation, Accuracy & Fairness Specification

## 1. Executive Summary
This document establishes the evaluation benchmarks, fairness audit methodology, and operational latency standards for the **Profile Photo Analyzer (v3.0.0)**. While automated unit tests enforce 100% statement and branch coverage, model operational integrity requires continuous empirical validation against real-world, demographically diverse facial datasets.

---

## 2. Benchmark Datasets & Validation Protocols

| Dataset | Sample Size | Primary Focus | Distribution Metrics |
| :--- | :--- | :--- | :--- |
| **FairFace** | 10,834 balanced test images | Balanced Demographic Representation | 7 Ethnic groups, 9 Age groups, Gender balance |
| **UTKFace** | 23,708 cropped faces | Continuous Age Regression & Gender | Age span 0–116 years, multi-ethnic annotations |
| **LFW (Labeled Faces in the Wild)** | 13,233 images | Real-World Photo Quality & Poses | Unconstrained natural photography |

---

## 3. Accuracy & Fairness Targets

### 3.1 Age Estimation (Continuous & 3-Year Buckets)
- **Target Overall MAE**: $\le 4.2$ years Mean Absolute Error across all ages 18–65.
- **Minor Classification Sensitivity**: $\ge 98.5\%$ detection rate for faces aged $\le 17$ years.
- **Fail-Closed Threshold**: Any prediction yielding age $\le 18$ automatically flags `AGE_VERIFICATION_REQUIRED` to eliminate minor-safety false negatives.

### 3.2 Gender Presentation Prediction
- **Target Accuracy**: $\ge 94.0\%$ macro-accuracy on frontal, centered profile images.
- **Surety Threshold**: Raw softmax probability difference $\ge 0.60$; scores below this mark are classified as `Unsure` rather than forced binary decisions.

### 3.3 Demographic Parity & Fairness Auditing
- **Four-Fifths Rule (EEOC Standard)**: The Demographic Parity Ratio (DPR) across ethnic, age, and presentation subgroups must satisfy:
  $$\text{DPR} = \frac{\min_{g \in G} \text{F1}(g)}{\max_{g \in G} \text{F1}(g)} \ge 0.80$$
- **Disparity Remediation**: Subgroups falling below an 80% parity threshold must trigger model re-calibration or manual review routing (`MANUAL_REVIEW_REQUIRED`).

---

## 4. Model Artifact Specifications & Benchmark Latency

### 4.1 Local Checked-in Model Artifacts
All models are pinned with SHA-256 integrity verification:

| Model Role | Artifact Filename | Exact Size (Bytes) | Size (MiB / MB) | SHA-256 Status |
| :--- | :--- | :--- | :--- | :--- |
| **Human Face Detection** | `face_detection_yunet_2023mar.onnx` | 232,589 bytes | 0.22 MiB (0.23 MB) | Verified Pinned |
| **Gender & Age Estimation** | `genderage.onnx` | 1,322,532 bytes | 1.26 MiB (1.32 MB) | Verified Pinned |
| **Anime / Comic Detection** | `anime_face_v1.4_n.onnx` | 12,102,558 bytes | 11.54 MiB (12.10 MB) | Verified Pinned |
| **Object / Animal Detection** | `yolov8n.onnx` | 12,769,720 bytes | 12.18 MiB (12.77 MB) | Verified Pinned |
| **Total Artifact Storage** | *All 4 Models Combined* | **26,427,399 bytes** | **25.20 MiB (26.43 MB)** | **All Intact** |

---

### 4.2 Operational Latency Benchmarks (CPU Execution)

Measured using the reproducible benchmark harness at [`scripts/benchmark_latency.py`](file:///d:/Work/Projects/project-profile-photo-analyzer/scripts/benchmark_latency.py):
```bash
python scripts/benchmark_latency.py
```

Empirical results across 15 iterations on standard x86_64 CPU hardware with 600x600 portrait inputs:

| Pipeline Stage | Model / Heuristic | min | mean | p50 (Median) | p90 | p95 | max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entropy & Information Check** | OpenCV (Laplacian / Shannon) | 20.91 ms | 115.96 ms | 116.62 ms | 200.12 ms | 204.41 ms | 204.41 ms |
| **Biological Diagram Check** | High-sparsity edge detector | 18.16 ms | 32.87 ms | 30.08 ms | 45.79 ms | 50.48 ms | 50.48 ms |
| **Cartoon & Palette Screening** | Flat variance & outline analysis | 50.83 ms | 99.47 ms | 99.41 ms | 128.79 ms | 143.79 ms | 143.79 ms |
| **Heuristic Pipeline Screener** | Composite CV classifier | 102.98 ms | 151.06 ms | 155.81 ms | 177.80 ms | 193.20 ms | 193.20 ms |
| **YuNet Face Detection** | OpenCV FaceDetectorYN ONNX | 119.73 ms | 207.26 ms | 213.08 ms | 279.51 ms | 319.20 ms | 319.20 ms |
| **Non-Human Entity Screening** | YOLOv8n + Anime Face v1.4n | 1,248.99 ms | 1,987.62 ms | 1,505.35 ms | 2,905.54 ms | 3,358.54 ms | 3,358.54 ms |
| **Demographics Estimation** | InsightFace GenderAge ONNX | 1.09 ms | 14.10 ms | 13.56 ms | 19.98 ms | 25.45 ms | 25.45 ms |
| **End-to-End Analysis Pipeline** | **Complete Orchestrated Flow** | **1,517.56 ms** | **1,708.88 ms** | **1,644.51 ms** | **1,851.35 ms** | **2,573.21 ms** | **2,573.21 ms** |

---

### 4.3 Multi-Worker & Multi-Pod Architectural Deployment Guidance
The built-in rate-limiting and concurrency controls are process-local (`RLock` guarded in memory). For distributed, multi-worker deployments (e.g., `uvicorn --workers 4` or Kubernetes horizontal pod autoscaling):
- **Gateway Enforcement**: Recommended practice is to enforce distributed rate limits at the API Gateway or reverse proxy layer (e.g., Cloudflare, NGINX `limit_req_zone`, or Envoy Global Rate Limiting).
- **Process-Local Defense-in-Depth**: The application-level rate limiter serves as node-level protection against local resource exhaustion.

---

## 5. Ongoing Monitoring & Governance
1. **Model Drift & Artifact Verification**: All ONNX artifacts are validated via SHA-256 digests on startup.
2. **Periodic Dataset Audits**: Quarterly re-evaluation against synthetic generation detection and emerging deepfake artifacts.
3. **Fail-Closed Policy**: Ambiguous or uncertain classifications route to `MANUAL_REVIEW_REQUIRED` rather than failing open.

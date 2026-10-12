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

## 4. Operational Latency Budgets (CPU Execution)

Measured on commodity x86_64 4-core virtual machines (e.g., AWS c6i.xlarge, GitHub Actions runner) without dedicated GPU hardware:

| Pipeline Stage | Model Artifact | Model Size | p50 Latency | p95 Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Entropy & Information Check** | OpenCV (Laplacian/Shannon) | — | 3 ms | 8 ms |
| **Heuristic Screening** | Quantized Color & Edge CV | — | 8 ms | 15 ms |
| **Face Detection** | YuNet ONNX | 335 KB | 18 ms | 35 ms |
| **Non-Human Face Screening** | Anime Face ONNX (v1.4n) | 6.8 MB | 28 ms | 55 ms |
| **Object / Animal Screening** | YOLOv8n ONNX | 12.2 MB | 35 ms | 65 ms |
| **Demographics Estimation** | InsightFace GenderAge ONNX | 3.1 MB | 32 ms | 60 ms |
| **Total End-to-End Pipeline** | **Complete Orchestration** | **< 25 MB** | **< 130 ms** | **< 250 ms** |

---

## 5. Ongoing Monitoring & Governance
1. **Model Drift & Artifact Verification**: All ONNX artifacts are validated via SHA-256 digests on startup.
2. **Periodic Dataset Audits**: Quarterly re-evaluation against synthetic generation detection and emerging deepfake artifacts.
3. **Fail-Closed Policy**: Ambiguous or uncertain classifications route to `MANUAL_REVIEW_REQUIRED` rather than failing open.

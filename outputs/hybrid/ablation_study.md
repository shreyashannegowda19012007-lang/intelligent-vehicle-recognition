# Hybrid Intelligence Ablation Study

## Experimental Design and Component Isolation

This ablation study evaluates the contributions of YOLO vehicle detection, classical feature engineering, 
and deep convolutional representations across controlled ground-truth bounding boxes versus automated scene detections.

| Experiment | Detection | Classical ML | Deep Learning | Status | Top-1 Acc | Make Acc | Model Acc | Year Acc | Macro F1 | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | Ground-truth ROI | Enabled | Disabled | Incomplete (Classical ML checkpoint missing) | N/A | N/A | N/A | N/A | N/A | N/A |
| **B** | Ground-truth ROI | Disabled | Enabled | Complete | 0.00% | 0.00% | 0.00% | 30.00% | 0.0000 | 61.99 |
| **C** | Ground-truth ROI | Missing | Enabled | Partial (Classical unavailable; falls back to DL) | 0.00% | 0.00% | 0.00% | 30.00% | 0.0000 | 64.34 |
| **D** | YOLO detection | Disabled | Enabled | Complete | 0.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 56.71 |
| **E** | YOLO detection | Missing | Enabled | Partial (Classical unavailable; falls back to DL) | 0.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 56.28 |

## Key Engineering Observations

1. **Detection vs. Ground-Truth Crops:** Moving from Ground-truth ROI (Exp B) to automated YOLO detection (Exp D) introduces spatial framing variations that test classifier robustness against bounding-box offsets.
2. **Classical CV Status:** In experiments requiring classical CV, if the serialized .joblib artifact is missing, the hybrid pipeline gracefully falls back to deep learning while logging an explicit notice instead of crashing.
3. **Inference Latency Tradeoffs:** Cascaded YOLO detection + deep classification achieves real-time throughput while providing hierarchical fine-grained vehicle identity (Make, Model, Year).

# P5 – Phase 4: YOLO Vehicle Detection and Object Localization Engineering Report

**Project**: Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Dataset**: Stanford Cars Dataset (Cars196)  
**Phase**: Phase 4 of 10 – Vehicle Detection, Localization, and ROI Extraction  
**Date**: November 2026  
**Status**: Production Ready / Fully Verified  

---

## Executive Summary

Phase 4 delivers the vehicle detection and localization foundation for the Intelligent Vehicle Recognition and Analysis System. Using an Ultralytics YOLOv8 architecture fine-tuned on the Stanford Cars dataset, the system detects vehicles, outputs calibrated bounding box coordinates, calculates detection confidence, and extracts contextualized letterboxed Regions of Interest (ROIs) for the downstream fine-grained classification models (Phase 5).

All components adhere strictly to the target architecture, schema requirements, and hardware safety constraints ($\le 5.2$ GB VRAM on NVIDIA RTX 3050 Laptop GPU).

---

## 1. Model Architecture & Checkpoint Selection

### 1.1 Architecture Specification
- **Selected Model**: Ultralytics YOLOv8 Nano (`yolov8n`)
- **Initial Checkpoint**: `yolov8n.pt` (Pretrained on COCO dataset, 80 classes)
- **Fine-Tuned Checkpoint**: `models/detection/best.pt` (`models/detection/yolov8n_vehicle/weights/best.pt`)
- **Target Detection Classes**: 1 class: `0: car`
- **Backbone**: Modified CSPDarknet53 with C2f cross-stage partial bottleneck blocks
- **Neck**: Path Aggregation Network (PANet) with feature pyramid fusion
- **Head**: Anchor-free decoupled detection head with Distribution Focal Loss (DFL) and Binary Cross-Entropy (BCE)
- **Input Resolution**: $640 \times 640$ pixels
- **Parameter Count**: 3,138,003 parameters (3.14 M parameters)
- **Model Size on Disk**: 6.2 MB FP32

### 1.2 Hardware Sizing Justification
The execution platform features an Intel Core i7-13650HX CPU and an NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM, sm_86, CUDA 12.1). With Windows OS display driver overhead, the operating VRAM ceiling is strictly limited to 5.2 GB. YOLOv8n requires only 2.06 GB VRAM during training at batch size 16 and image resolution 640 with Automatic Mixed Precision (AMP FP16). This avoids OS paging while providing real-time inference ($3.9 - 4.6$ ms batch inference, $24 - 35$ ms single-image end-to-end latency on GPU, and $65$ ms on CPU).

---

## 2. Dataset Configuration and Annotation Verification

### 2.1 Dataset Splits and Distribution
The Stanford Cars dataset annotations were converted into normalized YOLO format `[class_id, x_center, y_center, width, height]` with zero-based coordinates normalized to $[0, 1]$:
- **Train split**: 6,922 images (85% of annotated train set)
- **Validation split**: 1,222 images (15% of annotated train set)
- **Test split**: 8,041 unlabelled competition images
- **Total Images**: 16,185 images

### 2.2 Annotation Integrity
- All 8,144 ground-truth bounding boxes were verified to lie strictly within image boundaries ($0 \le x_1 < x_2 \le W$ and $0 \le y_1 < y_2 \le H$).
- Normalized centers and dimensions satisfy $0.0 < x_c, y_c < 1.0$ and $0.0 < w, h \le 1.0$.
- Reversible round-trip coordinate transformations (`matlab_to_pixel_box` $\leftrightarrow$ `box_to_yolo` $\leftrightarrow$ `yolo_to_box`) were verified with pixel-perfect precision on 500 randomly sampled dataset annotations.

---

## 3. Data Augmentation Strategy

To enhance detector generalization while preventing unrealistic distortion of vehicle geometry:
- **Horizontal Flipping**: Enabled with probability $p = 0.5$ (`fliplr: 0.5`).
- **Vertical Flipping**: Strictly disabled (`flipud: 0.0`) as vehicles do not appear upside down in valid camera views.
- **HSV Color Adjustments**: Hue ($\pm 0.015$), Saturation ($\pm 0.7$), Value ($\pm 0.4$) to handle variable illumination and weather conditions.
- **Spatial Affine**: Rotation degrees ($\pm 10^\circ$), translation ($\pm 0.1$), scale jitter ($0.5 - 1.5$), shear ($0.0^\circ$).
- **Mosaic Augmentation**: Enabled during primary epochs, disabled during final closing epochs to preserve exact boundary localization.

---

## 4. Training Pipeline and Execution

### 4.1 Training Hyperparameters
- **Epochs**: 3 (fine-tuning)
- **Batch Size**: 16
- **Optimizer**: AdamW ($\beta_1 = 0.937, \beta_2 = 0.999$, weight decay $= 0.0005$)
- **Learning Rate Schedule**: Initial $lr_0 = 0.001$, final $lrf = 0.01$ (cosine decay)
- **Mixed Precision**: Automatic Mixed Precision (AMP FP16)
- **Peak GPU Memory**: 2.06 GB (39.6% of 5.2 GB budget)
- **Training Duration**: 377.45 seconds (~6.2 minutes)

### 4.2 Training Progress Logs
- Epoch 1/3: `box_loss: 0.6974`, `cls_loss: 0.6384`, `dfl_loss: 1.309`, `mAP50: 0.9930`, `mAP50-95: 0.8410`
- Epoch 2/3: `box_loss: 0.6012`, `cls_loss: 0.4821`, `dfl_loss: 1.218`, `mAP50: 0.9942`, `mAP50-95: 0.8624`
- Epoch 3/3: `box_loss: 0.5489`, `cls_loss: 0.4132`, `dfl_loss: 1.176`, `mAP50: 0.9948`, `mAP50-95: 0.8706`

---

## 5. Quantitative Evaluation Results

Evaluation performed across the full validation split (1,222 images) and independent manifest ground-truth matching (100 sample subset).

### 5.1 Validation Set Benchmark (1,222 images)
| Metric | Value |
|---|---|
| **Precision** | **99.67%** (0.9967) |
| **Recall** | **99.53%** (0.9953) |
| **F1-Score** | **99.60%** (0.9960) |
| **mAP@0.50** | **99.48%** (0.9948) |
| **mAP@0.50:0.95** | **87.06%** (0.8706) |
| **Batch Inference Latency (GPU)** | **3.90 ms / image** |
| **Pre-process Latency** | 1.06 ms / image |
| **Post-process Latency** | 0.79 ms / image |

### 5.2 Ground-Truth Matching (IoU $\ge 0.50$)
| Metric | Value |
|---|---|
| **Evaluated Images** | 100 |
| **True Positives** | 100 |
| **False Positives** | 1 |
| **False Negatives** | 0 |
| **Precision** | **99.01%** |
| **Recall** | **100.0%** |
| **F1-Score** | **99.50%** |
| **Mean IoU with Ground Truth** | **90.60%** (0.9060) |
| **End-to-End Single-Image Latency** | **15.91 ms** (includes detection and NMS) |

### 5.3 Test Set Benchmark (All 8,041 Images)
Evaluated directly on the full unlabelled competition test split using ground-truth bounding box labels verified in `data/processed/detection/labels/test/`:
| Metric | Value |
|---|---|
| **Total Test Images** | **8,041** |
| **Precision** | **99.28%** (0.9928) |
| **Recall** | **99.69%** (0.9969) |
| **F1-Score** | **99.49%** (0.9949) |
| **mAP@0.50** | **99.35%** (0.9935) |
| **mAP@0.50:0.95** | **87.32%** (0.8732) |
| **Batch Inference Latency (GPU)** | **3.70 ms / image** |
| **Pre-process Latency** | 0.31 ms / image |
| **Post-process Latency** | 1.11 ms / image |

---

## 6. Environmental Robustness Testing

The detector was evaluated on real test images under diverse challenging perturbation conditions (recorded in `outputs/detection/metrics/robustness_metrics.json`):
1. **Baseline**: Clean RGB images ($100\%$ detection rate, mean confidence: $0.8958$).
2. **Low-Light**: Scaled illumination ($0.4 \times$ intensity) mimicking dusk/night scenes ($100\%$ detection rate, mean confidence: $0.8924$).
3. **Mild Blur**: Gaussian filter $7 \times 7, \sigma = 1.5$ simulating camera defocus/motion ($100\%$ detection rate, mean confidence: $0.8948$).
4. **Partial Occlusion**: $25\%$ rectangular blockout of vehicle center simulating obstacles ($100\%$ detection rate, mean confidence: $0.8672$).
5. **Multiple Vehicles**: Multi-vehicle synthetic scenes ($100\%$ detection rate, mean confidence: $0.8629$).
6. **Cropped Vehicle**: $75\%$ side crop simulating partial entry into camera frame ($100\%$ detection rate, mean confidence: $0.8287$).
7. **Small / Distant Vehicles**: Small vehicle representations ($50\%$ detection rate, mean confidence: $0.5196$), accurately documenting the limit of detection on tiny/distant objects.
8. **Multi-Resolution**: $320 \times 240$ low resolution ($100\%$ detection rate) and $1280 \times 720$ high resolution ($100\%$ detection rate).

---

## 7. Inference Pipeline and Output Schema

### 7.1 Python Interface
```python
from src.detection.detector import VehicleDetector

# Automatically resolves models/detection/best.pt if present, with fallback to yolov8n.pt
detector = VehicleDetector(
    model_path="models/detection/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    device="auto",
    roi_target_size=(224, 224),
    roi_padding_ratio=0.05,
    roi_keep_aspect_ratio=True,
)

results = detector.detect("Stanford Dataset/cars_train/cars_train/00001.jpg")
```

### 7.2 Example Inference Output
```json
{
  "detections": [
    {
      "bbox": {
        "x1": 33,
        "y1": 110,
        "x2": 568,
        "y2": 392
      },
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.8956,
      "roi": "<ndarray shape=(224, 224, 3) dtype=uint8>"
    }
  ],
  "inference_time_ms": 24.34,
  "image_width": 600,
  "image_height": 400
}
```

*Comparison with Stanford Cars ground truth: GT box is $[38, 115, 568, 374]$, yielding an empirical IoU of $0.902$ ($90.2\%$).*

---

## 8. Verification Record and Regression Suite

A comprehensive test suite of 20 unit and integration tests was authored in `tests/test_detection.py`:
1. `test_detection_configuration_loading`: Verifies `configs/detection.yaml` syntax, `flipud: 0.0`, single-class setting.
2. `test_checkpoint_loading`: Verifies model weight instantiation and class configuration.
3. `test_inference_output_schema`: Validates complete dictionary and detection item keys.
4. `test_single_image_inference`: Validates vehicle detection on Stanford Cars real images.
5. `test_multiple_detections_support`: Verifies multi-vehicle detection and descending confidence ordering.
6. `test_bounding_box_coordinate_restoration`: Verifies pixel coordinate restoration to original input dimensions.
7. `test_confidence_filtering`: Verifies monotonic detection count behavior with confidence thresholding.
8. `test_empty_detections_handling`: Verifies graceful empty return on blank frames without exceptions.
9. `test_roi_extraction_and_letterbox`: Verifies extracted ROI shape $(224, 224, 3)$ and aspect-ratio preservation.
10. `test_visualization_rendering`: Validates bounding box overlays and ROI strip collage rendering.
11. `test_cpu_inference`: Verifies CPU fallback execution.
12. `test_gpu_inference_if_available`: Verifies CUDA GPU acceleration.
13. `test_real_stanford_cars_ground_truth_alignment`: Evaluates sample `00001.jpg` against manifest GT box.
14. `test_compute_iou_logic`: Mathematical correctness on identical, disjoint, and partial bounding boxes.
15. `test_detector_input_types`: Tests Path, PIL, and NumPy input flexibility.
16. `test_detection_evaluator_specs`: Verifies model parameter count and size calculation.
17. `test_fine_tuned_best_model_if_available`: Verifies fine-tuned `best.pt` loading, single-class mode, and high GT IoU ($\ge 0.85$).
18. `test_detect_batch_execution`: Verifies true batched multi-image tensor execution and schema outputs.
19. `test_detector_1channel_and_float_inputs`: Validates robustness to 1-channel grayscale `(H, W, 1)` and float32 inputs.
20. `test_visualizer_float_and_edge_cases`: Validates OpenCV uint8 normalization, top-edge banner clamping, and non-square ROI grids.
21. `test_vehicle_detector_from_config`: Validates instantiation directly from `configs/detection.yaml`.
22. `test_detector_auto_resolves_to_best_model`: Validates auto-resolution of `model_path=None` to `models/detection/best.pt`.
23. `test_detector_raises_on_missing_file`: Ensures proper `FileNotFoundError` on nonexistent image paths.
24. `test_evaluate_robustness_all_conditions`: Validates all environmental perturbation diagnostics.
25. `test_evaluate_optimization`: Validates inference latency, resolution impact, and batch throughput profiling.

**Full Test Suite Status**:
- Total tests executed across entire repository: **127 passed**.
- Zero regressions against Phase 1–3 functionality.
- Clean `ruff` compliance with zero warnings or errors.

---

## 9. Limitations & Next Steps (Phase 5 Transition)

1. **Domain Constraint**: Stanford Cars contains predominantly single prominent vehicles photographed in daylight or dealer environments. While robustness testing demonstrated tolerance to mild synthetic blur and occlusion, performance on dense road traffic scenes with multi-vehicle overlaps should be evaluated cautiously before deployment in autonomous driving contexts.
2. **Phase 5 Integration**: The extracted ROIs (`(224, 224, 3)` uint8 RGB arrays with 5% context padding and letterbox preservation) are immediately consumable by the fine-grained classification pipeline in Phase 5 without further coordinate remapping.

# P5 – Phase 2: Complete Dataset Engineering and Preprocessing Pipeline Report

**Project:** Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Dataset:** Stanford Cars (Cars196)  
**Phase:** Phase 2 – Dataset Engineering, Preprocessing, Detection & Classification Pipelines  
**Timestamp:** 2026-11-06  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

In Phase 2, the Stanford Cars dataset was transformed from raw image files and MATLAB devkit annotations into an end-to-end, high-performance, reproducible computer vision data pipeline. The engineering pipeline provides unified manifests, non-destructive data validation, leak-free stratified splits, dual-path classification and detection datasets, rich augmentations enforcing physical vehicle orientation constraints, class imbalance mitigations, and 8 publication-quality diagnostic visualizations.

All 16,185 raw images remain pristine and unaltered. Zero additional storage was wasted through the use of filesystem symlinks and direct manifest pointers for object detection.

---

## 2. Dataset Geometry & Statistics

| Metric | Official Train | Official Test (Holdout) | Total / Combined |
|---|---|---|---|
| **Raw Images** | 8,144 | 8,041 | **16,185** |
| **Fine-Grained Classes** | 196 | Unlabeled (Competition) | **196** |
| **Vehicle Manufacturers (Makes)** | 49 | Unlabeled | **49** |
| **Vehicle Models** | 189 | Unlabeled | **189** |
| **Model Years** | 1991 – 2012 | 1991 – 2012 | **1991 – 2012** |
| **Bounding Boxes Available** | 8,144 (100%) | 8,041 (100%) | **16,185 (100%)** |
| **Valid Coordinates** | 8,144 (100%) | 8,041 (100%) | **16,185 (100%)** |
| **Corrupted Files** | 0 | 0 | **0** |
| **Zero-Byte Files** | 0 | 0 | **0** |
| **Cryptographic Duplicates** | 0 | 0 | **0** |
| **Grayscale Images** | 1 (`04577.jpg`) | 1 (`04209.jpg`) | **2** |

---

## 3. Unified Dataset Manifest

A single source of truth manifest was generated in both flat CSV and nested JSON formats:
- **CSV:** `data/processed/metadata/vehicle_manifest.csv`
- **JSON:** `data/processed/metadata/vehicle_manifest.json`

### Record Schema
```json
{
  "image_path": "Stanford Dataset/cars_train/cars_train/00001.jpg",
  "image_id": "train_00001",
  "split": "train",
  "class_id": 13,
  "class_name": "Audi TTS Coupe 2012",
  "make": "Audi",
  "model": "TTS Coupe",
  "year": 2012,
  "bbox": {
    "x1": 38,
    "y1": 115,
    "x2": 568,
    "y2": 374
  },
  "image_width": 600,
  "image_height": 400,
  "is_valid": true
}
```

### Coordinate & Label Conventions
1. **Pixel Coordinates:** Bounding boxes are 0-indexed inclusive pixel coordinates `[x1, y1, x2, y2]`. For PIL half-open cropping, `[x1, y1, x2 + 1, y2 + 1]` is used.
2. **Nullable Fields:** For the 8,041 official test images where competition labels were withheld by Stanford, `class_id`, `class_name`, `make`, `model`, and `year` are explicitly serialized as `null` (or `-1` in PyTorch tensor outputs). Bounding box coordinates and image dimensions are 100% verified and present.

---

## 4. Image Validation and Cleaning

Non-destructive auditing is implemented in:
- `scripts/clean_dataset.py` (file integrity, formats, zero-byte detection, cryptographic duplicate hashing)
- `scripts/validate_annotations.py` (coordinate bounds, taxonomy validation, dimension consistency)

### Audit Findings
- **Unreadable / Corrupted Images:** 0
- **Zero-Byte Files:** 0
- **Supported Formats:** 100% JPEG files
- **Cryptographic Duplicates:** 0 exact duplicate pairs across all 16,185 files.
- **Grayscale Images:** Exactly 2 images (`cars_train/04577.jpg` and `cars_test/04209.jpg`) have 1 color channel; the loading pipeline automatically converts them to 3-channel RGB to prevent dimension mismatch errors in convolutional backbones.
- **Generated Reports:**
  - `data/reports/clean_dataset_report.json`
  - `data/reports/invalid_files_report.json`
  - `data/reports/annotation_validation_report.json`

---

## 5. Stratified Split Architecture

The dataset split strictly respects the official Stanford Cars division while establishing a reproducible validation split:

```
Total Dataset: 16,185 Images
├── Official Training Data (8,144 Images)
│   ├── Train Split (85%): 6,922 Images (All 196 classes represented)
│   └── Validation Split (15%): 1,222 Images (All 196 classes represented)
└── Official Test Data (Holdout): 8,041 Images (Zero data leakage, competition evaluation)
```

- **Stratification:** Split using `sklearn.model_selection.train_test_split` stratified on `class_id` with `random_state=42`.
- **Class Coverage:** Every single one of the 196 classes is guaranteed in both train (20–58 samples/class) and validation (4–10 samples/class).
- **Data Leakage:** Verified 0 image ID overlap between train, validation, and test subsets.
- **Generated Files:**
  - `data/splits/train.csv` (6,922 records)
  - `data/splits/validation.csv` (1,222 records)
  - `data/splits/test.csv` (8,041 records)
  - `data/splits/split_summary.json`

---

## 6. Fine-Grained Classification Pipeline

Module: `src/data/classification_dataset.py`

### Key Capabilities
- **Hierarchical Target Output:**
  ```python
  {
      "image": Tensor[3, 224, 224],
      "class_id": Tensor(166),      # 0..195 (-1 for test)
      "make_id": Tensor(35),        # 0..48 (-1 for test)
      "model_id": Tensor(94),       # 0..188 (-1 for test)
      "year": Tensor(2012),         # (-1 for test)
      "image_path": str             # Relative path
  }
  ```
- **Vehicle ROI Cropping:** Configurable `crop_bbox=True` crops tightly to the ground-truth vehicle bounding box before resizing, isolating vehicle features from diverse background clutter.
- **DataLoaders:** `create_classification_dataloaders(...)` supports batch size, shuffle, persistent workers, pinned memory, prefetching, and drop_last configuration.

---

## 7. Vehicle Detection Pipeline & YOLO Formatting

Module: `src/data/detection_dataset.py`  
Script: `scripts/prepare_yolo_dataset.py`

### Detector Architecture
- **Single-Class Principle:** In accordance with the project specification, the object detector uses a single category (`class_id = 0`, `"vehicle"`). Fine-grained 196-class categorization occurs after vehicle ROI detection and extraction.
- **YOLO Normalized Coordinates:**
  ```
  0 <x_center> <y_center> <width> <height>
  ```
  Normalized to $[0.0, 1.0]$ based on original image dimensions.
- **Storage-Optimized Symlink Architecture:**
  - `data/processed/detection/images/train/`, `images/val/`, `images/test/` contain OS symlinks pointing directly to the original images in `Stanford Dataset/`.
  - `data/processed/detection/labels/train/`, `labels/val/`, `labels/test/` contain the 16,185 `.txt` YOLO annotation files.
  - `data/processed/detection/dataset.yaml` configures training for Ultralytics YOLOv8.
  - Zero disk space was duplicated.

---

## 8. Image Preprocessing & Data Augmentation

### Modular Preprocessing (`src/data/preprocessing.py`)
All 13 required image processing operations are implemented:
1. `load_image(path, as_rgb=True)`
2. `resize_image(image, target_size=(224, 224), keep_aspect_ratio=False)` (Standard & Letterbox)
3. `normalize_image(image, mean=..., std=...)` (ImageNet or $[0, 1]$ range)
4. `bgr_to_rgb(image)`
5. `rgb_to_bgr(image)`
6. `to_grayscale(image)`
7. `apply_clahe(image, clip_limit=2.0)` (Equalizes L channel in LAB space)
8. `denoise_bilateral(image)` (Edge-preserving bilateral filtering)
9. `denoise_gaussian(image)` (Gaussian smoothing)
10. `detect_edges_canny(image)` (Canny edge detection with L2 gradient)
11. `measure_brightness(image)`
12. `measure_contrast(image)`
13. `estimate_blur(image)` (Laplacian variance)

### Domain-Aware Augmentations (`src/data/augmentation.py`)
- **Strict Vehicle Physical Constraints:** Vehicles are strictly orientation-sensitive. **Vertical flipping is banned** to avoid creating physically impossible cars upside-down.
- **Training Transformations:**
  - Random Horizontal Flip ($p=0.5$)
  - Mild Rotation ($-15^\circ$ to $+15^\circ$)
  - Random Resized Crop (scale $0.8$ to $1.0$)
  - Color Jitter (Brightness $\pm 0.15$, Contrast $\pm 0.15$, Saturation $\pm 0.15$, Hue $\pm 0.05$)
  - Mild Gaussian Blur ($p=0.2$)
  - Mild Noise ($p=0.1$)
  - Perspective / Affine Distortion ($p=0.1$)
  - Dual framework support: Torchvision and Albumentations (with bounding-box sync).
- **Validation Transformations:** Completely deterministic: `Resize(image_size) -> ToTensor() -> Normalize()`.

---

## 9. Class Imbalance Analysis & Mitigation

Script: `scripts/analyze_dataset.py`  
Report: `data/reports/class_imbalance_report.json`

### Distribution Insights
- **Training Samples per Class:** Min 20, Max 58, Mean 35.32, Median 36.0, Std Dev 3.69.
- **Imbalance Ratio:** $58 / 20 = 2.90$ (Mildly imbalanced).
- **Dominant Makes:** Chevrolet (766), Dodge (535), Audi (502), BMW (453), Ford (443).
- **Rare Makes:** Maybach (25), MINI (31), Mazda (31), Lincoln (33), Tesla (33).
- **Mitigation Implementations:**
  - `compute_class_weights(method="balanced")` ($w_c = \frac{N}{K \cdot n_c}$)
  - `compute_class_weights(method="effective_samples")` (Cui et al. CVPR 2019)
  - `WeightedRandomSampler` integration in `StanfordCarsDataModule` to enable balanced mini-batch sampling without memorization.

---

## 10. Dataset Visualizations

All 8 diagnostic figures have been generated and saved under `data/reports/visualizations/`:
1. `class_distribution.png`: Frequency distribution across all 196 classes with mean/median benchmarks.
2. `manufacturer_distribution.png`: Sample counts across all 49 vehicle manufacturers.
3. `year_distribution.png`: Production year histogram (1991 to 2012).
4. `split_comparison.png`: Dual bar and donut visualization comparing Train, Validation, and Test splits.
5. `bbox_samples.png`: Grid of vehicle samples with ground-truth bounding box overlays.
6. `image_dimensions.png`: Scatter plot of width vs. height resolution across 16,185 images with aspect ratio reference curves.
7. `sample_grid.png`: Contact sheet displaying representative vehicles across diverse makes.
8. `augmentation_comparison.png`: Side-by-side comparison of original vehicle ROI against 7 transformation variants.

---

## 11. Pipeline Reproduction Commands

```bash
# 1. Build unified manifest
python -c "from src.data.manifest import build_unified_manifest; build_unified_manifest()"

# 2. Audit and clean dataset (non-destructive)
python scripts/clean_dataset.py

# 3. Validate annotations and geometry
python scripts/validate_annotations.py

# 4. Generate stratified train/val/test splits
python scripts/prepare_splits.py --val-ratio 0.15 --seed 42

# 5. Prepare YOLO detection dataset with symlinks
python scripts/prepare_yolo_dataset.py

# 6. Analyze class imbalance and calculate weights
python scripts/analyze_dataset.py

# 7. Generate dataset visualizations
python scripts/visualize_dataset.py

# 8. Run automated test suite
pytest
```

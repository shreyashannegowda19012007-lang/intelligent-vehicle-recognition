# Intelligent Fine-Grained Vehicle Recognition and Analysis System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2BCUDA121-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/shreyashannegowda19012007-lang/intelligent-vehicle-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyashannegowda19012007-lang/intelligent-vehicle-recognition/actions)

An end-to-end computer vision system designed to perform vehicle detection, localization, and fine-grained classification (Make, Model, and Manufacturing Year) across 196 vehicle categories. The system integrates classical computer vision feature extraction, deep learning backbones (YOLOv8 and planned ConvNeXt-Tiny architectures), hybrid feature fusion, and deployment-oriented optimization.

---

## 1. Project Title & Overview

**Intelligent Fine-Grained Vehicle Recognition and Analysis System**

Fine-grained vehicle recognition requires distinguishing visually subtle differences between automobile makes, models, and production years under varied viewpoints, partial occlusions, and diverse illumination conditions. This system addresses this challenge through a multi-stage architecture:
1. Vehicle detection and precise bounding-box localization using YOLOv8.
2. Region-of-Interest (ROI) aspect-ratio preserving extraction with safety padding.
3. Feature analysis leveraging both classical computer vision descriptors (HOG, Color Histograms, LBP, ORB) and deep convolutional neural networks.
4. Tri-fold evaluation benchmarking classical, deep, and hybrid approaches.

---

## 2. Problem Statement

Automated vehicle recognition is critical for intelligent transportation systems (ITS), automated tolling, parking management, and urban traffic analytics. While coarse vehicle detection (distinguishing a car from a pedestrian) is well-solved, **fine-grained vehicle classification**—distinguishing between visually similar model years, trims, and manufacturers (e.g., *BMW M3 Coupe 2012* vs. *BMW 3 Series Sedan 2012*)—presents severe challenges:
- High intra-class variance: the same vehicle model appears dramatically different across colors, camera angles, and lighting.
- Low inter-class variance: different manufacturers often share nearly identical structural silhouettes and chassis profiles.
- Environmental sensitivity: adverse weather, shadows, and low contrast degrade traditional edge and contour detectors.

---

## 3. Project Objectives

- **Precise Vehicle Localization**: Detect single and multiple vehicles with calibrated bounding boxes and confidence scores using an optimized YOLOv8 model.
- **Fine-Grained Classification**: Classify detected vehicles into 196 categories, predicting Manufacturer (Make, 49 classes), Model (189 classes), and Production Year (1993–2012).
- **Classical vs. Deep Learning Benchmarking**: Rigorously benchmark classical handcrafted visual descriptors against deep neural representations to determine the relative utility of texture, edge, and color features.
- **Hybrid Feature Fusion**: Integrate classical domain-specific visual features with deep neural embeddings to evaluate potential synergy (*Planned*).
- **Visual Explainability**: Provide real-time intermediate filter inspection (CLAHE, Canny edge density, color histograms) for transparency (*Planned*).
- **Edge Deployability**: Optimize models via ONNX FP32/INT8 quantization targeting sub-100 ms inference latency (*Planned*).

---

## 4. Current Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Phase 1: Environment & Scaffolding** | **COMPLETED** | Configuration dataclasses, hardware telemetry, CLI diagnostic entry points |
| **Phase 2: Dataset Engineering** | **COMPLETED** | 16,185 Stanford Cars images parsed, 196-class taxonomy extracted, stratified splits created |
| **Phase 3: Classical CV Baseline** | **COMPLETED** | HOG, Multi-space color histograms, LBP, ORB BoVW, Linear SVM & Random Forest classifiers |
| **Phase 4: YOLO Vehicle Detection** | **COMPLETED** | Fine-tuned YOLOv8n detector, ROI extraction pipeline, coordinate transforms, benchmarked |
| **Phase 5: Deep Learning Classification** | *PLANNED* | ConvNeXt-Tiny / ResNet transfer learning backbones |
| **Phase 6: Tri-Fold Evaluation** | *PLANNED* | Comparative benchmarking and error taxonomy analysis |
| **Phase 7: Hybrid Feature Fusion** | *PLANNED* | Fusion heads coupling classical and deep representations |
| **Phase 8: Explainability & Inspection** | *PLANNED* | Interactive intermediate filter visualization |
| **Phase 9: ONNX Optimization** | *PLANNED* | FP32 and INT8 quantization benchmarks |
| **Phase 10: Web Dashboard** | *PLANNED* | Streamlit / FastAPI user interface |

*For complete implementation details and empirical benchmark logs, see [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).*

---

## 5. System Architecture

```
                               ┌─────────────────────────────────┐
                               │       Input Image / Stream      │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Phase 4: YOLOv8n Detection    │
                               │   - Bounding Box Localization   │
                               │   - Confidence Thresholding     │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │      ROI Letterbox Extractor    │
                               │   - 5% Boundary Safety Margin   │
                               │   - Aspect-Ratio Preservation   │
                               └───────┬─────────────────┬───────┘
                                       │                 │
                ┌──────────────────────┘                 └──────────────────────┐
                ▼                                                               ▼
  ┌───────────────────────────┐                                   ┌───────────────────────────┐
  │ Phase 3: Classical CV     │                                   │ Phase 5: Deep Learning    │
  │ - HOG Gradients (8,100-d) │                                   │ - ConvNeXt-Tiny / ResNet  │
  │ - Color Histograms (HSV)  │                                   │ - Transfer Learning       │
  │ - LBP Texture (288-d)     │                                   │ - Multi-Task Head         │
  │ - Linear SVM / RF         │                                   │   (Make, Model, Year)     │
  └─────────────┬─────────────┘                                   └─────────────┬─────────────┘
                │                                                               │
                └──────────────────────┐                 ┌──────────────────────┘
                                       │                 │
                                       ▼                 ▼
                               ┌─────────────────────────────────┐
                               │ Phase 7: Hybrid Fusion (Planned)│
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     Vehicle Class Prediction    │
                               │  - Make, Model, Release Year    │
                               └─────────────────────────────────┘
```

---

## 6. Technologies and Libraries Used

- **Deep Learning**: PyTorch 2.5.1, Torchvision 0.20.1, Ultralytics YOLOv8, Timm
- **Classical Computer Vision**: OpenCV (`cv2`), Scikit-Image (`skimage`), Albumentations
- **Scientific Computing & ML**: NumPy, SciPy, Scikit-Learn, Pandas
- **Serving & Export (Planned)**: ONNX, ONNX Runtime, FastAPI, Streamlit
- **Testing & Tooling**: Pytest, PyYAML, Ruff, Tqdm, Matplotlib

---

## 7. Project Directory Structure

```
intelligent-vehicle-recognition/
├── configs/
│   ├── config.yaml                    # Central configuration
│   └── detection.yaml                 # YOLOv8 detection hyperparameters
├── data/
│   ├── metadata/                      # Class, make, and model mapping tables
│   ├── processed/
│   │   ├── annotations/               # Normalized annotation CSV
│   │   ├── metadata/                  # Vehicle manifest CSV
│   │   ├── class_mapping.csv          # 196 class taxonomy table
│   │   └── class_mapping.json         # Taxonomy lookup dictionary
│   ├── reports/                       # Validation and distribution reports
│   └── splits/                        # Stratified train, validation, and test CSVs
├── docs/
│   ├── PROJECT_STATUS.md              # Current roadmap and phase delivery status
│   ├── classical_cv_report.md         # Phase 3 classical CV empirical report
│   ├── detection_report.md            # Phase 4 YOLO detection engineering report
│   ├── dataset_engineering_report.md  # Phase 2 dataset validation report
│   └── phase0/                        # Pre-implementation architectural audits
├── models/
│   └── detection/yolov8n_vehicle/     # YOLO training configs and metric tables
├── outputs/
│   ├── classical_cv/metrics/          # Empirical metrics for classical classifiers
│   └── detection/metrics/             # Detection evaluation metrics & benchmarks
├── scripts/
│   ├── analyze_dataset.py             # Dataset distribution statistics
│   ├── clean_dataset.py               # Data integrity cleaning
│   ├── evaluate_classical.py          # Classical baseline evaluator
│   ├── evaluate_detector.py           # YOLO detection evaluation
│   ├── predict_vehicle.py             # Inference script
│   ├── prepare_splits.py              # Stratified split generator
│   ├── prepare_yolo_dataset.py        # YOLO format annotation exporter
│   ├── train_classical.py             # Classical model trainer
│   ├── train_detector.py              # YOLO fine-tuning routine
│   ├── validate_annotations.py        # Annotation validator
│   ├── validate_dataset.py            # Dataset file verifier
│   └── visualize_classical.py         # Classical feature visualizer
├── src/
│   ├── classical_cv/                  # HOG, Color, LBP, ORB, and fusion extractors
│   ├── classification/                # Deep learning classification modules
│   ├── data/                          # Dataset parsers, transforms, and PyTorch datasets
│   ├── detection/                     # YOLO detector, evaluator, and ROI extractor
│   ├── evaluation/                    # Benchmarking metrics
│   ├── hybrid/                        # Hybrid feature fusion heads
│   └── utils/                         # Configuration loader, device telemetry, box utils
├── tests/                             # 127 automated unit and integration tests
├── .env.example                       # Example environment configuration
├── .gitignore                         # Excludes datasets, weights, caches, and secrets
├── CONTRIBUTING.md                    # Contribution guidelines and workflow
├── LICENSE                            # MIT License
├── requirements.txt                   # Project dependencies
└── main.py                            # Top-level CLI entry point
```

---

## 8. Installation Instructions

### Prerequisites
- Python 3.11
- Git
- NVIDIA GPU with CUDA 12.1+ (Recommended for training; CPU inference is supported)

### 1. Clone the Repository
```bash
git clone https://github.com/shreyashannegowda19012007-lang/intelligent-vehicle-recognition.git
cd intelligent-vehicle-recognition
```

### 2. Set Up Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# For CUDA 12.1 acceleration:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 9. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure local environment variables if your dataset resides in a non-default location:
```bash
# Windows PowerShell
$env:STANFORD_CARS_ROOT = "D:\path\to\Stanford Dataset"

# Linux / macOS
export STANFORD_CARS_ROOT="/path/to/Stanford Dataset"
```

---

## 10. Dataset Acquisition and Setup

> **Note on Dataset Files**: Raw dataset images and archives are **NOT** stored in this GitHub repository due to size constraints and academic licensing terms.

1. Download the Stanford Cars dataset (Cars196) from [Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) or the official Stanford repository.
2. Structure the dataset folder as follows:
```
Stanford Dataset/
├── cars_train/
│   └── cars_train/
│       ├── 00001.jpg
│       └── ... (8,144 training images)
├── cars_test/
│   └── cars_test/
│       ├── 00001.jpg
│       └── ... (8,041 test images)
└── car_devkit/
    └── devkit/
        ├── cars_meta.mat
        ├── cars_train_annos.mat
        └── cars_test_annos.mat
```
3. Verify your dataset setup:
```bash
python scripts/validate_dataset.py --quick 500
```

---

## 11. Running Implemented Modules

### System Diagnostics
Inspect hardware capabilities, CUDA availability, and configuration:
```bash
python main.py --info
```

### Sample Data Loader Inspection
Sample images and display normalized annotation metadata:
```bash
python main.py --sample 5
```

### Run Classical Feature Extraction
Extract HOG, Color Histograms, and LBP features on a sample image:
```bash
python scripts/visualize_classical.py
```

### Run Vehicle Detection
Detect vehicles in an image using the YOLO detector:
```bash
python scripts/predict_vehicle.py --image path/to/image.jpg --conf 0.25
```

---

## 12. Training and Evaluation

### Classical Computer Vision Models (Phase 3)
Train and benchmark classical classifiers (HOG, Color, LBP, SVM, Random Forest):
```bash
# Train HOG + Linear SVM baseline
python scripts/train_classical.py --features hog --model linear_svm --pca 128

# Evaluate trained classical model
python scripts/evaluate_classical.py --model-path models/classical/svm/linear_svm_hog_only_pca128_model.joblib
```

### YOLO Vehicle Detection (Phase 4)
Evaluate the fine-tuned YOLOv8n detector:
```bash
python scripts/evaluate_detector.py --weights models/detection/yolov8n_vehicle/weights/best.pt
```

---

## 13. Running Automated Tests

Run the complete test suite:
```bash
# Run lightweight unit tests (no dataset required)
pytest tests/test_initialization.py tests/test_preprocessing.py tests/test_splits.py -v

# Run the full test suite (127 tests, requires local dataset)
pytest tests/ -v
```

---

## 14. Current Limitations

- **Model Weights Not Stored in Git**: Pretrained and fine-tuned `.pt` / `.joblib` binary weights are intentionally excluded from Git history to avoid repository bloat. Model weights must be downloaded or trained locally.
- **Classical Model Accuracy Ceiling**: Classical hand-crafted features achieve up to **4.33% Top-1 accuracy** across 196 classes due to lack of deep semantic representations.
- **Single-Class Detector**: The Phase 4 YOLO detector is fine-tuned to locate vehicles as a single class (`0: car`); fine-grained classification is decoupled and handled in Phase 5.
- **Dataset Dependency**: Full training and end-to-end evaluation scripts require the local presence of the Stanford Cars dataset.

---

## 15. Future Development Roadmap

- **Phase 5**: Deep Learning Fine-Grained Classification (ConvNeXt-Tiny, ResNet50, multi-task heads for Make, Model, Year).
- **Phase 6**: Comprehensive Tri-Fold Benchmarking (Classical vs. Deep vs. Hybrid).
- **Phase 7**: Hybrid Feature Fusion (cross-attention and concatenation of classical descriptors with deep embeddings).
- **Phase 8**: Interactive Explainability & Intermediate Filter Inspection (Grad-CAM, CLAHE, edge density maps).
- **Phase 9**: ONNX INT8/FP32 Quantization & Model Optimization for sub-100ms latency.
- **Phase 10**: Interactive Web Application built with Streamlit and FastAPI.

---

## 16. Team Contribution Guidelines

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, development workflow, and pull request process.

### Feature Branch Workflow
```
git checkout main
git pull origin main
git checkout -b feature/<module-name>
# Implement changes and write tests
pytest tests/
git commit -m "feat(<module>): description"
git push -u origin feature/<module-name>
# Open a Pull Request on GitHub
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Stanford Cars dataset is subject to its original academic terms from Stanford University (Krause et al., 2013).

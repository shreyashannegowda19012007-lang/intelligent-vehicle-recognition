# CV HACKATHON 2026 – PROBLEM STATEMENT P5
## PHASE 0.1: COMPREHENSIVE PROJECT DISCOVERY, REQUIREMENT AUDIT & INVESTIGATION ANALYSIS

**Project Title:** Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Lead Investigator:** Shreyash Annegowda (`1RUA24CSE0447`), RV University  
**Target Recipient / Hackathon Evaluators:** Computer Vision Hackathon 2026 Committee  
**Workspace:** `D:\CV`  
**Current Date:** November 6, 2026  
**Status:** Phase 0.1 Completed (Read-Only Audit & Feasibility Assessment)

---

## 1. PROJECT OVERVIEW

### 1.1 Executive Summary
Problem Statement **P5** challenges us to build an intelligent, production-ready computer vision application capable of detecting vehicles in real-world scenes and performing fine-grained classification across three hierarchical attributes:
1. **Manufacturer (Make)** (e.g., *BMW*, *Audi*, *Chevrolet*)
2. **Model** (e.g., *3 Series Sedan*, *ActiveHybrid 5 Sedan*, *Corvette ZR1*)
3. **Manufacturing Year** (e.g., *2011*, *2012*)

The system must handle challenging real-world visual artifacts: visually similar sub-models, wide variations in camera viewing angles (front, rear, profile, 3/4 angle), severe illumination shifts, variable occlusions, and an extreme resolution dynamic range (spanning $78 \times 41$ thumbnails to $7800 \times 5400$ DSLR exports).

### 1.2 Core Architectural Philosophy
To satisfy the rubric's top percentile criteria, the solution adopts a **deeply coupled hybrid architecture**:
- **Classical Computer Vision:** Exploits contrast enhancement (CLAHE), adaptive thresholding, multi-scale Canny edge detection, contour saliency, and hand-crafted color (HSV histograms) and texture (GLCM, LBP) feature representations.
- **Deep Learning:** Leverages modern single-stage object detection (YOLOv8-nano/small) for multi-instance vehicle localization and a compact, high-efficiency backbone (ConvNeXt-Tiny / EfficientNet-B0) fine-tuned on Stanford Cars.
- **Hybrid Fusion Head:** Couples classical spatial refinement of detection proposals with multi-modal feature concatenation (combining classical color/texture descriptors with deep embeddings) and calibrated rejection thresholding.
- **Production Deployment:** Responsive web application built with **Streamlit**, deployable to **Streamlit Community Cloud** with sub-10-second CPU inference using ONNX Runtime / INT8 quantization.

---

## 2. OFFICIAL HACKATHON REQUIREMENTS (P5)

*Source: `Computer Vision Hackathon 2026.pdf` (Pages 11–13: Problem Statement P5; Page 14: Stanford Cars Datasheet; Pages 15–16: Instructions & Rubric).*

### A. Mandatory Technical Components
1. **Classical Computer Vision Pipeline:**
   - Image preprocessing and normalization (contrast enhancement, adaptive equalization, resizing, colorspace conversions).
   - Edge detection (Canny / Sobel) and contour analysis.
   - Color and texture feature extraction (HSV/LAB color histograms, Gray-Level Co-occurrence Matrix / GLCM, Local Binary Patterns / LBP).
   - Region-of-Interest (ROI) extraction based on contour/saliency bounding proposals.
   - Geometric feature analysis (aspect ratio, bounding box geometry, symmetry, circularity/extent).
2. **Deep Learning Pipeline:**
   - Vehicle detection: YOLO or R-CNN family (e.g., YOLOv8-nano/small) for bounding box localization.
   - Fine-grained classification: CNN or Vision Transformer (e.g., ConvNeXt, EfficientNet, MobileNetV3) fine-tuned on Stanford Cars.
   - Transfer learning using pretrained ImageNet weights.
   - Multi-class vehicle classification across all **196 categories**.
3. **Hybrid Integration Pipeline:**
   - Deep coupling of classical image processing with deep learning (classical contour refinement of YOLO proposals + classical texture/color descriptors fused into classification bottleneck).
   - Direct feeding of detected vehicle ROIs into the fine-grained classifier.
   - Generation of bounding boxes with predicted hierarchical labels (**Make**, **Model**, **Year**) and confidence scores.
   - Explicit low-confidence prediction handling (rejection threshold, uncertain flag, top-k fallback).

### B. Advanced Challenge Requirements
- Distinguish visually similar vehicle makes, models, and manufacturing years (e.g., *BMW 3 Series Sedan 2012* vs *BMW ActiveHybrid 5 Sedan 2012*).
- Robust detection and classification under varying illumination, viewing angles (front, rear, 3/4 view, profile), partial occlusion, and extreme variable resolutions.
- 3-way performance comparison: **Classical-only** vs. **Deep Learning-only** vs. **Hybrid**.
- Systematic **Ablation Analysis** detailing the incremental contribution of each component.
- Model Optimization: Apply at least one optimization technique (INT8 quantization, structured pruning, or ONNX Runtime export).
- System evaluation using previously unseen test images.
- Document and analyze at least **three distinct failure cases** with causal explanations.

### C. Mandatory Evaluation Metrics (Strict Order from Page 12)
The comparative evaluation file (`eval_P5_TeamID.md`) MUST present metrics in this exact order:
1. **Classification performance:** Top-1 Accuracy and Top-5 Accuracy
2. **Object localization:** Intersection over Union (IoU)
3. **Object detection:** Mean Average Precision (mAP @ 0.5 and mAP @ 0.5:0.95)
4. **Model efficiency:** Model size (MB) and Inference latency (ms/image)
5. **Robustness:** Performance under varying conditions (illumination, noise, resolution)
6. **Confidence:** Prediction confidence calibration and rejection threshold analysis

### D. Application Requirements
1. Responsive Web UI (accessible on both phone and laptop).
2. Drag-and-drop image upload (JPEG/PNG/WEBP).
3. Bounding box overlay on input image with hierarchical labels and confidence scores.
4. Structured prediction breakdown: **Make**, **Model**, **Manufacturing Year**, and **Confidence Score**.
5. Interactive inspection of intermediate classical image-processing results (edge maps, contours, color histograms, masked ROIs).
6. Performance comparison tab contrasting Classical, Deep Learning, and Hybrid pipelines.
7. Sub-10-second end-to-end response time on public hosting.

### E. Mandatory Deliverables & Specific Naming Conventions
- **Live Hosted Application URL** (publicly accessible on mobile and desktop).
- **Public GitHub Repository** containing:
  - Complete, reproducible source code.
  - System Architecture Diagram matching the implementation.
  - Comprehensive Design Document covering all required sections.
  - Comparative Evaluation Results file named strictly: `eval_P5_TeamID.md` (e.g., `eval_P5_T01.md`).
    - *Crucial Formatting Rule:* The first line MUST contain all team USNs comma-separated (e.g., `1RUA24CSE0447, ...`).
    - Metrics MUST follow the exact sequence specified in Section 5 of Problem Statement P5.
  - Ablation study and failure-case analysis report (min. 3 failure cases).

### F. Rubric & Evaluation Breakdown (100 Points Total)
- **Demo and Viva:** **25 points** (Top band: 21–25). 3-minute strict live presentation on public URL, handling judge's unseen image, individual defense by every member.
- **Evaluation Metrics:** **25 points** (Top band: 12–15). Comparative benchmarks among top percentiles.
- **Implementation:** **20 points** (Top band: 12–15). Unseen input handling, proper train/validation split, clean modular code, bad input handling.
- **UI/UX and Deployment:** **15 points** (Top band: 12–15). Public URL responds <10s on phone/laptop, clear loading/error states, reproducible with one command or Dockerfile, no secrets committed.
- **Project Documentation:** **15 points** (Top band: 12–15). Architecture diagram matches code, design doc has all sections, README reproduces results, limitations/ethics stated honestly.
- **Deductions:** Late submission: -5 points per 15 minutes.
- **Caps for Missing Deliverables:** No live URL caps UI/UX at 0; no GitHub link caps Implementation at 0; no diagram/design doc caps Documentation at 5.
- **Tie-Break Priority:** 1st priority: Demo & Viva score; 2nd priority: Implementation score.

### G. Strict Non-Requirements
- **No License Plate Recognition (ALPR):** Explicitly excluded on Page 13: *"It should not be assumed to provide number-plate recognition, vehicle tracking or traffic-event annotations. These should not be mandatory requirements for this challenge."*

---

## 3. EXISTING REPOSITORY STRUCTURE & ASSETS

### 3.1 Workspace Directory Tree (`D:\CV`)
```
D:\CV\
└── Stanford Dataset\
    ├── Stanford_Cars.zip                                 (1,959,400,431 bytes ~ 1.82 GB)
    ├── cars_train\
    │   └── cars_train\                                   (8,144 .jpg files, 78x57 to 5616x3744)
    ├── cars_test\
    │   └── cars_test\                                    (8,041 .jpg files, 78x41 to 7800x5400)
    └── car_devkit\
        └── devkit\
            ├── cars_meta.mat                             (3,177 bytes - 196 class labels)
            ├── cars_train_annos.mat                      (187,916 bytes - 8,144 annotations with labels)
            ├── cars_test_annos.mat                       (155,831 bytes - 8,041 annotations without labels)
            ├── eval_train.m                              (1,248 bytes - MATLAB evaluation script)
            ├── README.txt                                (1,654 bytes - Official devkit documentation)
            └── train_perfect_preds.txt                   (28,123 bytes - Sample predictions)
```

### 3.2 External Unseen Image Corpus
- **Location:** `C:\Users\shrey\Favorites\Downloads\carz`
- **File Count:** 74 files (73 `.jpg` + 1 `.png` `Untitled design.png`).
- **Critical Finding:** Stanford Cars classes only cover model years **1991 to 2012** across 196 specific models. The `carz` folder contains post-2012 vehicles (e.g., `2013-Audi-R8`, `2014-McLaren-P1`, `2015-Audi-RS-Q3`, `2015-Land-Rover-Discovery-Sport`, `2014-Skoda-Octavia-RS`). These models do **NOT** exist in the 196 Stanford Cars classes.
- **Evaluation Purpose:** These images will be used strictly for:
  1. Multi-vehicle detection localization testing (IoU / bounding box sanity).
  2. Out-of-Distribution (OOD) rejection threshold validation (ensuring the model flags unknown/unsupported cars instead of making confident erroneous predictions).
  3. Interactive web UI demo testing during the live viva.

---

## 4. EXISTING IMPLEMENTATION STATUS & AUDIT FINDINGS

### 4.1 Component Status Matrix

| Component | Path | Current Status | Issues / Deficiencies | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| **Git Repository** | `D:\CV\.git` | **Missing** | Untracked workspace; no commit history. | Initialize git; create comprehensive `.gitignore` excluding dataset zips, raw images, `.venv`, `.mat`, and checkpoint files. |
| **Class Metadata** | `car_devkit\devkit\cars_meta.mat` | Valid | 196 class strings stored as MATLAB cell array. | Parse into structured JSON/CSV dictionary mapping `class_id -> {make, model, year, body_type}`. |
| **Train Annotations** | `car_devkit\devkit\cars_train_annos.mat` | Valid | 8,144 annotations with `bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname`. Coordinates are 1-based MATLAB pixels. | Convert to 0-based Python coordinates ($x_1 = \text{bbox\_x1} - 1$) and export to YOLO format and PyTorch annotation index. |
| **Test Annotations** | `car_devkit\devkit\cars_test_annos.mat` | Incomplete for offline evaluation | Contains bounding boxes and filenames, but **class field is absent** (official competition test set). | Implement stratified 80/20 train/validation split (6,515 train / 1,629 val) from labeled images to calculate Top-1/Top-5 accuracy. |
| **Train Images** | `cars_train\cars_train\` | Valid (8,144 files) | (1) Redundant nested folder structure. (2) **18 images are single-channel grayscale ('L' mode)**. | Retain; add `image.convert('RGB')` guard in all data loaders; simplify path references. |
| **Test Images** | `cars_test\cars_test\` | Valid (8,041 files) | (1) Redundant nested folder structure. (2) **16 images are single-channel grayscale ('L' mode)**. | Retain; guard against grayscale during inference testing. |
| **Raw Archive** | `Stanford Dataset\Stanford_Cars.zip` | Redundant archive | Consumes 1.82 GB in active workspace. | Retain as backup, exclude via `.gitignore`. |
| **Application Code** | None | **Missing** | No frontend, backend, models, or evaluation scripts exist. | Implement modular structure in Phase 1+. |

### 4.2 Verified Ground Truth Dataset Statistics
- **Total Classes:** 196
- **Unique Makes:** 49 (e.g., `Aston Martin`, `Bentley`, `BMW`, `Bugatti`, `Ferrari`, `Lamborghini`, `Mercedes-Benz`, `Rolls-Royce`, `Tesla`)
- **Unique Model Years:** 16 non-contiguous years: `[1991, 1993, 1994, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012]`
- **Class Representation:**
  - Minimum: **Hyundai Accent Sedan 2012** (Class 136) with **24 images**.
  - Maximum: **GMC Savana Van 2012** (Class 119) with **68 images**.
  - Average: 41.55 images/class.
- **Grayscale Images:** Exactly **34 images** across the dataset (18 in `cars_train`, 16 in `cars_test`).
- **Coordinate Indexing:** MATLAB 1-based indexing confirmed (`min bbox_x1 = 1, min bbox_y1 = 1`). Converting to 0-based Python indices requires subtracting 1: $x_1 = \text{bbox\_x1} - 1, y_1 = \text{bbox\_y1} - 1, w = \text{bbox\_x2} - x_1, h = \text{bbox\_y2} - y_1$.

### 4.3 Hardware & Python Environment Audit
- **CPU:** 13th Gen Intel Core i7-13650HX (14 cores: 6 P-cores + 8 E-cores, 20 logical threads).
- **RAM:** 16.0 GB DDR5 (15.78 GB visible, ~3.2 GB currently available).
- **GPU:** **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM)**.
  - Driver: `616.92`, WDDM 3.1
  - CUDA Compiler (`nvcc`): Installed (CUDA 13.2 / 13.4 driver API)
  - Dedicated VRAM: 6,144 MiB (5,260 MiB currently free).
- **Disk Storage:** Drive `D:\` has **115.45 GB free** (ample space for PyTorch weights, checkpoints, and ONNX models).
- **Python Installations:**
  - Python 3.14.4 (`C:\Python314\python.exe` - default PATH, CPU-only torch, unstable PyTorch/OpenCV wheels for Windows).
  - Python 3.11.9 (`C:\Users\shrey\AppData\Local\Programs\Python\Python311\python.exe` - target base Python, scientific packages pre-installed).
  - **Bottleneck Identified:** Default global PyTorch installations are CPU-only (`torch 2.14.0+cpu`).
  - **Remediation:** Create a dedicated `.venv` using Python 3.11.9 and install CUDA-accelerated PyTorch (`cu121`).

---

## 5. OUTSTANDING QUESTIONS & RESOLVED DECISIONS

### 5.1 Confirmed Decisions
- **Lead Developer:** Shreyash Annegowda (USN: `1RUA24CSE0447`, RV University).
- **Deployment Platform & Frontend:** **Streamlit deployed to Streamlit Community Cloud** (user confirmed; native Python, sub-10s latency, interactive filters and comparison tabs).
- **Model Training Execution:** **Local GPU training on NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)** using FP16 Automatic Mixed Precision (AMP) and batch size 16–32 (user confirmed).
- **Existing Code Preservation:** Clean slate; no existing code files exist in `D:\CV`.

### 5.2 Configurable Placeholders for Phase 1 Deliverables
- **Team Identifier:** Assigned as `eval_P5_T01.md` (configurable to official hackathon team code once issued).
- **Teammates:** Lead USN `1RUA24CSE0447` recorded as primary; comma-separated header line in `eval_P5_TeamID.md` formatted to accept additional team members if assigned.

---

## 6. INITIAL TECHNICAL RISKS & MITIGATION STRATEGY

| Risk Description | Severity | Impact | Architectural Mitigation |
| :--- | :---: | :--- | :--- |
| **VRAM Ceiling (6 GB RTX 3050)** | Medium | Out-of-memory (OOM) during fine-tuning of large models. | Select compact, parameter-efficient backbones (**ConvNeXt-Tiny** or **EfficientNet-B0**). Use PyTorch FP16 Automatic Mixed Precision (`torch.amp.autocast`), gradient accumulation (physical batch size 16, accumulation steps 2), and pin memory. |
| **Unlabeled Official Test Set** | High | Cannot compute offline Top-1/Top-5 accuracy on `cars_test`. | Partition the 8,144 labeled training images into an 80/20 stratified split (6,515 train / 1,629 val) preserving class frequencies across all 196 classes. This satisfies the rubric's top-band requirement for a rigorous train/val split. |
| **Single-Vehicle Dataset vs Multi-Vehicle Requirement** | High | Stanford Cars images typically feature one vehicle; hackathon mandates detecting multiple vehicles per image. | Utilize a decoupled two-stage pipeline: (1) Single-stage YOLOv8 vehicle detector localizes all $N$ vehicles in the frame; (2) Each cropped vehicle bounding box is independently passed to the fine-grained classifier; (3) Multi-box overlay displays Make/Model/Year and confidence for each vehicle. |
| **Public Cloud CPU Latency (<10s SLA)** | High | Streamlit Community Cloud runs on shared CPU cores; unoptimized deep models may breach 10s timeout. | Optimize inference pipeline: (1) Export classifier to **ONNX Runtime** / apply **INT8 Dynamic Quantization**; (2) Use ultra-lightweight **YOLOv8s** detector (<12M parameters, <45ms CPU inference); (3) Cache preprocessed models in Streamlit using `@st.cache_resource`. |
| **Superficial Hybrid Integration** | Medium | Deductions in evaluation if classical features are merely cosmetic. | Implement true bidirectional hybrid coupling: (1) Classical contour saliency and edge density refine and filter YOLO bounding box proposals; (2) Handcrafted HSV color histograms and GLCM texture vectors are concatenated with deep bottleneck embeddings before the final classification head; (3) Perform a rigorous 3-way ablation study (Classical vs Deep vs Hybrid). |
| **Grayscale Image Pipeline Crashing** | Low | 34 images in dataset are single-channel grayscale (`L` mode), which break 3-channel RGB expectations. | Enforce strict RGB conversion in all data loaders (`image.convert('RGB')` or `cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)`). |
| **Off-by-One MATLAB Bounding Boxes** | Low | Spatial misalignment of bounding box crops. | Apply 1-based to 0-based coordinate transformation ($x_1 = \text{bbox\_x1} - 1, y_1 = \text{bbox\_y1} - 1$). |

---

## 7. PHASE 0.1 SIGN-OFF SUMMARY

Phase 0.1 establishes a complete, unambiguous requirement mapping, verifies all hardware and dataset baseline constraints, and provides a clear architectural trajectory for P5. Proceed to forensic dataset verification (Phase 0.2), environment verification (Phase 0.3), and architectural design (Phase 0.4).

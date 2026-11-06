# Stanford Cars Dataset Forensic Validation Report
**Dataset:** Stanford Cars (Cars196)  
**Generated At:** 2026-11-06 14:14:14  
**Status:** VALID  

---

## 1. Dataset Geometry & Statistics
- **Root Directory:** `D:\CV\Stanford Dataset`
- **Total Images:** **16,185** (Train: 8,144, Test: 8,041)
- **Corrupted Images:** **0**
- **Grayscale Images Detected:** **2** (Requires RGB 3-channel conversion)
- **Resolution Range:** Min `78x58` px, Max `3817x2880` px

## 2. Annotations & Bounding Boxes
- **Annotation Source:** MATLAB `.mat` structs (`cars_train_annos.mat`, `cars_test_annos.mat`, `devkit/cars_meta.mat`)
- **Normalized Schema:** `data/processed/annotations/normalized_annotations.csv`
- **Total Annotation Records:** **16,185**
- **Train Split (Labeled):** **8,144**
- **Test Split (Competition Unlabeled):** **8,041**
- **Bounding Boxes Valid:** **True** (100% within image bounds)
- **Cryptographic Duplicate Pairs:** **0**

## 3. Taxonomy & Class Extraction
- **Total Classes:** **196** (196 fine-grained vehicle categories)
- **Total Manufacturers (Makes):** **49** (Including multi-word brands like Aston Martin, Land Rover, Mercedes-Benz, Rolls-Royce)
- **Unique Vehicle Models:** **189**
- **Make/Model/Year Parsing:** Verified complete without naive string splitting.
- *Notice:* Stanford Cars provides fine-grained image-level class labels and bounding boxes. Make, model, and year are parsed deterministically from class taxonomy.

## 4. Hardware & Runtime Telemetry
- **OS:** Windows 10 (10.0.26200)
- **Python:** 3.11.9 | **PyTorch:** 2.5.1+cu121
- **CPU:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (14 physical, 20 logical cores)
- **RAM:** 2.77 GB available / 15.78 GB total
- **CUDA Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB VRAM, CUDA 12.1)

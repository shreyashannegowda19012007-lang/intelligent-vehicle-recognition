# CV HACKATHON 2026 – PROBLEM STATEMENT P5
## PHASE 0.2: STANFORD CARS DATASET FORENSIC AUDIT REPORT

**Dataset Candidate:** Stanford Cars (Cars196)  
**Source Mirror:** [Kaggle: eduardo4jesus/stanford-cars-dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)  
**Local Candidate Path:** `D:\CV\Stanford Dataset`  
**Lead Dataset Engineer & Computer Vision Data Scientist:** Antigravity AI  
**Audit Status:** Fully Verified & Audited (Read-Only Forensic Evaluation)  

---

## 1. LOCATE THE EXTRACTED DATASET

### 1.1 Candidate Identification
A recursive discovery across `D:\CV` identified exactly one candidate dataset distribution:
- **Location:** `D:\CV\Stanford Dataset`
- **Compressed Archive:** `D:\CV\Stanford Dataset\Stanford_Cars.zip` (1,959,400,431 bytes / ~1.825 GB)
- **Extracted Root:** `D:\CV\Stanford Dataset`
- **Extracted Payloads:**
  - `cars_train\cars_train\`: 8,144 JPEG images
  - `cars_test\cars_test\`: 8,041 JPEG images
  - `car_devkit\devkit\`: 3 `.mat` annotation files, 1 MATLAB evaluation script, 2 `.txt` files
- **Derived Workspace Assets:** The workspace also contains derived artifacts generated during Phase 1 under `D:\CV\data\` (`classes.json`, `train_annotations.json`, `val_annotations.json`, and `data/yolo/`).
- **Archive Completeness Check:**
  - Total entries inside `.zip`: 16,191 files (1,990,032,852 uncompressed bytes).
  - Total extracted files on disk: 16,191 files (+ 1 `.zip` archive).
  - Delta: 0 missing files, 0 corrupted archives, 0 byte-size discrepancies.

---

## 2. DATASET STRUCTURE INSPECTION

### 2.1 Recursive Hierarchy
```text
D:\CV\Stanford Dataset\
├── Stanford_Cars.zip                      [1.825 GB compressed archive]
├── cars_train\
│   └── cars_train\                        [8,144 files: 00001.jpg - 08144.jpg]
├── cars_test\
│   └── cars_test\                         [8,041 files: 00001.jpg - 08041.jpg]
└── car_devkit\
    └── devkit\
        ├── README.txt                     [1,654 bytes]
        ├── cars_meta.mat                  [3,177 bytes]
        ├── cars_train_annos.mat           [187,916 bytes]
        ├── cars_test_annos.mat            [155,831 bytes]
        ├── eval_train.m                   [1,248 bytes]
        └── train_perfect_preds.txt        [28,123 bytes]
```

### 2.2 Organization Paradigm
- **Directory Paradigm:** Flat sequential image directories indexed via external metadata. Images are **not** partitioned into subdirectories by class.
- **Naming Pattern:** 5-digit zero-padded numbers:
  - Training: `00001.jpg` to `08144.jpg`
  - Testing: `00001.jpg` to `08041.jpg`
- **Filename Collision Hazard:** Filenames `00001.jpg` through `08041.jpg` exist in **both** splits. File ingestion pipelines must prepend the folder prefix (`cars_train` vs `cars_test`) to prevent collisions.
- **Double-Nesting Caveat:** The archive created double-nested image folders (`cars_train/cars_train/` and `cars_test/cars_test/`).

---

## 3. IMAGE INTEGRITY VALIDATION

Every single image file (16,185 files) was programmatically loaded, decoded, and analyzed via `PIL.Image` and `cv2.imread`.

### 3.1 Verification Summary
- **Zero-byte files:** 0
- **Corrupted / Unreadable files:** 0
- **Format:** 100.0% JPEG (`image/jpeg`)
- **Color Channels:**
  - `cars_train`: 8,126 RGB images (99.78%), **18 Grayscale (`L` mode) images (0.22%)**
  - `cars_test`: 8,025 RGB images (99.80%), **16 Grayscale (`L` mode) images (0.20%)**
  - **Total Grayscale:** Exactly 34 images.
- **Grayscale Roster:**
  - *Train (18):* `00266.jpg`, `01085.jpg`, `02176.jpg`, `03048.jpg`, `03439.jpg`, `03469.jpg`, `03539.jpg`, `04577.jpg`, `04848.jpg`, `05177.jpg`, `05502.jpg`, `05713.jpg`, `06947.jpg`, `07383.jpg`, `07693.jpg`, `07774.jpg`, `08137.jpg`, `08144.jpg`.
  - *Test (16):* `00034.jpg`, `00545.jpg`, `00726.jpg`, `01569.jpg`, `01917.jpg`, `02408.jpg`, `03216.jpg`, `03246.jpg`, `04006.jpg`, `04209.jpg`, `04288.jpg`, `04316.jpg`, `04858.jpg`, `05635.jpg`, `06912.jpg`, `07043.jpg`.
  - *Impact:* Passing single-channel tensors into CNN/ViT architectures triggers shape mismatch exceptions. The loader must call `img.convert('RGB')` at load time (implemented and verified in `src/data/dataset.py:88`).

### 3.2 Resolution and Aspect Ratio Distribution

| Split | Metric | Width ($W$) | Height ($H$) | Aspect Ratio ($W/H$) | Exemplar File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cars_train**<br>(8,144 imgs) | Min | 78 px | 57 px | 0.5931 (Extreme Portrait) | `03423.jpg` (650x1096) |
| | Max | 5,616 px | 3,744 px | 3.8875 (Extreme Panorama) | `02344.jpg` (1279x329) |
| | Mean | 699.82 px | 482.75 px | 1.4736 | — |
| | Median | 637.00 px | 424.00 px | 1.3485 | — |
| **cars_test**<br>(8,041 imgs) | Min | 78 px | 41 px | 0.5967 (Extreme Portrait) | `01829.jpg` (179x300) |
| | Max | 7,800 px | 5,400 px | 3.6064 (Extreme Panorama) | `03189.jpg` (898x249) |
| | Mean | 701.18 px | 483.75 px | 1.4708 | — |
| | Median | 640.00 px | 426.00 px | 1.3402 | — |

---

## 4. ANNOTATION FORENSIC ANALYSIS

### 4.1 Annotation Files & Schemas
Inspected via `scipy.io.loadmat`:
1. `cars_meta.mat`:
   - Contains cell array `'class_names'` of shape `(1, 196)`.
   - Maps 1-based class ID $i \in [1, 196]$ to canonical vehicle strings (e.g. `'AM General Hummer SUV 2000'`).
2. `cars_train_annos.mat`:
   - Struct array `'annotations'` of shape `(1, 8144)`.
   - Fields: `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `class`, `fname`.
   - `class`: integers in $[1, 196]$.
   - `fname`: string matching `00001.jpg` to `08144.jpg`.
3. `cars_test_annos.mat`:
   - Struct array `'annotations'` of shape `(1, 8041)`.
   - Fields: `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `fname`.
   - **CRITICAL OMISSION:** Field `class` is **COMPLETELY ABSENT**.
4. `train_perfect_preds.txt`:
   - 8,144 lines matching `cars_train_annos.mat` ground-truth classes with **100.0% exact agreement**.
5. `eval_train.m`:
   - MATLAB evaluation script demonstrating the benchmark evaluation protocol.
6. `README.txt`:
   - Official devkit documentation confirming that test labels were withheld per the 2013 competition design.

### 4.2 Test Label Recovery Status
- **Direct Match Recovery (19 images):** 19 test images match training set images byte-for-byte; their labels are known from `cars_train_annos.mat`.
- **Full Test Recovery (8,041 images):** The withheld competition labels were released academically in `cars_test_annos_withlabels.mat` (used by torchvision and Hugging Face `cars196`). However, they are absent from Eduardo Reis's Kaggle mirror.

---

## 5. CLASS MAPPING VALIDATION

### 5.1 Hierarchical Decomposition
Every class follows the template: `[Make] [Model / Trim / Sub-brand] [Year]`
- **Years (16):** 100% of the 196 classes terminate with a 4-digit year: `1991, 1993, 1994, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012`.
- **Manufacturers (49 Unique Makes):**
  - *Multi-Word Makes (3):* `AM General`, `Aston Martin`, `Land Rover`.
  - *Hyphenated Makes (2):* `Mercedes-Benz`, `Rolls-Royce`.
  - *Lowercase Make (1):* `smart`.
  - *Full Roster:* AM General, Acura, Aston Martin, Audi, BMW, Bentley, Bugatti, Buick, Cadillac, Chevrolet, Chrysler, Daewoo, Dodge, Eagle, FIAT, Ferrari, Fisker, Ford, GMC, Geo, HUMMER, Honda, Hyundai, Infiniti, Isuzu, Jaguar, Jeep, Lamborghini, Land Rover, Lincoln, MINI, Maybach, Mazda, McLaren, Mercedes-Benz, Mitsubishi, Nissan, Plymouth, Porsche, Ram, Rolls-Royce, Scion, Spyker, Suzuki, Tesla, Toyota, Volkswagen, Volvo, smart.
- **Body Styles (14 Canonical Styles):** `SuperCab`, `Regular Cab`, `Extended Cab`, `Crew Cab`, `Quad Cab`, `Club Cab`, `Convertible`, `Coupe`, `Sedan`, `Hatchback`, `SUV`, `Minivan`, `Van`, `Wagon`.
- **Implicit Overrides (14 Models):** Performance trims without explicit body keywords in their raw names require deterministic mapping (e.g., `Acura TL Type-S 2008` $\rightarrow$ Sedan, `Corvette ZR1 2012` $\rightarrow$ Coupe, `Chrysler 300 SRT-8 2010` $\rightarrow$ Sedan).

---

## 6. CLASS DISTRIBUTION ANALYSIS

```text
Total Training Images: 8,144
Total Unique Classes:  196
Missing Classes:       0
Duplicate Class Names: 0
Minimum Class Count:   24   (Class 136: Hyundai Accent Sedan 2012)
Maximum Class Count:   68   (Class 119: GMC Savana Van 2012)
Mean Class Count:      41.551
Median Class Count:    42.000
Standard Deviation:    4.323
Imbalance Ratio:       2.833 (Max / Min = 68 / 24)
```
- **Quartiles:** Q1 = 39.75, Median = 42.0, Q3 = 44.0 (IQR = 4.25).
- **Assessment:** Exceptionally well balanced for a 196-class dataset. 50% of classes contain between 40 and 44 images. Standard Cross-Entropy loss will train stably without class re-weighting.

---

## 7. BOUNDING-BOX VALIDATION

### 7.1 Geometry & Integrity
- All 16,185 bounding boxes (8,144 train + 8,041 test) were verified:
  - $1 \le x_1 \le x_2 \le W$ and $1 \le y_1 \le y_2 \le H$ (MATLAB 1-based inclusive pixel indices).
  - Degenerate / zero-area boxes: 0.
  - Out of bounds: 0.
- **Conversion to Continuous YOLO Coordinates:**
  $$x_{\text{center}} = \frac{x_1 - 1 + x_2}{2W}, \quad y_{\text{center}} = \frac{y_1 - 1 + y_2}{2H}, \quad w = \frac{x_2 - x_1 + 1}{W}, \quad h = \frac{y_2 - y_1 + 1}{H}$$
  - Round-trip accuracy verified across all 8,144 boxes: **0 pixel mismatches**.

### 7.2 Foreground Area Coverage (Corrected Formula)
- **Train:** Min = 4.28%, Max = **100.0%** (full image), Mean = **54.82%**, Median = **54.60%**.
- **Test:** Min = 3.18%, Max = **100.0%** (full image), Mean = **55.25%**, Median = **54.79%**.

---

## 8. TRAIN/VALIDATION/TEST FEASIBILITY & DUPLICATE DEFECTS

An exhaustive cryptographic MD5 hash audit across all 16,185 files discovered **39 duplicate pairs (78 images)** across the dataset:

### 8.1 Duplicate Breakdown
1. **19 Cross-Split Duplicate Pairs (Train $\leftrightarrow$ Test):** Exactly identical image bytes exist in both `cars_train` and `cars_test` (e.g., `08130.jpg` == `07855.jpg`, `00114.jpg` == `07699.jpg`, `02967.jpg` == `02449.jpg`). Evaluating on these 19 test images measures memorization rather than generalization (+0.24% evaluation bias).
2. **8 Internal Test Set Duplicate Pairs:** Within `cars_test`, 8 duplicate pairs (16 images) exist with identical bounding boxes (e.g., `00288.jpg` == `06136.jpg`, `01236.jpg` == `05844.jpg`).
3. **12 Internal Training Duplicate Pairs with Conflicting Labels:** Within `cars_train`, 12 duplicate pairs exist. Every pair is labeled as two different classes (e.g., `00050.jpg` Bentley 2012 vs `03436.jpg` Bentley 2007; `00151.jpg` Acura TSX vs `04892.jpg` Acura RL; `02045.jpg` Rolls-Royce Phantom vs `04665.jpg` Rolls-Royce Ghost).
   - *Discovery:* Pair 10 (`04922.jpg` Jeep Liberty vs `05994.jpg` Jeep Patriot) has **divergent bounding box coordinates** (`[276, 83, 654, 352]` vs `[275, 80, 657, 351]`), proving independent human annotation of duplicate web crawls.

### 8.2 Active Pipeline Train/Val Contamination
In the existing project split in `D:\CV\data`, a naive stratified 80/20 split partitioned **5 of the 12 conflicting duplicate pairs across the train/val boundary**:
- `00050.jpg` (VAL) $\leftrightarrow$ `03436.jpg` (TRAIN)
- `00595.jpg` (TRAIN) $\leftrightarrow$ `03516.jpg` (VAL)
- `02045.jpg` (VAL) $\leftrightarrow$ `04665.jpg` (TRAIN)
- `02843.jpg` (TRAIN) $\leftrightarrow$ `06839.jpg` (VAL)
- `04223.jpg` (TRAIN) $\leftrightarrow$ `06153.jpg` (VAL)

*Correction:* A group-aware split (grouping duplicate image hashes into the same split partition) is required before final model training to eliminate artificial validation penalties.

---

## 9. KAGGLE SOURCE VERIFICATION

- **URL:** `https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset`
- **Uploader:** Eduardo Reis (`eduardo4jesus`).
- **Archive Size:** ~1.825 GB compressed.
- **License:** Unknown / Academic Research.
- **Upstream Provenance:** Krause et al., ICCV 2013, Stanford AI Lab.
- **Fidelity:** Exact byte-for-byte re-upload of the original Stanford release tarballs, which is why official test set labels are withheld.

---

## 10. MULTI-AGENT SYNTHESIS

- **Agent A (Images):** 16,185 files 100% verified. 0 corrupted, 34 grayscale, extreme resolutions cataloged.
- **Agent B (Annotations):** Validated MAT structs. Verified 196 training classes. Confirmed lack of test labels in `cars_test_annos.mat`.
- **Agent C (Metadata):** Parsed 49 Makes, 14 Body Styles, 16 Years. Cataloged multi-word brands and implicit body style overrides.
- **Agent D (Quality):** Balanced class distribution (2.833 ratio). Uncovered 39 duplicate pairs and the 5-pair train/val cross-split leak.

---

## 11. REMAINING QUESTIONS & GAPS

1. **Test Set Evaluation:** Official test set labels are withheld in `cars_test_annos.mat`. Unlocking full 16,185-image evaluation requires community-recovered labels (`cars_test_annos_withlabels.mat`).
2. **Duplicate Remediation Policy:** The project team should decide whether to prune the 12 conflicting training duplicate pairs or absorb them using label smoothing ($\epsilon=0.1$).
3. **Train/Val Split Fix:** Group duplicate image hashes together to eliminate the 5 cross-split leaked pairs.

---

## 12. FINAL AUDIT DECISION

### **FINAL VERDICT: C. USABLE WITH LIMITATIONS**
*(Upgradable to **Verdict B: Usable with annotation recovery** upon recovery of test labels)*

**Evidence-Based Justification:**
1. **Core Viability:** The 8,144 labeled training images across 196 classes are 100% uncorrupted and exceptionally well balanced (imbalance ratio 2.833:1).
2. **Annotation Soundness:** Bounding boxes are 100% valid and verified with 0 round-trip pixel mismatches.
3. **Limitation 1 (Test Set):** Ground truth labels for the 8,041 test images are missing from this Kaggle mirror, making offline test classification accuracy calculation impossible without annotation recovery.
4. **Limitation 2 (Data Contamination):** 19 cross-split duplicates, 12 conflicting train duplicate pairs, and 5 train/val split leakage pairs.
5. **Phase Approval:** Approved for development using the internal stratified 80/20 train/validation split.

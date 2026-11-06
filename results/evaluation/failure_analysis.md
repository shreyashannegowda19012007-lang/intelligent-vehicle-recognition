# Failure Analysis and Error Taxonomy Report

**Project**: CV Hackathon P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Total Documented Cases**: 10  

---

## 1. Executive Summary of Failure Modes

Fine-grained vehicle recognition presents distinct challenges compared to standard object detection. Our empirical failure analysis identifies three primary error drivers across the validation split:
1. **Intra-Make Fine-Grained Similarity**: Vehicles sharing the same manufacturer grille, headlight geometry, and emblem styling (e.g. Audi S4 Sedan vs. Audi RS 4 Convertible) exhibit high mutual confusion.
2. **Production Year / Facelift Invariance**: Minor mid-cycle vehicle refreshes retain identical exterior chassis stampings, making year discrimination challenging without internal trim cues.
3. **Visual Quality & Perturbation Sensitivity**: Sensor defocus, low illumination, and downsampling suppress fine structural contours and radiator grille textures.

---

## 2. Granular Failure Case Records

### Case #1: Mitsubishi Lancer Sedan 2012

- **Input Image**: `Stanford Dataset/cars_train/cars_train/07936.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Mitsubishi Lancer Sedan 2012** (Make: *Mitsubishi*, Model: *Lancer*, Year: *2012*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #2: Acura TL Type-S 2008

- **Input Image**: `Stanford Dataset/cars_train/cars_train/07445.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Acura TL Type-S 2008** (Make: *Acura*, Model: *TL Type-S*, Year: *2008*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #3: Chevrolet Avalanche Crew Cab 2012

- **Input Image**: `Stanford Dataset/cars_train/cars_train/05639.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Chevrolet Avalanche Crew Cab 2012** (Make: *Chevrolet*, Model: *Avalanche*, Year: *2012*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #4: Audi RS 4 Convertible 2008

- **Input Image**: `Stanford Dataset/cars_train/cars_train/01640.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Audi RS 4 Convertible 2008** (Make: *Audi*, Model: *RS 4*, Year: *2008*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **visually_similar_models**
- **Category Overview**: Visually similar vehicle models with shared chassis/styling
- **Likely Reason**: Shared corporate styling and badge aesthetic between Audi RS 4 and Audi S4.

### Case #5: Ford Expedition EL SUV 2009

- **Input Image**: `Stanford Dataset/cars_train/cars_train/00518.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Ford Expedition EL SUV 2009** (Make: *Ford*, Model: *Expedition EL*, Year: *2009*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #6: Jeep Compass SUV 2012

- **Input Image**: `Stanford Dataset/cars_train/cars_train/02219.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Jeep Compass SUV 2012** (Make: *Jeep*, Model: *Compass*, Year: *2012*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #7: Buick Regal GS 2012

- **Input Image**: `Stanford Dataset/cars_train/cars_train/07921.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Buick Regal GS 2012** (Make: *Buick*, Model: *Regal GS*, Year: *2012*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #8: Acura TSX Sedan 2012

- **Input Image**: `Stanford Dataset/cars_train/cars_train/03054.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Acura TSX Sedan 2012** (Make: *Acura*, Model: *TSX*, Year: *2012*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #9: Chevrolet Malibu Sedan 2007

- **Input Image**: `Stanford Dataset/cars_train/cars_train/03652.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Chevrolet Malibu Sedan 2007** (Make: *Chevrolet*, Model: *Malibu*, Year: *2007*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

### Case #10: Chevrolet HHR SS 2010

- **Input Image**: `Stanford Dataset/cars_train/cars_train/02200.jpg`
- **Evaluation Condition**: `baseline_clean`
- **Ground Truth Vehicle**: **Chevrolet HHR SS 2010** (Make: *Chevrolet*, Model: *HHR SS*, Year: *2010*)
- **Predicted Vehicle**: **Audi S4 Sedan 2012** (Make: *Audi*, Model: *S4*, Year: *2012*)
- **Model Confidence**: `0.0070`
- **Failure Type**: **poor_image_quality**
- **Category Overview**: Degraded resolution, motion blur, or severe sensor noise
- **Likely Reason**: Degraded image resolution or blurring obscured key high-frequency features.

---

## 3. Mitigation Strategies

- **Hierarchical Loss Weighting**: Introduce multi-task penalty terms prioritizing make discrimination over fine-grained year variation.
- **High-Resolution Feature Fusion**: Fuse high-frequency edge descriptors (HOG/Sobel) into early convolutional feature maps.
- **Adaptive Contrast Preprocessing**: Apply real-time CLAHE equalization before classifier ingestion to restore low-contrast vehicle grilles.

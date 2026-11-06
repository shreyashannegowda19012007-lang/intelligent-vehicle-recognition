# Hybrid Intelligence Failure Analysis Report

## Executive Summary

This report documents systematic edge-case failures across the end-to-end vehicle recognition pipeline.
Each failure case is analyzed to pinpoint whether the defect originated during **detection**, **cropping**, 
**fine-grained classification**, or **hybrid decision fusion**.

### failure_01_severe_blur_feature_loss: Classification Feature Degradation

- **Failure Origin Stage:** `classification`
- **Likely Technical Cause:** Severe Gaussian blur removed high-frequency gradient signatures (grille mesh, headlight facets, badge contours), collapsing top-1 classification confidence.
- **Ground Truth:** Audi TTS 2012 (`Audi TTS Coupe 2012`)
- **Classical ML Prediction:** Unknown Unknown 0 (Score: 0.0)
- **Deep Learning Prediction:** GMC Savana 2012 (Confidence: 0.2469)
- **Final Hybrid Prediction:** GMC Savana 2012 (Confidence: 0.2469)
- **Bounding Box Coordinates:** `x1=26, y1=224, x2=807, y2=740`
- **Saved Artifacts:** `[outputs/hybrid/failure_cases/failure_01_severe_blur_feature_loss_image.jpg]`, `[outputs/hybrid/failure_cases/failure_01_severe_blur_feature_loss_crop.jpg]`

### failure_02_partial_vehicle_occlusion: Partial Occlusion / Bounding Box Truncation

- **Failure Origin Stage:** `cropping`
- **Likely Technical Cause:** Synthetic foreground obstruction obscured 45% of vehicle body, preventing the detector from capturing canonical vehicle contours and causing partial ROI cropping.
- **Ground Truth:** Audi TTS 2012 (`Audi TTS Coupe 2012`)
- **Classical ML Prediction:** Unknown Unknown 0 (Score: 0.0)
- **Deep Learning Prediction:** Bentley Continental GT 2007 (Confidence: 0.7059)
- **Final Hybrid Prediction:** Bentley Continental GT 2007 (Confidence: 0.7059)
- **Bounding Box Coordinates:** `x1=22, y1=230, x2=809, y2=858`
- **Saved Artifacts:** `[outputs/hybrid/failure_cases/failure_02_partial_vehicle_occlusion_image.jpg]`, `[outputs/hybrid/failure_cases/failure_02_partial_vehicle_occlusion_crop.jpg]`

### failure_03_extreme_underexposure: Low Illumination & Contrast Reduction

- **Failure Origin Stage:** `detection`
- **Likely Technical Cause:** Extreme darkness suppressed contrast below detector activation threshold, resulting in lowered detection confidence and localized ROI boundary shrinkage.
- **Ground Truth:** Audi TTS 2012 (`Audi TTS Coupe 2012`)
- **Classical ML Prediction:** Unknown Unknown 0 (Score: 0.0)
- **Deep Learning Prediction:** MINI Cooper Roadster 2012 (Confidence: 0.0314)
- **Final Hybrid Prediction:** MINI Cooper Roadster 2012 (Confidence: 0.0314)
- **Bounding Box Coordinates:** `x1=23, y1=231, x2=808, y2=755`
- **Saved Artifacts:** `[outputs/hybrid/failure_cases/failure_03_extreme_underexposure_image.jpg]`, `[outputs/hybrid/failure_cases/failure_03_extreme_underexposure_crop.jpg]`

### failure_04_fine_grained_model_confusion: Visually Similar Vehicle Make/Model Ambiguity

- **Failure Origin Stage:** `classification`
- **Likely Technical Cause:** Subtle inter-model visual distinctions between consecutive production years and closely related trim packages became indiscernible after severe resolution reduction.
- **Ground Truth:** Audi TTS 2012 (`Audi TTS Coupe 2012`)
- **Classical ML Prediction:** None None 0 (Score: 0.0)
- **Deep Learning Prediction:** None None 0 (Confidence: 0.0)
- **Final Hybrid Prediction:** None None 0 (Confidence: 0.0)
- **Bounding Box Coordinates:** `x1=0, y1=0, x2=0, y2=0`
- **Saved Artifacts:** `[outputs/hybrid/failure_cases/failure_04_fine_grained_model_confusion_image.jpg]`, `[outputs/hybrid/failure_cases/failure_04_fine_grained_model_confusion_crop.jpg]`

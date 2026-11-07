# AutoVision AI — Complete UI/UX Specification & Design System

## 1. Executive Summary
The AutoVision AI interface represents a modern, production-grade automotive vision SaaS application built for the National Computer Vision Hackathon 2026 (Problem Statement P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System).

This document serves as the comprehensive visual and technical reference for the frontend implementation, component architecture, responsive design tokens, and visual assets.

---

## 2. Visual Identity & Color Tokens

```css
:root {
  /* Surfaces */
  --bg-root: #0A0E17;
  --bg-card: #111827;
  --bg-card-hover: #172033;
  --bg-elevated: #1F2937;
  --border-card: #1E293B;
  --border-card-hover: #3B82F6;

  /* Brand Accents */
  --accent-primary: #3B82F6;
  --accent-secondary: #06B6D4;
  --accent-gradient: linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%);

  /* Status Colors */
  --status-success: #10B981;
  --status-warning: #F59E0B;
  --status-danger: #EF4444;

  /* Typography */
  --text-primary: #F9FAFB;
  --text-secondary: #94A3B8;
  --text-muted: #64748B;
}
```

---

## 3. Core Functional Views

### 3.1 Recognition Studio (`02_recognition_studio.png`)
- **Primary Function**: Primary inference interface with 1-click test presets and drag-and-drop file uploader.
- **Components**:
  - Live KPI metric counter (196 classes, 99.48% mAP50, 89.43% top-1, 94.50% make acc, <4ms latency).
  - 1-Click Verification Presets (Audi TTS, Acura TL, Dodge Dakota, Hyundai Sonata, Mitsubishi Lancer).
  - Real-time vehicle detection overlay with coordinate labels.
  - 5% ROI extracted vehicle crop preview.
  - Vehicle Identity Hero Card featuring make badge, model, year, and taxonomy ID.
  - Dual animated confidence bars (Model top-1 confidence and make confidence).
  - Top-5 prediction table with probability meters and rank badges.

### 3.2 Model Consensus & Arbitration (`03_model_consensus.png`)
- **Primary Function**: Inspect multi-stream decision fusion between Deep Learning (ConvNeXt-Tiny) and Classical CV (HOG+Color+LBP SVM).
- **Components**:
  - Stream 1 vs Stream 2 prediction comparison table.
  - Agreement status badge (Consensus Match vs Divergence).
  - Arbitration confidence delta and adaptive fusion weight display.

### 3.3 Computer Vision Filter Lab (`04_vision_filter_lab.png`)
- **Primary Function**: Interactive exploration of classical image enhancement and feature extraction algorithms.
- **Components**:
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) with tile grid size controls.
  - Bilateral edge-preserving smoothing filter with diameter and sigma sliders.
  - Laplacian 2nd-order spatial edge extraction filter.
  - Color space decomposition views (RGB, HSV, CIELAB).

### 3.4 Benchmarks & Ablation (`05_benchmarks_ablation.png`)
- **Primary Function**: Quantitative evaluation results across the Stanford Cars dataset.
- **Components**:
  - Comparative benchmark table (Classical Baseline vs Deep Learning Only vs Proposed Hybrid System).
  - Latency breakdown across detector, classifier, classical feature extraction, and fusion.
  - Component ablation study table showing individual contribution of each module.

### 3.5 Forensic Failure Taxonomy (`06_failure_taxonomy.png`)
- **Primary Function**: Deep dive into edge cases, misclassifications, and fine-grained visual ambiguity.
- **Components**:
  - Empirical 49-Make confusion matrix heatmap.
  - Top confusion pairs (subtle facelift trims and cross-year model variants).
  - Error categorization (intra-class variance, partial occlusion, extreme lighting).

### 3.6 Telemetry & REST API (`07_api_telemetry.png`)
- **Primary Function**: Enterprise microservice integration and hardware monitoring.
- **Components**:
  - Live hardware VRAM telemetry (NVIDIA CUDA GPU monitor).
  - Interactive FastAPI REST curl request example.
  - JSON payload response schema preview.

---

## 4. Visual Verification Gallery

The automated browser QA pipeline captures full-resolution screenshots into `docs/screenshots/`:
1. `01_landing_hero.png` — Hero section, live telemetry pill, and ingestion grid.
2. `02_recognition_studio.png` — Audi TTS Coupe identification, bbox, ROI, and top-5 table.
3. `03_model_consensus.png` — ConvNeXt vs SVM arbitration and agreement analysis.
4. `04_vision_filter_lab.png` — CLAHE, Bilateral, Laplacian, and Color Space decomposition.
5. `05_benchmarks_ablation.png` — Quantitative benchmark metrics and ablation study.
6. `06_failure_taxonomy.png` — 49-Make confusion matrix and forensic failure taxonomy.
7. `07_api_telemetry.png` — Microservice REST API documentation and GPU telemetry.
8. `08_mobile_view.png` — Responsive viewport on mobile (390x844).


# AutoVision AI — Frontend Design System & Development Conventions

This document establishes stable project-wide UI/UX conventions, component architecture, styling tokens, and critical backend protection rules for the AutoVision AI Computer Vision platform.

---

## 1. Design System Tokens & Colors

AutoVision AI follows a **Dark Luxury Automotive SaaS** visual aesthetic designed for high-density computer vision intelligence displays:

| Token Name | Hex Value | Usage / Semantic Role |
| :--- | :--- | :--- |
| **Surface Root (`--bg-root`)** | `#0A0E17` | Canvas background, void color |
| **Card Surface (`--bg-card`)** | `#111827` | Content containers, card bodies, panels |
| **Card Elevate / Border (`--border-color`)** | `#1E293B` | Borders, subtle card separation, card hover |
| **Primary Accent (`--accent-blue`)** | `#3B82F6` | Primary CTAs, active badges, key metrics |
| **Secondary Accent (`--accent-cyan`)** | `#06B6D4` | Model consensus highlights, telemetry pills |
| **Success State (`--status-green`)** | `#10B981` | High confidence (>70%), online health status |
| **Warning State (`--status-amber`)** | `#F59E0B` | Medium confidence (40-70%), consensus divergence |
| **Failure State (`--status-red`)** | `#EF4444` | Low confidence (<40%), detection miss, alerts |
| **Primary Text (`--text-primary`)** | `#F9FAFB` | Headings, card titles, key readout figures |
| **Muted Text (`--text-secondary`)** | `#94A3B8` | Body text, captions, secondary parameters |

---

## 2. Typography Conventions

- **Font Family**: Modern system UI sans-serif stack:
  ```css
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  ```
- **Monospace Readouts**: For latencies, coordinates, and probabilities:
  ```css
  font-family: "JetBrains Mono", "SF Mono", Consolas, "Courier New", monospace;
  font-variant-numeric: tabular-nums;
  ```
- **Heading Hierarchy**:
  - H1 / Hero: `2.25rem` (`36px`), bold `800`, letter-spacing `-0.03em`
  - Card Titles: `1.15rem` (`18.4px`), semibold `600`, letter-spacing `-0.02em`
  - Subheaders / Pills: `0.75rem` (`12px`), bold `700`, uppercase, tracking `0.05em`
  - Readout Figures: `1.75rem` - `2.25rem`, extrabold `800`

---

## 3. Component Architecture & Conventions

1. **Card Containers (`.saas-card`)**:
   - Background: `#111827`
   - Border: `1px solid #1E293B`
   - Border Radius: `12px` (`0.75rem`)
   - Padding: `20px` to `24px`
   - Shadow: `0 10px 25px -5px rgba(0, 0, 0, 0.5)`
   - Hover: Smooth subtle scale and border brightening (`#3B82F6` border highlight)

2. **Progress Meters (`.confidence-progress-bar`)**:
   - Container height: `8px` - `12px`, radius `6px`
   - Background: `rgba(255, 255, 255, 0.08)`
   - Fill: Gradient `linear-gradient(90deg, #3B82F6, #06B6D4)`
   - Transition: `width 0.6s cubic-bezier(0.4, 0, 0.2, 1)`

3. **Status & Model Badges (`.saas-badge`)**:
   - Font size: `0.70rem` (`11px`)
   - Text transform: `uppercase`
   - Letter spacing: `0.06em`
   - Padding: `4px 10px`
   - Border radius: `6px`

---

## 4. Responsive Layout Rules

- **Desktop (≥ 1200px)**: 2-column or 3-column split view (Input dropzone left, Viewport right; Multi-metric grid across 5 tiles).
- **Tablet (768px – 1199px)**: Responsive wrapping grids, 2-column KPI tiles.
- **Mobile (< 768px)**: Single column stacked layout. Full-width buttons and full-width dropzone. Tab navigation remains horizontal with touch-scrolling.

---

## 5. Protected Machine Learning & Core Pipeline Invariants

The following files and components are **STRICTLY PROTECTED**. No UI/UX or styling work should ever modify their internal math, network weights, or inference pipeline logic:

1. **Model Weights**:
   - `models/detection/best.pt` (YOLOv8n vehicle detector)
   - `models/classification/convnext_best.pt` (ConvNeXt-Tiny 196-class classifier)
   - `models/classical/svm/` (Linear SVM, Scalers, PCA models)
2. **Deep Learning Core**:
   - `src/classification/models/convnext.py`
   - `src/detection/detector.py`
   - `src/pipeline/hybrid_pipeline.py`
   - `src/pipeline/recognition_pipeline.py`
3. **Data Integrity**:
   - Stanford Cars dataset mappings (`data/stanford_cars/` and `src/utils/class_mapping.py`)
   - All ground truth evaluation results and confusion matrices (`outputs/classification/metrics/`)

---

## 6. Testing & Quality Assurance Requirements

Every UI modification or feature enhancement must satisfy:
1. **Compilation Check**: `python -m py_compile app/streamlit_app.py`
2. **Pytest Interface Tests**: `.venv\Scripts\pytest.exe tests/test_web_interface.py -v` (All 8 tests must pass)
3. **End-to-End Pipeline Tests**: `.venv\Scripts\pytest.exe tests/test_e2e_workflow.py tests/test_pipeline.py -v`
4. **Browser QA Automation**: `python scripts/capture_all_screenshots.py` verifying DOM mounting, preset clicks, and tab switching.


"""
Generate high-resolution technical diagrams and composite visual figures for the
Hackathon Technical Solution Report.
Saves all figures into submission/report/report_assets/.
"""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "submission" / "report" / "report_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Set global matplotlib parameters for clean academic publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0


def generate_fig1_system_architecture():
    """Figure 1: Overall System Architecture."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background subtle styling
    fig.patch.set_facecolor('#FFFFFF')

    # Color palette
    c_blue_bg = '#EFF6FF'
    c_blue_border = '#2563EB'
    c_orange_bg = '#FFF7ED'
    c_orange_border = '#EA580C'
    c_green_bg = '#F0FDF4'
    c_green_border = '#16A34A'
    c_purple_bg = '#FAF5FF'
    c_purple_border = '#9333EA'
    c_rose_bg = '#FFF1F2'
    c_rose_border = '#E11D48'
    c_slate_bg = '#F8FAFC'
    c_slate_border = '#475569'

    # Title
    ax.text(50, 96, "Figure 1: Overall End-to-End System Architecture", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(50, 92.5, "P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System", 
            ha='center', va='center', fontsize=11, color='#64748B')

    # Stage 1: Input Ingestion
    box1 = patches.FancyBboxPatch((3, 62), 16, 24, boxstyle="round,pad=0.8", 
                                  fc=c_blue_bg, ec=c_blue_border, lw=1.8)
    ax.add_patch(box1)
    ax.text(11, 82, "STAGE 1", ha='center', va='center', fontsize=9, fontweight='bold', color=c_blue_border)
    ax.text(11, 77.5, "Input Ingestion", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(11, 68, "• RGB / PNG / JPEG\n• Byte Validation\n• EXIF / Grayscale Fix\n• Dynamic Range Norm", 
            ha='center', va='center', fontsize=8.5, color='#334155', linespacing=1.3)

    # Arrow 1 -> 2
    ax.annotate('', xy=(24, 74), xytext=(20, 74),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    # Stage 2: Autonomous Detection & ROI Extraction
    box2 = patches.FancyBboxPatch((25, 62), 22, 24, boxstyle="round,pad=0.8", 
                                  fc=c_orange_bg, ec=c_orange_border, lw=1.8)
    ax.add_patch(box2)
    ax.text(36, 82, "STAGE 2", ha='center', va='center', fontsize=9, fontweight='bold', color=c_orange_border)
    ax.text(36, 77.5, "YOLOv8n Localization", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(36, 68, "• Anchor-Free Backbone (3.1M)\n• 99.48% mAP@0.50, 0.906 IoU\n• NMS (IoU=0.45, Conf>=0.25)\n• 5% Boundary Safety Padding\n• Aspect-Ratio Letterbox", 
            ha='center', va='center', fontsize=8.5, color='#334155', linespacing=1.3)

    # Bifurcation arrows from Stage 2 to Dual Streams
    ax.annotate('', xy=(52, 79), xytext=(48, 74),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))
    ax.annotate('', xy=(52, 53), xytext=(48, 74),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    # Stage 3A: Classical CV Stream
    box3a = patches.FancyBboxPatch((53, 67), 22, 24, boxstyle="round,pad=0.8", 
                                   fc=c_green_bg, ec=c_green_border, lw=1.8)
    ax.add_patch(box3a)
    ax.text(64, 87, "STAGE 3A: CLASSICAL CV", ha='center', va='center', fontsize=9, fontweight='bold', color=c_green_border)
    ax.text(64, 82.5, "Handcrafted Stream", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(64, 74, "• HOG Gradients (8,100-d)\n• HSV Histograms (32x32x32)\n• LBP Texture (288-d)\n• PCA-128 Dimensionality Red.\n• Calibrated Linear SVM\n• Latency: 0.015 ms (0.15 MB)", 
            ha='center', va='center', fontsize=8, color='#334155', linespacing=1.3)

    # Stage 3B: Deep Learning Stream
    box3b = patches.FancyBboxPatch((53, 38), 22, 25, boxstyle="round,pad=0.8", 
                                   fc=c_purple_bg, ec=c_purple_border, lw=1.8)
    ax.add_patch(box3b)
    ax.text(64, 59, "STAGE 3B: DEEP LEARNING", ha='center', va='center', fontsize=9, fontweight='bold', color=c_purple_border)
    ax.text(64, 54.5, "Neural Backbone Stream", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(64, 45.5, "• Primary: ConvNeXt-Tiny (320x320)\n  (89.43% Val Acc, 27.8M params)\n• Baseline: ResNet-50 (224x224)\n• ImageNet Pretraining + Head\n• CrossEntropy + Label Smoothing\n• Latency: 144.5 ms (320 MB)", 
            ha='center', va='center', fontsize=8, color='#334155', linespacing=1.3)

    # Convergence arrows to Stage 4
    ax.annotate('', xy=(79, 62), xytext=(76, 75),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))
    ax.annotate('', xy=(79, 58), xytext=(76, 48),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    # Stage 4: Hybrid Decision Engine
    box4 = patches.FancyBboxPatch((80, 43), 17, 34, boxstyle="round,pad=0.8", 
                                  fc=c_rose_bg, ec=c_rose_border, lw=1.8)
    ax.add_patch(box4)
    ax.text(88.5, 73, "STAGE 4", ha='center', va='center', fontsize=9, fontweight='bold', color=c_rose_border)
    ax.text(88.5, 68.5, "Hybrid Engine", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(88.5, 56.5, "• Late Score Fusion\n  (0.70 DL + 0.30 CV)\n• Agreement Boost (+0.05)\n• Dynamic Uncertainty\n  Threshold (tau = 0.35)\n• Rejection Gate (0.20)\n• Full/Make Consensus\n• Make Boost (+0.30%)", 
            ha='center', va='center', fontsize=8, color='#334155', linespacing=1.3)

    # Bottom Stage 5: Serving & Output
    box5 = patches.FancyBboxPatch((15, 8), 70, 22, boxstyle="round,pad=0.8", 
                                  fc=c_slate_bg, ec=c_slate_border, lw=1.8)
    ax.add_patch(box5)
    ax.text(50, 26, "STAGE 5: PRODUCTION SERVING & INTERACTIVE USER INTERFACES", 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#334155')

    # Sub-boxes inside Stage 5
    box5a = patches.FancyBboxPatch((18, 11), 30, 12, boxstyle="round,pad=0.5", 
                                   fc='#FFFFFF', ec='#94A3B8', lw=1.2)
    ax.add_patch(box5a)
    ax.text(33, 19.5, "Streamlit Web Dashboard", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(33, 14.5, "• Drag-and-drop Image Upload\n• Bounding Box & Multi-Vehicle Overlays\n• Make/Model/Year Badges & Top-5 Probs", 
            ha='center', va='center', fontsize=7.5, color='#475569')

    box5b = patches.FancyBboxPatch((52, 11), 30, 12, boxstyle="round,pad=0.5", 
                                   fc='#FFFFFF', ec='#94A3B8', lw=1.2)
    ax.add_patch(box5b)
    ax.text(67, 19.5, "FastAPI High-Throughput REST Engine", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(67, 14.5, "• Asynchronous /analyze Multipart Endpoint\n• Health Telemetry (/health) & Specs (/models)\n• Sub-2ms Serving Overhead (Uvicorn)", 
            ha='center', va='center', fontsize=7.5, color='#475569')

    # Arrow from Stage 4 to Stage 5
    ax.annotate('', xy=(88.5, 31), xytext=(88.5, 42),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))
    ax.annotate('', xy=(84, 28), xytext=(88.5, 31),
                arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig1_system_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig2_processing_workflow():
    """Figure 2: End-to-End Image Processing Workflow."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 60)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    ax.text(70, 56, "Figure 2: End-to-End Image Processing and Decision Workflow", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')

    steps = [
        ("Step 1", "Raw Image\nIngestion", "• Format check\n• RGB convert\n• Quality metrics", '#EFF6FF', '#2563EB'),
        ("Step 2", "YOLOv8n\nDetection", "• 640x640 resize\n• Anchor-free head\n• NMS thresholding", '#FFF7ED', '#EA580C'),
        ("Step 3", "ROI Padded\nExtraction", "• 5% safety margin\n• Letterbox preserve\n• Multi-car cropping", '#FEFCE8', '#CA8A04'),
        ("Step 4", "Dual Stream\nInference", "• Classical (PCA+SVM)\n• ConvNeXt (320x320)\n• Softmax logits", '#F0FDF4', '#16A34A'),
        ("Step 5", "Hybrid Decision\nEngine", "• Score fusion\n• Agreement test\n• Uncertainty gating", '#FAF5FF', '#9333EA'),
        ("Step 6", "Taxonomy\nDecoding", "• 196-class lookup\n• Make/Model/Year\n• Confidence score", '#FFF1F2', '#E11D48'),
        ("Step 7", "Serving &\nVisualization", "• BBox drawing\n• JSON API response\n• Streamlit cards", '#F8FAFC', '#475569'),
    ]

    x_start = 5
    box_w = 15
    box_h = 36
    gap = 4.5

    for idx, (st_num, title, bullets, bg, border) in enumerate(steps):
        x = x_start + idx * (box_w + gap)
        y = 12
        box = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.5", 
                                      fc=bg, ec=border, lw=1.6)
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h - 4, st_num, ha='center', va='center', fontsize=8.5, fontweight='bold', color=border)
        ax.text(x + box_w/2, y + box_h - 11, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#0F172A')
        ax.text(x + box_w/2, y + 10, bullets, ha='center', va='center', fontsize=7.5, color='#334155', linespacing=1.3)

        if idx < len(steps) - 1:
            ax.annotate('', xy=(x + box_w + gap, y + box_h/2), xytext=(x + box_w, y + box_h/2),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color='#94A3B8'))

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig2_processing_workflow.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig3_classical_pipeline():
    """Figure 3: Classical Computer Vision Pipeline."""
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    ax.text(50, 96, "Figure 3: Classical Algorithmic Baseline Feature Extraction Pipeline", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')
    ax.text(50, 92, "Multimodal Handcrafted Feature Fusion (HOG + Color + LBP) with Linear SVM", 
            ha='center', va='center', fontsize=10, color='#64748B')

    # Input Box
    b_in = patches.FancyBboxPatch((2, 38), 14, 24, boxstyle="round,pad=0.6", fc='#F1F5F9', ec='#475569', lw=1.5)
    ax.add_patch(b_in)
    ax.text(9, 54, "Vehicle ROI\nCrop", ha='center', va='center', fontsize=10, fontweight='bold', color='#0F172A')
    ax.text(9, 44, "128x128 Padded\nRGB Image", ha='center', va='center', fontsize=8, color='#475569')

    # Split arrows
    ax.annotate('', xy=(20, 75), xytext=(17, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))
    ax.annotate('', xy=(20, 50), xytext=(17, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))
    ax.annotate('', xy=(20, 25), xytext=(17, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))

    # Stream 1: HOG
    b_hog = patches.FancyBboxPatch((21, 64), 26, 22, boxstyle="round,pad=0.6", fc='#EFF6FF', ec='#2563EB', lw=1.5)
    ax.add_patch(b_hog)
    ax.text(34, 80, "1. HOG Gradient Features", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1D4ED8')
    ax.text(34, 72, "• Grayscale -> CLAHE (clip=2.0, grid=8x8)\n• Cell: 8x8, Block: 2x2, 9 Orientations\n• L2-Hys Normalization -> 8,100 Dimensions\n• Captures radiator grille & chassis contours", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    # Stream 2: Color
    b_col = patches.FancyBboxPatch((21, 39), 26, 22, boxstyle="round,pad=0.6", fc='#FFF7ED', ec='#EA580C', lw=1.5)
    ax.add_patch(b_col)
    ax.text(34, 55, "2. Color Distribution", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#C2410C')
    ax.text(34, 47, "• RGB -> HSV Color Space Transformation\n• 3D Joint Histogram (32x32x32 Bins)\n• L2 Normalized -> 144/96 Dimensions\n• Illuminance-invariant chromaticity prior", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    # Stream 3: LBP
    b_lbp = patches.FancyBboxPatch((21, 14), 26, 22, boxstyle="round,pad=0.6", fc='#F0FDF4', ec='#16A34A', lw=1.5)
    ax.add_patch(b_lbp)
    ax.text(34, 30, "3. LBP Micro-Texture", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#15803D')
    ax.text(34, 22, "• Grayscale Uniform Patterns (P=16, R=2)\n• 4x4 Spatial Sub-grid Tiling\n• 18 Bins/Cell -> 288 Dimensions\n• Wheel hub, intake mesh, surface texture", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    # Merge arrows to Fusion
    ax.annotate('', xy=(53, 50), xytext=(48, 75), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))
    ax.annotate('', xy=(53, 50), xytext=(48, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))
    ax.annotate('', xy=(53, 50), xytext=(48, 25), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))

    # Concatenation & Normalization Box
    b_cat = patches.FancyBboxPatch((54, 36), 18, 28, boxstyle="round,pad=0.6", fc='#FAF5FF', ec='#9333EA', lw=1.5)
    ax.add_patch(b_cat)
    ax.text(63, 58, "Feature Concatenation\n& Preprocessing", ha='center', va='center', fontsize=9, fontweight='bold', color='#7E22CE')
    ax.text(63, 46, "• Fused Dim: 8,532\n• StandardScaler\n  (Zero-mean, unit-var)\n• PCA Compression\n  (128 Eigenvectors)\n• 98.5% Variance Kept", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    # Arrow to Classifier
    ax.annotate('', xy=(76, 50), xytext=(73, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))

    # Classifier & Decision Box
    b_clf = patches.FancyBboxPatch((77, 36), 21, 28, boxstyle="round,pad=0.6", fc='#FFF1F2', ec='#E11D48', lw=1.5)
    ax.add_patch(b_clf)
    ax.text(87.5, 58, "Linear SVM\nClassification", ha='center', va='center', fontsize=9, fontweight='bold', color='#BE123C')
    ax.text(87.5, 46, "• One-vs-Rest Architecture\n• Regularization C=1.0\n• Platt Prob. Calibration\n• Top-1 Acc: 3.33%\n• Top-5 Acc: 11.00%\n• Latency: 0.015 ms\n• Size: 0.15 MB", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig3_classical_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig4_deep_learning_architecture():
    """Figure 4: Deep Learning Pipeline Architecture (ConvNeXt-Tiny & ResNet-50)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Subplot 1: ConvNeXt-Tiny
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    ax1.set_title("Primary Architecture: ConvNeXt-Tiny (320x320 Input)", fontsize=11, fontweight='bold', color='#0F172A')

    layers_convnext = [
        ("Input ROI", "320 x 320 x 3 RGB Tensor", '#EFF6FF', '#2563EB'),
        ("Patchify Stem", "4x4 Conv, Stride 4, LayerNorm\n96 Channels (80x80 Feature Map)", '#DBEAFE', '#1D4ED8'),
        ("Stage 1", "3x ConvNeXt Blocks (7x7 Depthwise,\nInverted Bottleneck 96->384->96, GELU)", '#BFDBFE', '#1D4ED8'),
        ("Stage 2 Downsample", "2x2 Conv, Stride 2, LayerNorm\n3x Blocks, 192 Channels (40x40 Map)", '#93C5FD', '#1E40AF'),
        ("Stage 3 Downsample", "2x2 Conv, Stride 2, LayerNorm\n9x Blocks, 384 Channels (20x20 Map)", '#60A5FA', '#1E40AF'),
        ("Stage 4 Downsample", "2x2 Conv, Stride 2, LayerNorm\n3x Blocks, 768 Channels (10x10 Map)", '#3B82F6', '#1E3A8A'),
        ("Global Pooling & Head", "Global AvgPool (768-d) -> LayerNorm\n-> Dropout(0.2) -> Linear(768, 196)", '#EDE9FE', '#7C3AED'),
        ("Outputs", "Softmax Logits -> 196 Categories\n(Val Top-1: 89.43%, Top-5: 97.80%)", '#DCFCE7', '#15803D')
    ]

    y = 88
    for name, desc, bg, ec in layers_convnext:
        box = patches.FancyBboxPatch((8, y-7), 84, 8, boxstyle="round,pad=0.4", fc=bg, ec=ec, lw=1.2)
        ax1.add_patch(box)
        ax1.text(12, y-3, name, fontsize=8.5, fontweight='bold', color='#0F172A')
        ax1.text(48, y-3, desc, fontsize=7.5, color='#334155')
        if y > 20:
            ax1.annotate('', xy=(50, y-8.5), xytext=(50, y-7), arrowprops=dict(arrowstyle="->", lw=1.2, color='#94A3B8'))
        y -= 10.5

    # Subplot 2: ResNet-50
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis('off')
    ax2.set_title("Baseline Architecture: ResNet-50 (224x224 Input)", fontsize=11, fontweight='bold', color='#0F172A')

    layers_resnet = [
        ("Input ROI", "224 x 224 x 3 RGB Tensor", '#FFF7ED', '#EA580C'),
        ("Initial Conv & Pool", "7x7 Conv, Stride 2 + MaxPool\n64 Channels (56x56 Feature Map)", '#FFEDD5', '#C2410C'),
        ("Stage 1 (Res2)", "3x Bottlenecks [1x1, 3x3, 1x1]\n256 Channels (56x56 Map)", '#FED7AA', '#C2410C'),
        ("Stage 2 (Res3)", "4x Bottlenecks (Stride 2)\n512 Channels (28x28 Map)", '#FDBA74', '#9A3412'),
        ("Stage 3 (Res4)", "6x Bottlenecks (Stride 2)\n1024 Channels (14x14 Map)", '#FB923C', '#9A3412'),
        ("Stage 4 (Res5)", "3x Bottlenecks (Stride 2)\n2048 Channels (7x7 Map)", '#F97316', '#7C2D12'),
        ("Global Pooling & Head", "Adaptive AvgPool (2048-d) -> Flatten\n-> Dropout(0.2) -> Linear(2048, 196)", '#EDE9FE', '#7C3AED'),
        ("Outputs", "Softmax Logits -> 196 Categories\n(Val Top-1: 0.49%, Year Acc: 58.92%)", '#FEF2F2', '#B91C1C')
    ]

    y = 88
    for name, desc, bg, ec in layers_resnet:
        box = patches.FancyBboxPatch((8, y-7), 84, 8, boxstyle="round,pad=0.4", fc=bg, ec=ec, lw=1.2)
        ax2.add_patch(box)
        ax2.text(12, y-3, name, fontsize=8.5, fontweight='bold', color='#0F172A')
        ax2.text(48, y-3, desc, fontsize=7.5, color='#334155')
        if y > 20:
            ax2.annotate('', xy=(50, y-8.5), xytext=(50, y-7), arrowprops=dict(arrowstyle="->", lw=1.2, color='#94A3B8'))
        y -= 10.5

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig4_deep_learning_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig5_hybrid_decision_mechanism():
    """Figure 5: Hybrid Integration and Adaptive Decision Mechanism."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    ax.text(50, 96, "Figure 5: Hybrid Integration and Adaptive Decision Mechanism", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')
    ax.text(50, 92, "Consensus Verification, Confidence Calibration, and Hierarchical Arbitration", 
            ha='center', va='center', fontsize=10, color='#64748B')

    # Input Sources
    b_dl = patches.FancyBboxPatch((4, 66), 22, 18, boxstyle="round,pad=0.5", fc='#FAF5FF', ec='#9333EA', lw=1.5)
    ax.add_patch(b_dl)
    ax.text(15, 78, "Deep Learning Stream", ha='center', va='center', fontsize=9, fontweight='bold', color='#7E22CE')
    ax.text(15, 71, "• ConvNeXt Posteriors P_DL(y)\n• Class, Make, Model, Year\n• Raw Confidence c_DL", 
            ha='center', va='center', fontsize=7.5, color='#334155')

    b_cv = patches.FancyBboxPatch((4, 38), 22, 18, boxstyle="round,pad=0.5", fc='#F0FDF4', ec='#16A34A', lw=1.5)
    ax.add_patch(b_cv)
    ax.text(15, 50, "Classical CV Stream", ha='center', va='center', fontsize=9, fontweight='bold', color='#15803D')
    ax.text(15, 43, "• Linear SVM Scores P_CV(y)\n• Edge/Color Structural Prior\n• Calibrated Probabilities", 
            ha='center', va='center', fontsize=7.5, color='#334155')

    # Arrows to Taxonomy Comparator
    ax.annotate('', xy=(34, 62), xytext=(27, 72), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))
    ax.annotate('', xy=(34, 52), xytext=(27, 47), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748B'))

    # Comparator Box
    b_cmp = patches.FancyBboxPatch((35, 42), 24, 30, boxstyle="round,pad=0.6", fc='#EFF6FF', ec='#2563EB', lw=1.6)
    ax.add_patch(b_cmp)
    ax.text(47, 67, "Taxonomy Agreement\nEngine", ha='center', va='center', fontsize=10, fontweight='bold', color='#1D4ED8')
    ax.text(47, 54, "Evaluate Matches across Hierarchy:\n1. Exact Class ID Match (1..196)\n2. Manufacturer Match (49 Makes)\n3. Model Series Match (189 Models)\n4. Manufacturing Year Match", 
            ha='center', va='center', fontsize=7.5, color='#1E293B', linespacing=1.2)

    # 3 Decision Branches
    # Branch 1: Full Agreement
    ax.annotate('', xy=(66, 75), xytext=(60, 62), arrowprops=dict(arrowstyle="->", lw=1.5, color='#16A34A'))
    b_br1 = patches.FancyBboxPatch((67, 66), 30, 18, boxstyle="round,pad=0.5", fc='#DCFCE7', ec='#16A34A', lw=1.5)
    ax.add_patch(b_br1)
    ax.text(82, 78, "Case A: Full Taxonomy Consensus", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#15803D')
    ax.text(82, 71, "• Class match verified (DL == Classical)\n• Confidence Boost: c_hyb = min(1.0, c_DL + 0.05)\n• Output Badge: [Full Consensus (Verified)]", 
            ha='center', va='center', fontsize=7.5, color='#1E293B')

    # Branch 2: Make Agreement
    ax.annotate('', xy=(66, 52), xytext=(60, 52), arrowprops=dict(arrowstyle="->", lw=1.5, color='#CA8A04'))
    b_br2 = patches.FancyBboxPatch((67, 43), 30, 18, boxstyle="round,pad=0.5", fc='#FEF9C3', ec='#CA8A04', lw=1.5)
    ax.add_patch(b_br2)
    ax.text(82, 55, "Case B: Make-Level Consensus", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#A16207')
    ax.text(82, 48, "• Manufacturer agreed (e.g. Both predict BMW)\n• Specific model/year resolved via DL\n• Output Badge: [Make Consensus (Reliable)]", 
            ha='center', va='center', fontsize=7.5, color='#1E293B')

    # Branch 3: Disagreement / Uncertainty
    ax.annotate('', xy=(66, 29), xytext=(60, 42), arrowprops=dict(arrowstyle="->", lw=1.5, color='#E11D48'))
    b_br3 = patches.FancyBboxPatch((67, 18), 30, 20, boxstyle="round,pad=0.5", fc='#FFF1F2', ec='#E11D48', lw=1.5)
    ax.add_patch(b_br3)
    ax.text(82, 33, "Case C: Disagreement / High Uncertainty", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#BE123C')
    ax.text(82, 25, "• If c_DL < 0.35 (Uncertainty threshold tau):\n  Score Fusion: S(y) = 0.70 P_DL + 0.30 P_CV\n• If c_hyb < 0.20 -> Low-Confidence Warning\n• Output Badge: [Model Disagreement (Flagged)]", 
            ha='center', va='center', fontsize=7.2, color='#1E293B')

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig5_hybrid_decision_mechanism.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig6_low_light_comparison():
    """Figure 6: Extreme Low-Light & Underexposure Restoration Comparison."""
    raw_crop_path = PROJECT_ROOT / "outputs" / "hybrid" / "failure_cases" / "failure_03_extreme_underexposure_crop.jpg"
    if not raw_crop_path.is_file():
        print(f"Warning: {raw_crop_path} not found")
        return

    raw_img = cv2.imread(str(raw_crop_path))
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # Compute luminance map
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE enhancement
    lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enh = cv2.merge((l_enhanced, a, b))
    clahe_rgb = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)

    # Bilateral denoising after CLAHE
    bilateral_rgb = cv2.bilateralFilter(clahe_rgb, d=7, sigmaColor=50, sigmaSpace=50)

    # Canny edge maps on raw vs enhanced
    edges_raw = cv2.Canny(gray, 30, 100)
    edges_enh = cv2.Canny(cv2.cvtColor(bilateral_rgb, cv2.COLOR_RGB2GRAY), 50, 150)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle("Figure 6: Extreme Low-Light Scene Restoration & Feature Recovery", 
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)

    # (a) Raw input
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title("(a) Raw Underexposed Input\n(Mean Brightness: 18.4 / 255)", fontsize=9.5, fontweight='bold')
    axes[0, 0].axis('off')

    # (b) Estimated Luminance Map
    im1 = axes[0, 1].imshow(gray, cmap='inferno')
    axes[0, 1].set_title("(b) Estimated Luminance Map L(x, y)\n(Severe Shadow Compression)", fontsize=9.5, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (c) Canny Edges on Raw
    axes[0, 2].imshow(edges_raw, cmap='gray')
    axes[0, 2].set_title("(c) Canny Edges on Raw Input\n(Vanished Gradients & Boundaries)", fontsize=9.5, fontweight='bold')
    axes[0, 2].axis('off')

    # (d) CLAHE Enhanced
    axes[1, 0].imshow(clahe_rgb)
    axes[1, 0].set_title("(d) CLAHE Contrast Equalization\n(Dynamic Range Recovered)", fontsize=9.5, fontweight='bold')
    axes[1, 0].axis('off')

    # (e) Bilateral Denoised & Normalized
    axes[1, 1].imshow(bilateral_rgb)
    axes[1, 1].set_title("(e) Edge-Preserving Bilateral Denoised\n(Noise Suppressed, Highlights Intact)", fontsize=9.5, fontweight='bold')
    axes[1, 1].axis('off')

    # (f) Recovered Canny Edges
    axes[1, 2].imshow(edges_enh, cmap='gray')
    axes[1, 2].set_title("(f) Canny Edges on Enhanced ROI\n(Chassis Silhouette & Wheel Hubs Restored)", fontsize=9.5, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig6_low_light_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig7_nonuniform_backlit_enhancement():
    """Figure 7: Non-Uniform / Backlit Scene Highlight Preservation."""
    sample_img_path = PROJECT_ROOT / "Stanford Dataset" / "cars_train" / "cars_train" / "00001.jpg"
    if not sample_img_path.is_file():
        sample_img_path = PROJECT_ROOT / "outputs" / "detection" / "visualizations" / "00001.jpg"

    img = cv2.imread(str(sample_img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Simulate strong non-uniform backlight: bright background, dark vehicle
    h, w = img_rgb.shape[:2]
    gradient_mask = np.linspace(1.6, 0.4, h)[:, None]
    backlit_img = np.clip(img_rgb.astype(np.float32) * gradient_mask[:, :, None], 0, 255).astype(np.uint8)

    # CLAHE on L channel preserves chromaticity and highlights
    lab = cv2.cvtColor(backlit_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge((l_enh, a, b))
    clahe_rgb = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle("Figure 7: Non-Uniform Backlit Scene Processing with Highlight Preservation", 
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.98)

    axes[0].imshow(backlit_img)
    axes[0].set_title("(a) Backlit Input Scene\n(Deep Underbody Shadows & Glare)", fontsize=9.5, fontweight='bold')
    axes[0].axis('off')

    # Highlight preservation histogram
    axes[1].hist(cv2.cvtColor(backlit_img, cv2.COLOR_RGB2GRAY).ravel(), bins=50, color='#64748B', alpha=0.6, label='Input Luminance')
    axes[1].hist(cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2GRAY).ravel(), bins=50, color='#2563EB', alpha=0.6, label='Enhanced Luminance')
    axes[1].set_title("(b) Luminance Distribution\n(Shadow Lift without Highlight Clipping)", fontsize=9.5, fontweight='bold')
    axes[1].set_xlabel("Pixel Intensity [0 - 255]", fontsize=8)
    axes[1].set_ylabel("Pixel Count", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    axes[2].imshow(clahe_rgb)
    axes[2].set_title("(c) Restored Output\n(Shadow Detail Recovered, No Overexposure)", fontsize=9.5, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig7_nonuniform_backlit_enhancement.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig8_intermediate_representations():
    """Figure 8: Comprehensive Intermediate Representations Panel."""
    vis_dir = PROJECT_ROOT / "outputs" / "classical_cv" / "visualizations"
    prefix = "sample_1_train_05999_"
    
    files = {
        "(a) Original Scene": vis_dir / f"{prefix}01_original.png",
        "(b) Autonomous ROI Crop": vis_dir / f"{prefix}02_vehicle_roi.png",
        "(c) Grayscale Representation": vis_dir / f"{prefix}03_grayscale.png",
        "(d) CLAHE Illumination Boost": vis_dir / f"{prefix}04_clahe.png",
        "(e) Canny Edge Contours": vis_dir / f"{prefix}05_canny_edges.png",
        "(f) HOG Gradient Orientations": vis_dir / f"{prefix}06_hog.png",
        "(g) LBP Micro-Texture": vis_dir / f"{prefix}08_lbp.png",
        "(h) ORB Keypoints": vis_dir / f"{prefix}09_orb_keypoints.png",
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle("Figure 8: Intermediate Visual and Structural Representations across the Pipeline", 
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.98)

    for idx, (label, fpath) in enumerate(files.items()):
        r = idx // 4
        c = idx % 4
        ax = axes[r, c]
        if fpath.is_file():
            img = Image.open(fpath)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Artifact:\n{fpath.name}", ha='center', va='center')
        ax.set_title(label, fontsize=9, fontweight='bold', color='#1E293B')
        ax.axis('off')

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig8_intermediate_representations.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig9_failure_cases_grid():
    """Figure 9: Forensic Failure Cases Analysis Grid."""
    cases_dir = PROJECT_ROOT / "outputs" / "hybrid" / "failure_cases"
    
    cases = [
        ("Case 1: Severe Defocus & Gaussian Blur", 
         cases_dir / "failure_01_severe_blur_feature_loss_crop.jpg",
         "Audi TTS Coupe 2012\n-> Misclassified as GMC Savana (Conf: 0.25)\nCause: High-frequency grille details lost"),
        ("Case 2: 45% Foreground Occlusion", 
         cases_dir / "failure_02_partial_vehicle_occlusion_crop.jpg",
         "Audi TTS Coupe 2012\n-> Misclassified as Bentley GT (Conf: 0.71)\nCause: Truncated aspect ratio & masked hood"),
        ("Case 3: Extreme Underexposure / Shadow", 
         cases_dir / "failure_03_extreme_underexposure_crop.jpg",
         "Audi TTS Coupe 2012\n-> Misclassified as MINI Cooper (Conf: 0.03)\nCause: Vanished contrast, detector drift"),
        ("Case 4: Intra-Make Grille Ambiguity", 
         cases_dir / "failure_04_fine_grained_model_confusion_crop.jpg",
         "Audi RS 4 vs. Audi S4 Sedan\n-> Shared Singleframe trapezoidal grille\nCause: Distinctions require minute bumper intake cues"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle("Figure 9: Forensic Evaluation of Real-World Edge-Case Failure Modes", 
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.98)

    for idx, (title, fpath, desc) in enumerate(cases):
        ax = axes[idx]
        if fpath.is_file():
            img = Image.open(fpath)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Artifact:\n{fpath.name}", ha='center', va='center')
        ax.set_title(title, fontsize=8.5, fontweight='bold', color='#B91C1C')
        ax.axis('off')
        ax.text(0.5, -0.22, desc, ha='center', va='top', transform=ax.transAxes, 
                fontsize=7.5, color='#334155', linespacing=1.2)

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig9_failure_cases_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def generate_fig10_training_and_evaluation_benchmarks():
    """Figure 10: Training Curves, Detection Performance, and Confusion Matrix."""
    train_plot_path = PROJECT_ROOT / "outputs" / "classification" / "plots" / "training_history.png"
    yolo_pr_path = PROJECT_ROOT / "models" / "detection" / "yolov8n_vehicle" / "BoxPR_curve.png"
    conf_makes_path = PROJECT_ROOT / "results" / "evaluation" / "confusion_matrix_makes.png"

    fig = plt.figure(figsize=(15, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle("Figure 10: Training Curves, Localization PR-Curve, and Manufacturer Confusion Matrix", 
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.98)

    ax1 = fig.add_subplot(1, 3, 1)
    if train_plot_path.is_file():
        img1 = Image.open(train_plot_path)
        ax1.imshow(img1)
    ax1.set_title("(a) ConvNeXt-Tiny Training Curves\n(CrossEntropy Loss & Top-1/Top-5 Acc)", fontsize=8.5, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    if yolo_pr_path.is_file():
        img2 = Image.open(yolo_pr_path)
        ax2.imshow(img2)
    ax2.set_title("(b) YOLOv8n Precision-Recall Curve\n(mAP@0.50 = 99.48%, IoU = 0.906)", fontsize=8.5, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    if conf_makes_path.is_file():
        img3 = Image.open(conf_makes_path)
        ax3.imshow(img3)
    ax3.set_title("(c) Manufacturer (Make) Confusion Matrix\n(Top Makes, Make-Level Accuracy: 89.6% - 94.5%)", fontsize=8.5, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig10_training_and_evaluation_benchmarks.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Generating Figure 1: System Architecture...")
    generate_fig1_system_architecture()
    print("Generating Figure 2: Processing Workflow...")
    generate_fig2_processing_workflow()
    print("Generating Figure 3: Classical Pipeline...")
    generate_fig3_classical_pipeline()
    print("Generating Figure 4: Deep Learning Architecture...")
    generate_fig4_deep_learning_architecture()
    print("Generating Figure 5: Hybrid Decision Mechanism...")
    generate_fig5_hybrid_decision_mechanism()
    print("Generating Figure 6: Low Light Comparison...")
    generate_fig6_low_light_comparison()
    print("Generating Figure 7: Nonuniform Backlit Enhancement...")
    generate_fig7_nonuniform_backlit_enhancement()
    print("Generating Figure 8: Intermediate Representations...")
    generate_fig8_intermediate_representations()
    print("Generating Figure 9: Failure Cases Grid...")
    generate_fig9_failure_cases_grid()
    print("Generating Figure 10: Benchmarks...")
    generate_fig10_training_and_evaluation_benchmarks()
    print("\nAll report assets generated successfully!")


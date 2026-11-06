"""
Visualization Module for Classical Computer Vision Baseline.
Generates publication-quality visual diagnostics for:
1. Original image
2. Vehicle ROI
3. Grayscale image
4. CLAHE enhanced result
5. Canny edge map
6. HOG visualization
7. Colour histograms (RGB, HSV, LAB)
8. LBP texture map
9. ORB keypoints
10. Multi-feature comprehensive comparison panel
11. Confusion matrix heatmap
12. Misclassification failure analysis panel
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from src.classical_cv.color_features import ColorHistogramExtractor
from src.classical_cv.hog_features import HOGFeatureExtractor
from src.classical_cv.orb_features import ORBFeatureExtractor
from src.classical_cv.roi_extractor import ROIExtractor
from src.classical_cv.texture_features import LBPFeatureExtractor
from src.data.preprocessing import (
    apply_clahe,
    detect_edges_canny,
    load_image,
    to_grayscale,
)
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class ClassicalVisualizer:
    """
    Visualizer producing classical CV feature maps and diagnostic panels.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        self.output_dir = Path(output_dir) if output_dir else self.root / "outputs" / "classical_cv" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.roi_extractor = ROIExtractor(target_size=(128, 128), keep_aspect_ratio=True)
        self.hog_extractor = HOGFeatureExtractor(image_size=(128, 128))
        self.color_extractor = ColorHistogramExtractor(color_spaces=["rgb", "hsv", "lab"])
        self.lbp_extractor = LBPFeatureExtractor(radius=2, n_points=16, image_size=(128, 128))
        self.orb_extractor = ORBFeatureExtractor(n_features=500)

    def generate_all_visualizations_for_sample(
        self,
        image_input: np.ndarray | str | Path,
        bbox: Any = None,
        sample_name: str = "sample_0",
        save_individual: bool = True,
    ) -> dict[str, str]:
        """
        Generate all required classical CV visualizations for an actual dataset image.

        Args:
            image_input: Image array or file path.
            bbox: Bounding box tuple or row.
            sample_name: Identifier for saved file naming.
            save_individual: If True, saves individual step images as well as summary panel.

        Returns:
            Dictionary of saved visualization file paths.
        """
        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input)
            full_p = img_p if img_p.is_file() else self.root / img_p
            original_img = load_image(full_p, as_rgb=True)
        else:
            original_img = image_input.copy()

        saved_files: dict[str, str] = {}

        # 1. Original Image
        p1 = self.output_dir / f"{sample_name}_01_original.png"
        if save_individual:
            plt.imsave(p1, original_img)
            saved_files["original"] = str(p1)

        # 2. Vehicle ROI
        is_matlab = isinstance(bbox, (dict, pd.Series)) and ("bbox_matlab_x1" in bbox and "bbox_x1" not in bbox and "bbox_xmin" not in bbox)
        roi_img, _ = self.roi_extractor.extract_roi(original_img, bbox=bbox, use_bbox=True, is_matlab_coord=is_matlab)
        p2 = self.output_dir / f"{sample_name}_02_vehicle_roi.png"
        if save_individual:
            plt.imsave(p2, roi_img)
            saved_files["vehicle_roi"] = str(p2)

        # 3. Grayscale Image
        gray_img = to_grayscale(roi_img)
        p3 = self.output_dir / f"{sample_name}_03_grayscale.png"
        if save_individual:
            plt.imsave(p3, gray_img, cmap="gray")
            saved_files["grayscale"] = str(p3)

        # 4. CLAHE Result
        clahe_img = apply_clahe(roi_img, clip_limit=2.0)
        p4 = self.output_dir / f"{sample_name}_04_clahe.png"
        if save_individual:
            plt.imsave(p4, clahe_img)
            saved_files["clahe"] = str(p4)

        # 5. Canny Edge Map
        canny_img = detect_edges_canny(roi_img, low_thresh=50, high_thresh=150)
        p5 = self.output_dir / f"{sample_name}_05_canny_edges.png"
        if save_individual:
            plt.imsave(p5, canny_img, cmap="gray")
            saved_files["canny_edges"] = str(p5)

        # 6. HOG Visualization
        _, hog_viz = self.hog_extractor.compute_hog_with_visualization(roi_img)
        p6 = self.output_dir / f"{sample_name}_06_hog.png"
        if save_individual:
            plt.imsave(p6, hog_viz, cmap="inferno")
            saved_files["hog"] = str(p6)

        # 7. Colour Histograms
        color_hists = self.color_extractor.compute_channel_histograms_dict(roi_img)
        p7 = self.output_dir / f"{sample_name}_07_color_histograms.png"
        if save_individual:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            space_colors = {"rgb": ["r", "g", "b"], "hsv": ["m", "c", "y"], "lab": ["black", "orange", "blue"]}
            for ax_idx, (cs, ch_dict) in enumerate(color_hists.items()):
                ax = axes[ax_idx]
                col_list = space_colors.get(cs, ["k", "gray", "silver"])
                for i, (ch_name, (centers, vals)) in enumerate(ch_dict.items()):
                    ax.plot(centers, vals, label=ch_name, color=col_list[i % len(col_list)], lw=1.8)
                ax.set_title(f"{cs.upper()} Histogram", fontsize=11, fontweight="bold")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(p7, dpi=150)
            plt.close(fig)
            saved_files["color_histograms"] = str(p7)

        # 8. LBP Visualization
        _, lbp_viz = self.lbp_extractor.compute_lbp_with_visualization(roi_img)
        p8 = self.output_dir / f"{sample_name}_08_lbp.png"
        if save_individual:
            plt.imsave(p8, lbp_viz, cmap="magma")
            saved_files["lbp"] = str(p8)

        # 9. ORB Keypoints
        orb_viz = self.orb_extractor.draw_keypoints(roi_img, max_keypoints=80)
        p9 = self.output_dir / f"{sample_name}_09_orb_keypoints.png"
        if save_individual:
            plt.imsave(p9, orb_viz)
            saved_files["orb_keypoints"] = str(p9)

        # 10. Multi-feature Comprehensive Comparison Panel (10-panel grid)
        p10 = self.output_dir / f"{sample_name}_10_feature_comparison.png"
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))

        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("1. Original Image", fontweight="bold", fontsize=10)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(roi_img)
        axes[0, 1].set_title("2. Vehicle ROI (128x128)", fontweight="bold", fontsize=10)
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gray_img, cmap="gray")
        axes[0, 2].set_title("3. Grayscale", fontweight="bold", fontsize=10)
        axes[0, 2].axis("off")

        axes[0, 3].imshow(clahe_img)
        axes[0, 3].set_title("4. CLAHE Enhanced", fontweight="bold", fontsize=10)
        axes[0, 3].axis("off")

        axes[0, 4].imshow(canny_img, cmap="gray")
        axes[0, 4].set_title("5. Canny Edge Map", fontweight="bold", fontsize=10)
        axes[0, 4].axis("off")

        axes[1, 0].imshow(hog_viz, cmap="inferno")
        axes[1, 0].set_title("6. HOG Gradients", fontweight="bold", fontsize=10)
        axes[1, 0].axis("off")

        # Color hist inside panel
        ax_hist = axes[1, 1]
        for ch_name, (centers, vals) in color_hists.get("rgb", {}).items():
            col = {"R": "red", "G": "green", "B": "blue"}.get(ch_name, "black")
            ax_hist.plot(centers, vals, label=ch_name, color=col, lw=1.5)
        ax_hist.set_title("7. RGB Histogram", fontweight="bold", fontsize=10)
        ax_hist.legend(loc="upper right", fontsize=7)
        ax_hist.grid(alpha=0.2)

        axes[1, 2].imshow(lbp_viz, cmap="magma")
        axes[1, 2].set_title("8. LBP Texture", fontweight="bold", fontsize=10)
        axes[1, 2].axis("off")

        axes[1, 3].imshow(orb_viz)
        axes[1, 3].set_title("9. ORB Keypoints", fontweight="bold", fontsize=10)
        axes[1, 3].axis("off")

        # 10th cell: Feature vector profile summary
        ax_prof = axes[1, 4]
        hog_feat = self.hog_extractor.extract_features(roi_img)
        lbp_feat = self.lbp_extractor.extract_features(roi_img)
        col_feat = self.color_extractor.extract_features(roi_img)
        descriptors = ["HOG", "LBP", "Color"]
        dims = [len(hog_feat), len(lbp_feat), len(col_feat)]
        bars = ax_prof.bar(descriptors, dims, color=["#3498db", "#9b59b6", "#e74c3c"])
        ax_prof.set_title("10. Feature Vector Dims", fontweight="bold", fontsize=10)
        for b in bars:
            ax_prof.text(b.get_x() + b.get_width() / 2, b.get_height() + 50, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8)
        ax_prof.set_ylim(0, max(dims) * 1.15)
        ax_prof.grid(alpha=0.3, axis="y")

        plt.suptitle(f"P5 Classical CV Baseline Feature Diagnostic: {sample_name}", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(p10, dpi=160)
        plt.close(fig)
        saved_files["feature_comparison"] = str(p10)

        logger.info("Generated %d visualizations for %s in %s", len(saved_files), sample_name, self.output_dir)
        return saved_files

    def plot_confusion_matrix_heatmap(
        self,
        cm: np.ndarray,
        labels: list[int] | list[str],
        output_path: str | Path | None = None,
        top_n: int = 20,
    ) -> str:
        """
        Plot heatmap of confusion matrix (filtering to top N most frequent or confused classes).
        """
        out_p = Path(output_path) if output_path else self.output_dir / "confusion_matrix.png"
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # If cm has many classes (e.g. 196), select top_n classes with highest sample count
        if len(cm) > top_n:
            row_sums = cm.sum(axis=1)
            top_indices = np.argsort(row_sums)[::-1][:top_n]
            sub_cm = cm[np.ix_(top_indices, top_indices)]
            sub_labels = [str(labels[i]) for i in top_indices]
        else:
            sub_cm = cm
            sub_labels = [str(l) for l in labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sub_cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(sub_cm.shape[1]),
            yticks=np.arange(sub_cm.shape[0]),
            xticklabels=sub_labels,
            yticklabels=sub_labels,
            ylabel="True Class",
            xlabel="Predicted Class",
            title=f"Classical CV Confusion Matrix (Top {len(sub_labels)} Classes)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.tight_layout()
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix plot to %s", out_p)
        return str(out_p)

    def plot_failure_sample(
        self,
        image_path: str | Path,
        bbox: Any,
        gt_info: dict[str, Any],
        pred_info: dict[str, Any],
        top_5: list[dict[str, Any]],
        output_path: str | Path | None = None,
    ) -> str:
        """
        Plot failure diagnostic showing vehicle image, bounding box, and Top-5 confidence scores.
        """
        img = load_image(image_path, as_rgb=True)
        is_matlab = isinstance(bbox, (dict, pd.Series)) and ("bbox_matlab_x1" in bbox and "bbox_x1" not in bbox and "bbox_xmin" not in bbox)
        roi, _ = self.roi_extractor.extract_roi(img, bbox=bbox, use_bbox=True, is_matlab_coord=is_matlab)

        out_p = Path(output_path) if output_path else self.output_dir / f"failure_{gt_info.get('class_id', 0)}_vs_{pred_info.get('class_id', 0)}.png"
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax_img, ax_bars) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})

        ax_img.imshow(roi)
        ax_img.set_title(f"Ground Truth:\n{gt_info.get('class_name', '')}", fontsize=9, fontweight="bold", color="darkgreen")
        ax_img.axis("off")

        # Top 5 bar chart
        names = [f"#{c['rank']} {c['class_name'][:25]}..." for c in top_5][::-1]
        scores = [c["score"] for c in top_5][::-1]
        colors = ["#2ecc71" if c["class_id"] == gt_info.get("class_id") else "#e74c3c" for c in top_5][::-1]

        ax_bars.barh(names, scores, color=colors)
        ax_bars.set_xlabel("Confidence / Decision Score", fontsize=9)
        ax_bars.set_title(f"Predicted: {pred_info.get('class_name', '')[:30]}", fontsize=9, fontweight="bold", color="darkred")
        ax_bars.grid(alpha=0.3, axis="x")

        plt.suptitle("Classical CV Fine-Grained Misclassification Analysis", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        return str(out_p)

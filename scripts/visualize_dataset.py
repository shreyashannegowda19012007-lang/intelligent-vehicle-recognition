"""
Dataset visualization suite for Stanford Cars (Cars196).
Generates high-resolution visualization figures using real dataset annotations and images:
1. Class distribution chart
2. Manufacturer distribution chart
3. Manufacturing year distribution chart
4. Train/validation/test split comparison
5. Bounding box overlay samples
6. Image dimension distribution
7. Random sample contact sheet
8. Augmentation comparison grid
Saves all charts into data/reports/visualizations/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import load_manifest_df
from src.data.preprocessing import apply_clahe
from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize_dataset")


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot sample counts across all 196 fine-grained classes."""
    train_df = df[df["split"] == "train"]
    counts = train_df["class_id"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(14, 6), dpi=150)
    plt.bar(range(len(counts)), counts.values, color="#1f77b4", width=0.8, edgecolor="none")
    plt.axhline(counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean ({counts.mean():.1f})")
    plt.axhline(counts.median(), color="orange", linestyle=":", linewidth=1.5, label=f"Median ({counts.median():.1f})")

    plt.title("Stanford Cars - Class Sample Distribution (196 Classes in Train Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Class Index (Sorted by Frequency)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_manufacturer_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot samples per manufacturer (all 49 makes)."""
    train_df = df[df["split"] == "train"]
    make_counts = train_df["make"].value_counts(ascending=True)

    plt.figure(figsize=(10, 14), dpi=150)
    bars = plt.barh(range(len(make_counts)), make_counts.values, color="#2ca02c", height=0.7)
    plt.yticks(range(len(make_counts)), make_counts.index, fontsize=8)
    plt.title("Stanford Cars - Samples per Manufacturer (49 Makes)", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Count", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, make_counts.values, strict=False):
        plt.text(val + 3, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_year_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot vehicle production year frequencies."""
    train_df = df[df["split"] == "train"]
    years = train_df["year"].dropna().astype(int).value_counts().sort_index()

    plt.figure(figsize=(10, 5), dpi=150)
    bars = plt.bar(years.index.astype(str), years.values, color="#ff7f0e", width=0.7)
    plt.title("Stanford Cars - Vehicle Manufacturing Year Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Model Year", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, years.values, strict=False):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 20, str(val), ha="center", fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_split_comparison(df: pd.DataFrame, splits_dir: Path, output_path: Path) -> None:
    """Compare counts and class coverage across splits."""
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    counts = {
        "Train (Stratified 85%)": len(pd.read_csv(train_csv)) if train_csv.is_file() else 0,
        "Validation (15%)": len(pd.read_csv(val_csv)) if val_csv.is_file() else 0,
        "Test (Official Holdout)": len(pd.read_csv(test_csv)) if test_csv.is_file() else 0,
    }

    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Bar chart
    bars = ax1.bar(labels, values, color=colors, width=0.5)
    ax1.set_title("Sample Counts per Split", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Images", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, v in zip(bars, values, strict=False):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 100, f"{v:,}", ha="center", fontweight="bold")

    # Donut chart
    ax2.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.5},
    )
    ax2.set_title("Dataset Split Proportion", fontsize=12, fontweight="bold")

    plt.suptitle("Stanford Cars Dataset Partition Architecture", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_bounding_box_samples(df: pd.DataFrame, root: Path, output_path: Path) -> None:
    """Overlay bounding box coordinates on representative car images."""
    train_df = df[df["split"] == "train"].sample(8, random_state=42)

    _fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    axes = axes.flatten()

    for i, (_, row) in enumerate(train_df.iterrows()):
        ax = axes[i]
        img_p = root / Path(str(row["image_path"]))
        x1 = float(row.get("bbox_x1", row.get("bbox_xmin", 0)))
        y1 = float(row.get("bbox_y1", row.get("bbox_ymin", 0)))
        x2 = float(row.get("bbox_x2", row.get("bbox_xmax", 0)))
        y2 = float(row.get("bbox_y2", row.get("bbox_ymax", 0)))

        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB")
            ax.imshow(img_rgb)

        # Draw bbox rectangle
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor="#00ff00", facecolor="none")
        ax.add_patch(rect)

        cname = str(row.get("class_name", "Vehicle"))
        # Shorten if long
        title = cname if len(cname) <= 28 else cname[:25] + "..."
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Stanford Cars - Bounding Box Ground Truth Overlays", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_image_dimensions(df: pd.DataFrame, output_path: Path) -> None:
    """Plot resolution distribution (width vs height)."""
    widths = df["image_width"]
    heights = df["image_height"]

    plt.figure(figsize=(9, 7), dpi=150)
    plt.scatter(widths, heights, alpha=0.3, color="#9467bd", s=15, edgecolors="none")

    # Annotate standard aspect ratios (4:3, 16:9, etc.)
    max_dim = max(widths.max(), heights.max())
    x_line = np.linspace(50, min(1600, max_dim), 100)
    plt.plot(x_line, x_line * (3 / 4), "r--", alpha=0.6, label="Aspect 4:3")
    plt.plot(x_line, x_line * (9 / 16), "g--", alpha=0.6, label="Aspect 16:9")

    plt.title("Stanford Cars - Image Dimension Distribution (N=16,185)", fontsize=13, fontweight="bold")
    plt.xlabel("Image Width (px)", fontsize=11)
    plt.ylabel("Image Height (px)", fontsize=11)
    plt.xlim(0, 2200)
    plt.ylim(0, 1600)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_sample_grid(df: pd.DataFrame, root: Path, output_path: Path) -> None:
    """Create a contact sheet showing 12 diverse vehicle categories."""
    # Pick 12 distinct manufacturers
    distinct_makes = df[df["split"] == "train"]["make"].unique()[:12]
    samples = [df[(df["split"] == "train") & (df["make"] == m)].iloc[0] for m in distinct_makes]

    _fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=150)
    axes = axes.flatten()

    for i, row in enumerate(samples):
        ax = axes[i]
        img_p = root / Path(str(row["image_path"]))
        with Image.open(img_p) as img:
            ax.imshow(img.convert("RGB"))

        cname = str(row.get("class_name", "Vehicle"))
        title = cname if len(cname) <= 30 else cname[:27] + "..."
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Stanford Cars - Representative Sample Contact Sheet", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_augmentation_comparison(df: pd.DataFrame, root: Path, output_path: Path) -> None:
    """Display side-by-side comparison of original image vs various augmentations."""
    sample_row = df[df["split"] == "train"].iloc[4]
    img_p = root / Path(str(sample_row["image_path"]))

    with Image.open(img_p) as raw_img:
        orig = raw_img.convert("RGB")

    # Crop vehicle ROI first
    x1 = int(sample_row.get("bbox_x1", sample_row.get("bbox_xmin", 0)))
    y1 = int(sample_row.get("bbox_y1", sample_row.get("bbox_ymin", 0)))
    x2 = int(sample_row.get("bbox_x2", sample_row.get("bbox_xmax", 0))) + 1
    y2 = int(sample_row.get("bbox_y2", sample_row.get("bbox_ymax", 0))) + 1
    crop_img = orig.crop((x1, y1, x2, y2)).resize((224, 224))

    # Various transformations
    import torchvision.transforms.functional as TF

    variants = [
        ("Original Crop", crop_img),
        ("Horizontal Flip", TF.hflip(crop_img)),
        ("Rotation (+15°)", TF.rotate(crop_img, 15)),
        ("Color Jitter (Bright/Sat)", TF.adjust_brightness(TF.adjust_saturation(crop_img, 1.4), 1.25)),
        ("Contrast Boost", TF.adjust_contrast(crop_img, 1.5)),
        ("Gaussian Blur", TF.gaussian_blur(crop_img, [5, 5])),
        ("CLAHE Equalization", Image.fromarray(apply_clahe(np.array(crop_img)))),
        ("Perspective / Affine", TF.affine(crop_img, angle=5, translate=[10, 5], scale=0.95, shear=5)),
    ]

    _fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)
    axes = axes.flatten()

    for ax, (title, v_img) in zip(axes, variants, strict=False):
        ax.imshow(v_img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Stanford Cars - Preprocessing & Augmentation Transformations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved: %s", output_path)


def generate_all_visualizations(
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all 8 required dataset visualizations."""
    root = find_project_root()
    splits_dir = root / "data" / "splits"
    vis_dir = Path(output_dir) if output_dir else (root / "data" / "reports" / "visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest_df(manifest_path)
    logger.info("Generating 8 visualization figures in: %s", vis_dir)

    outputs = [
        vis_dir / "class_distribution.png",
        vis_dir / "manufacturer_distribution.png",
        vis_dir / "year_distribution.png",
        vis_dir / "split_comparison.png",
        vis_dir / "bbox_samples.png",
        vis_dir / "image_dimensions.png",
        vis_dir / "sample_grid.png",
        vis_dir / "augmentation_comparison.png",
    ]

    plot_class_distribution(df, outputs[0])
    plot_manufacturer_distribution(df, outputs[1])
    plot_year_distribution(df, outputs[2])
    plot_split_comparison(df, splits_dir, outputs[3])
    plot_bounding_box_samples(df, root, outputs[4])
    plot_image_dimensions(df, outputs[5])
    plot_sample_grid(df, root, outputs[6])
    plot_augmentation_comparison(df, root, outputs[7])

    logger.info("All 8 visualizations successfully generated.")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset visualizations")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output directory for charts")
    args = parser.parse_args()

    generate_all_visualizations(manifest_path=args.manifest, output_dir=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

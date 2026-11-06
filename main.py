"""
CV Hackathon 2026 - Problem Statement P5
Intelligent Fine-Grained Vehicle Recognition and Analysis System.
Main entry point for project diagnostics, validation, and sample loading.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StanfordCarsDataset, get_default_transforms
from src.utils.config import get_config
from src.utils.device import get_device, get_hardware_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("P5_VehicleRecognition")


def show_info() -> None:
    """Print project configuration and hardware telemetry."""
    cfg = get_config()
    hw = get_hardware_telemetry()
    device = get_device("auto")

    print("\n" + "=" * 70)
    print("  P5 INTELLIGENT VEHICLE RECOGNITION SYSTEM - RUNTIME INFO")
    print("=" * 70)
    print(f"  Project Name        : {cfg.name} (v{cfg.version})")
    print(f"  Project Root        : {cfg.project_root}")
    print(f"  Dataset Name        : {cfg.dataset.name}")
    print(f"  Dataset Root        : {cfg.dataset.root_dir}")
    print(
        f"  Expected Images     : {cfg.dataset.expected_total_images:,} ({cfg.dataset.expected_train_images:,} train / {cfg.dataset.expected_test_images:,} test)"
    )
    print(
        f"  Classes / Makes     : {cfg.dataset.num_classes} classes, {cfg.dataset.num_makes} makes"
    )
    print("-" * 70)
    print(f"  Operating System    : {hw['os']}")
    print(f"  Python Version      : {hw['python_version']}")
    print(f"  PyTorch Version     : {hw['torch_version']}")
    print(
        f"  CPU Model           : {hw['cpu_model']} ({hw['cpu_cores_logical']} threads)"
    )
    print(
        f"  System RAM          : {hw['ram_available_gb']} GB available / {hw['ram_total_gb']} GB total"
    )
    print(f"  Selected Device     : {device}")
    if hw["cuda_available"]:
        print(
            f"  GPU Accelerator     : {hw['gpu_name']} ({hw['gpu_total_vram_gb']} GB VRAM, {hw['gpu_free_vram_gb']} GB free)"
        )
        print(
            f"  CUDA Version        : {hw['cuda_version']} (Compute {hw['gpu_compute_capability']})"
        )
    print("=" * 70 + "\n")


def run_sample_loader(num_samples: int = 5) -> None:
    """Load sample images via StanfordCarsDataset and verify outputs."""
    cfg = get_config()
    norm_csv = (
        cfg.project_root
        / "data"
        / "processed"
        / "annotations"
        / "normalized_annotations.csv"
    )
    if not norm_csv.is_file():
        logger.error("Normalized annotations CSV not found at: %s", norm_csv)
        sys.exit(1)

    import pandas as pd

    df = pd.read_csv(norm_csv)
    train_records = df[df["split"] == "train"].to_dict(orient="records")

    train_tf, _ = get_default_transforms(image_size=cfg.preprocessing.image_size[0])
    ds = StanfordCarsDataset(
        annotations=train_records[:num_samples],
        image_root=None,
        transform=train_tf,
        crop_bbox=True,
        target_type="multi",
    )

    print("\n" + "=" * 70)
    print(f"  SAMPLE DATASET LOADER TEST ({len(ds)} samples)")
    print("=" * 70)
    for i in range(len(ds)):
        img_tensor, targets = ds[i]
        filename = targets["filename"]
        class_id = targets["class_id"].item()
        bbox = targets["bbox"].tolist()
        print(
            f"  [{i + 1}/{len(ds)}] File: {filename:<12s} | Tensor Shape: {img_tensor.shape!s:<18s} | Class ID: {class_id:<3d} | BBox: {bbox}"
        )

    print("=" * 70)
    print("  Sample loader successfully verified PyTorch tensor pipeline!")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P5 Intelligent Fine-Grained Vehicle Recognition System CLI."
    )
    parser.add_argument(
        "--info", action="store_true", help="Display system and environment information"
    )
    parser.add_argument(
        "--sample",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Test DataLoader with N samples (default: 5)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Execute dataset validation suite"
    )
    parser.add_argument(
        "--quick",
        type=int,
        default=500,
        help="Number of images to verify during validation",
    )
    parser.add_argument("--version", action="store_true", help="Show project version")

    args = parser.parse_args()

    if args.version:
        cfg = get_config()
        print(f"{cfg.name} v{cfg.version}")
        return

    if args.validate:
        from scripts.validate_dataset import validate_all

        validate_all(quick_sample=args.quick)
        return

    if args.sample is not None:
        run_sample_loader(num_samples=args.sample)
        return

    # Default action: show info
    show_info()


if __name__ == "__main__":
    main()

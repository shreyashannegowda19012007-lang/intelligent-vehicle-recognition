"""
Model Checkpoint Downloader and Verification Utility.
Downloads required model weights from external release storage for deployment
environments without committing large binaries to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_models")

DEFAULT_STORAGE_BASE = os.getenv(
    "MODEL_STORAGE_URL",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0",
)

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "yolov8n_detector": {
        "local_path": PROJECT_ROOT / "yolov8n.pt" if (PROJECT_ROOT / "yolov8n.pt").is_file() else PROJECT_ROOT / "models" / "detection" / "yolov8n.pt",
        "url": f"{DEFAULT_STORAGE_BASE}/yolov8n.pt",
        "size_mb": 6.2,
        "required_for": "Vehicle localization and bounding box extraction",
    },
    "resnet50_stanford_cars": {
        "local_path": PROJECT_ROOT / "outputs" / "classification" / "checkpoints" / "best_checkpoint.pt",
        "url": os.getenv("RESNET50_CHECKPOINT_URL", ""),
        "size_mb": 97.5,
        "required_for": "Deep learning fine-grained vehicle classification (Stanford Cars 196 classes)",
    },
    "convnext_tiny_stanford_cars": {
        "local_path": PROJECT_ROOT / "models" / "classification" / "convnext_best.pt",
        "url": os.getenv("CONVNEXT_CHECKPOINT_URL", ""),
        "size_mb": 320.32,
        "required_for": "Primary deep learning fine-grained vehicle classification (ConvNeXt-Tiny, 196 classes)",
    },
}


def compute_file_sha256(path: Path) -> str:
    """Calculate SHA256 checksum of file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def download_file(url: str, destination: Path) -> bool:
    """Download file with progress logging."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s...", url, destination)
    try:
        def report_hook(count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                if percent % 25 == 0 and percent <= 100:
                    logger.debug("Download progress: %d%%", percent)

        urllib.request.urlretrieve(url, destination, reporthook=report_hook)
        logger.info("Successfully downloaded %s (%d bytes).", destination.name, destination.stat().st_size)
        return True
    except Exception as e:
        logger.error("Failed to download from %s: %s", url, e)
        if destination.is_file():
            destination.unlink(missing_ok=True)
        return False


def ensure_models_available(force: bool = False) -> bool:
    """
    Verify all required model checkpoints exist locally, downloading any missing items.
    """
    all_ready = True
    for model_key, meta in MODEL_CATALOG.items():
        loc_path: Path = meta["local_path"]
        if loc_path.is_file() and not force:
            logger.info("Model '%s' is ready at %s (size: %.2f MB)", model_key, loc_path, loc_path.stat().st_size / (1024 * 1024))
            continue

        url = meta["url"]
        if not url:
            logger.info("Model '%s' has no remote URL configured (RESNET50_CHECKPOINT_URL). Train via Colab or provide checkpoint locally at: %s", model_key, loc_path)
            continue
        success = download_file(url, loc_path)
        if not success:
            logger.warning("Could not automatically download '%s' from %s.", model_key, url)
            all_ready = False

    return all_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model checkpoints for production deployment")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    parser.add_argument("--check-only", action="store_true", help="Only check availability without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        ready = True
        for m_key, meta in MODEL_CATALOG.items():
            loc: Path = meta["local_path"]
            exists = loc.is_file()
            status = "FOUND" if exists else "MISSING"
            logger.info("Model [%s]: %s (%s)", m_key, status, loc)
            if not exists:
                ready = False
        return 0 if ready else 1

    success = ensure_models_available(force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


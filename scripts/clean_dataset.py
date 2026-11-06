"""
Image validation and cleaning pipeline for Stanford Cars (Cars196).
Performs non-destructive integrity scanning, file size checks, image readability,
cryptographic duplicate detection, and generates a structured invalid-file report.
Never deletes or alters raw image files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_parser import StanfordCarsDatasetParser, resolve_dataset_root
from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_dataset")


def compute_file_md5(file_path: Path, block_size: int = 65536) -> str:
    """Compute MD5 hash of a file efficiently."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_image_cleaning_audit(
    dataset_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    quick_sample: int | None = None,
) -> dict[str, Any]:
    """
    Run complete non-destructive image validation and duplicate detection.

    Args:
        dataset_root: Optional root directory override.
        output_dir: Directory where reports will be saved.
        quick_sample: If provided, audits a subsample of images for speed.

    Returns:
        Dictionary report summarizing findings.
    """
    start_time = time.time()
    root = resolve_dataset_root(dataset_root)
    proj_root = find_project_root()
    reports_dir = Path(output_dir) if output_dir else (proj_root / "data" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    parser = StanfordCarsDatasetParser(dataset_root=root, base_dir=proj_root)
    manifest = parser.get_manifest()

    train_dir = manifest.train_image_dir
    test_dir = manifest.test_image_dir

    all_train_files = sorted(train_dir.glob("*.jpg")) if train_dir else []
    all_test_files = sorted(test_dir.glob("*.jpg")) if test_dir else []
    all_files = all_train_files + all_test_files

    logger.info("Found %d total image files (%d train, %d test)", len(all_files), len(all_train_files), len(all_test_files))

    files_to_check = all_files
    if quick_sample and quick_sample < len(all_files):
        step = max(1, len(all_files) // quick_sample)
        files_to_check = all_files[::step][:quick_sample]
        logger.info("Subsampling %d images for fast audit", len(files_to_check))

    zero_byte_files: list[str] = []
    unreadable_files: list[dict[str, str]] = []
    invalid_format_files: list[str] = []
    grayscale_files: list[str] = []
    suspicious_dims: list[dict[str, Any]] = []

    hashes_by_file: dict[str, str] = {}
    files_by_hash: dict[str, list[str]] = defaultdict(list)
    files_by_size: dict[int, list[str]] = defaultdict(list)

    supported_extensions = {".jpg", ".jpeg", ".png"}

    min_w, min_h = 999999, 999999
    max_w, max_h = 0, 0

    for idx, fpath in enumerate(files_to_check):
        rel_str = str(fpath.relative_to(proj_root) if fpath.is_relative_to(proj_root) else fpath).replace("\\", "/")

        # 1. Format check
        if fpath.suffix.lower() not in supported_extensions:
            invalid_format_files.append(rel_str)
            continue

        # 2. File size & zero-byte check
        try:
            fsize = fpath.stat().st_size
        except OSError as e:
            unreadable_files.append({"file": rel_str, "error": str(e)})
            continue

        if fsize == 0:
            zero_byte_files.append(rel_str)
            continue

        files_by_size[fsize].append(rel_str)

        # 3. Readability & Integrity check
        try:
            with Image.open(fpath) as img:
                img.verify()

            # Re-open after verify() to inspect mode and size
            with Image.open(fpath) as img:
                w, h = img.size
                mode = img.mode

                if mode != "RGB":
                    grayscale_files.append(rel_str)

                if w < 50 or h < 50:
                    suspicious_dims.append({"file": rel_str, "width": w, "height": h})

                min_w = min(min_w, w)
                min_h = min(min_h, h)
                max_w = max(max_w, w)
                max_h = max(max_h, h)

        except Exception as e:  # noqa: BLE001
            unreadable_files.append({"file": rel_str, "error": str(e)})
            continue

        # 4. Hash computation for duplicate detection
        # Optimize: only compute hash for files where same size exists or across all if not huge
        # For Stanford cars (16k images), MD5 is fast (~2-3 sec)
        md5_hash = compute_file_md5(fpath)
        hashes_by_file[rel_str] = md5_hash
        files_by_hash[md5_hash].append(rel_str)

    # 5. Extract duplicates
    duplicate_groups = {h: flist for h, flist in files_by_hash.items() if len(flist) > 1}
    duplicate_pairs_count = sum(len(flist) - 1 for flist in duplicate_groups.values())

    elapsed = round(time.time() - start_time, 2)

    report: dict[str, Any] = {
        "status": "CLEAN" if (len(unreadable_files) == 0 and len(zero_byte_files) == 0) else "ISSUES_FOUND",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_duration_seconds": elapsed,
        "total_images_scanned": len(files_to_check),
        "total_images_in_dataset": len(all_files),
        "corrupted_or_unreadable_count": len(unreadable_files),
        "unreadable_files": unreadable_files,
        "zero_byte_count": len(zero_byte_files),
        "zero_byte_files": zero_byte_files,
        "invalid_format_count": len(invalid_format_files),
        "invalid_format_files": invalid_format_files,
        "grayscale_count": len(grayscale_files),
        "grayscale_files": grayscale_files[:20],  # sample
        "cryptographic_duplicate_groups": len(duplicate_groups),
        "cryptographic_duplicate_pairs": duplicate_pairs_count,
        "duplicate_groups": duplicate_groups,
        "resolution_range": {
            "min_width": min_w if min_w < 999999 else 0,
            "min_height": min_h if min_h < 999999 else 0,
            "max_width": max_w,
            "max_height": max_h,
        },
        "suspicious_dimensions_count": len(suspicious_dims),
        "suspicious_dimensions": suspicious_dims,
    }

    # Save clean dataset report
    clean_report_path = reports_dir / "clean_dataset_report.json"
    with open(clean_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved clean dataset report to %s", clean_report_path)

    # Save invalid files report (even if empty, for audit trail)
    invalid_report_path = reports_dir / "invalid_files_report.json"
    invalid_report_data = {
        "status": report["status"],
        "unreadable_files": unreadable_files,
        "zero_byte_files": zero_byte_files,
        "invalid_format_files": invalid_format_files,
        "suspicious_dimensions": suspicious_dims,
        "duplicate_groups": duplicate_groups,
    }
    with open(invalid_report_path, "w", encoding="utf-8") as f:
        json.dump(invalid_report_data, f, indent=2)
    logger.info("Saved invalid files report to %s", invalid_report_path)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and clean Stanford Cars dataset (non-destructive)")
    parser.add_argument("--root", type=str, default=None, help="Root path to Stanford Cars dataset")
    parser.add_argument("--output", type=str, default=None, help="Directory for reports")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for quick check")
    args = parser.parse_args()

    report = run_image_cleaning_audit(
        dataset_root=args.root,
        output_dir=args.output,
        quick_sample=args.sample,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["corrupted_or_unreadable_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

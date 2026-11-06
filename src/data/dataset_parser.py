"""
Dataset parser and directory integration module for Stanford Cars (Cars196).
Provides robust dataset root configuration, directory layout discovery, and path validation.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard environment variables for dataset root configuration
ENV_STANFORD_CARS_DIR = "STANFORD_CARS_DIR"
ENV_STANFORD_DATASET_ROOT = "STANFORD_DATASET_ROOT"

# Default candidate relative / absolute folder names to search
DEFAULT_CANDIDATE_DIRS: list[str] = [
    "Stanford Dataset",
    "stanford_cars",
    "stanford-cars-dataset",
    "data/stanford_cars",
    "data/Stanford Dataset",
]


@dataclass(frozen=True)
class DatasetManifest:
    """
    Immutable manifest capturing verified paths and summary metrics of the Stanford Cars dataset.
    """

    root_dir: Path
    train_image_dir: Path
    test_image_dir: Path
    devkit_dir: Path
    meta_mat_path: Path
    train_annos_mat_path: Path
    test_annos_mat_path: Path
    test_annos_withlabels_mat_path: Path | None
    num_train_images: int
    num_test_images: int
    is_valid: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        d = asdict(self)
        d["root_dir"] = str(self.root_dir)
        d["train_image_dir"] = str(self.train_image_dir)
        d["test_image_dir"] = str(self.test_image_dir)
        d["devkit_dir"] = str(self.devkit_dir)
        d["meta_mat_path"] = str(self.meta_mat_path)
        d["train_annos_mat_path"] = str(self.train_annos_mat_path)
        d["test_annos_mat_path"] = str(self.test_annos_mat_path)
        d["test_annos_withlabels_mat_path"] = (
            str(self.test_annos_withlabels_mat_path)
            if self.test_annos_withlabels_mat_path
            else None
        )
        return d


def resolve_dataset_root(
    custom_root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """
    Resolve the root directory of the Stanford Cars dataset using configurable priority:
    1. Explicit argument `custom_root` if passed.
    2. Environment variable `STANFORD_CARS_DIR`.
    3. Environment variable `STANFORD_DATASET_ROOT`.
    4. Known candidate subdirectories relative to `base_dir` (default: repository root / cwd).

    Args:
        custom_root: Explicit path to the dataset root folder.
        base_dir: Optional base path for resolving relative candidates.

    Returns:
        Resolved Path object pointing to an existing dataset root directory.

    Raises:
        FileNotFoundError: If the resolved path does not exist or candidate search fails.
    """
    # 1. Explicit argument
    if custom_root is not None:
        p = Path(str(custom_root).strip("\"'")).resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"Specified dataset root directory does not exist: {p}"
            )
        return p

    # 2. Environment variable: STANFORD_CARS_DIR
    env_dir = os.environ.get(ENV_STANFORD_CARS_DIR)
    if env_dir and env_dir.strip():
        p = Path(env_dir.strip("\"'")).resolve()
        if p.is_dir():
            return p
        logger.warning(
            "Environment variable %s points to non-existent directory: %s",
            ENV_STANFORD_CARS_DIR,
            env_dir,
        )

    # 3. Environment variable: STANFORD_DATASET_ROOT
    env_root = os.environ.get(ENV_STANFORD_DATASET_ROOT)
    if env_root and env_root.strip():
        p = Path(env_root.strip("\"'")).resolve()
        if p.is_dir():
            return p
        logger.warning(
            "Environment variable %s points to non-existent directory: %s",
            ENV_STANFORD_DATASET_ROOT,
            env_root,
        )

    # 4. Search candidates relative to base_dir or project root
    search_base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    # Ascend up to 6 levels checking for direct dataset layout or candidates
    curr = search_base
    for _ in range(6):
        if (curr / "cars_train").is_dir() or (curr / "car_devkit").is_dir():
            return curr

        for candidate in DEFAULT_CANDIDATE_DIRS:
            cand_path = (curr / candidate).resolve()
            if cand_path.is_dir():
                return cand_path

        if curr.parent == curr:
            break
        curr = curr.parent

    # Also search relative to this file's repository root
    file_repo_root = Path(__file__).resolve().parent.parent.parent
    if file_repo_root != search_base:
        if (file_repo_root / "cars_train").is_dir() or (
            file_repo_root / "car_devkit"
        ).is_dir():
            return file_repo_root
        for candidate in DEFAULT_CANDIDATE_DIRS:
            cand_path = (file_repo_root / candidate).resolve()
            if cand_path.is_dir():
                return cand_path

    raise FileNotFoundError(
        "Could not locate Stanford Cars dataset root. Looked in environment variables "
        f"('{ENV_STANFORD_CARS_DIR}', '{ENV_STANFORD_DATASET_ROOT}') and candidates: "
        f"{DEFAULT_CANDIDATE_DIRS} relative to '{search_base}'."
    )


class StanfordCarsDatasetParser:
    """
    Dataset integration manager and directory structure parser for Stanford Cars.
    Detects flat/nested image folders, devkit files, processed artifacts, and PyTorch loaders.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        base_dir: str | Path | None = None,
        auto_validate: bool = True,
    ) -> None:
        """
        Initialize the dataset parser and discover dataset paths.

        Args:
            dataset_root: Optional explicit path to the dataset folder.
            base_dir: Optional base path for auto-discovery fallback.
            auto_validate: If True, validates existence of core folders upon initialization.
        """
        self.root_dir: Path = resolve_dataset_root(
            custom_root=dataset_root, base_dir=base_dir
        )
        self._discover_paths()

        if auto_validate:
            self.validate(check_all_images=False)

    def _discover_paths(self) -> None:
        """Discover image folders, devkit, and annotation files handling various nesting patterns."""
        # 1. Training image directory (prioritize directory containing actual .jpg images)
        train_nested = self.root_dir / "cars_train" / "cars_train"
        train_flat = self.root_dir / "cars_train"
        if train_nested.is_dir() and any(train_nested.glob("*.jpg")):
            self.train_image_dir = train_nested
        elif train_flat.is_dir() and any(train_flat.glob("*.jpg")):
            self.train_image_dir = train_flat
        elif train_nested.is_dir():
            self.train_image_dir = train_nested
        elif train_flat.is_dir():
            self.train_image_dir = train_flat
        else:
            self.train_image_dir = train_nested  # default expected

        # 2. Testing image directory (prioritize directory containing actual .jpg images)
        test_nested = self.root_dir / "cars_test" / "cars_test"
        test_flat = self.root_dir / "cars_test"
        if test_nested.is_dir() and any(test_nested.glob("*.jpg")):
            self.test_image_dir = test_nested
        elif test_flat.is_dir() and any(test_flat.glob("*.jpg")):
            self.test_image_dir = test_flat
        elif test_nested.is_dir():
            self.test_image_dir = test_nested
        elif test_flat.is_dir():
            self.test_image_dir = test_flat
        else:
            self.test_image_dir = test_nested  # default expected

        # 3. Devkit directory (handle car_devkit/devkit, devkit, car_devkit, or root directly)
        devkit_nested = self.root_dir / "car_devkit" / "devkit"
        devkit_sub = self.root_dir / "devkit"
        devkit_flat = self.root_dir / "car_devkit"
        if devkit_nested.is_dir():
            self.devkit_dir = devkit_nested
        elif devkit_sub.is_dir():
            self.devkit_dir = devkit_sub
        elif devkit_flat.is_dir():
            self.devkit_dir = devkit_flat
        elif (self.root_dir / "cars_meta.mat").is_file():
            self.devkit_dir = self.root_dir
        else:
            self.devkit_dir = devkit_nested

        # 4. Core MAT annotation files
        self.meta_mat_path = self._find_file(
            ["cars_meta.mat", "devkit/cars_meta.mat"],
            search_dirs=[self.devkit_dir, self.root_dir],
        )
        self.train_annos_mat_path = self._find_file(
            ["cars_train_annos.mat", "devkit/cars_train_annos.mat"],
            search_dirs=[self.devkit_dir, self.root_dir],
        )
        self.test_annos_mat_path = self._find_file(
            ["cars_test_annos.mat", "devkit/cars_test_annos.mat"],
            search_dirs=[self.devkit_dir, self.root_dir],
        )

        # 5. Optional withlabels test annotations (if recovered/available)
        self.test_annos_withlabels_mat_path: Path | None = None
        for d in [self.devkit_dir, self.root_dir]:
            cand = d / "cars_test_annos_withlabels.mat"
            if cand.is_file():
                self.test_annos_withlabels_mat_path = cand
                break

    def _find_file(self, candidate_names: list[str], search_dirs: list[Path]) -> Path:
        """Search for a file among multiple directories and candidate names."""
        for d in search_dirs:
            for name in candidate_names:
                p = (d / name).resolve()
                if p.is_file():
                    return p
        # Fall back to first candidate in first search dir if not yet found
        return (search_dirs[0] / candidate_names[0]).resolve()

    def validate(self, check_all_images: bool = False) -> dict[str, Any]:
        """
        Validate dataset paths and file existence.

        Args:
            check_all_images: If True, verifies all image files exist and can be read.

        Returns:
            Validation report dictionary with status and diagnostic counts.

        Raises:
            FileNotFoundError: If any mandatory directory or annotation file is missing.
        """
        errors: list[str] = []

        if not self.root_dir.is_dir():
            errors.append(f"Root dataset directory does not exist: {self.root_dir}")

        if not self.train_image_dir.is_dir():
            errors.append(
                f"Training image directory does not exist: {self.train_image_dir}"
            )

        if not self.test_image_dir.is_dir():
            errors.append(
                f"Testing image directory does not exist: {self.test_image_dir}"
            )

        if not self.meta_mat_path.is_file():
            errors.append(f"cars_meta.mat does not exist at: {self.meta_mat_path}")

        if not self.train_annos_mat_path.is_file():
            errors.append(
                f"cars_train_annos.mat does not exist at: {self.train_annos_mat_path}"
            )

        if not self.test_annos_mat_path.is_file():
            errors.append(
                f"cars_test_annos.mat does not exist at: {self.test_annos_mat_path}"
            )

        if errors:
            raise FileNotFoundError(
                "Stanford Cars dataset validation failed:\n  " + "\n  ".join(errors)
            )

        train_imgs = self.list_train_images()
        test_imgs = self.list_test_images()

        report: dict[str, Any] = {
            "root_dir": str(self.root_dir),
            "train_image_dir": str(self.train_image_dir),
            "test_image_dir": str(self.test_image_dir),
            "devkit_dir": str(self.devkit_dir),
            "num_train_images": len(train_imgs),
            "num_test_images": len(test_imgs),
            "meta_mat_path": str(self.meta_mat_path),
            "train_annos_mat_path": str(self.train_annos_mat_path),
            "test_annos_mat_path": str(self.test_annos_mat_path),
            "has_test_labels": self.test_annos_withlabels_mat_path is not None,
            "is_valid": True,
        }

        if check_all_images:
            from PIL import Image

            corrupt_train = 0
            for p in train_imgs:
                try:
                    with Image.open(p) as img:
                        _ = img.size
                except (OSError, ValueError):
                    corrupt_train += 1

            corrupt_test = 0
            for p in test_imgs:
                try:
                    with Image.open(p) as img:
                        _ = img.size
                except (OSError, ValueError):
                    corrupt_test += 1

            report["corrupt_train_images"] = corrupt_train
            report["corrupt_test_images"] = corrupt_test
            report["all_images_readable"] = corrupt_train == 0 and corrupt_test == 0

        return report

    def list_train_images(self) -> list[Path]:
        """Return sorted list of all training image Paths."""
        if not self.train_image_dir.is_dir():
            return []
        return sorted(self.train_image_dir.glob("*.jpg"))

    def list_test_images(self) -> list[Path]:
        """Return sorted list of all testing image Paths."""
        if not self.test_image_dir.is_dir():
            return []
        return sorted(self.test_image_dir.glob("*.jpg"))

    def get_manifest(self) -> DatasetManifest:
        """Construct and return a complete DatasetManifest."""
        report = self.validate(check_all_images=False)
        return DatasetManifest(
            root_dir=self.root_dir,
            train_image_dir=self.train_image_dir,
            test_image_dir=self.test_image_dir,
            devkit_dir=self.devkit_dir,
            meta_mat_path=self.meta_mat_path,
            train_annos_mat_path=self.train_annos_mat_path,
            test_annos_mat_path=self.test_annos_mat_path,
            test_annos_withlabels_mat_path=self.test_annos_withlabels_mat_path,
            num_train_images=report["num_train_images"],
            num_test_images=report["num_test_images"],
            is_valid=report["is_valid"],
            details=report,
        )

    def get_processed_data_status(
        self, data_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Inspect the status of existing processed artifacts under data/.

        Args:
            data_dir: Path to the data directory (default: repo root / 'data').

        Returns:
            Dictionary detailing existing processed JSON/CSV/YOLO artifacts.
        """
        if data_dir is None:
            # Try to resolve relative to workspace
            target_data = Path.cwd().resolve() / "data"
        else:
            target_data = Path(data_dir).resolve()

        expected_files = {
            "classes_json": target_data / "classes.json",
            "train_annotations_json": target_data / "train_annotations.json",
            "val_annotations_json": target_data / "val_annotations.json",
            "all_train_annotations_json": target_data / "all_train_annotations.json",
            "normalized_annotations_csv": target_data
            / "processed"
            / "annotations"
            / "normalized_annotations.csv",
            "class_mapping_csv": target_data / "metadata" / "class_mapping.csv",
            "make_mapping_csv": target_data / "metadata" / "make_mapping.csv",
            "model_mapping_csv": target_data / "metadata" / "model_mapping.csv",
            "yolo_dataset_yaml": target_data / "yolo" / "dataset.yaml",
        }

        status: dict[str, Any] = {
            "data_dir": str(target_data),
            "data_dir_exists": target_data.is_dir(),
            "artifacts": {},
        }

        for key, p in expected_files.items():
            exists = p.is_file()
            size_bytes = p.stat().st_size if exists else 0
            status["artifacts"][key] = {
                "path": str(p),
                "exists": exists,
                "size_bytes": size_bytes,
            }

        return status

    def get_dataloader_status(self) -> dict[str, Any]:
        """
        Check availability of existing PyTorch dataset and data loader modules.

        Returns:
            Dictionary with import status and class references.
        """
        status: dict[str, Any] = {
            "module_available": False,
            "dataset_class": None,
            "dataloader_builder": None,
            "error": None,
        }

        try:
            mod = importlib.import_module("src.data.dataset")
            has_dataset = hasattr(mod, "StanfordCarsDataset")
            has_builder = hasattr(mod, "build_dataloaders")
            status["module_available"] = has_dataset and has_builder
            status["dataset_class"] = "StanfordCarsDataset" if has_dataset else None
            status["dataloader_builder"] = "build_dataloaders" if has_builder else None
        except (ImportError, AttributeError, RuntimeError) as e:
            status["error"] = str(e)

        return status


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Inspect and validate Stanford Cars dataset directory structure."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Explicit path to Stanford Cars dataset root.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Perform deep readability checks on all images.",
    )
    args = parser.parse_args()

    ds_parser = StanfordCarsDatasetParser(dataset_root=args.root)
    manifest = ds_parser.get_manifest()
    print("========================================")
    print("Stanford Cars Dataset Manifest")
    print("========================================")
    print(json.dumps(manifest.to_dict(), indent=2))

    if args.check_images:
        print("\nChecking all image files...")
        report = ds_parser.validate(check_all_images=True)
        print(
            f"Train images: {report['num_train_images']} (corrupt: {report.get('corrupt_train_images', 0)})"
        )
        print(
            f"Test images:  {report['num_test_images']} (corrupt: {report.get('corrupt_test_images', 0)})"
        )
        print(f"All readable: {report.get('all_images_readable', True)}")

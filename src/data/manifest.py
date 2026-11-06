"""
Unified Dataset Manifest implementation for Stanford Cars (Cars196).
Builds, validates, serializes, and loads the unified dataset manifest
containing normalized annotations, bounding boxes, taxonomy, and validity flags.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


@dataclass
class ManifestBBox:
    """Bounding box coordinates in 0-indexed pixel space (inclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, int]:
        return {"x1": int(self.x1), "y1": int(self.y1), "x2": int(self.x2), "y2": int(self.y2)}

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class ManifestRecord:
    """Unified Stanford Cars manifest record schema."""

    image_path: str
    image_id: str
    split: str
    class_id: int | None
    class_name: str | None
    make: str | None
    model: str | None
    year: int | None
    bbox: ManifestBBox
    image_width: int
    image_height: int
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-compatible dictionary."""
        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "split": self.split,
            "class_id": int(self.class_id) if self.class_id is not None and pd.notna(self.class_id) else None,
            "class_name": str(self.class_name) if self.class_name is not None and pd.notna(self.class_name) else None,
            "make": str(self.make) if self.make is not None and pd.notna(self.make) else None,
            "model": str(self.model) if self.model is not None and pd.notna(self.model) else None,
            "year": int(self.year) if self.year is not None and pd.notna(self.year) else None,
            "bbox": self.bbox.to_dict(),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "is_valid": bool(self.is_valid),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert record to flat dictionary for CSV serialization."""
        d = self.to_dict()
        bbox = d.pop("bbox")
        d["bbox_x1"] = bbox["x1"]
        d["bbox_y1"] = bbox["y1"]
        d["bbox_x2"] = bbox["x2"]
        d["bbox_y2"] = bbox["y2"]
        return d


def build_unified_manifest(
    source_df_or_path: pd.DataFrame | str | Path | None = None,
    project_root: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> tuple[Path, Path, list[ManifestRecord]]:
    """
    Build the unified dataset manifest from normalized annotations.

    Args:
        source_df_or_path: DataFrame or path to normalized_annotations.csv.
        project_root: Root directory of the repository.
        output_csv: Path to write vehicle_manifest.csv.
        output_json: Path to write vehicle_manifest.json.

    Returns:
        tuple of (Path to CSV, Path to JSON, list of ManifestRecords)
    """
    root = Path(project_root).resolve() if project_root else find_project_root()

    if source_df_or_path is None:
        source_path = root / "data" / "processed" / "annotations" / "normalized_annotations.csv"
        df = pd.read_csv(source_path)
    elif isinstance(source_df_or_path, (str, Path)):
        df = pd.read_csv(source_df_or_path)
    else:
        df = source_df_or_path.copy()

    records: list[ManifestRecord] = []

    for _, row in df.iterrows():
        # Handle nullable taxonomy fields for unlabeled test data
        class_id = int(row["class_id"]) if pd.notna(row.get("class_id")) else None
        class_name = str(row["class_name"]) if pd.notna(row.get("class_name")) else None
        make = str(row["make"]) if pd.notna(row.get("make")) else None
        model = str(row["model"]) if pd.notna(row.get("model")) else None
        year = int(row["year"]) if pd.notna(row.get("year")) else None

        # Pixel bounding box (0-based)
        x1 = int(row["bbox_xmin"])
        y1 = int(row["bbox_ymin"])
        x2 = int(row["bbox_xmax"])
        y2 = int(row["bbox_ymax"])
        img_w = int(row["image_width"])
        img_h = int(row["image_height"])

        # Determine validity: image existence + valid bbox + valid dimensions
        raw_valid = bool(row.get("annotation_valid", True))
        img_p = Path(str(row["image_path"]))
        full_p = img_p if img_p.is_absolute() else (root / img_p)

        bbox_valid = 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h
        dim_valid = img_w > 0 and img_h > 0
        file_valid = full_p.is_file()

        is_valid = bool(raw_valid and bbox_valid and dim_valid and file_valid)

        # Standardize relative image path using forward slashes
        norm_img_path = str(img_p).replace("\\", "/")

        record = ManifestRecord(
            image_path=norm_img_path,
            image_id=str(row["image_id"]),
            split=str(row["split"]),
            class_id=class_id,
            class_name=class_name,
            make=make,
            model=model,
            year=year,
            bbox=ManifestBBox(x1=x1, y1=y1, x2=x2, y2=y2),
            image_width=img_w,
            image_height=img_h,
            is_valid=is_valid,
        )
        records.append(record)

    # Resolve output destinations
    meta_dir = root / "data" / "processed" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    csv_dest = Path(output_csv) if output_csv else meta_dir / "vehicle_manifest.csv"
    json_dest = Path(output_json) if output_json else meta_dir / "vehicle_manifest.json"

    # Save CSV
    df_manifest = pd.DataFrame([r.to_flat_dict() for r in records])
    df_manifest.to_csv(csv_dest, index=False)
    logger.info("Saved unified manifest CSV to %s (%d rows)", csv_dest, len(df_manifest))

    # Save JSON
    json_payload = [r.to_dict() for r in records]
    with open(json_dest, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Saved unified manifest JSON to %s (%d records)", json_dest, len(json_payload))

    return csv_dest, json_dest, records


def load_manifest(json_or_csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load unified manifest records as a list of dictionaries.
    Supports either vehicle_manifest.json or vehicle_manifest.csv.
    """
    if json_or_csv_path is None:
        root = find_project_root()
        json_or_csv_path = root / "data" / "processed" / "metadata" / "vehicle_manifest.json"

    p = Path(json_or_csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # From CSV, reconstruct nested bbox dict
    df = pd.read_csv(p)
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["bbox"] = {
            "x1": int(rec.pop("bbox_x1")),
            "y1": int(rec.pop("bbox_y1")),
            "x2": int(rec.pop("bbox_x2")),
            "y2": int(rec.pop("bbox_y2")),
        }
        # Cast None/NaN properly
        for k in ["class_id", "class_name", "make", "model", "year"]:
            if pd.isna(rec[k]):
                rec[k] = None
        records.append(rec)
    return records


def load_manifest_df(csv_or_json_path: str | Path | None = None) -> pd.DataFrame:
    """Load unified manifest as a flat pandas DataFrame."""
    if csv_or_json_path is None:
        root = find_project_root()
        csv_or_json_path = root / "data" / "processed" / "metadata" / "vehicle_manifest.csv"

    p = Path(csv_or_json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    # If JSON, flatten bbox dict
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat_data = []
    for item in data:
        row = dict(item)
        bbox = row.pop("bbox")
        row["bbox_x1"] = bbox["x1"]
        row["bbox_y1"] = bbox["y1"]
        row["bbox_x2"] = bbox["x2"]
        row["bbox_y2"] = bbox["y2"]
        flat_data.append(row)
    return pd.DataFrame(flat_data)


__all__ = [
    "ManifestBBox",
    "ManifestRecord",
    "build_unified_manifest",
    "load_manifest",
    "load_manifest_df",
]

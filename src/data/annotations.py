"""
Annotation management interface for P5 Vehicle Recognition.
Provides convenient access to normalized annotations, taxonomy mapping,
and bounding-box validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.annotation_parser import (
    ALL_49_MAKES,
    AnnotationParser,
    NormalizedAnnotationRecord,
    TaxonomyParser,
    load_normalized_annotations,
    validate_annotation_integrity,
)
from src.data.dataset_parser import StanfordCarsDatasetParser, resolve_dataset_root
from src.utils.config import find_project_root


def get_normalized_annotations_df(
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load normalized annotations CSV into a pandas DataFrame."""
    if csv_path is None:
        root = find_project_root()
        csv_path = (
            root / "data" / "processed" / "annotations" / "normalized_annotations.csv"
        )
    return pd.read_csv(csv_path)


def get_class_mapping_dict(json_path: Path | str | None = None) -> dict[str, Any]:
    """Load class mapping JSON into a dictionary."""
    if json_path is None:
        root = find_project_root()
        json_path = root / "data" / "processed" / "class_mapping.json"
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Class mapping file not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def export_processed_class_mappings(
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """
    Export class_mapping.json and class_mapping.csv to data/processed/.

    Returns:
        tuple of (json_path, csv_path)
    """
    root = find_project_root()
    if output_dir is None:
        out_p = root / "data" / "processed"
    else:
        out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Ingest classes from metadata or parse fresh
    meta_csv = root / "data" / "metadata" / "class_mapping.csv"
    if meta_csv.is_file():
        df_classes = pd.read_csv(meta_csv)
    else:
        parser = StanfordCarsDatasetParser(base_dir=root)
        ann_parser = AnnotationParser(parser)
        ann_parser.parse_all()
        df_classes = pd.DataFrame([r.to_dict() for r in ann_parser.class_mappings])

    # Save CSV
    csv_target = out_p / "class_mapping.csv"
    df_classes.to_csv(csv_target, index=False)

    # Save JSON (both list of records and by-id lookup)
    json_target = out_p / "class_mapping.json"
    records = df_classes.to_dict(orient="records")
    by_class_id = {str(r["class_id"]): r for r in records}
    mapping_payload = {
        "num_classes": len(records),
        "classes": records,
        "by_class_id": by_class_id,
        "note": "Stanford Cars provides fine-grained image-level class labels and bounding boxes. Make, model, and year are parsed from the verified 196-class taxonomy.",
    }
    with open(json_target, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)

    return json_target, csv_target


__all__ = [
    "ALL_49_MAKES",
    "AnnotationParser",
    "NormalizedAnnotationRecord",
    "StanfordCarsDatasetParser",
    "TaxonomyParser",
    "export_processed_class_mappings",
    "get_class_mapping_dict",
    "get_normalized_annotations_df",
    "load_normalized_annotations",
    "resolve_dataset_root",
    "validate_annotation_integrity",
]

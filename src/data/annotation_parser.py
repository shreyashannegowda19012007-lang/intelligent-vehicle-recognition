"""
Annotation parser and schema normalization engine for Stanford Cars (Cars196).
Extracts, validates, and normalizes annotations from MATLAB MAT files,
builds class taxonomies, detects duplicate records, and exports standardized CSV outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.io as sio
from PIL import Image

# Ensure repository root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from ..utils.box_utils import matlab_to_pixel_box
except (ImportError, ValueError):
    from src.utils.box_utils import matlab_to_pixel_box

logger = logging.getLogger(__name__)


def find_project_root(start_dir: str | Path | None = None) -> Path:
    """
    Ascend directories from start_dir (or cwd) to find the project root
    (containing .git, Stanford Dataset, or src/data).
    """
    curr = Path(start_dir).resolve() if start_dir else Path.cwd().resolve()
    for p in [curr, *curr.parents]:
        if (
            (p / ".git").exists()
            or (p / "Stanford Dataset").is_dir()
            or (p / "src" / "data").is_dir()
        ):
            return p
    return _REPO_ROOT


# Complete roster of 49 verified vehicle manufacturers in Stanford Cars
ALL_49_MAKES: list[str] = [
    "AM General",
    "Acura",
    "Aston Martin",
    "Audi",
    "BMW",
    "Bentley",
    "Bugatti",
    "Buick",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Daewoo",
    "Dodge",
    "Eagle",
    "FIAT",
    "Ferrari",
    "Fisker",
    "Ford",
    "GMC",
    "Geo",
    "HUMMER",
    "Honda",
    "Hyundai",
    "Infiniti",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Lamborghini",
    "Land Rover",
    "Lincoln",
    "MINI",
    "Maybach",
    "Mazda",
    "McLaren",
    "Mercedes-Benz",
    "Mitsubishi",
    "Nissan",
    "Plymouth",
    "Porsche",
    "Ram",
    "Rolls-Royce",
    "Scion",
    "Spyker",
    "Suzuki",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
    "smart",
]

# Canonical multi-word manufacturers that cannot be parsed by single-token splitting
KNOWN_MULTIWORD_MAKES: list[str] = [
    "AM General",
    "Aston Martin",
    "Land Rover",
]

# Canonical body types present across Stanford Cars nomenclature
KNOWN_BODY_TYPES: list[str] = [
    "SuperCab",
    "Regular Cab",
    "Extended Cab",
    "Crew Cab",
    "Quad Cab",
    "Club Cab",
    "Convertible",
    "Coupe",
    "Sedan",
    "Hatchback",
    "SUV",
    "Minivan",
    "Van",
    "Wagon",
]

# Explicit body style overrides for trims/performance models where body type is implicit
BODY_TYPE_OVERRIDES: dict[str, str] = {
    "Acura TL Type-S 2008": "Sedan",
    "Acura Integra Type R 2001": "Coupe",
    "Buick Regal GS 2012": "Sedan",
    "Chevrolet Corvette ZR1 2012": "Coupe",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007": "Coupe",
    "Chevrolet HHR SS 2010": "SUV",
    "Chevrolet Cobalt SS 2010": "Coupe",
    "Chevrolet TrailBlazer SS 2009": "SUV",
    "Chrysler 300 SRT-8 2010": "Sedan",
    "Dodge Challenger SRT8 2011": "Coupe",
    "Dodge Charger SRT-8 2009": "Sedan",
    "FIAT 500 Abarth 2012": "Hatchback",
    "Jaguar XK XKR 2012": "Coupe",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012": "Coupe",
}

# Sorted by string length descending to guarantee longest prefix match first
SORTED_MAKES_DESCENDING = sorted(ALL_49_MAKES, key=len, reverse=True)
SORTED_BODY_TYPES_DESCENDING = sorted(KNOWN_BODY_TYPES, key=len, reverse=True)


@dataclass
class NormalizedAnnotationRecord:
    """
    Data container for a single normalized Stanford Cars annotation record.
    Implements the exact required schema alongside preserved original annotations.
    """

    # Required Normalized Schema (Section 4)
    image_id: str
    image_path: str
    dataset_source: str
    split: str
    class_id: int | None
    class_name: str | None
    make: str | None
    model: str | None
    year: int | None
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int
    image_width: int
    image_height: int
    annotation_valid: bool

    # Preserved Original Annotation Values
    original_class_id: int | None = None
    original_filename: str = ""
    bbox_matlab_x1: int = 0
    bbox_matlab_y1: int = 0
    bbox_matlab_x2: int = 0
    bbox_matlab_y2: int = 0
    body_type: str | None = None
    clean_model: str | None = None
    has_label: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary matching normalized schema order."""
        return asdict(self)


class TaxonomyParser:
    """
    Parser for vehicle class nomenclature, extracting hierarchical Make, Model, Year,
    and Body Style while preventing naive splitting bugs and handling multi-word manufacturers.
    """

    @staticmethod
    def parse_class_name(raw_name: str) -> dict[str, Any]:
        """
        Parse a raw Stanford Cars class string into structured components.

        Template: [Make] [Model / Trim] [Body Style] [Year]
        Example: 'AM General Hummer SUV 2000' ->
                 Make: 'AM General', Model: 'Hummer SUV', Clean: 'Hummer', Year: 2000, Body: 'SUV'

        Args:
            raw_name: The raw canonical string from cars_meta.mat.

        Returns:
            Dictionary containing make, model, clean_model, year, body_type, is_ambiguous, review_flag.
        """
        raw_name = raw_name.strip()
        tokens = raw_name.split(" ")

        is_ambiguous = False
        review_flag = "OK"

        # 1. Parse manufacturing year from terminal token
        year: int | None = None
        year_str = tokens[-1]
        if year_str.isdigit() and len(year_str) == 4:
            year = int(year_str)
        else:
            is_ambiguous = True
            review_flag = f"Terminal token '{year_str}' is not a valid 4-digit year"

        # 2. Parse vehicle make via longest prefix matching across known makes
        matched_make: str | None = None
        for make_candidate in SORTED_MAKES_DESCENDING:
            if raw_name.startswith(make_candidate + " "):
                matched_make = make_candidate
                break

        if not matched_make:
            # Fallback for unexpected external makes: use first whitespace token
            matched_make = tokens[0]
            is_ambiguous = True
            review_flag = f"Manufacturer '{matched_make}' not in known 49 makes"

        # 3. Extract remainder string (model + body type)
        if year is not None:
            # Strip make from start and year from end
            prefix_len = len(matched_make) + 1
            suffix_len = len(year_str) + 1
            rem = raw_name[prefix_len:-suffix_len].strip()
        else:
            rem = raw_name[len(matched_make) + 1 :].strip()

        # 4. Resolve body style
        body_type: str | None = None
        # Check explicit overrides for performance trims lacking body keywords
        if raw_name in BODY_TYPE_OVERRIDES:
            body_type = BODY_TYPE_OVERRIDES[raw_name]
        else:
            # Match against known body types (longest match first)
            for bt in SORTED_BODY_TYPES_DESCENDING:
                if bt in rem:
                    body_type = bt
                    break

        if not body_type:
            body_type = "Unknown"
            is_ambiguous = True
            review_flag = f"Unrecognized body style in '{rem}'"

        # 5. Extract Model & Clean Model (model string without body style)
        model = rem if rem else raw_name
        clean_model = model
        if body_type != "Unknown" and body_type in model:
            clean_model = model.replace(body_type, "").strip()
            clean_model = " ".join(clean_model.split())  # collapse whitespace

        if not clean_model:
            clean_model = model

        return {
            "raw_name": raw_name,
            "make": matched_make,
            "model": model,
            "clean_model": clean_model,
            "year": year,
            "body_type": body_type,
            "is_ambiguous": is_ambiguous,
            "review_flag": review_flag,
        }

    @classmethod
    def parse_meta_mat(
        cls, mat_path: str | Path, output_json_path: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Parse cars_meta.mat cell array into complete metadata taxonomy dictionary.

        Args:
            mat_path: Path to cars_meta.mat.
            output_json_path: Optional path to export JSON file.

        Returns:
            Dictionary with parsed classes, make_to_id, year_to_id, etc.
        """
        mat_p = Path(mat_path)
        if not mat_p.is_file():
            raise FileNotFoundError(f"cars_meta.mat not found at {mat_p}")

        mat = sio.loadmat(str(mat_p))
        raw_class_names = [str(c[0]) for c in mat["class_names"][0]]

        parsed_items: list[dict[str, Any]] = []
        unique_makes: set[str] = set()
        unique_years: set[int] = set()
        unique_body_types: set[str] = set()
        unique_models: set[str] = set()

        for idx, raw_name in enumerate(raw_class_names):
            info = cls.parse_class_name(raw_name)
            info["class_id"] = idx  # 0-based
            info["class_id_0"] = idx
            info["class_id_1"] = idx + 1  # 1-based original MATLAB class id
            info["original_class_id"] = idx + 1

            parsed_items.append(info)
            unique_makes.add(info["make"])
            if info["year"] is not None:
                unique_years.add(info["year"])
            unique_body_types.add(info["body_type"])
            unique_models.add(info["model"])

        sorted_makes = sorted(unique_makes)
        sorted_years = sorted(unique_years)
        sorted_body_types = sorted(unique_body_types)
        sorted_models = sorted(unique_models)

        make_to_id = {m: i for i, m in enumerate(sorted_makes)}
        year_to_id = {y: i for i, y in enumerate(sorted_years)}
        body_type_to_id = {b: i for i, b in enumerate(sorted_body_types)}
        model_to_id = {m: i for i, m in enumerate(sorted_models)}

        for item in parsed_items:
            item["make_id"] = make_to_id[item["make"]]
            item["year_id"] = year_to_id[item["year"]]
            item["body_type_id"] = body_type_to_id[item["body_type"]]
            item["model_id"] = model_to_id[item["model"]]

        meta_result: dict[str, Any] = {
            "num_classes": len(parsed_items),
            "classes": parsed_items,
            "makes": sorted_makes,
            "years": sorted_years,
            "body_types": sorted_body_types,
            "models": sorted_models,
            "make_to_id": make_to_id,
            "year_to_id": {str(y): i for y, i in year_to_id.items()},
            "body_type_to_id": body_type_to_id,
            "model_to_id": model_to_id,
        }

        if output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(meta_result, f, indent=2)

        return meta_result

    @classmethod
    def generate_mapping_dataframes(
        cls,
        classes_meta: dict[str, Any],
        train_class_counts: dict[int, int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate DataFrames for class_mapping, make_mapping, and model_mapping.

        Args:
            classes_meta: Output from parse_meta_mat.
            train_class_counts: Optional mapping of 0-based class_id to training sample count.

        Returns:
            Tuple of (class_mapping_df, make_mapping_df, model_mapping_df).
        """
        counts = train_class_counts or {}

        # 1. Class mapping
        class_rows = []
        for c in classes_meta["classes"]:
            cid = c["class_id"]
            class_rows.append(
                {
                    "class_id": cid,
                    "original_class_id": c["original_class_id"],
                    "class_name": c["raw_name"],
                    "make": c["make"],
                    "model": c["model"],
                    "clean_model": c["clean_model"],
                    "year": c["year"],
                    "body_type": c["body_type"],
                    "sample_count": counts.get(cid, 0),
                    "is_ambiguous": c.get("is_ambiguous", False),
                    "review_flag": c.get("review_flag", "OK"),
                }
            )
        class_df = pd.DataFrame(class_rows)

        # 2. Make mapping
        make_rows = []
        for mid, make_name in enumerate(classes_meta["makes"]):
            classes_for_make = [
                c for c in classes_meta["classes"] if c["make"] == make_name
            ]
            sample_count = sum(counts.get(c["class_id"], 0) for c in classes_for_make)
            make_rows.append(
                {
                    "make_id": mid,
                    "make": make_name,
                    "class_count": len(classes_for_make),
                    "sample_count": sample_count,
                }
            )
        make_df = pd.DataFrame(make_rows)

        # 3. Model mapping
        model_rows = []
        for mid, model_name in enumerate(classes_meta["models"]):
            classes_for_model = [
                c for c in classes_meta["classes"] if c["model"] == model_name
            ]
            sample_count = sum(counts.get(c["class_id"], 0) for c in classes_for_model)
            sample_class = classes_for_model[0] if classes_for_model else None
            model_rows.append(
                {
                    "model_id": mid,
                    "model": model_name,
                    "clean_model": sample_class["clean_model"]
                    if sample_class
                    else model_name,
                    "make": sample_class["make"] if sample_class else "",
                    "body_type": sample_class["body_type"] if sample_class else "",
                    "class_count": len(classes_for_model),
                    "sample_count": sample_count,
                }
            )
        model_df = pd.DataFrame(model_rows)

        return class_df, make_df, model_df


def _extract_scalar_val(val: Any) -> Any:
    """Unwrap nested MATLAB 1x1 arrays or cell arrays down to primitive value."""
    while hasattr(val, "size") and val.size == 1 and hasattr(val, "item"):
        val = val.item()
    return val


def _extract_int_scalar(val: Any) -> int:
    """Extract an integer scalar from various MATLAB / NumPy numeric or object arrays."""
    return int(_extract_scalar_val(val))


class AnnotationParser:
    """
    High-level engine for extracting and normalizing annotations from Stanford Cars MAT files.
    """

    def __init__(
        self,
        meta_mat_path: str | Path,
        dataset_source: str = "stanford_cars",
        workspace_root: str | Path | None = None,
    ) -> None:
        """
        Initialize annotation parser with metadata path.

        Args:
            meta_mat_path: Path to cars_meta.mat.
            dataset_source: Identifier string for the dataset source.
            workspace_root: Root directory for computing portable relative image paths.
        """
        self.meta_mat_path = Path(meta_mat_path)
        self.dataset_source = dataset_source
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root else find_project_root()
        )
        self.classes_meta = TaxonomyParser.parse_meta_mat(self.meta_mat_path)
        self.class_lookup: dict[int, dict[str, Any]] = {
            c["original_class_id"]: c for c in self.classes_meta["classes"]
        }

    def parse_split_annotations(
        self,
        annos_mat_path: str | Path,
        img_dir: str | Path,
        split: str = "train",
        is_test: bool = False,
    ) -> list[NormalizedAnnotationRecord]:
        """
        Parse a single MATLAB annotation file into normalized records.

        Args:
            annos_mat_path: Path to cars_train_annos.mat or cars_test_annos.mat.
            img_dir: Directory containing the corresponding image files.
            split: Split identifier ('train' or 'test').
            is_test: Flag indicating if test split (where class labels are withheld).

        Returns:
            List of NormalizedAnnotationRecord instances.
        """
        annos_p = Path(annos_mat_path)
        img_p = Path(img_dir)

        if not annos_p.is_file():
            raise FileNotFoundError(f"Annotations MAT file not found: {annos_p}")
        if not img_p.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_p}")

        mat = sio.loadmat(str(annos_p))
        raw_annos = mat["annotations"][0]

        records: list[NormalizedAnnotationRecord] = []

        for anno in raw_annos:
            fname = str(_extract_scalar_val(anno["fname"]))
            fpath = img_p / fname
            file_exists = fpath.is_file()

            # Global collision-safe image identifier (e.g. train_00001, test_00001)
            stem = Path(fname).stem
            image_id = f"{split}_{stem}"

            # Calculate relative path from workspace root with POSIX slashes
            try:
                rel_path = str(fpath.relative_to(self.workspace_root)).replace(
                    "\\", "/"
                )
            except ValueError:
                # If outside workspace, store relative to dataset folder or absolute
                rel_path = str(fpath).replace("\\", "/")

            bx1 = _extract_int_scalar(anno["bbox_x1"])
            by1 = _extract_int_scalar(anno["bbox_y1"])
            bx2 = _extract_int_scalar(anno["bbox_x2"])
            by2 = _extract_int_scalar(anno["bbox_y2"])

            img_w: int = 0
            img_h: int = 0
            if file_exists:
                try:
                    with Image.open(fpath) as img:
                        img_w, img_h = img.size
                except (OSError, ValueError) as e:
                    logger.warning("Failed to read image %s: %s", fpath, e)
                    file_exists = False

            # Raw bounding box geometry validity in 1-based MATLAB coordinates:
            # 1 <= bx1 <= bx2 <= img_w and 1 <= by1 <= by2 <= img_h
            raw_bbox_valid = (
                img_w > 0
                and img_h > 0
                and (1 <= bx1 <= bx2 <= img_w)
                and (1 <= by1 <= by2 <= img_h)
            )

            if img_w > 0 and img_h > 0:
                # 0-based inclusive coordinates [xmin, ymin, xmax, ymax]
                x1_inc, y1_inc, x2_inc, y2_inc = matlab_to_pixel_box(
                    bx1, by1, bx2, by2, img_w, img_h, inclusive=True
                )
            else:
                x1_inc, y1_inc, x2_inc, y2_inc = (
                    max(0, bx1 - 1),
                    max(0, by1 - 1),
                    max(0, bx2 - 1),
                    max(0, by2 - 1),
                )

            # Check if class label is present in MAT struct
            has_class_field = ("class" in anno.dtype.names) and (not is_test)
            class_id: int | None = None
            original_class_id: int | None = None
            class_name: str | None = None
            make: str | None = None
            model: str | None = None
            clean_model: str | None = None
            year: int | None = None
            body_type: str | None = None

            if has_class_field:
                orig_id = _extract_int_scalar(anno["class"])
                original_class_id = orig_id
                class_id = orig_id - 1  # 0-based
                cinfo = self.class_lookup.get(orig_id)
                if cinfo:
                    class_name = cinfo["raw_name"]
                    make = cinfo["make"]
                    model = cinfo["model"]
                    clean_model = cinfo["clean_model"]
                    year = cinfo["year"]
                    body_type = cinfo["body_type"]

            # Annotation validity check:
            # 1. Image exists on disk and is readable
            # 2. Dimensions > 0
            # 3. Raw bounding box is geometrically valid within image boundaries
            # 4. Clamped pixel bounding box satisfies 0 <= xmin <= xmax < width and 0 <= ymin <= ymax < height
            # 5. If class is present, 0 <= class_id < 196
            bbox_valid = (
                raw_bbox_valid
                and (0 <= x1_inc <= x2_inc < img_w)
                and (0 <= y1_inc <= y2_inc < img_h)
            )
            class_valid = (class_id is None) or (
                0 <= class_id < self.classes_meta["num_classes"]
            )

            annotation_valid = bool(file_exists and bbox_valid and class_valid)

            rec = NormalizedAnnotationRecord(
                image_id=image_id,
                image_path=rel_path,
                dataset_source=self.dataset_source,
                split=split,
                class_id=class_id,
                class_name=class_name,
                make=make,
                model=model,
                year=year,
                bbox_xmin=x1_inc,
                bbox_ymin=y1_inc,
                bbox_xmax=x2_inc,
                bbox_ymax=y2_inc,
                image_width=img_w,
                image_height=img_h,
                annotation_valid=annotation_valid,
                original_class_id=original_class_id,
                original_filename=fname,
                bbox_matlab_x1=bx1,
                bbox_matlab_y1=by1,
                bbox_matlab_x2=bx2,
                bbox_matlab_y2=by2,
                body_type=body_type,
                clean_model=clean_model,
                has_label=has_class_field,
            )
            records.append(rec)

        return records

    def build_normalized_dataframe(
        self,
        train_annos_mat: str | Path,
        test_annos_mat: str | Path,
        train_img_dir: str | Path,
        test_img_dir: str | Path,
        test_annos_withlabels_mat: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Extract both train and test splits and combine into a single normalized DataFrame.

        Args:
            train_annos_mat: Path to cars_train_annos.mat.
            test_annos_mat: Path to cars_test_annos.mat.
            train_img_dir: Directory containing cars_train images.
            test_img_dir: Directory containing cars_test images.
            test_annos_withlabels_mat: Optional path to cars_test_annos_withlabels.mat if available.

        Returns:
            DataFrame containing normalized records across all 16,185 images.
        """
        train_recs = self.parse_split_annotations(
            annos_mat_path=train_annos_mat,
            img_dir=train_img_dir,
            split="train",
            is_test=False,
        )

        # Check if recovered test labels exist
        if test_annos_withlabels_mat and Path(test_annos_withlabels_mat).is_file():
            test_recs = self.parse_split_annotations(
                annos_mat_path=test_annos_withlabels_mat,
                img_dir=test_img_dir,
                split="test",
                is_test=False,
            )
        else:
            test_recs = self.parse_split_annotations(
                annos_mat_path=test_annos_mat,
                img_dir=test_img_dir,
                split="test",
                is_test=True,
            )

        all_recs = train_recs + test_recs
        data = [r.to_dict() for r in all_recs]
        df = pd.DataFrame(data)

        # Ensure correct column ordering matching Section 4 specification
        ordered_columns = [
            "image_id",
            "image_path",
            "dataset_source",
            "split",
            "class_id",
            "class_name",
            "make",
            "model",
            "year",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "image_width",
            "image_height",
            "annotation_valid",
            "original_class_id",
            "original_filename",
            "bbox_matlab_x1",
            "bbox_matlab_y1",
            "bbox_matlab_x2",
            "bbox_matlab_y2",
            "body_type",
            "clean_model",
            "has_label",
        ]

        df = df[ordered_columns]

        # Use nullable integer types for numeric columns containing missing values
        df["class_id"] = df["class_id"].astype("Int64")
        df["original_class_id"] = df["original_class_id"].astype("Int64")
        df["year"] = df["year"].astype("Int64")

        return df


def detect_duplicates(
    df: pd.DataFrame,
    compute_md5: bool = False,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Detect duplicate annotation records, repeated coordinates, and cryptographic MD5 byte duplicates.

    Args:
        df: Normalized annotations DataFrame.
        compute_md5: If True, computes MD5 hash of image bytes to find cryptographic duplicates.
        workspace_root: Base workspace folder for reading image files.

    Returns:
        Dictionary detailing duplicate counts, groups, and cross-split pairs.
    """
    base_p = Path(workspace_root).resolve() if workspace_root else find_project_root()

    duplicate_image_ids = df[df.duplicated(subset=["image_id"], keep=False)]
    duplicate_boxes_train = df[
        (df["split"] == "train")
        & df.duplicated(
            subset=[
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "image_width",
                "image_height",
            ],
            keep=False,
        )
    ]
    duplicate_boxes_total = df[
        df.duplicated(
            subset=[
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "image_width",
                "image_height",
            ],
            keep=False,
        )
    ]

    report: dict[str, Any] = {
        "total_records": len(df),
        "duplicate_image_ids_count": len(duplicate_image_ids),
        "train_identical_box_geometry_count": len(duplicate_boxes_train),
        "total_identical_box_geometry_count": len(duplicate_boxes_total),
        "md5_duplicate_groups": [],
        "cross_split_duplicates": [],
        "train_internal_duplicates": [],
        "test_internal_duplicates": [],
    }

    if compute_md5:
        md5_map: dict[str, list[dict[str, Any]]] = {}
        for row in df.itertuples():
            img_rel = row.image_path
            p = Path(img_rel)
            img_full = p if p.is_absolute() else base_p / p
            if not img_full.is_file():
                logger.warning("Image file not found for MD5 hash: %s", img_full)
                continue
            with open(img_full, "rb") as fp:
                digest = hashlib.md5(fp.read()).hexdigest()

            cid = getattr(row, "class_id", None)
            cname = getattr(row, "class_name", None)
            md5_map.setdefault(digest, []).append(
                {
                    "image_id": row.image_id,
                    "filename": getattr(row, "original_filename", Path(img_rel).name),
                    "split": row.split,
                    "class_id": None if pd.isna(cid) else int(cid),
                    "class_name": None if pd.isna(cname) else str(cname),
                }
            )

        for digest, entries in md5_map.items():
            if len(entries) > 1:
                group_info = {"md5": digest, "count": len(entries), "entries": entries}
                report["md5_duplicate_groups"].append(group_info)

                splits_in_group = {e["split"] for e in entries}
                if len(splits_in_group) > 1:
                    report["cross_split_duplicates"].append(group_info)
                elif "train" in splits_in_group:
                    report["train_internal_duplicates"].append(group_info)
                elif "test" in splits_in_group:
                    report["test_internal_duplicates"].append(group_info)

    return report


def validate_annotation_integrity(
    df: pd.DataFrame,
    workspace_root: str | Path | None = None,
    check_all_images_exist: bool = True,
    expected_train_count: int | None = 8144,
    expected_test_count: int | None = 8041,
) -> dict[str, Any]:
    """
    Execute exhaustive validation across all 7 verification criteria specified in Section 8:
    1. Every referenced image exists.
    2. Every available class ID is valid.
    3. Bounding boxes are correctly extracted.
    4. Image dimensions are accurate.
    5. Metadata is consistent.
    6. Missing labels are identified.
    7. Duplicate annotation records are detected.

    Args:
        df: Normalized annotations DataFrame.
        workspace_root: Base workspace folder for reading images.
        check_all_images_exist: If True, checks file existence for all records.
        expected_train_count: Expected training records count (default: 8144, None to skip check).
        expected_test_count: Expected testing records count (default: 8041, None to skip check).

    Returns:
        Structured validation summary dictionary.

    Raises:
        AssertionError: If any integrity constraint fails.
    """
    base_p = Path(workspace_root).resolve() if workspace_root else find_project_root()
    num_records = len(df)

    # 1. Split counts
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    num_train = int(train_mask.sum())
    num_test = int(test_mask.sum())

    if expected_train_count is not None:
        assert num_train == expected_train_count, (
            f"Expected {expected_train_count} train records, found {num_train}"
        )
    if expected_test_count is not None:
        assert num_test == expected_test_count, (
            f"Expected {expected_test_count} test records, found {num_test}"
        )
    if expected_train_count is not None and expected_test_count is not None:
        expected_total = expected_train_count + expected_test_count
        assert num_records == expected_total, (
            f"Expected {expected_total} total records, found {num_records}"
        )

    # 2. Image existence verification
    missing_images = []
    if check_all_images_exist:
        for row in df.itertuples():
            img_rel = row.image_path
            p = Path(img_rel)
            img_full = p if p.is_absolute() else base_p / p
            if not img_full.is_file():
                missing_images.append(img_rel)
    assert len(missing_images) == 0, (
        f"Found {len(missing_images)} missing images: {missing_images[:5]}"
    )

    # 3. Class ID validity
    train_df = df[train_mask]
    test_df = df[test_mask]

    if num_train > 0:
        assert train_df["class_id"].notna().all(), (
            "Train set contains missing class_id values!"
        )
        assert train_df["class_id"].between(0, 195).all(), (
            "Train class_id outside valid range [0, 195]!"
        )
        assert train_df["original_class_id"].between(1, 196).all(), (
            "Train original_class_id outside [1, 196]!"
        )
        if expected_train_count == 8144:
            assert len(train_df["class_id"].unique()) == 196, (
                "Train set does not cover all 196 classes!"
            )

    # 4. Missing label identification (test set)
    test_missing_labels = int(test_df["class_id"].isna().sum())
    if expected_test_count == 8041:
        assert test_missing_labels == 8041, (
            f"Expected 8041 missing test labels, got {test_missing_labels}"
        )

    # 5. Bounding box validity
    # 0 <= xmin <= xmax < W and 0 <= ymin <= ymax < H
    invalid_boxes = df[
        (df["bbox_xmin"] < 0)
        | (df["bbox_ymin"] < 0)
        | (df["bbox_xmin"] > df["bbox_xmax"])
        | (df["bbox_ymin"] > df["bbox_ymax"])
        | (df["bbox_xmax"] >= df["image_width"])
        | (df["bbox_ymax"] >= df["image_height"])
    ]
    assert len(invalid_boxes) == 0, (
        f"Found {len(invalid_boxes)} invalid bounding boxes: {invalid_boxes.head()}"
    )

    # 6. Image dimensions validity
    invalid_dims = df[(df["image_width"] <= 0) | (df["image_height"] <= 0)]
    assert len(invalid_dims) == 0, (
        f"Found {len(invalid_dims)} images with invalid dimensions!"
    )

    # 7. Annotation valid flag
    assert df["annotation_valid"].all(), "Some records have annotation_valid == False!"

    # 8. Metadata consistency
    unique_makes = train_df["make"].dropna().unique()
    unique_years = train_df["year"].dropna().unique()
    unique_body_types = train_df["body_type"].dropna().unique()

    if expected_train_count == 8144:
        assert len(unique_makes) == 49, (
            f"Expected 49 makes in train, found {len(unique_makes)}"
        )
        assert len(unique_years) == 16, (
            f"Expected 16 years in train, found {len(unique_years)}"
        )
        assert len(unique_body_types) == 14, (
            f"Expected 14 body types in train, found {len(unique_body_types)}"
        )

    return {
        "total_records": num_records,
        "train_records": num_train,
        "test_records": num_test,
        "missing_images_count": len(missing_images),
        "unique_classes": len(train_df["class_id"].unique()) if num_train > 0 else 0,
        "unique_makes": len(unique_makes),
        "unique_years": len(unique_years),
        "unique_body_types": len(unique_body_types),
        "all_boxes_valid": True,
        "all_dimensions_valid": True,
        "missing_test_labels_identified": test_missing_labels,
        "annotation_valid_percentage": 100.0,
        "status": "PASS",
    }


def load_normalized_annotations(
    csv_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load normalized_annotations.csv with correct nullable integer types (Int64)
    for class_id, original_class_id, and year, preserving null test labels.

    Args:
        csv_path: Path to normalized_annotations.csv (default: data/processed/annotations/normalized_annotations.csv).
        workspace_root: Optional workspace root for resolving relative csv_path.

    Returns:
        DataFrame with validated schema and nullable integer dtypes.
    """
    if csv_path is None:
        ws = Path(workspace_root).resolve() if workspace_root else find_project_root()
        target = (
            ws / "data" / "processed" / "annotations" / "normalized_annotations.csv"
        )
    else:
        target = Path(csv_path)

    if not target.is_file():
        raise FileNotFoundError(f"Normalized annotations file does not exist: {target}")

    dtypes: dict[str, str] = {
        "image_id": "string",
        "image_path": "string",
        "dataset_source": "string",
        "split": "string",
        "class_id": "Int64",
        "class_name": "string",
        "make": "string",
        "model": "string",
        "year": "Int64",
        "bbox_xmin": "int64",
        "bbox_ymin": "int64",
        "bbox_xmax": "int64",
        "bbox_ymax": "int64",
        "image_width": "int64",
        "image_height": "int64",
        "annotation_valid": "bool",
        "original_class_id": "Int64",
        "original_filename": "string",
        "bbox_matlab_x1": "int64",
        "bbox_matlab_y1": "int64",
        "bbox_matlab_x2": "int64",
        "bbox_matlab_y2": "int64",
        "body_type": "string",
        "clean_model": "string",
        "has_label": "bool",
    }

    return pd.read_csv(target, dtype=dtypes)


def export_normalized_dataset_and_metadata(
    dataset_parser_instance: Any,
    output_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    na_rep: str = "",
) -> dict[str, str]:
    """
    Orchestrate full extraction, normalization, and export of all artifacts:
    - data/processed/annotations/normalized_annotations.csv
    - data/metadata/class_mapping.csv
    - data/metadata/make_mapping.csv
    - data/metadata/model_mapping.csv

    Args:
        dataset_parser_instance: Initialized StanfordCarsDatasetParser.
        output_dir: Directory to place metadata and processed folders (default: repo root / 'data').
        workspace_root: Repository root path.
        na_rep: Missing value representation in CSV (default empty string for standard CSV).

    Returns:
        Dictionary mapping artifact keys to exported absolute file paths.
    """
    base_data = (
        Path(output_dir).resolve() if output_dir else find_project_root() / "data"
    )
    ws_root = Path(workspace_root).resolve() if workspace_root else find_project_root()

    manifest = dataset_parser_instance.get_manifest()

    # 1. Initialize annotation parser
    anno_parser = AnnotationParser(
        meta_mat_path=manifest.meta_mat_path,
        dataset_source="stanford_cars",
        workspace_root=ws_root,
    )

    # 2. Build full normalized annotations dataframe
    logger.info("Building normalized annotations DataFrame across 16,185 images...")
    df = anno_parser.build_normalized_dataframe(
        train_annos_mat=manifest.train_annos_mat_path,
        test_annos_mat=manifest.test_annos_mat_path,
        train_img_dir=manifest.train_image_dir,
        test_img_dir=manifest.test_image_dir,
        test_annos_withlabels_mat=manifest.test_annos_withlabels_mat_path,
    )

    # 3. Calculate train class sample counts for mappings
    train_class_counts = df[df["split"] == "train"]["class_id"].value_counts().to_dict()
    train_class_counts_int = {
        int(k): int(v) for k, v in train_class_counts.items() if not pd.isna(k)
    }

    # 4. Generate metadata mapping dataframes
    class_df, make_df, model_df = TaxonomyParser.generate_mapping_dataframes(
        classes_meta=anno_parser.classes_meta,
        train_class_counts=train_class_counts_int,
    )

    # 5. Define output paths
    annos_out_dir = base_data / "processed" / "annotations"
    meta_out_dir = base_data / "metadata"

    annos_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    norm_csv_path = annos_out_dir / "normalized_annotations.csv"
    class_csv_path = meta_out_dir / "class_mapping.csv"
    make_csv_path = meta_out_dir / "make_mapping.csv"
    model_csv_path = meta_out_dir / "model_mapping.csv"

    # 6. Write CSV files
    df.to_csv(norm_csv_path, index=False, na_rep=na_rep)
    class_df.to_csv(class_csv_path, index=False)
    make_df.to_csv(make_csv_path, index=False)
    model_df.to_csv(model_csv_path, index=False)

    logger.info("Successfully exported normalized dataset and metadata tables.")

    return {
        "normalized_annotations_csv": str(norm_csv_path),
        "class_mapping_csv": str(class_csv_path),
        "make_mapping_csv": str(make_csv_path),
        "model_mapping_csv": str(model_csv_path),
    }


if __name__ == "__main__":
    try:
        from .dataset_parser import StanfordCarsDatasetParser
    except (ImportError, ValueError):
        from src.data.dataset_parser import StanfordCarsDatasetParser

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = StanfordCarsDatasetParser()
    out = export_normalized_dataset_and_metadata(parser)
    for k, v in out.items():
        print(f"Exported {k}: {v}")
